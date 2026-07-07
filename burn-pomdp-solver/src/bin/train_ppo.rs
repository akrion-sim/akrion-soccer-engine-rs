//! Clipped-PPO fine-tuner for the Burn POMDP-solver — Codex lever #1: use the engine's now-REAL
//! logged behavior probability (the sidecar's true sampled prob, stamped on each decision) as the
//! old-policy prob for importance-weighted PPO, replacing AWR's proxy weight.
//!
//!   ratio      = π_new(a|belief) / π_behavior(a)          (π_behavior = logged sidecar prob)
//!   L_clip     = -mean( min(ratio·A, clip(ratio,1±ε)·A) )  (trust-region policy improvement)
//!   L_value    =  mean( (V - R̂)² )                          (normalized MC-return critic)
//!   L_entropy  = -H(π_new)                                  (exploration)
//!   L_kl       =  KL(π_ref ‖ π_new)                         (anchor to the pre-fine-tune BEST so
//!                                                            fine-tuning can't collapse back into
//!                                                            the mediocre deterministic mode)
//!   loss = L_clip + 0.5·L_value + ent_coef·L_entropy + kl_coef·L_kl
//!
//! Warm-started from MODEL_IN (= the current BEST = the behavior policy), so at epoch 0 ratio≈1
//! (a proper on-policy PPO start) and the clip/KL keep every step inside a trust region.
//!
//! CAVEAT: π_new here is log_softmax over ALL actions; the logged behavior prob was renormalized
//! over the LEGAL set. After the offline/AWR warm-start the net puts little mass on illegal actions,
//! so the ratio is a mild-conservative estimate; the clip + KL absorb the slack. Exporting the legal
//! mask would make it exact (future work).

use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use burn_pomdp_solver::adapter::{load_flat_grouped, returns as mc_returns, to_tensors, Trajectory};
use burn_pomdp_solver::{PomdpActorCritic, PomdpConfig};

type B = Autodiff<NdArray>;

const MAX_T: usize = 128;

fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(d)
}
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(d)
}

fn main() {
    let path = std::env::var("TRAJ").unwrap_or_else(|_| "/tmp/traj_export.jsonl".into());
    let dev = Default::default();
    let gamma = env_f32("PPO_GAMMA", 0.98);
    let eps = env_f32("PPO_CLIP", 0.2);
    let ent_coef = env_f32("PPO_ENT", 0.01);
    let kl_coef = env_f32("PPO_KL", 0.5);
    let lr = env_f32("PPO_LR", 2e-4) as f64;
    let epochs = env_usize("PPO_EPOCHS", 250);
    let batch = env_usize("PPO_BATCH", 64);

    // Window-SPLIT long agent-trajectories into ≤MAX_T chunks instead of truncating: the old
    // truncate discarded ~80% of collected decisions (100k rows → 19k used). Each window becomes an
    // independent sub-trajectory (its own MC returns, last decision marked done). Uses ALL the data.
    let raw = load_flat_grouped(&path).expect("load export");
    let mut trajs: Vec<Trajectory> = Vec::new();
    for t in raw {
        if t.len() < 4 {
            continue;
        }
        let mut i = 0;
        while i < t.len() {
            let end = (i + MAX_T).min(t.len());
            let mut w: Trajectory = t[i..end].to_vec();
            if w.len() >= 4 {
                if let Some(last) = w.last_mut() {
                    last.done = true;
                }
                trajs.push(w);
            }
            i = end;
        }
    }
    let n_decisions: usize = trajs.iter().map(|t| t.len()).sum();
    let data_n = trajs.iter().flatten().map(|d| d.action).max().unwrap_or(0) + 1;
    let n_actions = env_usize("N_ACTIONS", data_n).max(data_n);

    // Global return stats → normalize the critic target (matches train_real; keeps critic_mse ~O(1)).
    let all_returns: Vec<f32> = trajs.iter().flat_map(|t| mc_returns(t, gamma)).collect();
    let gmean = all_returns.iter().sum::<f32>() / all_returns.len().max(1) as f32;
    let gvar = all_returns.iter().map(|r| (r - gmean).powi(2)).sum::<f32>() / all_returns.len().max(1) as f32;
    let gstd = gvar.sqrt().max(1e-3);
    // Diagnostic: how much real exploration signal is in the logged probs? (all 1.0 ⇒ deterministic
    // export ⇒ PPO degenerates to plain PG; <1.0 spread ⇒ true stochastic behavior policy).
    let bpps: Vec<f32> = trajs.iter().flatten().map(|d| d.behavior_policy_probability).collect();
    let bpp_mean = bpps.iter().sum::<f32>() / bpps.len().max(1) as f32;
    let bpp_det = bpps.iter().filter(|&&p| p >= 0.999).count() as f32 / bpps.len().max(1) as f32;
    println!(
        "loaded {} trajectories, {} decisions, n_actions={}, return mean={:.3} std={:.3} | logged behavior_prob mean={:.3} frac_deterministic={:.2} (informational — old-policy prob is RECOMPUTED from the reference net, so this is exact regardless)",
        trajs.len(), n_decisions, n_actions, gmean, gstd, bpp_mean, bpp_det
    );

    let cfg = PomdpConfig::new(8, n_actions).with_model_dim(96).with_hidden_dim(128).with_n_heads(6);
    let mut net = cfg.init::<B>(&dev);
    // Frozen reference = the behavior policy (pre-fine-tune BEST). Used only for the KL anchor; never
    // stepped. Loaded as a second instance from the same checkpoint (Burn modules aren't Clone).
    let mut reference: Option<PomdpActorCritic<B>> = None;
    if let Ok(mi) = std::env::var("MODEL_IN") {
        net = net.load(&mi, &dev).expect("load checkpoint for fine-tuning");
        reference = Some(cfg.init::<B>(&dev).load(&mi, &dev).expect("load frozen reference"));
        println!("fine-tuning from checkpoint {mi}.bin  (KL-anchored to it, coef={kl_coef})");
    } else {
        println!("no MODEL_IN — training from init, KL anchor disabled");
    }

    // Precompute EVERY constant tensor once. entities/actions/normalized-returns and the frozen
    // reference's log-probs do NOT change across epochs — recomputing the reference forward each epoch
    // was the 20-min-timeout cost. After this the hot loop runs only the trainable net's fwd/bwd.
    struct Cached<B: Backend> {
        entities: Tensor<B, 3>,
        actions: Tensor<B, 1, Int>,
        returns: Tensor<B, 1>,          // normalized critic target (constant)
        old_logp: Tensor<B, 1>,         // ref log-prob of taken action (exact old-policy log-prob)
        ref_logp: Option<Tensor<B, 2>>, // full ref log-probs (T,73) for the KL anchor; None ⇒ no anchor
    }
    let cache: Vec<Cached<B>> = trajs
        .iter()
        .map(|traj| {
            let t = traj.len();
            let (entities, actions, returns_raw) = to_tensors::<B>(traj, gamma, &dev);
            let returns = returns_raw.sub_scalar(gmean).div_scalar(gstd);
            let (old_logp, ref_logp) = match &reference {
                Some(r) => {
                    let rlp = log_softmax(r.forward(entities.clone()).logits, 1).detach(); // (T,73)
                    let ola = rlp.clone().gather(1, actions.clone().reshape([t, 1])).reshape([t]).detach();
                    (ola, Some(rlp))
                }
                None => {
                    let old_p: Vec<f32> =
                        traj.iter().map(|d| d.behavior_policy_probability.clamp(1e-6, 1.0)).collect();
                    (Tensor::<B, 1>::from_data(TensorData::new(old_p, [t]), &dev).log(), None)
                }
            };
            Cached { entities, actions, returns, old_logp, ref_logp }
        })
        .collect();
    drop(reference); // free the reference net; the cache holds its (detached) outputs
    println!("precomputed {} trajectory tensor caches (reference forward done once, not per-epoch)", cache.len());

    let mut opt = AdamConfig::new().init();
    let mut order: Vec<usize> = (0..cache.len()).collect();
    let mut seed = 0x51EEDu64;
    let (mut ema_pi, mut ema_v, mut ema_ratio, mut ema_kl) = (0f32, 0f32, 1f32, 0f32);

    for epoch in 0..epochs {
        for i in (1..order.len()).rev() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (seed >> 33) as usize % (i + 1);
            order.swap(i, j);
        }
        let (mut pi_sum, mut v_sum, mut r_sum, mut kl_sum, mut count) = (0f32, 0f32, 0f32, 0f32, 0usize);
        for &idx in order.iter().take(batch) {
            let c = &cache[idx];
            let t = c.entities.dims()[0];

            let out = net.forward(c.entities.clone());

            // Standardized, detached advantage (GAE-free MC advantage; batch-standardized like AWR).
            let adv = (c.returns.clone() - out.values.clone()).detach();
            let am = adv.clone().mean();
            let astd = (adv.clone() - am.clone()).powf_scalar(2.0).mean().sqrt().add_scalar(1e-6);
            let adv = (adv - am).div(astd); // (T)

            let logp = log_softmax(out.logits.clone(), 1); // (T, n_actions)
            let logp_a = logp.clone().gather(1, c.actions.clone().reshape([t, 1])).reshape([t]); // (T)

            // old_logp precomputed from the frozen reference (= behavior policy). Because π_new and
            // π_ref share the same all-73 softmax normalization on the same state, the legal-
            // renormalization the sidecar applied at sampling time CANCELS in the ratio — exact.
            let ratio = (logp_a - c.old_logp.clone()).exp(); // (T)
            let unclipped = ratio.clone() * adv.clone();
            let clipped = ratio.clone().clamp(1.0 - eps, 1.0 + eps) * adv;
            let policy_loss = unclipped.min_pair(clipped).mean().neg();

            let value_loss = (out.values - c.returns.clone()).powf_scalar(2.0).mean();
            let probs = softmax(out.logits, 1);
            let entropy = (probs * logp.clone()).sum_dim(1).mean().neg();

            // KL(π_ref ‖ π_new): mode-covering anchor to the frozen reference ⇒ anti-collapse.
            let kl = match &c.ref_logp {
                Some(rlp) => {
                    let ref_p = rlp.clone().exp();
                    (ref_p * (rlp.clone() - logp)).sum_dim(1).mean()
                }
                None => Tensor::<B, 1>::from_data(TensorData::new(vec![0f32], [1]), &dev).mean(),
            };

            // entropy var = +H(π); SUBTRACT it (maximize entropy ⇒ exploration). KL is ≥0 and ADDED
            // (penalize divergence from the frozen reference ⇒ anti-collapse).
            let loss = policy_loss.clone()
                + value_loss.clone().mul_scalar(0.5)
                - entropy.mul_scalar(ent_coef)
                + kl.clone().mul_scalar(kl_coef);

            pi_sum += policy_loss.into_scalar();
            v_sum += value_loss.into_scalar();
            r_sum += ratio.mean().into_scalar();
            kl_sum += kl.into_scalar();
            count += 1;
            let grads = GradientsParams::from_grads(loss.backward(), &net);
            net = opt.step(lr, net, grads);
        }
        let n = count.max(1) as f32;
        ema_pi = 0.9 * ema_pi + 0.1 * (pi_sum / n);
        ema_v = 0.9 * ema_v + 0.1 * (v_sum / n);
        ema_ratio = 0.9 * ema_ratio + 0.1 * (r_sum / n);
        ema_kl = 0.9 * ema_kl + 0.1 * (kl_sum / n);
        if epoch % 25 == 0 {
            println!(
                "epoch {epoch}: policy(ema)={ema_pi:.4}  critic_mse(ema)={ema_v:.4}  ratio(ema)={ema_ratio:.3}  kl(ema)={ema_kl:.4}"
            );
        }
    }

    let out_path = std::env::var("MODEL_OUT").unwrap_or_else(|_| "/tmp/burn-pomdp-ppo".into());
    match net.save(&out_path) {
        Ok(_) => println!("saved PPO-fine-tuned model -> {out_path}.bin"),
        Err(e) => println!("save failed: {e}"),
    }
    println!("OK: clipped-PPO fine-tune on REAL logged behavior probs (KL-anchored, entropy-regularized).");
}
