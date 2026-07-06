//! Train the Burn POMDP-solver on REAL exported soccer trajectories — v2, per Codex's recipe:
//! AWR (advantage-weighted regression) instead of raw REINFORCE (which diverged), with GLOBALLY
//! NORMALIZED returns (so the critic can fit) + entropy. This is the stable offline objective for
//! logged behavior-policy data: `actor = -mean( clip(exp(adv/β),0,w_max)·logπ(logged_action) )`.

use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::{backend::Backend, Tensor};
use burn_pomdp_solver::adapter::{load_flat_grouped, returns as mc_returns, to_tensors, Trajectory};
use burn_pomdp_solver::PomdpConfig;

type B = Autodiff<NdArray>;

const MAX_T: usize = 128;

fn main() {
    let path = std::env::var("TRAJ").unwrap_or_else(|_| "/tmp/traj_export.jsonl".into());
    let dev = Default::default();
    let (gamma, beta, w_max, ent_coef) = (0.98f32, 1.0f32, 20.0f32, 0.01f32);

    let mut trajs = load_flat_grouped(&path).expect("load export");
    trajs.retain(|t| t.len() >= 4);
    for t in trajs.iter_mut() {
        if t.len() > MAX_T {
            t.truncate(MAX_T);
            if let Some(last) = t.last_mut() { last.done = true; }
        }
    }
    let n_decisions: usize = trajs.iter().map(|t| t.len()).sum();
    let n_actions = trajs.iter().flatten().map(|d| d.action).max().unwrap_or(0) + 1;

    // GLOBAL return stats → normalize the critic target (the fix for critic_mse ~600).
    let all_returns: Vec<f32> = trajs.iter().flat_map(|t| mc_returns(t, gamma)).collect();
    let gmean = all_returns.iter().sum::<f32>() / all_returns.len().max(1) as f32;
    let gvar = all_returns.iter().map(|r| (r - gmean).powi(2)).sum::<f32>() / all_returns.len().max(1) as f32;
    let gstd = gvar.sqrt().max(1e-3);
    println!(
        "loaded {} trajectories, {} decisions, n_actions={}, return mean={:.3} std={:.3}",
        trajs.len(), n_decisions, n_actions, gmean, gstd
    );

    let cfg = PomdpConfig::new(8, n_actions).with_model_dim(96).with_hidden_dim(128).with_n_heads(6);
    let mut net = cfg.init::<B>(&dev);
    let mut opt = AdamConfig::new().init();
    let mut order: Vec<usize> = (0..trajs.len()).collect();
    let mut seed = 0x51EEDu64;
    let (mut ema_actor, mut ema_critic, mut ema_w) = (0f32, 0f32, 0f32);

    for epoch in 0..400 {
        for i in (1..order.len()).rev() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (seed >> 33) as usize % (i + 1);
            order.swap(i, j);
        }
        let (mut a_sum, mut c_sum, mut w_sum, mut count) = (0f32, 0f32, 0f32, 0usize);
        for &idx in order.iter().take(64) {
            let traj: &Trajectory = &trajs[idx];
            let t = traj.len();
            let (entities, actions, returns_raw) = to_tensors::<B>(traj, gamma, &dev);
            let returns = returns_raw.sub_scalar(gmean).div_scalar(gstd); // normalized target
            let out = net.forward(entities);

            // standardized advantage, detached
            let adv = (returns.clone() - out.values.clone()).detach();
            let am = adv.clone().mean();
            let astd = (adv.clone() - am.clone()).powf_scalar(2.0).mean().sqrt().add_scalar(1e-6);
            let adv = (adv - am).div(astd);
            // AWR weight = clip(exp(adv/β), 0, w_max)
            let weight = adv.div_scalar(beta).exp().clamp(0.0, w_max).detach();

            let logp = log_softmax(out.logits.clone(), 1);
            let logp_a = logp.clone().gather(1, actions.reshape([t, 1])).reshape([t]);
            let actor_loss = (weight.clone() * logp_a).mean().neg();
            let probs = softmax(out.logits, 1);
            let entropy = (probs * logp).sum_dim(1).mean().neg();
            let critic_loss = (out.values - returns).powf_scalar(2.0).mean();
            let loss = actor_loss.clone() + critic_loss.clone().mul_scalar(0.5) - entropy.mul_scalar(ent_coef);

            a_sum += actor_loss.into_scalar();
            c_sum += critic_loss.into_scalar();
            w_sum += weight.mean().into_scalar();
            count += 1;
            let grads = GradientsParams::from_grads(loss.backward(), &net);
            net = opt.step(3e-4, net, grads);
        }
        let n = count.max(1) as f32;
        ema_actor = 0.9 * ema_actor + 0.1 * (a_sum / n);
        ema_critic = 0.9 * ema_critic + 0.1 * (c_sum / n);
        ema_w = 0.9 * ema_w + 0.1 * (w_sum / n);
        if epoch % 40 == 0 {
            println!("epoch {epoch}: actor(ema)={ema_actor:.4}  critic_mse(ema)={ema_critic:.4}  awr_w(ema)={ema_w:.3}");
        }
    }
    let out_path = std::env::var("MODEL_OUT").unwrap_or_else(|_| "/tmp/burn-pomdp-solver".into());
    match net.save(&out_path) {
        Ok(_) => println!("saved trained model -> {out_path}.bin"),
        Err(e) => println!("save failed: {e}"),
    }
    println!("OK: AWR training on REAL soccer trajectories (normalized returns, clamped weights).");
}
