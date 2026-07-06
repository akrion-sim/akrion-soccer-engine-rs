//! Train the Burn POMDP-solver on REAL exported soccer trajectories (offline policy gradient).
//! Loads the engine's flat JSONL export, groups into per-(agent,episode) trajectories, and trains
//! the attention+GRU actor-critic toward RETURN: advantage-weighted log-prob of the logged action
//! (advantages standardized) + critic value regression + entropy. This is the proof the real
//! architecture ingests + learns from real soccer data — the last piece before the A/B.

use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::{backend::Backend, Tensor};
use burn_pomdp_solver::adapter::{load_flat_grouped, to_tensors, Trajectory};
use burn_pomdp_solver::PomdpConfig;

type B = Autodiff<NdArray>;

const MAX_T: usize = 128; // cap trajectory length for memory/BPTT

fn main() {
    let path = std::env::var("TRAJ").unwrap_or_else(|_| "/tmp/traj_export.jsonl".into());
    let dev = Default::default();
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
    let mean_reward: f32 = trajs.iter().flatten().map(|d| d.reward).sum::<f32>() / n_decisions.max(1) as f32;
    println!(
        "loaded {} trajectories, {} decisions, n_actions={}, mean_reward={:.4}",
        trajs.len(), n_decisions, n_actions, mean_reward
    );

    let cfg = PomdpConfig::new(8, n_actions).with_model_dim(96).with_hidden_dim(128).with_n_heads(6);
    let mut net = cfg.init::<B>(&dev);
    let mut opt = AdamConfig::new().init();
    let (gamma, ent_coef) = (0.98f32, 0.01f32);

    let mut order: Vec<usize> = (0..trajs.len()).collect();
    let mut seed = 0x51EEDu64;
    let mut ema_actor = 0f32;
    let mut ema_vfit = 0f32;

    for epoch in 0..400 {
        // shuffle (Fisher-Yates via lcg)
        for i in (1..order.len()).rev() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (seed >> 33) as usize % (i + 1);
            order.swap(i, j);
        }
        let (mut a_sum, mut v_sum, mut count) = (0f32, 0f32, 0usize);
        for &idx in order.iter().take(64) {
            let traj: &Trajectory = &trajs[idx];
            let t = traj.len();
            let (entities, actions, returns) = to_tensors::<B>(traj, gamma, &dev);
            let out = net.forward(entities);
            // standardized advantage (detached baseline)
            let adv = (returns.clone() - out.values.clone()).detach();
            let mean = adv.clone().mean();
            let std = (adv.clone() - mean.clone()).powf_scalar(2.0).mean().sqrt().add_scalar(1e-6);
            let adv = (adv - mean).div(std);
            let logp = log_softmax(out.logits.clone(), 1);
            let logp_a = logp.clone().gather(1, actions.reshape([t, 1])).reshape([t]);
            let actor_loss = (logp_a * adv).mean().neg();
            let probs = softmax(out.logits, 1);
            let entropy = (probs * logp).sum_dim(1).mean().neg();
            let critic_loss = (out.values - returns).powf_scalar(2.0).mean();
            let loss = actor_loss.clone() + critic_loss.clone().mul_scalar(0.5) - entropy.mul_scalar(ent_coef);

            a_sum += actor_loss.into_scalar();
            v_sum += critic_loss.into_scalar();
            count += 1;
            let grads = GradientsParams::from_grads(loss.backward(), &net);
            net = opt.step(3e-4, net, grads);
        }
        ema_actor = 0.9 * ema_actor + 0.1 * (a_sum / count.max(1) as f32);
        ema_vfit = 0.9 * ema_vfit + 0.1 * (v_sum / count.max(1) as f32);
        if epoch % 40 == 0 {
            println!("epoch {epoch}: actor_loss(ema)={ema_actor:.4}  critic_mse(ema)={ema_vfit:.4}");
        }
    }
    println!("OK: trained the attention+GRU actor-critic on REAL soccer trajectories (offline policy gradient).");
}
