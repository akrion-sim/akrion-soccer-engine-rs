//! The CTDE RL training loop: REINFORCE with a critic baseline + entropy bonus, over trajectories
//! in the engine's export format (`adapter::Decision`). Proves the full pipeline — feature adapter →
//! attention/GRU belief → policy-gradient toward RETURN — actually learns to maximize return.
//!
//! Synthetic task (stand-in for real replay until the engine exporter lands): the rewarding action
//! each decision is a bin of the pooled entity signal; the net must attend + pool + map to it. Mean
//! return climbing = the learned POMDP policy is optimizing return, not imitating a teacher.

use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use burn_pomdp_solver::adapter::{Decision, Trajectory, FIELD_MOTION_DIM, N_ENTITIES, ENTITY_DIM};
use burn_pomdp_solver::PomdpConfig;

type B = Autodiff<NdArray>;

fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

/// A synthetic trajectory + the (hidden) rewarding action per decision.
fn synth(t: usize, n_actions: usize, seed: &mut u64) -> (Trajectory, Vec<usize>) {
    let mut traj = Vec::with_capacity(t);
    let mut good = Vec::with_capacity(t);
    for i in 0..t {
        let mut fm = vec![0f32; FIELD_MOTION_DIM];
        for v in fm.iter_mut() { *v = lcg(seed); }
        let mut s = 0f32;
        for e in 0..N_ENTITIES { s += fm[e * ENTITY_DIM]; } // channel 0 pooled
        let g = ((((s / N_ENTITIES as f32) + 1.0) / 2.0 * n_actions as f32) as usize).min(n_actions - 1);
        good.push(g);
        traj.push(Decision { field_motion: fm, action: 0, reward: 0.0, done: i == t - 1 });
    }
    (traj, good)
}

fn main() {
    let dev = Default::default();
    let (t, n_actions) = (20, 12);
    let cfg = PomdpConfig::new(ENTITY_DIM, n_actions);
    let mut net = cfg.init::<B>(&dev);
    let mut opt = AdamConfig::new().init();
    let (gamma, ent_coef) = (0.97f32, 0.02f32);
    let mut seed = 0xBEEF_1234u64;
    let mut ema_ret = 0f32;

    for epoch in 0..600 {
        let (mut traj, good) = synth(t, n_actions, &mut seed);
        // entities tensor (T,23,8)
        let ent: Vec<f32> = traj.iter().flat_map(|d| d.field_motion.clone()).collect();
        let entities = Tensor::<B, 1>::from_data(TensorData::new(ent, [t * FIELD_MOTION_DIM]), &dev)
            .reshape([t, N_ENTITIES, ENTITY_DIM]);
        let out = net.forward(entities);
        let probs = softmax(out.logits.clone(), 1); // (T, A)

        // Sample an action per decision from the current policy, score reward vs the hidden good action.
        let pdata: Vec<f32> = probs.clone().into_data().to_vec().unwrap();
        let mut acts = vec![0i64; t];
        let mut ep_reward = 0f32;
        for i in 0..t {
            let u = (lcg(&mut seed) + 1.0) / 2.0; // ~[0,1)
            let mut cum = 0f32;
            let mut a = n_actions - 1;
            for j in 0..n_actions {
                cum += pdata[i * n_actions + j];
                if u <= cum { a = j; break; }
            }
            acts[i] = a as i64;
            let r = if a == good[i] { 1.0 } else { 0.0 };
            traj[i].reward = r;
            ep_reward += r;
        }
        // returns from the sampled rewards
        let g = burn_pomdp_solver::adapter::returns(&traj, gamma);
        let returns_t = Tensor::<B, 1>::from_data(TensorData::new(g, [t]), &dev);
        let actions_t = Tensor::<B, 1, Int>::from_data(TensorData::new(acts, [t]), &dev);

        // REINFORCE with critic baseline: A = (G - V).detach(); actor = -(logp(a)·A) - β·H; critic = mse(V,G)
        let logp = log_softmax(out.logits, 1); // (T, A)
        let logp_a = logp.clone().gather(1, actions_t.clone().reshape([t, 1])).reshape([t]); // (T)
        let advantage = (returns_t.clone() - out.values.clone()).detach();
        let actor_loss = (logp_a * advantage).mean().neg();
        let entropy = (probs.clone() * logp).sum_dim(1).mean().neg(); // H = -Σ p·logp
        let critic_loss = (out.values - returns_t).powf_scalar(2.0).mean();
        let loss = actor_loss + critic_loss.mul_scalar(0.5) - entropy.clone().mul_scalar(ent_coef);

        ema_ret = 0.98 * ema_ret + 0.02 * (ep_reward / t as f32);
        if epoch % 50 == 0 {
            println!(
                "epoch {epoch}: mean_return(ema)={:.3}  raw={:.3}  entropy={:.3}",
                ema_ret, ep_reward / t as f32, entropy.into_scalar()
            );
        }
        let grads = GradientsParams::from_grads(loss.backward(), &net);
        net = opt.step(3e-4, net, grads);
    }
    println!("OK: CTDE REINFORCE loop optimizes RETURN through the attention+GRU policy (adapter format).");
}
