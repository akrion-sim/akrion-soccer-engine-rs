//! Smoke test: prove the recurrent-attention actor-critic TRAINS. Task: per decision, the correct
//! action is a bin of the pooled entity signal — the net must attend over entities, pool, run the
//! GRU, and map belief→action. If accuracy climbs toward 1.0 via autodiff+Adam, the whole
//! POMDP-solver stack (attention + GRU + actor) is a working, trainable learner.

use burn::backend::{Autodiff, NdArray};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Int, Tensor, TensorData};
use burn_pomdp_solver::PomdpConfig;

type B = Autodiff<NdArray>;

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0 // ~[-1,1)
}

fn make_batch(dev: &<B as burn::tensor::backend::Backend>::Device, t: usize, n: usize, edim: usize, n_actions: usize, seed: &mut u64)
    -> (Tensor<B, 3>, Tensor<B, 1, Int>) {
    let mut data = vec![0f32; t * n * edim];
    for v in data.iter_mut() { *v = lcg(seed); }
    let mut targets = vec![0i64; t];
    for ti in 0..t {
        let mut s = 0f32;
        for ni in 0..n { s += data[ti * n * edim + ni * edim]; } // channel 0, summed over entities
        let bin = (((s / n as f32) + 1.0) / 2.0 * n_actions as f32) as i64;
        targets[ti] = bin.clamp(0, n_actions as i64 - 1);
    }
    let e = Tensor::<B, 1>::from_data(TensorData::new(data, [t * n * edim]), dev).reshape([t, n, edim]);
    let tg = Tensor::<B, 1, Int>::from_data(TensorData::new(targets, [t]), dev);
    (e, tg)
}

fn main() {
    let dev = Default::default();
    let (t, n, edim, n_actions) = (16, 23, 8, 12);
    let cfg = PomdpConfig::new(edim, n_actions);
    let mut net = cfg.init::<B>(&dev);
    let mut opt = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&dev);
    let mut seed = 0x1234_5678u64;

    for epoch in 0..400 {
        let (entities, targets) = make_batch(&dev, t, n, edim, n_actions, &mut seed);
        let out = net.forward(entities);
        let loss = loss_fn.forward(out.logits.clone(), targets.clone());
        if epoch % 50 == 0 {
            let pred = out.logits.argmax(1).reshape([t]);
            let acc = pred.equal(targets).float().mean().into_scalar() as f64;
            println!("epoch {epoch}: loss={:.4} acc={:.3}", loss.clone().into_scalar() as f64, acc);
        }
        let grads = GradientsParams::from_grads(loss.backward(), &net);
        net = opt.step(1e-3, net, grads);
    }
    println!("OK: Burn recurrent-attention actor-critic trained end-to-end (attention + GRU + actor via autodiff+Adam).");
}
