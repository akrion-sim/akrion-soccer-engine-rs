//! Prove the sidecar's core call on the TRAINED model: load /tmp/burn-pomdp-big.bin, run a 3-tick
//! rollout via step_infer carrying the GRU belief, and show it serves an action+value per tick with
//! the belief evolving. This is exactly what the sidecar does per decision.

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use burn_pomdp_solver::adapter::{ENTITY_DIM, N_ENTITIES};
use burn_pomdp_solver::PomdpConfig;

type B = NdArray;

fn main() {
    let dev = Default::default();
    // Must match the training config (train_real): entity_dim=8, n_actions=73, model 96 / hidden 128 / 6 heads.
    let cfg = PomdpConfig::new(ENTITY_DIM, 73).with_model_dim(96).with_hidden_dim(128).with_n_heads(6);
    let path = std::env::var("MODEL").unwrap_or_else(|_| "/tmp/burn-pomdp-big".into());
    let net = cfg.init::<B>(&dev).load(&path, &dev).expect("load trained model");
    println!("loaded trained POMDP-solver from {path}.bin");

    let mut hidden = None;
    for tick in 0..3 {
        let data: Vec<f32> = (0..(N_ENTITIES * ENTITY_DIM))
            .map(|i| ((i + tick * 7) as f32 * 0.031).sin())
            .collect();
        let ent = Tensor::<B, 1>::from_data(TensorData::new(data, [N_ENTITIES * ENTITY_DIM]), &dev)
            .reshape([1, N_ENTITIES, ENTITY_DIM]);
        let (logits, value, new_hidden) = net.step_infer(ent, hidden.clone());
        let lg: Vec<f32> = logits.into_data().to_vec().unwrap();
        let action = lg.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap();
        let v: f32 = value.into_data().to_vec::<f32>().unwrap()[0];
        let bnorm = new_hidden.clone().powf_scalar(2.0).sum().into_scalar().sqrt();
        println!("tick {tick}: action={action} value={v:.4} belief_norm={bnorm:.4}");
        hidden = Some(new_hidden);
    }
    println!("SIDECAR INFERENCE OK — trained POMDP-solver serves per-tick decisions carrying belief.");
}
