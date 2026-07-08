//! Verify the snapshot bridge: init → save → load into a fresh module → identical outputs. This is
//! what lets the sidecar serve a trained policy the trainer produced.

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use burn_pomdp_solver::adapter::{ENTITY_DIM, N_ENTITIES};
use burn_pomdp_solver::PomdpConfig;

type B = NdArray;

fn main() {
    let dev = Default::default();
    let cfg = PomdpConfig::new(ENTITY_DIM, 12);
    let net = cfg.init::<B>(&dev);

    let data: Vec<f32> = (0..(3 * N_ENTITIES * ENTITY_DIM)).map(|i| (i as f32 * 0.013).sin()).collect();
    let x = Tensor::<B, 1>::from_data(TensorData::new(data, [3 * N_ENTITIES * ENTITY_DIM]), &dev)
        .reshape([3, N_ENTITIES, ENTITY_DIM]);

    let v1: Vec<f32> = net.forward(x.clone()).values.into_data().to_vec().unwrap();
    net.save("/tmp/burn-snap-test").expect("save");

    let net2 = cfg.init::<B>(&dev).load("/tmp/burn-snap-test", &dev).expect("load");
    let v2: Vec<f32> = net2.forward(x).values.into_data().to_vec().unwrap();

    let maxdiff = v1.iter().zip(&v2).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("snapshot round-trip: value max|delta| = {maxdiff:.2e}");
    println!(
        "{}",
        if maxdiff < 1e-5 { "SNAPSHOT BRIDGE OK — save+load reproduces the policy exactly" } else { "MISMATCH" }
    );
}
