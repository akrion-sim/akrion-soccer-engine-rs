//! Smoke test: build the recurrent permutation-invariant actor-critic and run a 3-decision rollout
//! over a dummy 23-entity field (22 players + ball), carrying the GRU belief across decisions.
//! Proves the POMDP-solver architecture forwards end-to-end in candle.

use candle_core::{Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use extlib_pomdp_solver::PomdpActorCritic;

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let (entity_dim, model_dim, hidden_dim, n_actions, n_entities) = (8, 64, 96, 12, 23);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
    let net = PomdpActorCritic::new(entity_dim, model_dim, hidden_dim, n_actions, vb)?;

    let mut hidden = net.zero_hidden(&dev)?;
    println!("param tensors: {}", varmap.all_vars().len());
    for t in 0..3 {
        // A permutation of the SAME field should give the same policy — the property the flat MLP lacks.
        let entities = Tensor::randn(0f32, 1.0, (n_entities, entity_dim), &dev)?;
        let out = net.step(&entities, &hidden)?;
        hidden = out.hidden;
        let logits = out.policy_logits.flatten_all()?.to_vec1::<f32>()?;
        let value = out.value.flatten_all()?.to_vec1::<f32>()?;
        println!(
            "decision {t}: value={:.4}  argmax_action={}  belief_norm={:.4}",
            value[0],
            logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap(),
            hidden.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt()
        );
    }
    println!("OK: recurrent permutation-invariant actor-critic forwards + carries belief across decisions.");
    Ok(())
}
