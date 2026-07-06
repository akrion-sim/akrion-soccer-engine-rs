//! A **proper POMDP solver** for the soccer engine, backed by `candle` — the architecture the
//! plateau analysis prescribes and the 1-hidden-layer hand-rolled MLP cannot express:
//!
//!   entities (22 players + ball)                     ← permutation-variant flat vector today
//!     └─ per-entity embed → SELF-ATTENTION           ← permutation-INVARIANT relational reasoning (GNN-like)
//!         └─ attention-pooled context
//!             └─ GRU belief cell (hidden carried across decisions)  ← POMDP MEMORY over history
//!                 ├─ ACTOR head  → π(family | belief)   ← trained by policy-gradient toward RETURN
//!                 └─ CRITIC head → V(belief)            ← centralized critic (CTDE) at train time
//!
//! This is a skeleton: real layers, real autodiff, real forward pass. It exists to prove the
//! architecture is expressible + buildable in candle before the (bigger) integration into the engine.

use candle_core::{Result, Tensor, D};
use candle_nn::{linear, ops::softmax, Linear, Module, VarBuilder};

/// A single GRU cell (explicit equations — robust, no rnn-API coupling). POMDP belief update.
pub struct GruCell {
    ir: Linear, hr: Linear, // reset gate:  from input x, from hidden h
    iz: Linear, hz: Linear, // update gate
    in_: Linear, hn: Linear, // candidate
}

impl GruCell {
    pub fn new(in_dim: usize, hid: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ir: linear(in_dim, hid, vb.pp("ir"))?, hr: linear(hid, hid, vb.pp("hr"))?,
            iz: linear(in_dim, hid, vb.pp("iz"))?, hz: linear(hid, hid, vb.pp("hz"))?,
            in_: linear(in_dim, hid, vb.pp("in"))?, hn: linear(hid, hid, vb.pp("hn"))?,
        })
    }
    /// x: (in_dim), h: (hid) → new hidden (hid).
    pub fn step(&self, x: &Tensor, h: &Tensor) -> Result<Tensor> {
        let r = candle_nn::ops::sigmoid(&(self.ir.forward(x)? + self.hr.forward(h)?)?)?;
        let z = candle_nn::ops::sigmoid(&(self.iz.forward(x)? + self.hz.forward(h)?)?)?;
        let n = (self.in_.forward(x)? + (&r * self.hn.forward(h)?)?)?.tanh()?;
        // h' = (1 - z) * n + z * h
        let one = Tensor::ones_like(&z)?;
        (&((&one - &z)? * &n)? + &(&z * h)?)
    }
}

/// Permutation-invariant self-attention encoder over the entity set.
pub struct EntityEncoder {
    embed: Linear,
    wq: Linear, wk: Linear, wv: Linear,
    dim: usize,
}

impl EntityEncoder {
    pub fn new(entity_dim: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embed: linear(entity_dim, dim, vb.pp("embed"))?,
            wq: linear(dim, dim, vb.pp("wq"))?,
            wk: linear(dim, dim, vb.pp("wk"))?,
            wv: linear(dim, dim, vb.pp("wv"))?,
            dim,
        })
    }
    /// entities: (n_entities, entity_dim) → attention-pooled context (dim). Order-invariant.
    pub fn forward(&self, entities: &Tensor) -> Result<Tensor> {
        let e = self.embed.forward(entities)?; // (N, d)
        let q = self.wq.forward(&e)?;
        let k = self.wk.forward(&e)?;
        let v = self.wv.forward(&e)?;
        let scale = (self.dim as f64).sqrt();
        let scores = (q.matmul(&k.t()?)? / scale)?; // (N, N)
        let attn = softmax(&scores, D::Minus1)?;
        let ctx = attn.matmul(&v)?; // (N, d)
        ctx.mean(0) // mean-pool over entities → permutation-invariant (d)
    }
}

/// The full recurrent, permutation-invariant actor-critic POMDP solver.
pub struct PomdpActorCritic {
    encoder: EntityEncoder,
    belief: GruCell,
    actor: Linear,  // → action-family logits
    critic: Linear, // → scalar value
    pub hidden_dim: usize,
}

pub struct AcOutput {
    pub policy_logits: Tensor, // (n_actions)
    pub value: Tensor,         // (1)
    pub hidden: Tensor,        // (hidden_dim) — carry to the next decision (belief)
}

impl PomdpActorCritic {
    pub fn new(entity_dim: usize, model_dim: usize, hidden_dim: usize, n_actions: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            encoder: EntityEncoder::new(entity_dim, model_dim, vb.pp("enc"))?,
            belief: GruCell::new(model_dim, hidden_dim, vb.pp("gru"))?,
            actor: linear(hidden_dim, n_actions, vb.pp("actor"))?,
            critic: linear(hidden_dim, 1, vb.pp("critic"))?,
            hidden_dim,
        })
    }
    /// One decision: attend over entities, update belief, emit policy + value.
    pub fn step(&self, entities: &Tensor, prev_hidden: &Tensor) -> Result<AcOutput> {
        let ctx = self.encoder.forward(entities)?;      // (model_dim)
        let hidden = self.belief.step(&ctx, prev_hidden)?; // (hidden_dim)  POMDP belief
        let policy_logits = self.actor.forward(&hidden)?;
        let value = self.critic.forward(&hidden)?;
        Ok(AcOutput { policy_logits, value, hidden })
    }
    pub fn zero_hidden(&self, dev: &candle_core::Device) -> Result<Tensor> {
        Tensor::zeros(self.hidden_dim, candle_core::DType::F32, dev)
    }
}
