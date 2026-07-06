//! A **learned POMDP solver** for the soccer engine, on Burn — the Lock-2 architecture the
//! 1-hidden-layer hand-rolled MLP cannot be:
//!
//!   per-decision entities (22 players + ball)
//!     └─ embed → LEARNED MultiHeadAttention over entities → mean-pool   ← permutation-invariant, relational
//!         └─ GRU over the decision SEQUENCE (belief carried across time) ← learned POMDP memory
//!             ├─ ACTOR  → π(action-family | belief)                      ← policy-gradient toward RETURN
//!             └─ CRITIC → V(belief)                                       ← centralized critic (CTDE)
//!
//! Unlike the candle inference-skeleton, this trains: Burn gives autodiff + Adam + real
//! MultiHeadAttention/GRU modules. Forward consumes a whole trajectory (BPTT over decisions).

pub mod adapter;

use burn::config::Config;
use burn::module::Module;
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::gru::{Gru, GruConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{backend::Backend, activation::softmax, Tensor};

#[derive(Config, Debug)]
pub struct PomdpConfig {
    pub entity_dim: usize,
    #[config(default = 64)]
    pub model_dim: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 96)]
    pub hidden_dim: usize,
    pub n_actions: usize,
}

#[derive(Module, Debug)]
pub struct PomdpActorCritic<B: Backend> {
    embed: Linear<B>,
    attn: MultiHeadAttention<B>,
    gru: Gru<B>,
    actor: Linear<B>,
    critic: Linear<B>,
}

/// Output over a trajectory of T decisions.
pub struct AcOut<B: Backend> {
    pub logits: Tensor<B, 2>, // (T, n_actions)
    pub values: Tensor<B, 1>, // (T)
}

impl PomdpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PomdpActorCritic<B> {
        PomdpActorCritic {
            embed: LinearConfig::new(self.entity_dim, self.model_dim).init(device),
            attn: MultiHeadAttentionConfig::new(self.model_dim, self.n_heads).init(device),
            gru: GruConfig::new(self.model_dim, self.hidden_dim, true).init(device),
            actor: LinearConfig::new(self.hidden_dim, self.n_actions).init(device),
            critic: LinearConfig::new(self.hidden_dim, 1).init(device),
        }
    }
}

impl<B: Backend> PomdpActorCritic<B> {
    /// entities: (T decisions, N entities, entity_dim). Returns per-decision policy logits + values.
    pub fn forward(&self, entities: Tensor<B, 3>) -> AcOut<B> {
        let [t, n, _] = entities.dims();
        let embedded = self.embed.forward(entities); // (T, N, model_dim)
        // Self-attention over the N entities, independently per decision (T is the MHA "batch").
        let attended = self.attn.forward(MhaInput::self_attn(embedded)).context; // (T, N, model_dim)
        let pooled = attended.mean_dim(1); // (T, 1, model_dim) — permutation-invariant entity pool
        let m = pooled.dims()[2];
        let seq = pooled.reshape([1, t, m]); // (1, T, model_dim) — one trajectory of length T
        let belief = self.gru.forward(seq, None); // (1, T, hidden_dim) — learned belief over decisions
        let h = belief.dims()[2];
        let belief = belief.reshape([t, h]); // (T, hidden_dim)
        let logits = self.actor.forward(belief.clone()); // (T, n_actions)
        let values = self.critic.forward(belief).reshape([t]); // (T)
        let _ = n;
        AcOut { logits, values }
    }

    /// Softmax action probabilities per decision — for sampling / entropy.
    pub fn policy(&self, entities: Tensor<B, 3>) -> Tensor<B, 2> {
        softmax(self.forward(entities).logits, 1)
    }
}
