//! PPO (clipped) + GAE training of a shared-weight multi-agent policy for
//! Team A against the scripted baseline. One actor MLP + one critic MLP, both
//! shared across all 5 players. Reward is team-level with potential shaping.

use crate::game::*;
use crate::nn::{masked_softmax, Mlp};
use crate::rng::Rng;

const GAMMA: f32 = 0.99;
const LAMBDA: f32 = 0.95;
const CLIP: f32 = 0.2;
const LR_ACTOR: f32 = 3e-4;
const LR_CRITIC: f32 = 1e-3;
const EPOCHS: usize = 4;
const MINIBATCH: usize = 1024;
const W_SHAPE: f32 = 2.0;
const ENT_BETA0: f32 = 0.02;

// Teammate-spacing reward weight. Overridable via SPACING_W env for tuning.
fn w_spacing() -> f32 {
    std::env::var("SPACING_W").ok().and_then(|s| s.parse().ok()).unwrap_or(0.02)
}

/// PER-PLAYER spacing reward as a function of a player's nearest-teammate
/// distance `d`. Steep penalties for bunching (the user's curve), reward around
/// the ~5-unit optimum. Applied to EACH outfielder individually, every tick, in
/// all phases — so bunching is directly attributable and can't hide in a shared
/// team average.
fn spacing_reward(d: f32) -> f32 {
    if d < 1.0 {
        -50.0
    } else if d < 2.0 {
        -20.0
    } else if d < 3.0 {
        -10.0
    } else if d < 4.0 {
        -3.0
    } else if d <= 6.0 {
        0.0 // ~5 units = optimal: NEUTRAL, so good spacing doesn't pay an
            // accumulating bonus that dwarfs goals (that caused corner-camping).
    } else if d < 8.0 {
        -(d - 6.0) * 3.0 // too far is also penalized -> ~5 is a STABLE optimum
    } else {
        -6.0
    }
}

#[derive(Clone)]
pub struct Policy {
    pub actor: Mlp,
    pub critic: Mlp,
}

impl Policy {
    pub fn new(rng: &mut Rng) -> Self {
        Policy {
            actor: Mlp::new(&[OBS_DIM, 64, 64, NA], rng),
            critic: Mlp::new(&[OBS_DIM, 64, 64, 1], rng),
        }
    }

    /// Greedy (argmax over legal actions) — used at evaluation.
    pub fn act_greedy(&self, obs: &[f32], mask: &[bool; NA]) -> usize {
        let logits = self.actor.predict(obs);
        let probs = masked_softmax(&logits, mask);
        let mut bi = 0;
        let mut bp = -1.0;
        for i in 0..NA {
            if mask[i] && probs[i] > bp {
                bp = probs[i];
                bi = i;
            }
        }
        bi
    }
}

struct Sample {
    obs: [f32; OBS_DIM],
    mask: [bool; NA],
    action: usize,
    old_logp: f32,
    adv: f32,
    ret: f32,
}

/// Collect one game of Team-A experience (5 players) vs the scripted baseline.
fn rollout(policy: &Policy, rng: &mut Rng) -> Vec<Sample> {
    let mut w = World::new();
    let w_spacing_coeff = w_spacing();
    if rng.f01() < 0.5 {
        w.kickoff(Team::B); // randomize opening possession to remove kickoff bias
    }
    // per-player trajectory buffers
    let mut obs_buf: Vec<[[f32; OBS_DIM]; N]> = Vec::with_capacity(STEPS);
    let mut mask_buf: Vec<[[bool; NA]; N]> = Vec::with_capacity(STEPS);
    let mut act_buf: Vec<[usize; N]> = Vec::with_capacity(STEPS);
    let mut logp_buf: Vec<[f32; N]> = Vec::with_capacity(STEPS);
    let mut val_buf: Vec<[f32; N]> = Vec::with_capacity(STEPS);
    let mut rew_buf: Vec<f32> = Vec::with_capacity(STEPS);
    let mut space_buf: Vec<[f32; N]> = Vec::with_capacity(STEPS); // per-player spacing reward

    let mut phi_prev = w.potential_a();

    for _ in 0..STEPS {
        let mut obs_t = [[0.0f32; OBS_DIM]; N];
        let mut mask_t = [[false; NA]; N];
        let mut act_a = [A_STAY; N];
        let mut logp_t = [0.0f32; N];
        let mut val_t = [0.0f32; N];

        for i in 1..N {
            // policy controls the 4 outfielders; GK (index 0) is rule-based.
            let obs = w.observe(Team::A, i);
            let mask = w.legal_mask(Team::A, i);
            let logits = policy.actor.predict(&obs);
            let probs = masked_softmax(&logits, &mask);
            let a = rng.sample_categorical(&probs);
            let p = probs[a].max(1e-8);
            obs_t[i] = obs;
            mask_t[i] = mask;
            act_a[i] = a;
            logp_t[i] = p.ln();
            val_t[i] = policy.critic.predict(&obs)[0];
        }

        let act_b = w.scripted_actions(Team::B);
        w.step(&act_a, &act_b, rng);

        // team reward (Team A perspective)
        let phi = w.potential_a();
        let shaping = GAMMA * phi - phi_prev;
        phi_prev = phi;
        let mut r = W_SHAPE * shaping;
        if w.ev_goal_a {
            r += 5.0;
        }
        if w.ev_goal_b {
            r -= 5.0;
        }
        // Only a tiny nudge for a completed pass (prefer it to a loose turnover);
        // forward progress is rewarded by the potential shaping above, and goals
        // dominate — so passing stays INSTRUMENTAL and the policy still attacks.
        if w.ev_pass_completed_a {
            // reward completing a pass, weighted by how much it advanced the ball
            // toward goal — line-breaking forward passes pay, lateral hoarding
            // doesn't. Small base keeps a completed pass better than a turnover.
            r += 0.1 + (w.last_pass_gain_a.max(0.0) * 0.12).min(1.2);
        }
        if w.ev_turnover_a {
            r -= 0.25;
        }
        // shot reward scaled by MPC-finish placement quality: rewards taking a
        // shot when a good corner exists AND placing it well.
        if w.ev_shot_on_a {
            r += 0.15 + 0.5 * w.last_shot_quality_a;
        }
        // reward winning the ball back (pressing / interceptions / tackles)
        if w.ev_win_ball_a {
            r += 0.3;
        }
        // PER-PLAYER teammate spacing (all phases): each outfielder is rewarded
        // for its OWN nearest-teammate distance, so bunching is directly credited.
        let mut sp_t = [0.0f32; N];
        for i in 1..N {
            let mut nd = f32::INFINITY;
            for j in 1..N {
                if i != j {
                    let d = w.a[i].pos.sub(w.a[j].pos).len();
                    if d < nd {
                        nd = d;
                    }
                }
            }
            if nd.is_finite() {
                sp_t[i] = w_spacing_coeff * spacing_reward(nd);
            }
        }

        obs_buf.push(obs_t);
        mask_buf.push(mask_t);
        act_buf.push(act_a);
        logp_buf.push(logp_t);
        val_buf.push(val_t);
        rew_buf.push(r);
        space_buf.push(sp_t);
    }

    // GAE per player (same team reward broadcast to all 5).
    let t = rew_buf.len();
    let mut samples = Vec::with_capacity(t * (N - 1));
    for i in 1..N {
        let mut adv = 0.0f32;
        let mut next_v = 0.0f32; // bootstrap 0 at horizon end
        for s in (0..t).rev() {
            let v = val_buf[s][i];
            // team reward (shared) + this player's OWN spacing reward
            let delta = rew_buf[s] + space_buf[s][i] + GAMMA * next_v - v;
            adv = delta + GAMMA * LAMBDA * adv;
            let ret = adv + v;
            next_v = v;
            samples.push(Sample {
                obs: obs_buf[s][i],
                mask: mask_buf[s][i],
                action: act_buf[s][i],
                old_logp: logp_buf[s][i],
                adv,
                ret,
            });
        }
    }
    samples
}

pub struct IterStats {
    pub avg_reward: f32,
    pub entropy: f32,
    pub value_loss: f32,
}

/// One PPO iteration: collect `games` rollouts, then EPOCHS of minibatch updates.
pub fn train_iter(policy: &mut Policy, games: usize, ent_beta: f32, rng: &mut Rng) -> IterStats {
    let mut data: Vec<Sample> = Vec::new();
    let mut total_r = 0.0f32;
    for _ in 0..games {
        let s = rollout(policy, rng);
        data.extend(s);
    }
    // returns/advs summed across all player-steps; report mean reward-to-go proxy
    for s in &data {
        total_r += s.ret;
    }
    let avg_reward = total_r / data.len().max(1) as f32;

    // normalize advantages (across the batch)
    let mean_adv = data.iter().map(|s| s.adv).sum::<f32>() / data.len().max(1) as f32;
    let var_adv =
        data.iter().map(|s| (s.adv - mean_adv).powi(2)).sum::<f32>() / data.len().max(1) as f32;
    let std_adv = var_adv.sqrt().max(1e-6);
    for s in data.iter_mut() {
        s.adv = (s.adv - mean_adv) / std_adv;
    }

    let mut idx: Vec<usize> = (0..data.len()).collect();
    let mut ent_accum = 0.0f32;
    let mut vloss_accum = 0.0f32;
    let mut count = 0.0f32;

    for _ in 0..EPOCHS {
        shuffle(&mut idx, rng);
        for chunk in idx.chunks(MINIBATCH) {
            for &si in chunk {
                let s = &data[si];
                // ---- actor ----
                let acts = policy.actor.forward(&s.obs);
                let logits = acts.last().unwrap();
                let probs = masked_softmax(logits, &s.mask);
                let p_a = probs[s.action].max(1e-8);
                let new_logp = p_a.ln();
                let ratio = (new_logp - s.old_logp).exp();
                let a = s.adv;
                // PPO clip: gradient coefficient on log-prob
                let coeff = if a >= 0.0 {
                    if ratio <= 1.0 + CLIP {
                        a * ratio
                    } else {
                        0.0
                    }
                } else if ratio >= 1.0 - CLIP {
                    a * ratio
                } else {
                    0.0
                };
                // entropy of the (masked) distribution
                let mut ent = 0.0f32;
                for i in 0..NA {
                    if s.mask[i] && probs[i] > 1e-8 {
                        ent -= probs[i] * probs[i].ln();
                    }
                }
                // dLoss/dlogit_j = -coeff*(1_{j=a}-p_j) + beta*p_j*(log p_j + H)
                let mut d_logits = vec![0.0f32; NA];
                for j in 0..NA {
                    if !s.mask[j] {
                        d_logits[j] = 0.0;
                        continue;
                    }
                    let ind = if j == s.action { 1.0 } else { 0.0 };
                    let pg = -coeff * (ind - probs[j]);
                    let eg = ent_beta * probs[j] * (probs[j].max(1e-8).ln() + ent);
                    d_logits[j] = pg + eg;
                }
                policy.actor.backward(&acts, &d_logits);

                // ---- critic (MSE to GAE return) ----
                let cacts = policy.critic.forward(&s.obs);
                let v = cacts.last().unwrap()[0];
                let dv = v - s.ret; // dL/dv for 0.5*(v-ret)^2
                policy.critic.backward(&cacts, &[dv]);

                ent_accum += ent;
                vloss_accum += 0.5 * (v - s.ret) * (v - s.ret);
                count += 1.0;
            }
            policy.actor.step(LR_ACTOR);
            policy.critic.step(LR_CRITIC);
        }
    }

    IterStats {
        avg_reward,
        entropy: ent_accum / count.max(1.0),
        value_loss: vloss_accum / count.max(1.0),
    }
}

pub fn ent_beta_at(iter: usize, total: usize) -> f32 {
    // linear decay of entropy bonus
    let frac = 1.0 - (iter as f32 / total.max(1) as f32);
    ENT_BETA0 * frac.max(0.1)
}

/// Evaluate greedy policy vs scripted over `games`. Returns
/// (avg_goal_diff, win_rate, avg_goals_a, avg_goals_b, avg_completed_passes_a,
///  avg_nearest_teammate_dist_during_A_possession).
pub fn evaluate(policy: &Policy, games: usize, rng: &mut Rng) -> (f32, f32, f32, f32, f32, f32) {
    let mut diff = 0.0f32;
    let mut wins = 0.0f32;
    let mut ga = 0.0f32;
    let mut gb = 0.0f32;
    let mut passes = 0.0f32;
    let mut space_sum = 0.0f32;
    let mut space_ticks = 0.0f32;
    for _ in 0..games {
        let mut w = World::new();
        if rng.f01() < 0.5 {
            w.kickoff(Team::B);
        }
        for _ in 0..STEPS {
            let mut act_a = [A_STAY; N];
            for i in 1..N {
                let obs = w.observe(Team::A, i);
                let mask = w.legal_mask(Team::A, i);
                act_a[i] = policy.act_greedy(&obs, &mask);
            }
            let act_b = w.scripted_actions(Team::B);
            w.step(&act_a, &act_b, rng);
            if w.ev_pass_completed_a {
                passes += 1.0;
            }
            // measure spacing on EVERY tick (all phases) to validate the whole match
            space_sum += w.avg_nearest_teammate_a();
            space_ticks += 1.0;
        }
        let d = w.goals_a as f32 - w.goals_b as f32;
        diff += d;
        ga += w.goals_a as f32;
        gb += w.goals_b as f32;
        if d > 0.0 {
            wins += 1.0;
        } else if d == 0.0 {
            wins += 0.5;
        }
    }
    let g = games.max(1) as f32;
    let spacing = if space_ticks > 0.0 { space_sum / space_ticks } else { 0.0 };
    (diff / g, wins / g, ga / g, gb / g, passes / g, spacing)
}

/// Baseline sanity check: scripted-vs-scripted goal difference (should be ~0).
pub fn evaluate_scripted_vs_scripted(games: usize, rng: &mut Rng) -> (f32, f32, f32) {
    let mut diff = 0.0f32;
    let (mut ga, mut gb) = (0.0f32, 0.0f32);
    for _ in 0..games {
        let mut w = World::new();
        if rng.f01() < 0.5 {
            w.kickoff(Team::B);
        }
        for _ in 0..STEPS {
            let act_a = w.scripted_actions(Team::A);
            let act_b = w.scripted_actions(Team::B);
            w.step(&act_a, &act_b, rng);
        }
        diff += w.goals_a as f32 - w.goals_b as f32;
        ga += w.goals_a as f32;
        gb += w.goals_b as f32;
    }
    let g = games.max(1) as f32;
    (diff / g, ga / g, gb / g)
}

fn shuffle(v: &mut [usize], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}
