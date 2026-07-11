//! PPO (clipped) + GAE training of a shared-weight multi-agent policy for
//! Team A against the scripted baseline. One actor MLP + one critic MLP, both
//! shared across all 5 players. Reward is team-level with potential shaping.

use crate::game::*;
use crate::nn::{masked_softmax, Mlp};
use crate::rng::Rng;

const GAMMA: f32 = 0.995; // per-tick discount retuned for 20 Hz (same per-second horizon)
const LAMBDA: f32 = 0.95;
const CLIP: f32 = 0.2;
const LR_ACTOR: f32 = 3e-4;
const LR_CRITIC: f32 = 1e-3;
const LR_BC: f32 = 8e-4;
const EPOCHS: usize = 4;
const MINIBATCH: usize = 1024;
const MAX_ROLLOUT_THREADS: usize = 4;
const W_SHAPE: f32 = 2.2; // potential shaping: strong pull to carry/pass the ball toward goal
const W_ADVANCE: f32 = 0.04; // OFFENSE: push upfield hard (no offsides — camp high)
const W_OPEN: f32 = 0.04; // OFFENSE: get into a CLEAR passing lane from the ball
const W_WIDTH: f32 = 0.045; // OFFENSE: use the width of the pitch (stretch wide)
const W_FLANK: f32 = 0.025; // OFFENSE: commit to a left/right channel (two-flank spread)
const W_GOALSIDE: f32 = 0.02; // DEFENSE: get goalside of the ball
const ENT_BETA0: f32 = 0.02;

// Teammate-spacing reward weight. Overridable via SPACING_W env for tuning.
// Default halved from the 10 Hz value (per-tick reward now fires twice as often).
fn w_spacing() -> f32 {
    std::env::var("SPACING_W")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.003)
}

/// PER-PLAYER spacing reward as a function of a player's nearest-teammate
/// distance `d`. Steep penalties for bunching (the user's curve), reward around
/// the ~5-unit optimum. Applied to EACH outfielder individually, every tick, in
/// all phases — so bunching is directly attributable and can't hide in a shared
/// team average.
fn spacing_reward(d: f32) -> f32 {
    // The ball is a major attractor, but only the closest player should chase;
    // stacking is never OK. We want teammates to hold real cartesian distance and
    // stretch the pitch, so the optimum sits WIDE at ~8 yds:
    //   < 1.5  : MAJOR penalty (overlap zone — sickening bunching)
    //   1.5-3  : strong penalty (bumped) — being within 3 yds is bad
    //   3-4    : real penalty (bumped) — 4 yds is still too close
    //   ~8     : peak reward (optimal spacing — genuine width/separation)
    //   > 12   : penalty (too far to combine)
    if d < 1.0 {
        -100.0
    } else if d < 1.5 {
        -50.0
    } else if d < 3.0 {
        -8.0 // within 3 yds: bumped penalty (was -3)
    } else if d < 4.0 {
        -3.0 // within 4 yds: now also penalized
    } else if d < 8.0 {
        (d - 4.0) * 2.0 // 0 at 4 -> +8 peak at 8
    } else if d < 12.0 {
        8.0 - (d - 8.0) * 2.0 // +8 at 8 -> 0 at 12
    } else {
        -(d - 12.0) * 4.0 // 12+ : escalating penalty
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
            // decentralized actor (per-agent field vector) + CENTRALIZED critic
            // (global state) = MAPPO / CTDE. The actor has TWO heads packed into one
            // output vector: the first NA are the macro-action logits (masked), the
            // last NS are the speed-gear logits (unmasked).
            actor: Mlp::new(&[OBS_DIM, 64, 64, NA + NS], rng),
            critic: Mlp::new(&[GLOBAL_DIM, 128, 64, 1], rng),
        }
    }

    /// Greedy (argmax action + argmax speed) — used at evaluation. Returns the
    /// PACKED action `action + speed*NA` ready to hand to `World::step`.
    pub fn act_greedy(&self, obs: &[f32], mask: &[bool; NA]) -> usize {
        let logits = self.actor.predict(obs);
        debug_assert_eq!(logits.len(), NA + NS);
        let probs = masked_softmax(&logits[0..NA], mask);
        let mut bi = 0;
        let mut bp = -1.0;
        for i in 0..NA {
            if mask[i] && probs[i] > bp {
                bp = probs[i];
                bi = i;
            }
        }
        // greedy speed gear (argmax over the speed head)
        let mut bs = 0;
        let mut bsp = f32::NEG_INFINITY;
        for k in 0..NS {
            if logits[NA + k] > bsp {
                bsp = logits[NA + k];
                bs = k;
            }
        }
        bi + bs * NA
    }
}

struct Sample {
    obs: [f32; OBS_DIM],       // decentralized actor observation
    gstate: [f32; GLOBAL_DIM], // centralized critic global state
    mask: [bool; NA],
    action: usize,          // macro action (0..NA)
    speed: usize,           // speed gear (0..NS)
    old_logp: f32,          // joint log-prob = ln p(action) + ln p(speed)
    adv: f32,
    ret: f32,
}

struct BcSample {
    obs: [f32; OBS_DIM],
    mask: [bool; NA],
    action: usize,
}

pub struct CloneStats {
    pub samples: usize,
    pub loss: f32,
    pub accuracy: f32,
}

/// Collect one game of Team-A experience (5 players) vs the scripted baseline.
fn rollout(policy: &Policy, rng: &mut Rng, opponent_noise: f32) -> Vec<Sample> {
    let mut w = World::new();
    let w_spacing_coeff = w_spacing();
    if rng.f01() < 0.5 {
        w.kickoff(Team::B); // randomize opening possession to remove kickoff bias
    }
    // per-player trajectory buffers
    let mut obs_buf: Vec<[[f32; OBS_DIM]; N]> = Vec::with_capacity(STEPS);
    let mut mask_buf: Vec<[[bool; NA]; N]> = Vec::with_capacity(STEPS);
    let mut act_buf: Vec<[usize; N]> = Vec::with_capacity(STEPS); // macro action (0..NA)
    let mut spd_buf: Vec<[usize; N]> = Vec::with_capacity(STEPS); // speed gear (0..NS)
    let mut logp_buf: Vec<[f32; N]> = Vec::with_capacity(STEPS);
    let mut val_buf: Vec<f32> = Vec::with_capacity(STEPS); // CENTRALIZED value (one per tick)
    let mut gstate_buf: Vec<[f32; GLOBAL_DIM]> = Vec::with_capacity(STEPS);
    let mut rew_buf: Vec<f32> = Vec::with_capacity(STEPS);
    let mut space_buf: Vec<[f32; N]> = Vec::with_capacity(STEPS); // per-player spacing reward

    let mut phi_prev = w.potential_a();

    for _ in 0..STEPS {
        let mut obs_t = [[0.0f32; OBS_DIM]; N];
        let mut mask_t = [[false; NA]; N];
        let mut mact_t = [A_STAY; N]; // macro action
        let mut spd_t = [SPD_STAND; N]; // speed gear
        let mut packed_a = [A_STAY; N]; // packed (action + speed*NA) for step
        let mut logp_t = [0.0f32; N];

        // CENTRALIZED critic: one value for the whole global state this tick.
        let gstate = w.global_state();
        let v_central = policy.critic.predict(&gstate)[0];

        for i in 1..N {
            // policy controls the 4 outfielders; GK (index 0) is rule-based.
            let obs = w.observe(Team::A, i);
            let mask = w.legal_mask(Team::A, i);
            let logits = policy.actor.predict(&obs);
            // action head (masked) and speed head (unmasked) — sampled independently,
            // joint log-prob is their sum.
            let aprobs = masked_softmax(&logits[0..NA], &mask);
            let a = rng.sample_categorical(&aprobs);
            let pa = aprobs[a].max(1e-8);
            let sprobs = masked_softmax(&logits[NA..NA + NS], &[true; NS]);
            let s = rng.sample_categorical(&sprobs);
            let ps = sprobs[s].max(1e-8);
            obs_t[i] = obs;
            mask_t[i] = mask;
            mact_t[i] = a;
            spd_t[i] = s;
            packed_a[i] = a + s * NA;
            logp_t[i] = pa.ln() + ps.ln();
        }

        let act_b = noisy_scripted_actions(&w, Team::B, opponent_noise, rng);
        w.step(&packed_a, &act_b, rng);

        // team reward (Team A perspective)
        let phi = w.potential_a();
        let shaping = GAMMA * phi - phi_prev;
        phi_prev = phi;
        let mut r = W_SHAPE * shaping;
        if w.ev_goal_a {
            r += 8.0; // GOALS are the prize — rewarded well above shots
        }
        if w.ev_goal_b {
            r -= 6.0;
        }
        // Only a tiny nudge for a completed pass (prefer it to a loose turnover);
        // forward progress is rewarded by the potential shaping above, and goals
        // dominate — so passing stays INSTRUMENTAL and the policy still attacks.
        if w.ev_pass_completed_a {
            let n = w.pass_streak_a; // completed passes so far in THIS possession
            // Small flat credit (prefer a completed pass to a loose turnover) PLUS a
            // forward-PROGRESS bonus. Progress is bounded by field length and lateral
            // recycling gains ~0, so it can't be farmed by tiki-taka.
            r += 0.06 + (w.last_pass_gain_a.max(0.0) * 0.1).min(0.6);
            // MILESTONE: the 2nd completed pass unlocks a legal shot. The pinball used
            // to farm this via pass-pass-shoot-repeat — now the SHOT COOLDOWN breaks
            // that loop (the rapid follow-up shot pays nothing), so the milestone is
            // safe to reward again for genuine build-up.
            if n == 2 {
                r += 0.3;
            }
            // Anti-recycle: escalating penalty for sterile long possessions.
            if n > 6 {
                r -= 0.2 * (n - 6) as f32;
            }
        }
        if w.ev_turnover_a {
            r -= 0.2; // real cost, but not so harsh the required passing is avoided
        }
        // SHOT ON GOAL from the final third after 2 passes is EARNED and rewarded
        // (not just goals) — a real base plus a chance-quality bonus. Shoot-spam is
        // stopped structurally by the cooldown: a rapid-fire repeat shot (fired while
        // a prior shot is still "hot") pays nothing, so 60-shots/game can't farm this.
        if w.ev_shot_on_a && !w.shot_was_rapid_a {
            r += 0.6 + 0.6 * w.last_shot_quality_a;
        }
        // reward winning the ball back (pressing / interceptions / tackles)
        if w.ev_win_ball_a {
            r += 0.3;
        }
        // dribbling = possessing the ball (less pinball): forward pays, lateral a
        // little. SMALL per-tick — it fires every tick you carry, so a big value
        // accumulates into ball-hoarding that dwarfs goals.
        if w.ev_dribble_fwd_a {
            r += 0.015; // forward carrying (also paid by potential shaping) — bounded, unfarmable
        }
        // (lateral dribble: no flat reward — potential shaping handles real progress)
        if w.ev_turnover_a && (w.ev_dribble_fwd_a || w.ev_dribble_lat_a) {
            r -= 0.4; // dispossessed while dribbling
        }
        // Ping-pong: A→B→A (one return, streak 1) is FINE. It only becomes a
        // problem from the SECOND return (A→B→A→B, streak 2), and worse each time.
        // 5x heavier than before, and heavier still if the exchange hasn't
        // advanced the ball 5+ yards upfield (pointless tapping in place).
        if w.ev_return_pass_a && w.return_streak_a >= 2 {
            let k = w.return_streak_a;
            let mut pen = 1.5 * 2f32.powi((k - 2) as i32); // 1.5, 3, 6, 12, ...
            if w.ball.x - w.return_start_x < 5.0 {
                pen *= 1.5; // no upfield progress -> heavier
            }
            r -= pen.min(24.0);
        }

        // Possession PHASE (covers ball-in-flight during our build-up, not just
        // strict ownership): our phase = we own OR loose ball we last touched.
        let a_owns = matches!(w.owner, Some(o) if matches!(o.team, Team::A));
        let b_owns = matches!(w.owner, Some(o) if matches!(o.team, Team::B));
        let our_phase = a_owns || (w.owner.is_none() && matches!(w.last_touch, Some(Team::A)));
        let their_phase = b_owns || (w.owner.is_none() && matches!(w.last_touch, Some(Team::B)));

        // PER-PLAYER coordination (MARL): teammate spacing (anti-bunch) PLUS a
        // possession-conditioned positioning reward —
        //   OFFENSE: advance UPFIELD and OPEN UP into space to receive a pass.
        //   DEFENSE: get GOALSIDE of the ball (between the ball and our own goal).
        let mut sp_t = [0.0f32; N];
        for i in 1..N {
            let pos = w.a[i].pos;
            let mut nd = f32::INFINITY;
            for j in 1..N {
                if i != j {
                    let d = w.a[j].pos.sub(pos).len();
                    if d < nd {
                        nd = d;
                    }
                }
            }
            if nd.is_finite() {
                sp_t[i] = w_spacing_coeff * spacing_reward(nd);
            }
            if our_phase {
                // Advance upfield (attack frame +x). No offsides rule, so reward
                // getting into the ATTACKING HALF and keep rewarding all the way to
                // the opponent goal — attackers should camp high, not hold at half.
                let advance = pos.x / FIELD_L - 0.5;
                // OPEN = a CLEAR passing lane from the ball to me (no defender in
                // between). NOT merely far from a marker — a player inline behind a
                // defender is NOT open. This is the MECHANISM for "opening up": the
                // SOLUTION the policy should learn is to move to a WIDER position,
                // which clears the lane (hence width is rewarded strongly too).
                let open = w.lane_clearness(Team::A, w.ball, pos) - 0.5;
                // WIDTH: how far off the central lane (0 = center, 1 = touchline).
                let wide = (pos.y - FIELD_W / 2.0).abs() / (FIELD_W / 2.0);
                let width = wide - 0.4; // penalize the central lane, reward stretching
                // FLANK affinity: convex bonus for genuinely committing to a
                // left/right channel. With the 8-yd anti-bunch this splits the
                // front line across BOTH flanks instead of clustering one side.
                let flank = if wide > 0.5 { (wide - 0.5) * 2.0 } else { 0.0 };
                sp_t[i] +=
                    W_ADVANCE * advance + W_OPEN * open + W_WIDTH * width + W_FLANK * flank;
            } else if their_phase {
                // goalside of the ball: our goal is at x=0, so reward being at a
                // LOWER x than the ball (between ball and own goal).
                let goalside = ((w.ball.x - pos.x) / 8.0).clamp(-1.0, 1.0);
                sp_t[i] += W_GOALSIDE * goalside;
            }
        }

        obs_buf.push(obs_t);
        mask_buf.push(mask_t);
        act_buf.push(mact_t);
        spd_buf.push(spd_t);
        logp_buf.push(logp_t);
        val_buf.push(v_central);
        gstate_buf.push(gstate);
        rew_buf.push(r);
        space_buf.push(sp_t);
    }

    // GAE per agent, baselined by the SHARED centralized value V(global_state).
    // Each agent's per-tick reward = team reward + its own spacing reward; the
    // centralized critic is trained (below) to predict these returns from the
    // global state — a valid, lower-variance baseline (MAPPO/CTDE).
    let t = rew_buf.len();
    let mut samples = Vec::with_capacity(t * (N - 1));
    for i in 1..N {
        let mut adv = 0.0f32;
        let mut next_v = 0.0f32; // bootstrap 0 at horizon end
        for s in (0..t).rev() {
            let v = val_buf[s]; // shared centralized value
            let delta = rew_buf[s] + space_buf[s][i] + GAMMA * next_v - v;
            adv = delta + GAMMA * LAMBDA * adv;
            let ret = adv + v;
            next_v = v;
            samples.push(Sample {
                obs: obs_buf[s][i],
                gstate: gstate_buf[s],
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

fn collect_rollouts(policy: &Policy, games: usize, rng: &mut Rng) -> Vec<Sample> {
    let jobs: Vec<(u64, f32)> = (0..games)
        .map(|_| (rng.next_u64(), opponent_noise(rng)))
        .collect();
    let workers = rollout_worker_count(games);
    if workers <= 1 {
        return run_rollout_jobs(policy, &jobs);
    }

    let chunk = jobs.len().div_ceil(workers);
    let mut handles = Vec::new();
    for job_chunk in jobs.chunks(chunk) {
        let local_policy = policy.clone();
        let local_jobs = job_chunk.to_vec();
        handles.push(std::thread::spawn(move || {
            run_rollout_jobs(&local_policy, &local_jobs)
        }));
    }

    let mut data = Vec::new();
    for handle in handles {
        data.extend(handle.join().expect("rollout worker panicked"));
    }
    data
}

fn run_rollout_jobs(policy: &Policy, jobs: &[(u64, f32)]) -> Vec<Sample> {
    let mut data = Vec::new();
    for &(seed, noise) in jobs {
        let mut local_rng = Rng::new(seed);
        data.extend(rollout(policy, &mut local_rng, noise));
    }
    data
}

pub fn rollout_worker_count(games: usize) -> usize {
    if games <= 1 {
        return 1;
    }
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let env_limit = std::env::var("FIVEASIDE_ROLLOUT_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(MAX_ROLLOUT_THREADS);
    games.min(available).min(env_limit).max(1)
}

fn opponent_noise(rng: &mut Rng) -> f32 {
    let r = rng.f01();
    if r < 0.15 {
        0.18
    } else if r < 0.35 {
        0.08
    } else {
        0.0
    }
}

fn noisy_scripted_actions(w: &World, team: Team, noise: f32, rng: &mut Rng) -> [usize; N] {
    let mut acts = w.scripted_actions(team);
    if noise <= 0.0 {
        return acts;
    }
    for i in 1..N {
        if rng.f01() < noise {
            let mask = w.legal_mask(team, i);
            let a = sample_legal_action(&mask, rng);
            acts[i] = a + scripted_gear(a) * NA; // keep the noisy move at a real speed
        }
    }
    acts
}

fn sample_legal_action(mask: &[bool; NA], rng: &mut Rng) -> usize {
    let n = mask.iter().filter(|&&ok| ok).count();
    if n == 0 {
        return A_STAY;
    }
    let pick = (rng.next_u64() % n as u64) as usize;
    let mut seen = 0usize;
    for (idx, &ok) in mask.iter().enumerate() {
        if ok {
            if seen == pick {
                return idx;
            }
            seen += 1;
        }
    }
    A_STAY
}

fn legal_or_first(action: usize, mask: &[bool; NA]) -> Option<usize> {
    if action < NA && mask[action] {
        Some(action)
    } else {
        mask.iter().position(|&ok| ok)
    }
}

pub fn behavior_clone_scripted(
    policy: &mut Policy,
    games: usize,
    epochs: usize,
    rng: &mut Rng,
) -> CloneStats {
    if games == 0 || epochs == 0 {
        return CloneStats {
            samples: 0,
            loss: 0.0,
            accuracy: 0.0,
        };
    }

    let mut data = Vec::new();
    for _ in 0..games {
        let mut w = World::new();
        if rng.f01() < 0.5 {
            w.kickoff(Team::B);
        }
        for _ in 0..STEPS {
            let mut act_a = w.scripted_actions(Team::A);
            for i in 1..N {
                let obs = w.observe(Team::A, i);
                let mask = w.legal_mask(Team::A, i);
                if let Some(action) = legal_or_first(act_a[i], &mask) {
                    act_a[i] = action;
                    data.push(BcSample { obs, mask, action });
                }
            }
            let act_b = w.scripted_actions(Team::B);
            w.step(&act_a, &act_b, rng);
        }
    }

    let mut idx: Vec<usize> = (0..data.len()).collect();
    let mut loss_accum = 0.0f32;
    let mut correct = 0.0f32;
    let mut count = 0.0f32;
    for _ in 0..epochs {
        shuffle(&mut idx, rng);
        for chunk in idx.chunks(MINIBATCH) {
            for &si in chunk {
                let s = &data[si];
                let acts = policy.actor.forward(&s.obs);
                let logits = acts.last().unwrap();
                let probs = masked_softmax(logits, &s.mask);
                let p = probs[s.action].max(1e-8);
                loss_accum += -p.ln();
                if probs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| s.mask[*i])
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .is_some_and(|(i, _)| i == s.action)
                {
                    correct += 1.0;
                }
                let mut d_logits = vec![0.0f32; NA];
                for j in 0..NA {
                    if s.mask[j] {
                        d_logits[j] = probs[j] - if j == s.action { 1.0 } else { 0.0 };
                    }
                }
                policy.actor.backward(&acts, &d_logits);
                count += 1.0;
            }
            policy.actor.step(LR_BC);
        }
    }

    CloneStats {
        samples: data.len(),
        loss: loss_accum / count.max(1.0),
        accuracy: correct / count.max(1.0),
    }
}

pub struct IterStats {
    pub avg_reward: f32,
    pub entropy: f32,
    pub value_loss: f32,
}

/// One PPO iteration: collect `games` rollouts, then EPOCHS of minibatch updates.
pub fn train_iter(policy: &mut Policy, games: usize, ent_beta: f32, rng: &mut Rng) -> IterStats {
    let mut data = collect_rollouts(policy, games, rng);
    let mut total_r = 0.0f32;
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

                // ---- centralized critic (MSE to GAE return), on GLOBAL state ----
                let cacts = policy.critic.forward(&s.gstate);
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

/// Rich per-evaluation analytics (all averaged per game unless noted).
#[derive(Clone, Default)]
pub struct Stats {
    pub goal_diff: f32,
    pub winrate: f32,
    pub ga: f32,
    pub gb: f32,
    pub spacing: f32,      // avg nearest-teammate distance (all ticks)
    pub bunch: f32,        // fraction of ticks a pair < 2.5
    pub possession: f32,   // fraction of ticks Team A holds the ball
    pub pass_att: f32,     // pass attempts / game
    pub pass_cmp: f32,     // completed passes / game
    pub pass_fwd: f32,     // forward pass attempts / game
    pub pass_lat: f32,     // lateral pass attempts / game
    pub pass_back: f32,    // backward pass attempts / game
    pub shots: f32,        // shot attempts / game
    pub shots_scored: f32, // goals (proxy for converted shots) / game
    pub turnovers: f32,    // A turnovers / game
    pub wins_won: f32,     // balls won / game (press/intercept/tackle)
}
impl Stats {
    pub fn pass_completion(&self) -> f32 {
        if self.pass_att > 0.0 {
            self.pass_cmp / self.pass_att
        } else {
            0.0
        }
    }
    pub fn conversion(&self) -> f32 {
        if self.shots > 0.0 {
            self.shots_scored / self.shots
        } else {
            0.0
        }
    }
}

/// Evaluate greedy policy vs scripted over `games`, collecting full analytics.
pub fn evaluate(policy: &Policy, games: usize, rng: &mut Rng) -> Stats {
    let mut s = Stats::default();
    let mut space_sum = 0.0f32;
    let mut ticks = 0.0f32;
    let mut poss_ticks = 0.0f32;
    let mut bunch_ticks = 0.0f32;
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
                s.pass_cmp += 1.0;
            }
            if w.ev_pass_attempt_a {
                s.pass_att += 1.0;
                match w.pass_dir_a {
                    1 => s.pass_fwd += 1.0,
                    -1 => s.pass_back += 1.0,
                    _ => s.pass_lat += 1.0,
                }
            }
            if w.ev_shot_attempt_a {
                s.shots += 1.0;
            }
            if w.ev_turnover_a {
                s.turnovers += 1.0;
            }
            if w.ev_win_ball_a {
                s.wins_won += 1.0;
            }
            space_sum += w.avg_nearest_teammate_a();
            ticks += 1.0;
            if matches!(w.owner, Some(o) if matches!(o.team, Team::A)) {
                poss_ticks += 1.0;
            }
            if w.closest_pair_a() < 2.5 {
                bunch_ticks += 1.0;
            }
        }
        let d = w.goals_a as f32 - w.goals_b as f32;
        s.goal_diff += d;
        s.ga += w.goals_a as f32;
        s.gb += w.goals_b as f32;
        s.shots_scored += w.goals_a as f32; // proxy: goals as converted shots
        if d > 0.0 {
            s.winrate += 1.0;
        } else if d == 0.0 {
            s.winrate += 0.5;
        }
    }
    let g = games.max(1) as f32;
    for v in [
        &mut s.goal_diff,
        &mut s.winrate,
        &mut s.ga,
        &mut s.gb,
        &mut s.pass_att,
        &mut s.pass_cmp,
        &mut s.pass_fwd,
        &mut s.pass_lat,
        &mut s.pass_back,
        &mut s.shots,
        &mut s.shots_scored,
        &mut s.turnovers,
        &mut s.wins_won,
    ] {
        *v /= g;
    }
    s.spacing = if ticks > 0.0 { space_sum / ticks } else { 0.0 };
    s.bunch = if ticks > 0.0 {
        bunch_ticks / ticks
    } else {
        0.0
    };
    s.possession = if ticks > 0.0 { poss_ticks / ticks } else { 0.0 };
    s
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spacing_reward_penalizes_overlap_and_prefers_useful_distance() {
        assert!(spacing_reward(0.8) < spacing_reward(2.0)); // overlap worst
        assert!(spacing_reward(3.5) < 0.0); // within 4 yds is penalized
        assert!(spacing_reward(2.5) < spacing_reward(3.5)); // within 3 penalized harder
        assert!(spacing_reward(8.0) > spacing_reward(5.0)); // optimum moved WIDE to 8
        assert!(spacing_reward(8.0) > spacing_reward(11.0)); // too far tapers off
        assert!(spacing_reward(13.0) < 0.0); // beyond 12: penalty
    }

    #[test]
    fn scripted_vs_scripted_is_seed_reproducible() {
        let mut a = Rng::new(1234);
        let mut b = Rng::new(1234);
        let left = evaluate_scripted_vs_scripted(8, &mut a);
        let right = evaluate_scripted_vs_scripted(8, &mut b);
        assert_eq!(left, right);
    }

    #[test]
    fn untrained_evaluation_metrics_stay_in_bounds() {
        let mut rng = Rng::new(99);
        let policy = Policy::new(&mut rng);
        let stats = evaluate(&policy, 2, &mut rng);
        assert!((0.0..=1.0).contains(&stats.winrate));
        assert!((0.0..=1.0).contains(&stats.bunch));
        assert!((0.0..=1.0).contains(&stats.possession));
        assert!(stats.pass_completion().is_finite());
        assert!(stats.conversion().is_finite());
    }

    #[test]
    fn behavior_clone_warm_start_collects_finite_teacher_signal() {
        let mut rng = Rng::new(2026);
        let mut policy = Policy::new(&mut rng);
        let stats = behavior_clone_scripted(&mut policy, 1, 1, &mut rng);
        assert!(stats.samples > 0);
        assert!(stats.loss.is_finite());
        assert!((0.0..=1.0).contains(&stats.accuracy));
    }
}
