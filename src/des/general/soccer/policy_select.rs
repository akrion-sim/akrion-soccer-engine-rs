//! Stochastic top-k policy selection.
//!
//! Every per-tick MDP/POMDP decision in the engine reduces to ranking a set of
//! candidate actions ("policies") by score and committing to the highest-ranked
//! feasible one — a deterministic *argmax* over the candidate surface. This
//! module turns that argmax into a **soft** top-k draw: instead of always taking
//! the single best option, the decision picks
//!
//!   * the **best** option ~70% of the time,
//!   * the **2nd best** ~20% of the time,
//!   * the **3rd best** ~10% of the time,
//!
//! with the weights renormalised over however many candidates actually exist
//! (so a 2-candidate decision is ~78/22, a 1-candidate decision is forced). The
//! exact weights live in [`crate::des::general::soccer::tunables`] under
//! `policy_selection`, env/PG-overridable; the defaults reproduce 70/20/10.
//!
//! ## Value-weighted (Boltzmann) mode
//!
//! The fixed 70/20/10 split is *rank*-weighted: it explores the 2nd-best as
//! often whether it trails the best by 0.001 or by 10. Set
//! `policy_selection.boltzmann_temperature > 0` to switch the same gated draw to
//! a **value-weighted** softmax over *all* candidates — probability
//! `∝ exp(score / temperature)` — so a near-tie explores readily while a runaway
//! best is taken almost always, and the explored option is picked by how good it
//! actually scores rather than merely by being 2nd in line. The temperature
//! default `0.0` leaves the rank-weighted split in force, so an unconfigured
//! process is byte-identical. The behaviour-policy probability reported for
//! learning honesty is the softmax probability of the chosen candidate in this
//! mode (vs. the renormalised rank weight in the legacy mode). This path needs
//! candidate *scores*, so it is only taken by the scored entry points
//! ([`reorder_with_scores_traced`], [`sampled_index_by_score`]); the score-less
//! [`reorder_by_draw`] always uses the rank-weighted split.
//!
//! ## Determinism
//!
//! Player decisions run off an immutable [`WorldSnapshot`] and have no mutable
//! RNG, and the engine is byte-identical-reproducible. So the "coin" is a pure
//! deterministic draw hashed from `(decision_seed, player_id, tick, site)` via a
//! SplitMix64 finaliser — reproducible for a given match seed, independent
//! across decision sites within a tick, and varying across matches because
//! `decision_seed` is the match `config.seed`.
//!
//! ## Gating
//!
//! The whole mechanism is behind [`stochastic_policy_topk_enabled`]
//! (`DD_SOCCER_ENABLE_STOCHASTIC_POLICY_TOPK`, default **OFF**). With the gate
//! off every helper here is a no-op that returns the input order / rank 0
//! unchanged, so an unconfigured process is byte-identical to before this module
//! existed.
//!
//! ## Learning-honesty caveat
//!
//! These helpers report the behaviour-policy probability of the **promoted**
//! (sampled) candidate. Downstream the engine "commits to the first *feasible*
//! candidate" in the reordered list, so the executed action equals the promoted
//! one **only when the promoted candidate passes its own feasibility/MPC check**;
//! if it is skipped, the executed action is a later candidate but the recorded
//! probability still refers to the promoted one. This is a property of the
//! promote-then-verify design shared by both the rank-weighted and value-weighted
//! draws, not specific to either. The value-weighted path narrows the gap by
//! giving ineligible (non-positive-weight) candidates zero exploration mass (see
//! [`reorder_with_scores_traced`]) so it does not *add* skip-prone promotions; the
//! residual (a legal candidate vetoed by a later MPC/target check) is left as a
//! known limitation rather than reconciled here.
//!
//! [`WorldSnapshot`]: crate::des::general::soccer::WorldSnapshot

const ENABLE_ENV: &str = "DD_SOCCER_ENABLE_STOCHASTIC_POLICY_TOPK";

use crate::des::general::soccer::POLICY_SELECTION_TOP_RANK_LIMIT;

/// Distinct per-decision-site salts so the draw at, say, the possession ranker is
/// statistically independent of the draw at the defensive ranker on the same tick
/// for the same player. The concrete values are arbitrary but must stay stable
/// (they seed the deterministic hash).
pub mod site {
    pub const POSSESSION: u64 = 0x5011_0001;
    pub const SUPPORT: u64 = 0x5011_0002;
    pub const DEFENSIVE: u64 = 0x5011_0003;
    pub const LOOSE_BALL: u64 = 0x5011_0004;
    pub const OFF_BALL: u64 = 0x5011_0005;
    pub const AERIAL_RECEPTION: u64 = 0x5011_0006;
    pub const BEAT_DEFENDER: u64 = 0x5011_0007;
    pub const DRIBBLE_KIND: u64 = 0x5011_0008;
    pub const FIRST_TOUCH: u64 = 0x5011_0009;
}

/// Whether stochastic top-k selection is live this process. OFF unless
/// `DD_SOCCER_ENABLE_STOCHASTIC_POLICY_TOPK` is set to a truthy value ⇒ off
/// restores the deterministic argmax byte-for-byte.
pub fn stochastic_policy_topk_enabled() -> bool {
    #[cfg(test)]
    {
        env_flag_enabled(ENABLE_ENV)
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| env_flag_enabled(ENABLE_ENV))
    }
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|raw| {
            let v = raw.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Deterministic unit draw in `[0, 1)` from a SplitMix64 finaliser over the
/// decision coordinates. Same inputs ⇒ same draw, so a replay of a match seed is
/// reproducible; `site` decorrelates simultaneous decisions of the same player.
pub fn decision_unit_draw(decision_seed: u64, player_id: usize, tick: u64, site: u64) -> f64 {
    let mut z = decision_seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add((player_id as u64).wrapping_add(1).wrapping_mul(0xbf58_476d_1ce4_e5b9))
        .wrapping_add(tick.wrapping_mul(0x94d0_49bb_1331_11eb))
        .wrapping_add(site.wrapping_mul(0xd6e8_feb8_6659_fd93));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    // 53-bit mantissa precision, strictly < 1.0.
    ((z >> 11) as f64) / ((1u64 << 53) as f64)
}

/// The per-rank selection weights `[top1, top2, top3]` from the tunables tree
/// (defaults `[0.70, 0.20, 0.10]`). Pulled through a thin accessor so the module
/// stays unit-testable without the global tunables registry.
fn rank_weights() -> [f64; POLICY_SELECTION_TOP_RANK_LIMIT] {
    crate::des::general::soccer::tunables()
        .policy_selection
        .rank_weights()
}

/// The Boltzmann (softmax) temperature from the tunables tree. `> 0` selects the
/// value-weighted draw; the default `0.0` (and any non-positive / non-finite
/// value) keeps the rank-weighted top-k split, preserving byte-identical
/// behaviour. Pulled through a thin accessor so the value-weighted helpers stay
/// unit-testable in isolation.
fn boltzmann_temperature() -> f64 {
    crate::des::general::soccer::tunables()
        .policy_selection
        .boltzmann_temperature
}

/// Value-weighted draw: pick an index into `scores` with probability
/// `∝ exp(score / temperature)` (numerically-stable softmax), using the
/// deterministic unit `draw`. Returns `(index, probability_of_that_index)` for
/// learning honesty.
///
/// Degenerate inputs fall back to the plain argmax with probability 1.0:
/// fewer than two candidates, a non-positive / non-finite temperature, no finite
/// score, or an all-zero weight mass. Non-finite scores contribute zero mass.
/// `scores` need not be sorted; the returned index is into the slice as given.
/// The probability is floored to [`SELECTION_PROB_FLOOR`] so an importance ratio
/// can never divide by zero.
fn boltzmann_pick(scores: &[f64], temperature: f64, draw: f64) -> (usize, f64) {
    let n = scores.len();
    if n < 2 {
        return (0, 1.0);
    }
    if !(temperature.is_finite() && temperature > 0.0) {
        return (argmax_index(scores), 1.0);
    }
    let max = scores
        .iter()
        .copied()
        .filter(|s| s.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return (argmax_index(scores), 1.0);
    }
    let mut weights = Vec::with_capacity(n);
    let mut total = 0.0;
    for &s in scores {
        let w = if s.is_finite() {
            ((s - max) / temperature).exp()
        } else {
            0.0
        };
        weights.push(w);
        total += w;
    }
    if !(total > 0.0) {
        return (argmax_index(scores), 1.0);
    }
    let draw = if draw.is_finite() { draw.clamp(0.0, 1.0) } else { 0.0 };
    let target = draw * total;
    let mut acc = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if target < acc {
            return (i, (w / total).max(SELECTION_PROB_FLOOR));
        }
    }
    // Floating-point slack: fall to the last positive-weight candidate.
    let last = (0..n).rev().find(|&i| weights[i] > 0.0).unwrap_or(n - 1);
    (last, (weights[last] / total).max(SELECTION_PROB_FLOOR))
}

/// Index of the highest finite score (lowest index on ties), or 0 if none is
/// finite — the deterministic argmax fallback shared by the value-weighted path.
fn argmax_index(scores: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (i, &s) in scores.iter().enumerate() {
        if s.is_finite() && s > best_score {
            best_score = s;
            best = i;
        }
    }
    best
}

/// Lower bound on any reported behaviour-policy probability, shared by the
/// rank-weighted and value-weighted paths so an off-policy importance ratio can
/// never divide by zero.
const SELECTION_PROB_FLOOR: f64 = 1e-3;

/// Pick a rank in `0..n.min(3)` from a unit `draw`, using the first `n` rank
/// weights **renormalised** over what is available. Non-finite / non-positive
/// weights are floored to zero; if everything is zero the best (rank 0) is taken.
///
/// `n` is the number of candidates actually present. With `n >= 3` the full
/// `[0.70, 0.20, 0.10]` applies; with `n == 2` it is `[0.70, 0.20]` renormalised
/// to `~[0.78, 0.22]`; with `n <= 1` rank 0 is forced.
pub fn sampled_rank(
    n: usize,
    weights: [f64; POLICY_SELECTION_TOP_RANK_LIMIT],
    draw: f64,
) -> usize {
    let k = n.min(POLICY_SELECTION_TOP_RANK_LIMIT);
    if k <= 1 {
        return 0;
    }
    let draw = if draw.is_finite() {
        draw.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut w = [0.0f64; POLICY_SELECTION_TOP_RANK_LIMIT];
    let mut total = 0.0;
    for i in 0..k {
        let wi = if weights[i].is_finite() {
            weights[i].max(0.0)
        } else {
            0.0
        };
        w[i] = wi;
        total += wi;
    }
    if total <= 0.0 {
        return 0;
    }
    let target = draw * total;
    let mut acc = 0.0;
    for i in 0..k {
        acc += w[i];
        if target < acc {
            return i;
        }
    }
    k - 1
}

/// Behaviour-policy probability of committing to the candidate at `rank` under the
/// top-k mix: the renormalised rank weight. This is the value PPO/MAPPO must use as
/// the **behaviour** probability of the executed action — the rank mix *is* the
/// behaviour policy, so the actor head's own softmax is not the distribution the
/// action was actually drawn from. Ranks past the available candidates (or past
/// the three weighted ranks) get a small positive floor so an importance ratio
/// never divides by zero. `n <= 1` ⇒ `1.0` (forced choice).
pub fn behavior_probability_for_rank(
    n: usize,
    weights: [f64; POLICY_SELECTION_TOP_RANK_LIMIT],
    rank: usize,
) -> f64 {
    let k = n.min(POLICY_SELECTION_TOP_RANK_LIMIT);
    if k <= 1 {
        return 1.0;
    }
    let mut total = 0.0;
    for i in 0..k {
        if weights[i].is_finite() && weights[i] > 0.0 {
            total += weights[i];
        }
    }
    if total <= 0.0 {
        return if rank == 0 { 1.0 } else { SELECTION_PROB_FLOOR };
    }
    if rank < k && weights[rank].is_finite() && weights[rank] > 0.0 {
        (weights[rank] / total).max(SELECTION_PROB_FLOOR)
    } else {
        SELECTION_PROB_FLOOR
    }
}

/// Whole-decision draw: the reordered candidate list **and** the behaviour-policy
/// probability of the now-front (sampled) candidate, for learning honesty. The
/// probability is `None` when the gate is off or there are fewer than two
/// candidates (a greedy argmax that needs no importance correction).
pub struct SampledOrder<T> {
    pub ordered: Vec<T>,
    pub behavior_probability: Option<f64>,
}

/// Like [`reorder_by_draw`] but also reports the behaviour-policy probability of
/// the promoted front candidate (see [`behavior_probability_for_rank`]).
///
/// This score-less entry point always uses the **rank-weighted** top-k split; to
/// get the value-weighted (Boltzmann) draw, pass candidate scores to
/// [`reorder_with_scores_traced`].
pub fn reorder_by_draw_traced<T>(
    ordered: Vec<T>,
    decision_seed: u64,
    player_id: usize,
    tick: u64,
    site: u64,
) -> SampledOrder<T> {
    // No scores ⇒ the value-weighted path is never eligible; delegate with an
    // empty score list so there is a single reorder code path.
    reorder_with_scores_traced(ordered, &[], decision_seed, player_id, tick, site)
}

/// Like [`reorder_by_draw_traced`] but value-weighted when configured: given the
/// candidate `scores` aligned with `ordered` (same length, same order), a
/// positive `policy_selection.boltzmann_temperature` draws over the candidates
/// with probability `∝ exp(score / temperature)`; otherwise the rank-weighted
/// top-k split applies. `scores.len() != ordered.len()` (e.g. the empty slice
/// from [`reorder_by_draw_traced`]) forces the rank-weighted path.
///
/// `scores` are the **ordering weights** of the agentic ranker, where a
/// non-positive weight marks an **ineligible** candidate (e.g. an illegal action
/// scored 0). Those receive **zero** exploration mass in value-weighted mode, so
/// the draw never promotes an option the downstream "commit to the first feasible
/// candidate" loop would only skip — which would otherwise waste the exploration
/// budget *and* mis-record the behaviour probability against a different executed
/// action. (The raw-score entry point [`sampled_index_by_score`] keeps full
/// softmax, since there a non-positive score is a legitimate value.)
///
/// The reported `behavior_probability` is the softmax probability of the chosen
/// candidate in value-weighted mode, or the renormalised rank weight otherwise —
/// honest for off-policy correction either way. No-op (and `None` probability)
/// when the gate is off or there are fewer than two candidates, so an
/// unconfigured process is byte-identical.
pub fn reorder_with_scores_traced<T>(
    ordered: Vec<T>,
    scores: &[f64],
    decision_seed: u64,
    player_id: usize,
    tick: u64,
    site: u64,
) -> SampledOrder<T> {
    // Read the (global) temperature only when the gate is actually on, so an
    // unconfigured process does no extra tunables lookup.
    if !stochastic_policy_topk_enabled() || ordered.len() < 2 {
        return SampledOrder {
            ordered,
            behavior_probability: None,
        };
    }
    reorder_with_scores_traced_at(
        ordered,
        scores,
        boltzmann_temperature(),
        decision_seed,
        player_id,
        tick,
        site,
    )
}

/// [`reorder_with_scores_traced`] with the Boltzmann `temperature` passed in
/// explicitly rather than read from the global tunables — the unit-testable core.
fn reorder_with_scores_traced_at<T>(
    mut ordered: Vec<T>,
    scores: &[f64],
    temperature: f64,
    decision_seed: u64,
    player_id: usize,
    tick: u64,
    site: u64,
) -> SampledOrder<T> {
    if !stochastic_policy_topk_enabled() || ordered.len() < 2 {
        return SampledOrder {
            ordered,
            behavior_probability: None,
        };
    }
    let n = ordered.len();
    let draw = decision_unit_draw(decision_seed, player_id, tick, site);
    let (rank, probability) = if temperature > 0.0 && scores.len() == n {
        // Value-weighted: spread softmax mass over *eligible* candidates only.
        // A non-positive ordering weight is ineligible; mapping it to
        // NEG_INFINITY makes `boltzmann_pick` treat it as zero mass (and if none
        // are eligible it falls back to the deterministic argmax). This keeps the
        // value draw consistent with the rank-weighted path, which never floats
        // an ineligible option up out of its sorted-to-the-bottom position.
        let eligible: Vec<f64> = scores
            .iter()
            .map(|&s| if s > 0.0 { s } else { f64::NEG_INFINITY })
            .collect();
        boltzmann_pick(&eligible, temperature, draw)
    } else {
        let weights = rank_weights();
        let rank = sampled_rank(n, weights, draw);
        (rank, behavior_probability_for_rank(n, weights, rank))
    };
    if rank > 0 && rank < n {
        let chosen = ordered.remove(rank);
        ordered.insert(0, chosen);
    }
    SampledOrder {
        ordered,
        behavior_probability: Some(probability),
    }
}

/// Reorder an already-ranked list so the **stochastically chosen** rank is moved
/// to the front, preserving the relative order of the rest. The downstream
/// "commit to the first feasible option" loops therefore try the sampled option
/// first and fall through to the next-best if it is not committable — graceful
/// degradation with no separate feasibility plumbing.
///
/// No-op (returns `ordered` unchanged) when the gate is off or there are fewer
/// than two candidates, preserving byte-identical behaviour.
pub fn reorder_by_draw<T>(
    ordered: Vec<T>,
    decision_seed: u64,
    player_id: usize,
    tick: u64,
    site: u64,
) -> Vec<T> {
    reorder_by_draw_traced(ordered, decision_seed, player_id, tick, site).ordered
}

/// Pick an index into a scored candidate slice (`scores` need not be sorted): the
/// indices are ranked by descending score, then the stochastic draw selects among
/// them — by rank (top-k 70/20/10) or, when `boltzmann_temperature > 0`, by value
/// (softmax over the scores). Returns the index of the chosen candidate. Used by
/// the specialised heads that argmax over a small fixed set of discrete
/// sub-actions.
///
/// Returns the plain argmax (gate off / <2 candidates / all-equal ties resolved
/// by lowest index) to preserve byte-identical behaviour when disabled. See
/// [`sampled_index_by_score_traced`] for the variant that also reports the
/// behaviour-policy probability needed for off-policy learning honesty.
pub fn sampled_index_by_score(
    scores: &[f64],
    decision_seed: u64,
    player_id: usize,
    tick: u64,
    site: u64,
) -> Option<usize> {
    sampled_index_by_score_traced(scores, decision_seed, player_id, tick, site).map(|(idx, _)| idx)
}

/// Like [`sampled_index_by_score`] but also returns the behaviour-policy
/// probability of the chosen index (`None` when the gate is off or there are
/// fewer than two candidates — a greedy argmax that needs no importance
/// correction). The probability is the softmax probability in value-weighted
/// mode, or the renormalised rank weight in the legacy rank-weighted mode.
pub fn sampled_index_by_score_traced(
    scores: &[f64],
    decision_seed: u64,
    player_id: usize,
    tick: u64,
    site: u64,
) -> Option<(usize, Option<f64>)> {
    // Read the (global) temperature only when the gate is on; gate-off ignores it.
    let temperature = if stochastic_policy_topk_enabled() {
        boltzmann_temperature()
    } else {
        0.0
    };
    sampled_index_by_score_traced_at(
        scores,
        temperature,
        decision_seed,
        player_id,
        tick,
        site,
    )
}

/// [`sampled_index_by_score_traced`] with the Boltzmann `temperature` passed in
/// explicitly rather than read from the global tunables — the unit-testable core.
fn sampled_index_by_score_traced_at(
    scores: &[f64],
    temperature: f64,
    decision_seed: u64,
    player_id: usize,
    tick: u64,
    site: u64,
) -> Option<(usize, Option<f64>)> {
    if scores.is_empty() {
        return None;
    }
    // Rank indices by descending score (stable on ties → lowest index first), so
    // the deterministic argmax is rank 0.
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        let sa = if scores[a].is_finite() { scores[a] } else { f64::NEG_INFINITY };
        let sb = if scores[b].is_finite() { scores[b] } else { f64::NEG_INFINITY };
        sb.total_cmp(&sa).then_with(|| a.cmp(&b))
    });
    if !stochastic_policy_topk_enabled() || idx.len() < 2 {
        return idx.first().copied().map(|i| (i, None));
    }
    let draw = decision_unit_draw(decision_seed, player_id, tick, site);
    let (rank, probability) = if temperature > 0.0 {
        // Softmax over the score-sorted scores; `rank` indexes the sorted order.
        let sorted_scores: Vec<f64> = idx.iter().map(|&i| scores[i]).collect();
        boltzmann_pick(&sorted_scores, temperature, draw)
    } else {
        let weights = rank_weights();
        let rank = sampled_rank(idx.len(), weights, draw);
        (rank, behavior_probability_for_rank(idx.len(), weights, rank))
    };
    idx.get(rank).copied().map(|i| (i, Some(probability)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // The gate is a process-wide env var, so gate-toggling tests must not run
    // concurrently with one another (cargo runs tests in parallel by default).
    static GATE_LOCK: Mutex<()> = Mutex::new(());

    fn with_gate<T>(on: bool, f: impl FnOnce() -> T) -> T {
        let _guard = GATE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        if on {
            std::env::set_var(ENABLE_ENV, "1");
        } else {
            std::env::remove_var(ENABLE_ENV);
        }
        let out = f();
        std::env::remove_var(ENABLE_ENV);
        out
    }

    #[test]
    fn draw_is_deterministic_and_unit_ranged() {
        for seed in [0u64, 7, 911, u64::MAX] {
            for pid in 0..23usize {
                for tick in [0u64, 1, 1000, 90_000] {
                    let d = decision_unit_draw(seed, pid, tick, site::POSSESSION);
                    assert!((0.0..1.0).contains(&d), "draw {d} out of range");
                    // Same inputs reproduce exactly.
                    assert_eq!(d, decision_unit_draw(seed, pid, tick, site::POSSESSION));
                }
            }
        }
    }

    #[test]
    fn distinct_sites_decorrelate_same_player_tick() {
        let a = decision_unit_draw(7, 4, 100, site::POSSESSION);
        let b = decision_unit_draw(7, 4, 100, site::DEFENSIVE);
        assert_ne!(a, b);
    }

    #[test]
    fn sampled_rank_partitions_the_unit_interval_70_20_10() {
        // Weights are renormalised over their (float) sum, so the band edges sit
        // at ~0.70 and ~0.90 modulo ~1e-16; assert clearly *inside* each band.
        let w = [0.70, 0.20, 0.10];
        assert_eq!(sampled_rank(3, w, 0.00), 0);
        assert_eq!(sampled_rank(3, w, 0.50), 0);
        assert_eq!(sampled_rank(3, w, 0.699), 0);
        assert_eq!(sampled_rank(3, w, 0.701), 1);
        assert_eq!(sampled_rank(3, w, 0.85), 1);
        assert_eq!(sampled_rank(3, w, 0.899), 1);
        assert_eq!(sampled_rank(3, w, 0.901), 2);
        assert_eq!(sampled_rank(3, w, 0.999), 2);
    }

    #[test]
    fn sampled_rank_renormalises_for_two_candidates() {
        let w = [0.70, 0.20, 0.10];
        // 0.70 / 0.90 ≈ 0.7778 boundary.
        assert_eq!(sampled_rank(2, w, 0.0), 0);
        assert_eq!(sampled_rank(2, w, 0.77), 0);
        assert_eq!(sampled_rank(2, w, 0.78), 1);
        assert_eq!(sampled_rank(2, w, 0.999), 1);
        // Never returns rank 2 with only two candidates.
        assert_eq!(sampled_rank(2, w, 1.0), 1);
    }

    #[test]
    fn sampled_rank_forces_single_candidate_and_handles_degenerate_weights() {
        assert_eq!(sampled_rank(1, [0.7, 0.2, 0.1], 0.99), 0);
        assert_eq!(sampled_rank(0, [0.7, 0.2, 0.1], 0.99), 0);
        // All-zero weights → best.
        assert_eq!(sampled_rank(3, [0.0, 0.0, 0.0], 0.99), 0);
        // Non-finite draw → best.
        assert_eq!(sampled_rank(3, [0.7, 0.2, 0.1], f64::NAN), 0);
    }

    #[test]
    fn reorder_is_a_noop_when_gate_off() {
        with_gate(false, || {
            let v = vec!["a", "b", "c", "d"];
            let out = reorder_by_draw(v.clone(), 7, 3, 100, site::POSSESSION);
            assert_eq!(out, v);
        });
    }

    #[test]
    fn reorder_moves_sampled_rank_to_front_preserving_rest() {
        with_gate(true, || {
            // Find inputs whose draw lands in each rank band, then check the
            // reorder rotates exactly that element to the front.
            let base = vec![10u32, 20, 30, 40];
            for (pid, tick) in [(0, 0), (1, 5), (2, 9), (3, 42), (4, 1234)] {
                let draw = decision_unit_draw(7, pid, tick, site::POSSESSION);
                let rank = sampled_rank(base.len(), [0.70, 0.20, 0.10], draw);
                let out = reorder_by_draw(base.clone(), 7, pid, tick, site::POSSESSION);
                let mut expected = base.clone();
                let chosen = expected.remove(rank);
                expected.insert(0, chosen);
                assert_eq!(out, expected, "rank {rank} pid {pid} tick {tick}");
            }
        });
    }

    #[test]
    fn behavior_probability_matches_renormalised_rank_weight() {
        let w = [0.70, 0.20, 0.10];
        // Three candidates: exactly the rank weights (sum ≈ 1).
        assert!((behavior_probability_for_rank(3, w, 0) - 0.70).abs() < 1e-9);
        assert!((behavior_probability_for_rank(3, w, 1) - 0.20).abs() < 1e-9);
        assert!((behavior_probability_for_rank(3, w, 2) - 0.10).abs() < 1e-9);
        // Two candidates renormalise to 0.70/0.90 and 0.20/0.90.
        assert!((behavior_probability_for_rank(2, w, 0) - 0.70 / 0.90).abs() < 1e-9);
        assert!((behavior_probability_for_rank(2, w, 1) - 0.20 / 0.90).abs() < 1e-9);
        // Forced single candidate is certain.
        assert_eq!(behavior_probability_for_rank(1, w, 0), 1.0);
        // A rank beyond the weighted ranks gets a small positive floor (never 0,
        // so an importance ratio can't divide by zero).
        assert!(behavior_probability_for_rank(3, w, 5) > 0.0);
    }

    #[test]
    fn sampled_index_is_plain_argmax_when_gate_off() {
        with_gate(false, || {
            let scores = [0.1, 0.9, 0.5];
            assert_eq!(
                sampled_index_by_score(&scores, 7, 1, 1, site::BEAT_DEFENDER),
                Some(1)
            );
        });
    }

    // ---- value-weighted (Boltzmann) mode ----

    #[test]
    fn boltzmann_pick_is_argmax_for_nonpositive_or_degenerate_inputs() {
        // Non-positive / non-finite temperature → greedy argmax, certain.
        assert_eq!(boltzmann_pick(&[0.1, 0.9, 0.5], 0.0, 0.99), (1, 1.0));
        assert_eq!(boltzmann_pick(&[0.1, 0.9, 0.5], -1.0, 0.99), (1, 1.0));
        assert_eq!(boltzmann_pick(&[0.1, 0.9, 0.5], f64::NAN, 0.99), (1, 1.0));
        // Fewer than two candidates is forced.
        assert_eq!(boltzmann_pick(&[42.0], 2.0, 0.99), (0, 1.0));
        // No finite score → argmax fallback at index 0.
        let nan = f64::NAN;
        assert_eq!(boltzmann_pick(&[nan, nan], 2.0, 0.99), (0, 1.0));
    }

    #[test]
    fn boltzmann_pick_partitions_the_unit_interval_by_softmax_mass() {
        // Two candidates, scores 2 and 0, temperature 1 ⇒ softmax = e^2 : e^0,
        // i.e. p0 = e^2/(e^2+1) ≈ 0.8808, p1 ≈ 0.1192. The cumulative sampler
        // must put the band edge at ~0.8808.
        let scores = [2.0, 0.0];
        let p0 = 1.0 / (1.0 + (-2.0_f64).exp()); // e^2/(e^2+1)
        let (i_lo, prob_lo) = boltzmann_pick(&scores, 1.0, 0.0);
        assert_eq!(i_lo, 0);
        assert!((prob_lo - p0).abs() < 1e-9, "prob {prob_lo} vs {p0}");
        // Just inside the first band stays on index 0.
        assert_eq!(boltzmann_pick(&scores, 1.0, p0 - 1e-6).0, 0);
        // Just past the edge crosses to index 1.
        let (i_hi, prob_hi) = boltzmann_pick(&scores, 1.0, p0 + 1e-6);
        assert_eq!(i_hi, 1);
        assert!((prob_hi - (1.0 - p0)).abs() < 1e-9, "prob {prob_hi}");
    }

    #[test]
    fn boltzmann_pick_explores_near_ties_more_than_clear_gaps() {
        // Probability mass on the 2nd-best grows as the score gap shrinks: a
        // near-tie should leave a much larger non-best mass than a clear gap.
        let near = boltzmann_pick(&[1.00, 0.99], 0.5, 1.0).1; // p(2nd) for a near-tie
        let clear = boltzmann_pick(&[1.00, 0.00], 0.5, 1.0).1; // p(2nd) for a clear gap
        assert!(near > clear, "near-tie {near} should exceed clear-gap {clear}");
        // And hotter temperature flattens (more exploration) vs. colder.
        let hot = boltzmann_pick(&[1.0, 0.0], 2.0, 1.0).1;
        let cold = boltzmann_pick(&[1.0, 0.0], 0.2, 1.0).1;
        assert!(hot > cold, "hot {hot} should exceed cold {cold}");
    }

    #[test]
    fn boltzmann_pick_ignores_non_finite_candidates() {
        // A NaN-scored candidate contributes zero mass and is never selected.
        let scores = [1.0, f64::NAN, 0.5];
        for draw in [0.0, 0.25, 0.5, 0.75, 0.999] {
            let (idx, prob) = boltzmann_pick(&scores, 1.0, draw);
            assert_ne!(idx, 1, "NaN candidate selected at draw {draw}");
            assert!(prob > 0.0);
        }
    }

    #[test]
    fn value_weighted_reorder_falls_back_to_rank_weighted_at_zero_temperature() {
        // With temperature 0 the scored reorder must reproduce the rank-weighted
        // reorder exactly (same front element and behaviour probability), so the
        // default-0 tunable stays byte-identical to the legacy path.
        with_gate(true, || {
            let base = vec![10u32, 20, 30, 40];
            let scores = [4.0, 3.0, 2.0, 1.0];
            for (pid, tick) in [(0, 0), (1, 5), (2, 9), (3, 42), (7, 1234)] {
                let legacy = reorder_by_draw_traced(base.clone(), 7, pid, tick, site::POSSESSION);
                let scored = reorder_with_scores_traced_at(
                    base.clone(),
                    &scores,
                    0.0,
                    7,
                    pid,
                    tick,
                    site::POSSESSION,
                );
                assert_eq!(scored.ordered, legacy.ordered, "order pid {pid}");
                assert_eq!(
                    scored.behavior_probability, legacy.behavior_probability,
                    "prob pid {pid}"
                );
            }
        });
    }

    #[test]
    fn value_weighted_reorder_falls_back_when_scores_misaligned() {
        // A score list whose length != candidate count can't be trusted, so the
        // rank-weighted path is used even at a positive temperature.
        with_gate(true, || {
            let base = vec![10u32, 20, 30, 40];
            let short_scores = [4.0, 3.0]; // wrong length
            let legacy = reorder_by_draw_traced(base.clone(), 7, 2, 9, site::SUPPORT);
            let scored = reorder_with_scores_traced_at(
                base.clone(),
                &short_scores,
                2.5,
                7,
                2,
                9,
                site::SUPPORT,
            );
            assert_eq!(scored.ordered, legacy.ordered);
            assert_eq!(scored.behavior_probability, legacy.behavior_probability);
        });
    }

    #[test]
    fn value_weighted_reorder_is_a_noop_when_gate_off() {
        with_gate(false, || {
            let base = vec!["a", "b", "c", "d"];
            let scores = [4.0, 3.0, 2.0, 1.0];
            let out =
                reorder_with_scores_traced_at(base.clone(), &scores, 2.5, 7, 3, 100, site::POSSESSION);
            assert_eq!(out.ordered, base);
            assert_eq!(out.behavior_probability, None);
        });
    }

    #[test]
    fn value_weighted_reorder_promotes_the_softmax_sampled_candidate() {
        // With the gate on and a positive temperature, the front element and its
        // reported probability must match a direct Boltzmann draw over the scores.
        // All scores positive (eligible) so the eligibility mask is a no-op here.
        with_gate(true, || {
            let base = vec![10u32, 20, 30, 40];
            let scores = [3.0, 2.5, 1.0, 0.5];
            let temperature = 1.5;
            for (pid, tick) in [(0, 0), (1, 5), (2, 9), (3, 42), (7, 1234)] {
                let draw = decision_unit_draw(7, pid, tick, site::DEFENSIVE);
                let (rank, prob) = boltzmann_pick(&scores, temperature, draw);
                let scored = reorder_with_scores_traced_at(
                    base.clone(),
                    &scores,
                    temperature,
                    7,
                    pid,
                    tick,
                    site::DEFENSIVE,
                );
                assert_eq!(scored.ordered[0], base[rank], "front pid {pid}");
                assert_eq!(scored.behavior_probability, Some(prob), "prob pid {pid}");
            }
        });
    }

    #[test]
    fn value_weighted_reorder_never_promotes_ineligible_candidates() {
        // A non-positive ordering weight is ineligible (illegal action): it must
        // get zero exploration mass, so the value draw never floats it to the
        // front, and the reported probability matches a softmax over the eligible
        // candidates only (here just the first two).
        with_gate(true, || {
            let base = vec![10u32, 20, 30, 40];
            let scores = [3.0, 2.0, 0.0, -1.0]; // only indices 0,1 eligible
            let masked = [3.0, 2.0, f64::NEG_INFINITY, f64::NEG_INFINITY];
            let temperature = 1.0;
            for (pid, tick) in [(0, 0), (1, 5), (2, 9), (3, 42), (7, 1234), (11, 77)] {
                let draw = decision_unit_draw(7, pid, tick, site::POSSESSION);
                let (rank, prob) = boltzmann_pick(&masked, temperature, draw);
                let scored = reorder_with_scores_traced_at(
                    base.clone(),
                    &scores,
                    temperature,
                    7,
                    pid,
                    tick,
                    site::POSSESSION,
                );
                assert!(
                    scored.ordered[0] == 10 || scored.ordered[0] == 20,
                    "promoted an ineligible candidate: {:?}",
                    scored.ordered[0]
                );
                assert_eq!(scored.ordered[0], base[rank], "front pid {pid}");
                assert_eq!(scored.behavior_probability, Some(prob), "prob pid {pid}");
            }
        });
    }

    #[test]
    fn value_weighted_reorder_with_no_eligible_candidates_keeps_argmax() {
        // All ordering weights non-positive ⇒ no eligible candidate ⇒ fall back to
        // the deterministic front (no reorder), reported with certainty.
        with_gate(true, || {
            let base = vec![10u32, 20, 30];
            let scores = [0.0, 0.0, -2.0];
            let out = reorder_with_scores_traced_at(
                base.clone(),
                &scores,
                1.0,
                7,
                3,
                9,
                site::LOOSE_BALL,
            );
            assert_eq!(out.ordered, base);
            assert_eq!(out.behavior_probability, Some(1.0));
        });
    }

    #[test]
    fn value_weighted_index_matches_boltzmann_over_sorted_scores() {
        // sampled_index_by_score_traced_at in value-weighted mode must return the
        // original index whose score the Boltzmann draw selected (scores need not
        // be pre-sorted), plus that candidate's softmax probability.
        with_gate(true, || {
            let scores = [0.5, 3.0, 2.5, 1.0]; // argmax at index 1
            let temperature = 1.25;
            for (pid, tick) in [(0, 0), (1, 5), (2, 9), (3, 42)] {
                let mut sorted: Vec<f64> = scores.to_vec();
                sorted.sort_by(|a, b| b.total_cmp(a)); // [3.0, 2.5, 1.0, 0.5]
                let draw = decision_unit_draw(7, pid, tick, site::DRIBBLE_KIND);
                let (rank, prob) = boltzmann_pick(&sorted, temperature, draw);
                let chosen_score = sorted[rank];
                let got = sampled_index_by_score_traced_at(
                    &scores,
                    temperature,
                    7,
                    pid,
                    tick,
                    site::DRIBBLE_KIND,
                )
                .unwrap();
                assert_eq!(scores[got.0], chosen_score, "score pid {pid}");
                assert_eq!(got.1, Some(prob), "prob pid {pid}");
            }
        });
    }

    #[test]
    fn value_weighted_index_is_argmax_when_gate_off() {
        with_gate(false, || {
            let scores = [0.1, 0.9, 0.5];
            // Even with a positive temperature, gate-off is the greedy argmax and
            // reports no behaviour probability.
            assert_eq!(
                sampled_index_by_score_traced_at(&scores, 3.0, 7, 1, 1, site::BEAT_DEFENDER),
                Some((1, None))
            );
        });
    }
}
