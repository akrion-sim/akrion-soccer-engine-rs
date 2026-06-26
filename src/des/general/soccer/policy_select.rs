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
//! [`WorldSnapshot`]: crate::des::general::soccer::WorldSnapshot

const ENABLE_ENV: &str = "DD_SOCCER_ENABLE_STOCHASTIC_POLICY_TOPK";

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
fn rank_weights() -> [f64; 3] {
    let t = &crate::des::general::soccer::tunables().policy_selection;
    [t.top1_weight, t.top2_weight, t.top3_weight]
}

/// Pick a rank in `0..n.min(3)` from a unit `draw`, using the first `n` rank
/// weights **renormalised** over what is available. Non-finite / non-positive
/// weights are floored to zero; if everything is zero the best (rank 0) is taken.
///
/// `n` is the number of candidates actually present. With `n >= 3` the full
/// `[0.70, 0.20, 0.10]` applies; with `n == 2` it is `[0.70, 0.20]` renormalised
/// to `~[0.78, 0.22]`; with `n <= 1` rank 0 is forced.
pub fn sampled_rank(n: usize, weights: [f64; 3], draw: f64) -> usize {
    let k = n.min(3);
    if k <= 1 {
        return 0;
    }
    let draw = if draw.is_finite() {
        draw.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut w = [0.0f64; 3];
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
pub fn behavior_probability_for_rank(n: usize, weights: [f64; 3], rank: usize) -> f64 {
    const FLOOR: f64 = 1e-3;
    let k = n.min(3);
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
        return if rank == 0 { 1.0 } else { FLOOR };
    }
    if rank < k && weights[rank].is_finite() && weights[rank] > 0.0 {
        (weights[rank] / total).max(FLOOR)
    } else {
        FLOOR
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
pub fn reorder_by_draw_traced<T>(
    mut ordered: Vec<T>,
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
    let weights = rank_weights();
    let draw = decision_unit_draw(decision_seed, player_id, tick, site);
    let rank = sampled_rank(n, weights, draw);
    if rank > 0 {
        let chosen = ordered.remove(rank);
        ordered.insert(0, chosen);
    }
    SampledOrder {
        ordered,
        behavior_probability: Some(behavior_probability_for_rank(n, weights, rank)),
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
/// indices are ranked by descending score, then the top-k stochastic draw selects
/// among the best three. Returns the index of the chosen candidate. Used by the
/// specialised heads that argmax over a small fixed set of discrete sub-actions.
///
/// Returns the plain argmax (gate off / <2 candidates / all-equal ties resolved
/// by lowest index) to preserve byte-identical behaviour when disabled.
pub fn sampled_index_by_score(
    scores: &[f64],
    decision_seed: u64,
    player_id: usize,
    tick: u64,
    site: u64,
) -> Option<usize> {
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
        return idx.first().copied();
    }
    let draw = decision_unit_draw(decision_seed, player_id, tick, site);
    let rank = sampled_rank(idx.len(), rank_weights(), draw);
    idx.get(rank).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_gate<T>(on: bool, f: impl FnOnce() -> T) -> T {
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
    fn sampled_index_is_plain_argmax_when_gate_off() {
        with_gate(false, || {
            let scores = [0.1, 0.9, 0.5];
            assert_eq!(
                sampled_index_by_score(&scores, 7, 1, 1, site::BEAT_DEFENDER),
                Some(1)
            );
        });
    }
}
