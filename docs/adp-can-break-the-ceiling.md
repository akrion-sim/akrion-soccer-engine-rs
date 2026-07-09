# Can Approximate DP break the ceiling?

Whether Dynamic Programming (value iteration on an aggregated state space) can solve
the conversion + defense gap that neural nets (with self-bootstrap + maxa) haven't
closed.

---

## The bottleneck has moved

Deferred pass credit (`DD_SOCCER_ENABLE_DEFERRED_PASS_CREDIT`) solved forward-pass
**selection** — it doubled forward passes and forward share while holding completion
rate. But payoff vs analytic stayed ~0.5. The bottleneck moved downstream:

- **Conversion**: the net gets forward but doesn't finish (goal/shot conversion)
- **Defense**: the opponent scores more than the candidate does

The self-bootstrap + maxa lever improves payoff by +0.050 (0.472→0.522, W=0.7). But
+0.050 isn't enough to clear the eval gate (Wilson LB ≥ 0.500). W=0.9 crashed (−0.020,
too much deadly-triad amplification).

**The question**: can Approximate Dynamic Programming provide better value targets
than the neural self-bootstrap, specifically for the sparse long-horizon signals
(goals, conceded goals) that conversion and defense depend on?

---

## What ADP is, and why it fits this problem

ADP = value iteration (or policy iteration) on an **aggregated / coarsely bucketed**
representation of the state space. The aggregation trades off bias (coarse buckets
lose detail) for variance (dense observations per bucket → accurate, stable estimates).

For soccer, the key action-relevant state can be bucketed coarsely:

| Dimension | Buckets | Example |
|---|---|---|
| Ball x-zone | 3 | defending, middle, attacking third |
| Ball y-zone | 3 | left, center, right |
| Possession | 3 | own, loose, opponent |
| Score diff | 3 | losing, level, winning |
| Phase | 6 | build, attack, defend, transition, set-piece, kickoff |

≈ 3×3×3×3×6 = **486 buckets** (the exact same schema the existing DP bootstrap uses).

On this 486-state MDP, value iteration converges in ~40 sweeps (microseconds on a
laptop). The resulting V(bucket) is the **true optimal value of the aggregate MDP** —
it captures the long-term consequences of being in each coarse state.

**Why this matters for conversion + defense:**

1. **A goal is a sparse terminal event** (±1, a few times per game). The per-step
   forward-pass reward is dense (~0.1-4.0, hundreds per game). The neural net's
   training target is dominated by the dense forward-pass signal; the goal signal is
   buried in noise. Value iteration on the aggregate MDP **propagates the goal
   reward backward efficiently**: each sweep moves the value one step further back,
   so after 40 sweeps, buckets that preceded a goal by 40 decision steps have
   elevated value.

2. **Dense aggregation cancels noise.** With 486 buckets and thousands of
   transitions per game, each bucket's mean reward and next-state distribution are
   accurate. The DP table's V(bucket) has far lower variance than the per-decision
   neural self-bootstrap target (which bootstraps off a noisy 24-unit MLP).

3. **The DP table explicitly values defensive states.** Bucket (our attacking
   third, opponent possession) correctly has negative value because it tends to
   lead to opponent goals. The neural net has to learn this from sparse experience;
   the DP table captures it from the aggregate statistics of thousands of
   transitions.

---

## What already exists

The codebase already ships `DD_SOCCER_ENABLE_DP_BOOTSTRAP` (`soccer.rs:22262`),
which builds a per-game DP table and uses it to bootstrap the **full-game team
return** (not the per-decision target):

| Parameter | Default | Range | Env var |
|---|---|---|---|
| Horizon | 16 ticks | 1..4096 | `DD_SOCCER_DP_BOOTSTRAP_HORIZON` |
| Sweeps | 40 | 1..1000 | `DD_SOCCER_DP_BOOTSTRAP_SWEEPS` |

It also ships `SOCCER_APPROX_DP_REPLAY_PASSES` (`world.rs:2422`), which runs
multiple forward/backward passes of the replay through the **tabular Q** — fitted
value iteration.

**Why neither currently solves the conversion gap:**

- **The DP bootstrap affects the full-game team return** (blended at weight 0.35
  into each transition's reward), which then trains the **tabular Q**. The neural
  net's per-decision target bootstraps off the tabular Q's `best_value_hierarchical`,
  so the DP effect is indirect, delayed, and diluted.

- **The DP table is per-game** (built fresh from one game's transitions). With
  ~10-30 observations per bucket per game, the value estimates are nearly as noisy
  as MC returns. A persistent DP table (accumulated across hundreds of games) would
  have thousands of observations per bucket.

- **The horizon is 16 ticks (~1 second)**. A goal often requires a sequence of
  3-10 seconds of attacking play. The n-step return truncates at 16 ticks, which
  may not reach the key pass/dribble decisions that created the chance.

- **Approx DP passes only train the tabular Q**, not the neural net. The neural
  net's self-bootstrap target (neural max-a blended with tabular Q) doesn't benefit
  directly.

---

## Proposal: Persistent DP Value Table for the Neural Bootstrap Target

Replace the **tabular Q** in the self-bootstrap blend with a **persistent DP
value table V_dp(bucket)** aggregated across many games. The neural target becomes:

```
max_next = w · V_neural_max_a(s', a)  +  (1-w) · V_dp(bucket(s'))
        = w · max_{a∈families} V_net(s', a)  +  (1-w) · V_dp(phi(s'))
```

where `phi(s) → bucket` maps the full state to a 486-bucket coarse key. The DP
table is maintained as:

- **Transition counts**: `N(b, a, b') ++` for each observed (s → s') transition
- **Reward sums**: `R(b, a) = mean(r | b, a)` of the dense per-tick reward
- **Value iteration**: periodic (every N games) solve V = T*V on the aggregate MDP

The V_dp(bucket) is the **optimal value of the aggregate MDP**: given the average
transition dynamics and rewards in each bucket, what is the expected discounted sum
of future rewards from being in this bucket?

### Why this helps conversion + defense

1. **Dense observations**: With 486 buckets and hundreds of games, each bucket has
   hundreds to thousands of observations. The aggregate statistics are accurate.

2. **Sparse reward propagation**: A goal (+1, rare) propagates backward through
   value iteration. After 40 sweeps at γ=0.98, buckets 40 decision steps before a
   goal have elevated value. The DP table captures "which states lead to goals"
   and "which states prevent opponent goals" without the net having to learn this
   from scratch.

3. **Stable target**: V_dp changes slowly (incremental game-by-game updates). The
   neural self-bootstrap target is stabilized by the DP anchor, preventing the
   deadly-triad divergence seen at W=0.9.

4. **Neural net learns the residual**: V_neural_residual(s) = V_true(s) - V_dp(bucket(s))
   is small and bounded. The net only needs to learn the within-bucket variation —
   how the exact player positions and velocities differ from the bucket average.
   This is a fundamentally easier learning problem.

### Comparison with the current approach

| Property | Current (W=0.7) | Proposed (DP table) |
|---|---|---|
| Bootstrap anchor | Tabular Q (35-dim, sparse, noisy) | DP V(bucket) (5-dim, dense, stable) |
| Goal propagation | Through Q-learning (per-step, slow) | Through value iteration (per-sweep, fast) |
| Target variance | High (self-bootstrap amplifies noise) | Low (DP anchors, residual is small) |
| Deadly-triad risk | Present (W=0.9 crashed) | Controlled (DP table is stable) |
| Generalization | Continuous (neural net) | Discrete (bucket averages) |

---

## Implementation paths (three tiers)

### Tier 1 — Enable the existing ADP path and tune `DD_SOCCER_ENABLE_DP_BOOTSTRAP`

Set env vars and run the existing treatment with DP ON. The effect is indirect
(DP helps the tabular Q, which feeds the neural bootstrap target), but it is the
fastest test.

```
DD_SOCCER_ENABLE_DP_BOOTSTRAP=1
DD_SOCCER_DP_BOOTSTRAP_HORIZON=64       # longer n-step (was 16)
DD_SOCCER_DP_BOOTSTRAP_SWEEPS=200       # more VI sweeps (was 40)
SOCCER_APPROX_DP_REPLAY_PASSES=8        # multi-pass Q training (was 1)
```

Implementation note: `SOCCER_APPROX_DP_REPLAY_PASSES` still overrides explicitly,
but forward-pass climb profiles now default this replay budget to 8 when any of
`DD_SOCCER_FORWARD_PASS_CLIMB_CURRICULUM`,
`SOCCER_POLICY_PROMOTION_FORWARD_PASS_PRIMARY`, or
`SOCCER_EVAL_REQUIRE_FORWARD_PASS_CLIMB` is active. Non-climb runs keep the old
default of 1 replay pass. The local league trainer and continuous RDS climb
manifest now also arm the Tier 1 DP bootstrap profile by default/preflight:
`DD_SOCCER_ENABLE_DP_BOOTSTRAP=1`, `DD_SOCCER_DP_BOOTSTRAP_HORIZON=64`, and
`DD_SOCCER_DP_BOOTSTRAP_SWEEPS=200`.

Run with W=0.7, 32 train, 90 eval. If payoff climbs from 0.522 to 0.545+, DP is
helping. If not, the indirect path is too weak.

Estimated: < 1 hour of code, 2 hours of compute.

### Tier 2 — Build a persistent DP table (no neural target change)

Add a global DP table (singleton / static) that accumulates (bucket, reward, next_bucket)
observations across all training games. Run value iteration on the merged table
after each game (or every N games). The table provides V(bucket) for any coarse
state.

Then replace the tabular Q anchor in the self-bootstrap target with the DP table's
V(bucket):

```rust
// Current (world.rs:18212-18247):
let tabular_max_next = policy.best_value_hierarchical(&next_state);
let max_next = w * neural_raw + (1-w) * tabular_max_next;

// Proposed:
use std::sync::LazyLock;
static DP_TABLE: LazyLock<Mutex<DpValueTable>> = ...;
let dp_value = DP_TABLE.lock().value(dp_bucket(&next_state));
let max_next = w * neural_raw + (1-w) * dp_value;
```

The DP value replaces the tabular Q's `best_value_hierarchical` in the self-bootstrap
blend. The tabular Q is still used for action selection (the actual policy), but
the **training target** for the neural net now uses the DP value as its anchor.

Estimated: 1-2 days of code, ~100 lines of Rust.

### Tier 3 — Full hierarchical value function (DP + neural residual)

The neural net explicitly learns V_residual(s) = V(s) - V_dp(bucket(s)). The
training target is:

```
target = r + γ · [V_dp(bucket(s')) + V_neural_residual(s')]
```

The DP table provides the coarse structure. The neural net learns the residual
(details within each bucket). Combined, the value function is:

```
V(s) = V_dp(phi(s)) + V_neural_residual(s)
```

This is the **theoretically cleanest** approach: the DP table handles the
coarse long-horizon structure (which buckets lead to goals, which don't), and
the neural net handles the fine-grained continuous residual (given the exact
player positions, is this better or worse than the bucket average?).

Estimated: 3-5 days of code, new neural head for the residual.

---

## What this does NOT solve

ADP on a coarse aggregation has **inherent bias**: two different states in the same
bucket are treated identically. If the ball is in the attacking third but one
state has the player with an open shot and the other has no shooting angle, the DP
table gives them the same value. The neural net (or residual) must distinguish them.

This means ADP alone (without a neural residual) is unlikely to beat analytic by
itself. The strength is in the **combination**: DP provides the stable backbone,
the neural net provides the fine-grained corrections.

---

## Promotion rule: best-of, not blended

The right way to include approximate DP alongside neural nets is:

1. **Inside learning**: let ADP/Bellman replay improve Q targets and provide the
   replay-backed prior that neural-authoritative scoring can consult.
2. **At promotion**: evaluate candidate artifacts through the same held-out
   HOME-vs-analytic / local-best gate and keep the stronger promoted artifact,
   whether the gain came mostly from neural weights, DP-improved Q tables, or the
   hybrid of both.
3. **Avoid action-distribution averaging**: do not average DP and neural actions
   at decision time. That can be worse than either parent. The gate should choose
   the winner for the next stage/promotion; the neural net can then learn from the
   DP-improved trajectory targets.

This keeps DP as a credit/target stabilizer and promotion contender, while the
neural net remains the deployable generalizer for full continuous POMDP state.

---

## Recommendation

**Start with Tier 1**: enable `DD_SOCCER_ENABLE_DP_BOOTSTRAP=1` with longer
horizon (64) and more sweeps (200). In forward-pass climb mode, the replay pass
budget now defaults to 8; set `SOCCER_APPROX_DP_REPLAY_PASSES=8` explicitly when
running outside that profile or when you want the manifest/logs to make the
choice obvious. Run as the next experiment with W=0.7, 32 training games.

If Tier 1 shows a measurable improvement (payoff +0.01 to +0.02), the DP signal is
reaching the neural net and Tier 2 is justified. If Tier 1 shows nothing, the
indirect path (DP → tabular Q → neural bootstrap) is too weak, and Tier 2 (direct
DP integration into the neural target) is the only real option.
