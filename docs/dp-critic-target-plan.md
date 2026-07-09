# DP-bootstrapped critic target

Replace the neural critic's tabular-Q bootstrap with a value-iterated
abstract-state DP table, breaking the dependency chain that caps the critic
at the tabular Q's ceiling.

## The problem

The neural critic target at `world.rs:18178-18205`:

```
target = adjusted_reward + gamma * tabular_max_next
```

where `tabular_max_next = policy.best_value_hierarchical(&next_state)` — the
tabular Q-value of the successor state. The neural net regresses onto the
tabular Q's aliased estimates by construction, inheriting its ceiling.

Two existing levers try to break this:
- **MC critic target** (`dd_soccer_enable_mc_critic_target`): realized returns —
  high-variance, goal-dominated (~100+ vs ~1), requires standardization
- **Neural self-bootstrap** (`dd_soccer_enable_neural_self_bootstrap`): blends
  net's own prediction (0.45) with tabular (0.55) — the tabular half still
  drags

Neither works because both still depend on the tabular Q either directly or
as the dominant blend component.

## The lever

The DP value-iterated table V(b) over the 486-bucket abstract MCD
(`soccer_dp_value_iteration`, `soccer.rs:14748`) is:

1. **Computed from raw (bucket, reward, next_bucket) transitions** —
   independent of the tabular Q's state abstraction and aliasing
2. **Propagates delayed rewards backward** — Jacobi sweeps distribute
   goal/shoot/turnover credit across the full sequence
3. **Always available** — no cold-start problem

But it is currently consumed only for the *team correlated reward* in
`soccer_dp_bootstrapped_replay_transitions` (`soccer.rs:14862`), NOT for the
neural critic target or decision-time value blend.

## Approach #1 — DP-bootstrapped critic target

In `neural_training_samples_filtered` (`world.rs:18080`), replace
`tabular_max_next` with `V_DP(b(s'))`:

```
target = adjusted_reward + gamma * V_DP(b')
       — no tabular Q anywhere in this target
```

### Changes

1. **Make DP functions `pub(crate)`** in `soccer.rs` so `world.rs` can call
   `soccer_dp_state_bucket` and `soccer_dp_value_iteration`.

2. **Add gate** `dd_soccer_enable_dp_critic_target()` in `world.rs` following
   the existing `soccer_env_flag_enabled` pattern near line 2446.

3. **Cache DP table on `SoccerMatch`**: add field
   `dp_value_table: Option<(BTreeMap<u32, f64>, usize, usize)>`. Populate
   in `apply_full_game_learning_if_ready` (after deferred credits, before
   neural training) from `self.episode_learning_transitions`.

4. **Consume in critic target**: in `neural_training_samples_filtered`, when
   the gate is on and the DP table is available, look up
   `V_DP(b(successor_state))` instead of calling
   `policy.best_value_hierarchical`.

### Gate

`DD_SOCCER_ENABLE_DP_CRITIC_TARGET` — default OFF, `soccer_env_flag_enabled`.
When OFF the code path is byte-identical to current main.

### Why this should work

The DP value propagation breaks free of the tabular Q's aliasing. The abstract
MCD state has 486 buckets covering ball zone, possession, score sign, and
phase — the same coarse structure the tabular Q already aliases over, but the
DP value iteration computes V(b) directly from rewards without the tabular Q's
limited-resolution max-Q lookup. So the critic gets a better bootstrap target,
and the neural net can learn the fine-grained residual on top.

## Approach #2 — DP-as-prior at decision time

In `neural_blended_action` (`world.rs:15944`), the blend is:

```
value(s,a) = lambda * V_neural(s,a) + (1-lambda) * Q_tabular(s,a)
```

Replace `Q_tabular` with `V_DP(b(s))`:

```
value(s,a) = lambda * V_neural(s,a) + (1-lambda) * V_DP(b(s))
```

Benefits: V_DP is always available (no cold-start ramp), never aliased by
the tabular Q's limitations.

### Changes

Smaller — just replace the tabular Q reference in the blend with the DP
bucket lookup for the current state. Gate behind the same
`DD_SOCCER_ENABLE_DP_CRITIC_TARGET` flag.

## Implementation order

1. Approach #1 only (DP-bootstrapped critic target) — the higher-leverage
   change because it fixes the training signal. The model will then need
   some number of games to learn from the improved targets, at which point
   the decision-time value improves automatically via the trained net.

2. Approach #2 can be added independently if Approach #1 doesn't move the
   needle alone — it gives the live policy a better fallback prior even
   before the net is warm.
