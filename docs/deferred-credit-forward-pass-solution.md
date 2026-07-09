# The forward-pass climb solution: deferred pass credit (attribution)

> **Status (2026-07-09):** This is the one **confirmed** lever that broke the forward-pass
> *selection* wall. It climbs the forward-pass advancement metric vs the learned base, but it does
> **not** by itself beat the pure-analytic engine on payoff/GD — the bottleneck moves downstream
> (conversion), see [§ What it does NOT solve](#what-it-does-not-solve). Companion to
> [how-to-climb-codex-conversation.md](how-to-climb-codex-conversation.md).

## TL;DR

The neural learner completes passes at ~90% but chooses **safe (lateral/back)** far more than the
analytic engine — a **selection** problem, not an exposure or completion problem (widening the
candidate cap 3→11 moved forward-share `0.00`). The fix is **attribution**: back-date turnover
blame (and completion credit) onto the *decision transition that launched the pass*, so the actor
can learn *which forward-pass decisions are good vs turnover-traps*. Enabled by
`DD_SOCCER_ENABLE_DEFERRED_PASS_CREDIT=1`.

## The mechanism

Gate: `dd_soccer_enable_deferred_pass_credit()` — [soccer.rs:22211](../src/des/general/soccer.rs#L22211).

Without it, a turnover penalty lands only on the per-tick transition where the ball is physically
lost (`deferred_reward_transitions`) — far from, and disconnected from, the *decision* that chose
the risky forward ball. The actor never learns the decision was bad.

With it ([world.rs:33936-33954](../src/des/general/soccer/world.rs#L33936)):

- the turnover penalty is **back-dated onto the actual decision transitions** via
  `deferred_reward_credits`, so it reaches the **full-game replay** the actor/critic train on;
- the per-tick `deferred_reward_transitions` path is **skipped** when the gate is on, to avoid
  double-penalizing.

Net effect: the launch transition that chose a forward pass gets credited/blamed by that pass's
**eventual outcome** (completed & retained vs turned over). That is the signal the POMDP actor
needs to *select* forward passes that actually work, instead of defaulting to safe recycles.

## Confirmed result

A/B (deferred-credit ON vs OFF, both on the winning stack, vs analytic; `/tmp/gate-ab-defcred`):

| metric | baseline (OFF) | candidate (ON) | Δ |
|---|---|---|---|
| completed forward passes/game | 1.4 | **2.8** | **+1.46** |
| forward share of completions | 5% | **10%** | **2×** |
| pass completion | 91% | 89% | ~flat (not spam) |
| advancement verdict | — | **CLIMB** (paired lower bound > 0) | out-progresses base |

So deferred-credit **doubles forward passing** and **doubles forward share** with completion held —
it genuinely moves the *selection* the net makes, not just volume. This is the real forward-pass
climb.

## The winning stack it sits on

Both arms of the confirming A/Bs use:

```
DD_SOCCER_ENABLE_DISCRETIZED_KICK=0
DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD=1        # graded chance-quality terminal reward
SOCCER_NEURAL_TARGET_SCALE=30
SOCCER_NEURAL_TARGET_CLIP=15                    # un-crushed critic window
SOCCER_NEURAL_TARGET_POPART=true               # PopArt over the widened target distribution
DD_SOCCER_ENABLE_LEARNED_MPC_OBJECTIVE=1
DD_SOCCER_ENABLE_DEFERRED_PASS_CREDIT=1         # <-- the forward-pass climb lever
```

(For A/Bs, the training regime is mcts-off: the pass-like margin gate only fires with mcts on and
would prune injected/edge candidates — [world.rs](../src/des/general/soccer/world.rs) pass-like
margin gate is `if blend.mcts_enabled`.)

## What it does NOT solve

**It broke forward-pass *selection* but did not convert to beating pure analytic.** In the
confirming run, overall `DECISION: REJECT` — payoff stayed ~0.5 and GD ~−7 vs analytic at low
power. The advancement metric climbs; the scoreline/payoff does not.

**This is the key strategic clue:** *selection fixed + payoff still flat* means the real gap is now
**downstream of passing** — conversion/finishing, defense, or off-ball movement — not the pass
choice itself. Chasing more forward-pass levers past this point has diminishing returns.

## Levers falsified around this result (for the record)

- **Forward-release candidate injection** (force the Cap-B-excluded forward pass into the scored
  root): fires (575 forced passes, all survive) but **GD −21** — forcing risky forward passes makes
  the net play more forward and *lose more*. Dropped. (branch `plateau-net`, not merged.)
- **Candidate exposure widening** (`SOCCER_NEURAL_MCTS_PASS_TARGET_CANDIDATES` 3→11): forward-share
  Δ `0.00`. Exposure was not the wall.
- **Reward-scale un-crush** (target_scale 120): degraded policy (GD −61).
- **Territory / pitch_value shaping:** policy-invariant by construction; cannot raise the ceiling.

## Next frontier + a known harness bug to fix first

Per Codex, the next lever is **conversion/capacity** (can the policy *use* the now-correct
attribution) — e.g. `SOCCER_LEAGUE_HIDDEN_UNITS=128`, framed as "can the critic/policy convert",
not another forward-pass hack.

**Per-decision specialist learning** (pass/dribble/shot specialist heads + curriculum, formation LP
coupling) is the natural way to push each POMDP decision independently — **but it currently cannot
be A/B'd through `soccer_outcome_ab_run`**: the persisted `SoccerNeuralNetworkSnapshot`
([soccer.rs](../src/des/general/soccer.rs) — has `policy_head`, `line_depth_head`,
`mpc_objective_head`) does **not** include `skill_policy_heads`, and `set_team_neural_brain`
([world.rs:12234](../src/des/general/soccer/world.rs#L12234)) restores only `policy_head`. So trained
specialists are **discarded at eval** (fresh untrained heads are created), and any specialist A/B is
invalid. **Fix required before testing per-decision learning:** persist `skill_policy_heads` in the
snapshot (a `SoccerPolicySpecialistHeadSnapshot` type already exists,
[soccer.rs:38812](../src/des/general/soccer.rs#L38812)) and restore them in `set_team_neural_brain`;
then run round-based (≥160-round curriculum, ≥600 eval games), not game-based.
