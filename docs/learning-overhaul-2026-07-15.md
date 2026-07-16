# Learning overhaul — 2026-07-15 audit vs proven RL systems

Why we plateau: this doc compares the engine's learning stack (both games) against
systems that demonstrably learn team sport at scale — MAPPO (Yu et al. 2021),
OpenAI Five, AlphaStar's league, Google Research Football (GRF) / TiZero — lists
the concrete divergences found by a five-agent source audit, and records what was
fixed today vs what remains. Everything cited was verified at source on `main`
(pre-fix state), not assumed from docs or memory.

## 1. What the successful systems all do

| Practice | MAPPO / Five / AlphaStar / GRF | This engine (pre-fix) |
|---|---|---|
| Value normalization (PopArt) | Always on — called "consistently helpful" in the MAPPO paper | Config default `false`; the continuous trainer reported `target_popart=false` locally; only the league/ratchet bins + the k8s manifest forced it on |
| Centralized critic V(s) | The "C" in MAPPO/CTDE — critic sees global state, actor doesn't | `DD_SOCCER_ENABLE_CENTRAL_CRITIC_V2` default OFF ⇒ default advantage baseline is a per-agent action-conditioned Q (SARSA-style) |
| Advantage normalization | Per-batch, always | Present (forced with outcome reward) ✓ |
| Actor learning rate | ~2.5e-4..5e-4, minibatched, LR schedule | **0.05**, per-sample SGD, no minibatching, no schedule (`SOCCER_POLICY_LEARNING_RATE`, soccer.rs:5680) |
| PPO surrogate | `min(rA, clip(r)A)` | hard-zero outside trust region (importance-weighted REINFORCE w/ cut) — workable but noisier |
| Reward design | Sparse goal ±1 + tiny ONE-TIME checkpoint shaping (GRF: 0.1×10, once each); dense shaping annealed to zero (Five) | goal=500 stacked with ~40 dense channels; several inert or double-paying paths (below) |
| One dependable terminal currency | goal/win constants, never rescaled | goal was triple-injected (~1500/conversion), turnover goals discounted to 250, and `DD_SOCCER_REWARD_KIND_SCALES` could rescale `Goal` ×[0.0001,4] |
| Self-play opponent selection | PFSP: fixtures ∝ p(1−p) ("challenge"); exploiter agents; anchored eval | flat product over the pool (`games_per_opp` each), no exploiters, no weighting |
| Population-based mutation evidence | PBT mutates on full eval windows | "final batch" bypass let a 1-game run compute a mutation (promotion gate was the only defense) |
| Episode/terminal hygiene | strict `done` semantics | `done=true` on every full-game replay row (overloaded); actor GAE re-derives terminals from a 0.3 s decision gap |
| Config | one canonical config, versioned | ~100 default-off env gates; the "real" stack exists only in launcher scripts + the k8s manifest — the historical source of the inert-lever bugs |
| Experience scale | 10⁶–10⁹ episodes | hundreds of games/run. With goals ≈ rare, the sparse-signal rate per update is orders of magnitude thinner — which is why the reward-path integrity + checkpoint bridge matter that much more |

## 2. Verified mistakes found (the "why no traction" list)

11v11 (all verified at source, pre-fix `main`):

1. **The conversion was not a dependable constant.** A normal goal entered
   learning through THREE stacked mechanisms — chain events + a cloned
   direct-shot transition + the contextual-deferred pool — each carrying the
   full `live_goal_reward_points()`: ~1500 total per conversion (plus buildup
   +24 and the on-target tier). A steal-and-finish goal paid 250 instead of
   500. `DD_SOCCER_REWARD_KIND_SCALES="Goal=…"`/learned calibration could
   rescale the anchor. And a broken `debug_assert`
   (`GOAL_CHAIN_REWARD_PATTERN` sums 160 vs `GOAL_REWARD_POINTS` 500 after the
   2026-07-13 constant bump) panicked every normal-goal test in debug builds.
2. **Completed backward passes were PENALIZED** (−4.0 own half / −3.4 opponent
   half). Punishing success on the safest ball teaches pass-avoidance and caps
   overall completion — the exact trap measured in the 5v5 (below).
3. **PopArt/standardization inconsistency.** League/ratchet forced them on,
   the continuous trainer defaulted off — local runs and the k8s runs trained
   under different critic scaling.
4. **No PFSP anywhere.** League fixtures were a flat product; the 5v5 ladder
   sampled history uniformly; no exploiter roles.
5. **Evolution final-batch bypass** could mutate tactics from one game when
   the promotion gate was disabled/lowered (witnessed historically:
   attack width 0.680→1.180 after one match).
6. **`done` overloaded** (full-game replay sets it on every row) — the actor
   GAE works around it via the 0.3 s decision-gap heuristic; the Bellman path
   honors it. Any new consumer must not trust `transition.done`.
7. Architecture reality check: the actor is a re-ranker over analytic
   candidates with default-ON `terrible_pass_veto`; hybrid authority caps what
   the net can discover (docs/5v5-vs-11v11-cross-pollination.md already calls
   this the liability to shed).

5v5 standalone (measured with the 12-iter same-seed smoke):

8. **Pass economics were upside down.** The untrained net passes 18.8×/game at
   71% with a healthy fwd/lat/back mix; 12 PPO iterations COLLAPSED that to
   ~3/game, 97% forward, 0 backward, 60% completion. EV of a pass attempt was
   negative (credit 2.5 vs turnover stack ≈ 25–75), the FIRST return pass was
   taxed 15 despite the comment saying it's fine, and `avg_reward ≈ −1400/game`
   (a punitive economy where inaction is optimal).
9. **Axis-flip reward bug**: `dribble_field_context(w.ball.x)` fed the WIDTH
   coordinate into the upfield-progress taper (x=width/y=length after
   f189deee) — the carry reward depended on which sideline the ball was near.
   The helper's own tests prove it expects a length coordinate.
10. **Console header mismatch**: the training table printed pass/shot columns
    under `entropy | val_loss` headers.
11. Ladder: no pool bound, held challengers discarded, uniform history
    sampling (LCB95 promotion + scripted anchor floor were already good ✓).

## 3. What changed today (all landed on `main`, tests green)

11v11:
- `record_goal_rewards`: **exactly ONE 500-point pool per conversion**,
  chain-distributed (recorder-normalized, sums exactly), turnover goals
  included — the 250 discount is gone. Old behavior kept for A/B only behind
  `DD_SOCCER_ENABLE_LEGACY_GOAL_MULTI_CREDIT`. Broken assert removed.
- `calibrated_reward_event_amount`: **Goal + MatchResult are
  calibration-immune** (the anchors are the currency). New invariant test
  `goal_pool_is_single_injection_and_calibration_immune`.
- Completed backward pass: **+0.6 own half / +0.9 opponent half** (was
  −4.0/−3.4). Ordering: forward ≫ lateral (1.2) > backward > 0 ≫ turnover.
- `SoccerNeuralLearningConfig::target_popart_enabled` default **true**;
  `DD_SOCCER_ENABLE_TARGET_STANDARDIZATION` default-ON in production
  (kill switch `=0`; tests stay env-driven). Every trainer now runs the same
  critic normalization as the k8s manifest.
- League: **PFSP fixture allocation** (`SOCCER_LEAGUE_PFSP`, default on) —
  fixtures ∝ p(1−p)+ε on a logistic squash of a per-opponent goal-diff EMA,
  coverage floor 1/opponent, analytic anchor keeps its FULL share (grounding
  is never traded away). `league_pfsp` line logs the allocation.
- Evolution: `should_evolve` now requires a **full evidence window**
  (`evolution_search_samples.len() >= evolution_window_games`) — a 1-game run
  can no longer mutate even with the promotion gate disabled.
- `measure_pass_completion`: attempts now counted **by direction at launch**
  (new MatchStats fields) → prints per-direction completion rates against the
  targets (85% fwd / 95% back / 90% overall).

5v5:
- Pass economics: `REW_PASS 2.5→6.0`, `REW_TURNOVER 23→14`,
  `REW_BAD_PASS_TURNOVER 15→8` (bounds unchanged; a 60%-confidence forward
  pass ≈ breakeven, an 85% one clearly pays, turnover still ≈ 2–4× a completed
  pass). First return pass free; escalation from the second.
- **GRF-style progression checkpoints**: first CONTROLLED entry per
  possession-cycle into midfield/final-third/box pays 6/10/15 (`REW_CHECKPOINT`,
  re-arm only after 6 yd regression — lose-and-regain can't farm it; turnover
  penalty > checkpoint sum). This is the sparse-goal gradient bridge.
- Axis bug fixed (`dribble_field_context(w.ball.y)`), console header fixed.
- Ladder: **PFSP sampling** over champions (payoff EMA, probe evals),
  **exploiter pool** (anchor-qualified held challengers retained, cap 8,
  `EXPLOITER_FRAC`), **champion pool cap 32** (evict most-beaten of the older
  half; disk keeps full lineage).

Measured effect (same seed 41, 12 iters): gd −0.467→−0.333, winrate
0.283→0.350, shots 0.1→0.2 (20% conversion). At 60 iters: passes 3.1→6.6
att/game, mix 97/3/0→77/19/4, shots ×6, quality trend −8.1→−2.9,
avg_reward −2331→−583 (@40). 300-iter run: gd −0.95 (untrained) → −0.225 @250,
goals 0→0.2/game. Traction is back; completion% now needs iterations + the
completion-targeted weights to climb toward 90/95/85.

11v11 2-game smoke (anchored + central-critic V2 + field-vector V2 + popart,
seed 0x2022): runs clean, 3 goals paid exactly one pool each, learned heads
consume samples with finite converging losses, checkpoints written.

## 4. Still open (ordered)

1. **Actor optimizer modernization**: LR 0.05 per-sample SGD → minibatched
   PPO with LR ~3e-4 + schedule, standard `min(rA, clip(r)A)`. Highest-risk
   highest-reward change; do it behind a gate with an A/B (it changes every
   run's dynamics).
2. **Central critic V2 + field vector V2 defaults**: currently opt-in
   (fc37e06e shipped them default-off with an A/B pending). Flip after the A/B
   confirms; they are the correct MAPPO shape.
3. **Hybrid authority**: flip `terrible_pass_veto` off in training stacks;
   extend actor-owned execution (shot placement gate exists; pass-aim next).
   The cross-pollination doc's "hermetic net" end-state stands.
4. **Recurrent belief state** (user-flagged): the POMDP obs is engineered
   history; nothing recurrent exists (`FeedForwardNetwork` only). Design:
   a small GRU (hidden 64) over the per-decision feature sequence per player,
   truncated BPTT 16 decisions, fed as 64 extra actor features behind
   `DD_SOCCER_ENABLE_RECURRENT_BELIEF` — implementable in the hand-rolled
   framework (it already has full backprop) but a multi-day change with its
   own stability battery; not attempted in this pass. GRF/Five evidence says
   feedforward + good rewards learns team play first; recurrence is a
   refinement, not the traction blocker.
5. **Eval noise**: ±0.13 payoff at 100 games swamps lever deltas (relstack
   memory). Promotion/verdicts need ≥300-game paired evals (the 5v5's
   LCB95+confirmation pattern is the reference).
6. **done-flag cleanup**: replace the overloaded `done` with an explicit
   `terminal` field on transitions; delete the 0.3 s heuristic.
7. 5v5 completion targets: rerun the 32-weight tuner under the new economics
   (`REW_PASS`/`REW_TURNOVER`/`REW_CHECKPOINT` bounds) toward 90/95/85.
