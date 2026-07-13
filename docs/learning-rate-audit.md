# Learning-rate audit & hardening

An audit of every step-size / stability knob in the soccer learner, the failure
modes they leave open, and the hardening applied. Companion to
[reward-shaping-audit.md](reward-shaping-audit.md) (what the reward *is*) and
[learnability-conversion-roadmap.md](learnability-conversion-roadmap.md) (which
surfaces learn). This doc is about *how fast and how safely* the nets move.

## The surface (all in `soccer.rs` unless noted)

| Knob | Value | Role |
|------|-------|------|
| `DEFAULT_SOCCER_NEURAL_LEARNING_RATE` | 0.015 | value/critic SGD step |
| `SOCCER_POLICY_LEARNING_RATE` | 0.05 | actor (policy-gradient) step |
| `SOCCER_WORLD_MODEL_LEARNING_RATE` | 0.02 | world-model regression step |
| `MAX_SOCCER_NEURAL_LEARNING_RATE` | 0.25 | hard clamp via `sanitized_learning_rate` |
| `SOCCER_NEURAL_GRAD_CLIP_NORM` / `SOCCER_POLICY_GRAD_CLIP_NORM` / world-model | 8.0 / 5.0 / 8.0 | per-step L2 gradient-norm ceilings |
| `DEFAULT_SOCCER_NEURAL_TARGET_SCALE` / `_TARGET_CLIP` | 30.0 / 3.0 | value target is `reward/scale` clipped to ±clip |
| `SOCCER_POLICY_GAE_LAMBDA` | 0.95 | advantage trace decay |
| `SOCCER_POLICY_ENTROPY_COEFF` | 0.01 | anti-collapse entropy bonus |
| `DEFAULT_SOCCER_MAPPO_CLIP_EPSILON` | 0.20 | PPO/MAPPO ratio clip |
| `SOCCER_FULL_GAME_RETURN_{DISCOUNT,BLEND,CLIP}` | 0.98/tick, 0.35, 400.0 | full-game return recursion (discount 0.98 per the Jul-2026 credit-assignment fix, soccer.rs:5590; clip 400.0, soccer.rs:5592) |
| advantage standardization | gated, default OFF | zero-mean/unit-variance over the policy batch ([world.rs](../src/des/general/soccer/world.rs)) |

**Already-solid guards (no change needed):** LR clamped ≤ 0.25; three grad-norm
ceilings; the non-finite-gradient guard in `FeedForwardNetwork::train_sample_clipped`
drops poisoned steps so a NaN can't reach live play; value targets bounded to ±3.0;
actor samples filter `advantage.is_finite()`; full-game return clamped to ±400.

## Findings

**F1 — Effective learning rate drifts with reward magnitude (the core issue).**
The policy-gradient step is `LR · advantage`, so the *effective* LR scales with the
advantage magnitude, which scales with reward magnitude. Advantage standardization —
the textbook fix that decouples the step from reward scale — is **OFF by default**.
Any change that shifts reward magnitude (a new shaping term, and especially the flat
terminal won-game label) silently changes the effective LR. With GAE, a *constant*
per-step label is worst early in training: the trace `A_i = Σ (γλ)^k δ_{i+k}` sums the
label over the trajectory before the critic learns to absorb it, so early advantages
can be large ∝ trajectory length.

**F2 — The keeper head is baseline-free and unnormalized.** `neural_keeper_policy_training_samples`
sets `advantage = transition.reward` directly — no critic baseline, no standardization
([world.rs](../src/des/general/soccer/world.rs)). The flat won-game label (≈±200:
`MATCH_OUTCOME_WIN_REWARD_POINTS`=200 plus a goal-margin bonus of 15/goal capped at 5 goals,
soccer.rs:5628-5631) then *dominates* the keeper's own shaping rewards, so the keeper would
learn "we won ⇒ every action I took was good," independent of quality.

**F3 — Value-target saturation (pre-existing, flagged not fixed).** Targets are
`reward/30` clipped to ±3.0, i.e. the critic cannot represent a return beyond ±90 raw,
while the full-game return clips at ±400. High-magnitude games saturate the critic and
bias the advantage baseline. **The won-game label now makes this worse, not better:** at
`MATCH_OUTCOME_WIN`=200 the label is 200/30 ≈ 6.67 in target space, which **exceeds** the
±3.0 target clip — so the outcome label itself saturates the critic. (This corrects an earlier
version of this note that assumed a ±12.5 label sitting safely inside the clip.)

**F4 — Other reward-fed trainers are safe.** The tabular Q trainer takes a flat
per-team label as a constant that cancels in per-team argmax; the value net clips (F3);
the actor specialists reuse the (now standardized) actor advantage; the pass-completion
/ long-pass / beat-defender heads train on their own targets, not the replay reward.

## Hardening applied

A flat Monte-Carlo outcome label is only safe *with* advantage standardization, so the
won-game reward now forces it on (default path unchanged):

- New `dd_soccer_standardize_policy_advantages() = dd_soccer_enable_advantage_normalization() || match_outcome_reward_enabled()`
  and a shared `policy_advantage_standardization(&[f64]) -> Option<(mean, std)>` helper
  ([world.rs](../src/des/general/soccer/world.rs)).
- **Actor** (`neural_policy_training_samples`): refactored onto the shared helper +
  the coupled gate. Byte-identical for every pre-existing gate state (the actor already
  standardized under the advantage-normalization gate); standardization is now also
  forced when the won-game reward is on.
- **Keeper** (`neural_keeper_policy_training_samples`): adds standardization gated on
  `match_outcome_reward_enabled()` alone — byte-identical for every pre-existing path
  (the keeper never standardized), and removes the common-mode label + bounds the step
  scale when the label is on. Standardization keeps the realised win-vs-loss *contrast*
  (winner's advantages stay above the loser's); it only removes the common mode.

Off ⇒ byte-identical. Covered by `policy_advantage_standardization_*` (degenerate /
non-finite / zero-mean+unit-variance + contrast-preserved) and the existing
advantage/keeper/actor/mappo suites.

## Recommendations (gated; validate via the promotion eval gate, not raw reward)

1. **Enable advantage normalization in trained runs** (`DD_SOCCER_ENABLE_ADVANTAGE_NORMALIZATION=1`)
   so the effective LR is reward-scale-invariant even without the won-game reward; A/B
   held-out Elo via `soccer_eval_gate_run` before making it the default.
2. **A/B the won-game magnitudes** (`MATCH_OUTCOME_WIN`=200.0, margin 15.0, cap 5) on
   held-out win-rate — the eval gate is the instrument. (Given F3, also consider lifting
   `target_clip` in tandem so the ±6.67 outcome label isn't clipped flat.)
3. **Track F3**: if high-scoring self-play games become common, lift `target_clip` or
   the return clip together so the critic baseline isn't saturated.

## 11v11 plateau anchor audit (2026-07-13)

The reward scale is not itself moving during one learner process. With
`SOCCER_DYNAMIC_REWARD_WEIGHTS=1`, reward helpers re-read environment variables rather
than caching them, but the process does not rewrite its own environment during a game or
training round. An external search can launch a later process with a different profile;
inside one run, the numeric ruler is stationary.

A requested match-win reward of `10_000` is also not a distinct treatment in the current
replay contract. The match outcome is added after dense-return blending and the resulting
transition is clamped to `SOCCER_FULL_GAME_RETURN_CLIP` (`400`). The largest meaningful
one-goal win-anchor treatment is therefore `400`; values above it collapse to the same
target before critic target scaling, target clipping, and advantage normalization.

The moving ruler that remained was the **acceptance opponent**:

- `soccer_league_train` trained on varying analytic genomes but promoted against only the
  immediately previous neural checkpoint.
- `main_soccer_learning_run` likewise ratcheted candidate versus previous neural anchor,
  without proving that either retained absolute performance against the analytic engine.

Both trainers now use a deterministic pure-analytic anchor (`0xA5A5_0007`) in addition to
the moving neural comparison. League training always includes the fixed opponent and its
checkpoint gate compares candidate and incumbent against paired analytic fixtures. The
continuous learner's promotion gate does the same; a candidate is archived when its
held-out goal difference versus the analytic anchor regresses beyond the configured
tolerance. Candidate/incumbent seeds and home-away assignments are paired so the comparison
measures policy drift instead of fixture drift.

The reproducible reward experiment is `scripts/run_reward_anchor_ab.sh`. It trains three
same-seed arms against an alternating analytic pool: no outcome label, the existing `200`
label, and the maximum effective `400` label, then evaluates every snapshot on the same
held-out analytic field. `WIN_REWARD_A` and `WIN_REWARD_B` select the two nonzero arms.

The first powered run trained each arm for 60 same-seed, `0.35`-minute games and evaluated
it over 32 held-out games of the same duration. These short fixtures are a fast screening
gate, not proof of full-match climb. Outcome-off was nominally best (payoff `0.500`, Elo delta `+15.5`, forward
pass margin `+0.00/game`); `200` reached payoff `0.438`, Elo delta `+4.3`, forward margin
`+0.16/game`; `400` reached payoff `0.453`, Elo delta `-40.3`, forward margin `-0.06/game`.
Every Wilson lower bound remained below the promotion floor (`0.282` to `0.336`). This
rejects “make the broadcast win label huge” as the next default, but does not by itself
prove that the outcome label should be removed.

The independent-seed replication used the same 60-game training and 32-game held-out
protocol. Outcome-off again led (payoff `0.516`, Elo delta `+11.5`, forward-pass margin
`+0.09/game`); the arm requested as `30` reached payoff `0.422`, Elo delta `-35.9`,
forward margin `+0.03/game`; `200` reached payoff `0.484`, Elo delta `+2.2`, forward
margin `-0.19/game`. Audit of the executed treatment found that the reward hierarchy
floors the win value at goal reward `160 + 20`, so the requested `30` arm actually ran
at `180`. The experiment binary now prints the effective post-hierarchy win reward to
prevent another mislabeled treatment. No arm passed its Wilson promotion floor. Across
both seeds, every executed flat per-transition outcome-label arm (`180`, `200`, and
`400`) underperformed outcome-off on payoff, so none is promoted from this experiment.

That result narrows the next test to credit-assignment structure, not another magnitude.
`scripts/run_outcome_credit_ab.sh` holds the seed, opponent pools, target transform, and
`200` win value constant while comparing outcome-off, the flat outcome broadcast, and
the experimental `DD_SOCCER_OUTCOME_CREDIT=1` replay. The latter still broadcasts the
result but replaces the correlated dense return with bounded positive milestones plus
pitch-value shaping. It is not the effective DP-bootstrap production path: when both
flags are on, DP replay takes precedence and bypasses outcome-credit replay. The experiment
therefore disables DP deliberately to isolate whether dense negative/correlated replay
conflicts with the fixed winning anchor. It must clear the same held-out gate on repeated
seeds and then survive a DP-compatible test before any default change; one nominal win is
not evidence of a durable climb.

The first 60-game outcome-credit screen produced 32 held-out fixtures per arm. Outcome-off
and flat `200` tied on the primary payoff at `0.547`; their Elo deltas were `+12.4` and
`+44.9`, and forward-pass margins were `-0.22/game` and `+0.00/game`, respectively. The
experimental outcome-credit replay regressed to payoff `0.422`, Elo delta `-49.8`, and
forward margin `-0.38/game`. Wilson lower bounds were `0.379`, `0.379`, and `0.268`, all
below promotion. Outcome-credit is rejected on this seed; the flat/off tie is not evidence
to change the existing outcome default.

The next DP-compatible test is `scripts/run_dp_terminal_outcome_ab.sh`. Its default-off
`DD_SOCCER_ENABLE_DP_TERMINAL_OUTCOME_CREDIT` treatment puts the fixed result on exactly
one terminal sample per team, then uses the existing fitted value iteration and n-step
bootstrap to propagate that signal through the field-state abstraction. It compares
DP-without-outcome, the existing DP flat broadcast, and DP terminal propagation at the
production target scale. This tests a fixed winning anchor without assigning identical
credit to every action; it remains experimental until held-out and repeated-seed gates pass.

The first powered DP screen trained 60 games and evaluated 32 held-out fixtures per arm.
DP-without-outcome reached payoff `0.422`, Elo delta `-17.6`, and forward margin
`-0.12/game`; existing DP flat `200` reached payoff `0.469`, Elo delta `+1.5`, and forward
margin `+0.19/game`; DP terminal propagation reached payoff `0.453`, Elo delta `+25.9`,
and forward margin `+0.00/game`. Wilson lower bounds (`0.268`, `0.309`, `0.295`) all failed
promotion. The terminal treatment therefore stays off. Within this matched DP seed, the
existing flat anchor led both alternatives on primary payoff and forward-pass margin, so
the evidence supports retaining it while the plateau investigation moves to a different
factor. The Elo ordering alone is not used to overturn the payoff result because the
analytic-baseline estimates varied across arms.

Reward placement is therefore no longer the next lever. A held-out net-influence probe on
the DP-flat checkpoint showed `76.6%` family changes on the neural-active, two-plus-candidate
subset, well above the documented `25-30%` high-influence threshold. Final commit influence
was lower, but the on-ball direction was decisive: fresh neural pass intent was `7.8%`, while
committed pass actions were `12.5%`. The downstream pipeline was adding rather than suppressing
passes; the actor itself rarely proposed them. This falsifies the shield-ownership explanation
for this checkpoint and points to action preference/credit.

`scripts/run_forward_select_learning_ab.sh` tests that diagnosis with the existing default-off,
per-checkpoint `forwardSelectLogitWeight`. It compares the incumbent, learning the scalar only
from executed forward-pass samples, and learning it with qualified missed forward opportunities
from the planner teacher. The third arm is specifically a cold-start test: a policy that rarely
chooses a forward pass cannot reliably learn a forward-only scalar from executed actions alone.
Every arm keeps the existing DP flat result anchor and is judged on the same held-out analytic
field; no gate is promoted from the diagnostic alone.

The first forward-selection screen trained 60 games and evaluated 32 held-out fixtures per
arm. Both learned treatments saturated `forwardSelectLogitWeight` at its hard maximum `8.0`.
The incumbent led primary payoff at `0.469` (Elo delta `-102.4`, forward-pass margin
`-0.19/game`); executed-sample learning reached payoff `0.453` (Elo delta `+7.6`, forward
margin `-0.12/game`); planner-teacher learning regressed to payoff `0.391` (Elo delta `-83.2`)
despite a nominal forward margin of `+0.06/game`. Wilson lower bounds were `0.309`, `0.295`,
and `0.242`; every arm was rejected. The analytic-baseline Elo estimates varied across arms,
so the payoff result remains primary. The saturated treatment is not promoted.

The scalar update currently sums eligible gradients across a batch and clamps the learned
weight even though score-time influence is independently capped at `0.25`; this makes the hard
maximum a likely scale artifact, not a calibrated preference. `scripts/run_forward_select_dose_ab.sh`
therefore freezes the incumbent checkpoint and evaluates weights `0`, `0.05`, and `0.20` on one
paired held-out field. This isolates the useful dose before changing the trainable update rule.

The first fixed-dose field used weights `0`, `0.05`, and `0.20`. All three produced exactly
the same 32 paired fixtures and diagnostics: payoff `0.406`, Elo delta `-48.5`, forward-pass
margin `-0.50/game`, and Wilson lower bound `0.255`. Thus weights through `0.20` did not cross
an action-ranking boundary on this checkpoint. The runner accepts `FORWARD_SELECT_WEIGHT_A`
and `FORWARD_SELECT_WEIGHT_B` so the next bounded probe can test the range between this inert
region and the saturated treatment without changing code or retraining the actor.

The follow-up used weights `0`, `0.50`, and `8.0`; all 32 paired fixtures were again exactly
identical to the control (`0.406` payoff, `-0.50/game` forward margin). A direct JSONL trace
explained why: one paired game produced 32 on-ball neural decisions and zero forward-pass
candidates in the scored set for both `0` and `8.0`. Across a broader four-game trace, 34 of
138 on-ball states qualified for a forward option; root injection created three to four forward
candidates, but the post-injection pass-like margin gate pruned every one. The learned scalar
was saturating on candidates it could never influence.

The existing bounded pass-family root floor fixes that exposure seam without forcing a pass.
In a four-game mechanism trace, control exposed two forward candidates and selected one;
`SOCCER_NEURAL_MCTS_MIN_PASS_LIKE_ROOT_CANDIDATES=1` exposed 22 and selected seven; adding
the fixed `8.0` scalar exposed 26 and selected 12. These are mechanism counts, not promotion
evidence. `scripts/run_forward_exposure_ab.sh` therefore holds the checkpoint fixed and compares
control, root exposure, and root exposure plus scalar on a full held-out analytic field before
the floor is allowed into a training treatment.

The 32-game frozen-checkpoint exposure screen did not pass promotion. Control reached payoff
`0.406`, candidate forward completions `0.2/game`, and forward margin `-0.16/game`. Root exposure
improved payoff to `0.422` and candidate forward completions to `0.3/game`, but its forward margin
was still `-0.19/game`. Exposure plus scalar also reached payoff `0.422` and `0.3` candidate forward
completions, with a weaker `-0.22/game` margin. Wilson lower bounds were `0.255`, `0.268`, and
`0.268`. Neither gate is promoted. The floor is nevertheless active and improves the frozen actor's
primary mean, so the next experiment tests whether training with that bounded exposure supplies
useful actor samples; the saturated scalar is excluded.

An experiment-stack audit then found a blocking train/eval mismatch in
`soccer_outcome_ab_run`: analytic training ran actor-critic on with MCTS off, while analytic
held-out evaluation inherited the tournament's actor-off/MCTS-on defaults. The tournament default
is correct for two learned teams because its actor sidecar is shared, but an analytic fixture has
exactly one learned team and can safely evaluate that actor. The runner now defaults both
`train-analytic` and `eval-analytic` to actor-critic plus MCTS, while retaining explicit env
overrides for ablations. An optimized end-to-end smoke proved both phases report
`actor_critic=true` and `mcts_enabled=true`.

This mismatch means earlier critic/reward screens remain diagnostics of their executed stacks, and
the fixed-checkpoint MCTS exposure traces remain valid, but actor-specific held-out conclusions are
not promotion evidence: the evaluator did not consult the trained actor. In particular, the first
forward-select learning comparison must be rerun under the aligned stack. The interrupted
exposure-learning run is discarded and restarted from its original seed after this correction.
