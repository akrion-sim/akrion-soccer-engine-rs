# Reward shaping audit (#8)

The per-step learning reward (assembled in `soccer.rs`, the
`reward += …` block beginning ~L19000) has accreted dozens of dense shaping
shards. This audits them against the **potential-based reward shaping (PBRS)**
criterion (Ng, Harada & Russell 1999): a shaping term is provably
policy-invariant iff it can be written

```
F(s, s') = γ·Φ(s') − Φ(s)
```

for some state potential `Φ`. Terms that fit are safe — they only change *when*
reward arrives. Terms that don't add a per-visit bias the policy can farm,
pulling the objective away from "win the match".

Helper: `des::general::soccer::reward_shaping::potential_based_shaping(gamma, Φ_before, Φ_after)`.
New state-conditioned dense terms should route through it.

## Classification

### A. Genuine outcome / action rewards — keep as-is
Not shaping; these *are* the objective or price a real action transition.
- Goal scored / conceded, shot on target, possession won/lost at a contest.
- `intentional_long_ball_loss_penalty`, `low_pressure_forced_pass_penalty`,
  blocked-lane floor-pass penalty — price the *quality of the action chosen*,
  not mere occupancy of a state.

### B. Already potential-based (`f(after) − f(before)`) — safe, γ=1 implicit
These are PBRS with `Φ` = the named scalar and `γ` folded to 1. Correct in
form; the only latent issue is `γ=1` vs the learner's true discount (see
"Follow-ups").
- **Pitch value** — `territorial_advantage(after) − territorial_advantage(before)`.
  ✅ Now routed through `potential_based_shaping` (this change) as the template.
- Yards-to-goal progress — `(before.yards_to_goal − after.yards_to_goal)` (~L19159).
- Ball-forward delta — `ball_forward.clamp(...)` measured as a positional change (~L19085/19087).
- Carry progress, spacing deltas (`spacing_delta.*`) — differences of a shape potential.

### C. Flat state-conditioned bonuses — NOT potential-based (review)
Constant reward for being in a state, with no canceling `−Φ(s)` on exit. Each
is a per-step bias the policy can accumulate by loitering in the rewarded state.
- `reward += 0.09` for a pass-state condition (~L19065), `reward += 0.08` (~L19233),
  `reward += 0.18 * role_multiplier * half_multiplier` (~L19205).
- Fixed penalties not tied to an action outcome: `reward -= 0.42` (~L19208),
  `-= 0.22` (~L19244), `-= 0.16` (~L19278).
- `EFFORT_REWARD_POINTS` / `LAZINESS_PENALTY_POINTS` (~L19016/19018) — effort
  bonuses; defensible as B if expressed as a *closing-distance delta*, a bias if
  flat per tick.

**Recommended conversion:** for each C term, define the implied potential `Φ`
(e.g. "being in the half-space" → `Φ = 1` there, `0` elsewhere) and emit
`potential_based_shaping(γ, Φ_before, Φ_after)` instead of the flat bonus. The
agent still gets pulled toward the state, but the pull cancels on exit, so it
cannot farm the term. This is behaviour-changing for the *trained* policy, so it
must be gated and A/B'd against the current learner, not flipped blind.

## Forward-pass primacy knobs (2026-07-08, gated, default byte-identical)

Operator lever: make **completed forward passes** the primary *dense* advancement signal instead of
shots. Two `OnceLock` env knobs, default 1.0 (identical):
- `DD_SOCCER_FORWARD_PASS_REWARD_SCALE` (0..20) — multiplies `completed_pass_reward_for_pitch`
  (`forward_pass_reward_scale()`, soccer.rs:23444).
- `DD_SOCCER_SHOT_SHAPING_REWARD_SCALE` (0..1) — dampens **only** the shot-TAKEN *shaping* proxy
  (`SHOT_ON_TARGET_REWARD_POINTS`, soccer.rs:36473); the goal (100) and terminal-outcome rewards are
  category-A and stay intact so finishing is never un-learned.

**PBRS status:** the completed-forward-pass reward is a **category-C** per-visit term (a completion
event, not a `Φ(s')−Φ(s)` delta), so scaling it up *is* increasing a farmable bias. At the DEFAULT
scale the per-pass reward sits below `SHOT_ON_TARGET_REWARD_POINTS = 80` / `GOAL_REWARD_POINTS = 160`
(soccer.rs:1167,1179), but the lever deliberately breaks that ordering: at `FWD_PASS_SCALE=6` a max
forward/flank pass component reaches **≈83.5**, exceeding the shot-shaping proxy once
`SHOT_SCALE=0.4` damps it to ≈32 (Codex r16). Goal (160) + terminal-outcome reward are **not** scaled
and still dominate, so finishing is never un-learned, and regression guards hold in
`soccer_learning.rs:9091` (analytic parity) and `:9105` (backward-recycle penalty). Because it is
**not** policy-invariant and now *can* out-earn damped shot shaping, the sterile-possession
discriminator (**held-out GD ≤ 0 while completed-forward-pass margin rises**) is mandatory in the
A/B; gated vs the confirmed base, never flipped blind. The clean PBRS alternative (deferred) is a learned
EPV potential routed through `potential_based_shaping`, which would value progression without the
farm risk but needs a possession-chain export first.

Current code enforces that discriminator in the promotion paths. `soccer_eval_gate_run` requires
the normal scoreline/WDL/Wilson promotion verdict, positive held-out goal difference by default
(`SOCCER_EVAL_MIN_GOAL_DIFF_MARGIN=0`), and forward-pass advancement. `soccer_league_train`
ranks/checkpoints on net forward passes (`completed_forward_passes - pass_turnovers`) while
validation also requires non-negative goal difference by default
(`SOCCER_LEAGUE_CHECKPOINT_VALIDATE_MIN_GOAL_DIFF_MARGIN=0.0`).

## Follow-ups (deferred, each its own gated change)
1. **Discount alignment.** B-terms (and pitch value) use `γ=1`; the returns use
   the learner's actual `γ` (`REWARD_SHAPING_DEFAULT_GAMMA = 0.99`). True
   invariance needs them to match. Low-risk but non-identical → gate + A/B.
2. **Convert the C-terms** to potential form one at a time behind a gate,
   measuring win-rate / Elo (#7), not raw reward.
3. **Single shaping budget.** Sum of |shaping| per step currently uncapped; a
   normaliser keeps shaping from dominating the sparse match-outcome signal.

## What this change lands
- `reward_shaping` module: the PBRS primitive + invariance tests (telescoping to
  a constant over a returning trajectory; non-finite neutralisation).
- Pitch-value reward routed through it (γ=1.0, byte-identical) as the template.
- This audit. The C→potential conversions are intentionally **not** applied
  blind; they are the gated follow-up work.
