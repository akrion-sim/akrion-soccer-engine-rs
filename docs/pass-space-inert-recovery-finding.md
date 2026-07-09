# Pass-space is INERT — `target_point` dropped at execution (2026-07-08 pm, recovery-A/B track)

Companion to [how-to-climb-codex-conversation.md](how-to-climb-codex-conversation.md). This is the
recovered pass-space verdict (the r13 "lost verdict"). The A/B proves the `DD_SOCCER_ENABLE_NEURAL_
PASS_SPACE` gate is a **no-op**; Codex then located the cause (repo-grounded): the distinct space
candidate **is** created and scored (99.5% distinct >1yd), but its `target_point` is **dropped when
the plan is lowered to `SoccerAction::Pass`** (no `target_point` field), so it never becomes a
different pass. Fix = make `target_point` survive execution. (My initial "never created at source"
read was wrong — corrected inline below.)

## The experiment

Frozen net `passspace-final.json` (198710 steps, trained WITH the gate), vs pure-analytic field,
identical held-out seeds (base `0xE7A10000`), 120 games/arm, `DD_SOCCER_ENABLE_NEURAL_PASS_SPACE`
OFF then ON in **separate processes** (gate is a process `OnceLock`), full `tee`
(`/tmp/passspace_recover.sh`, no grep — the original verdict was lost because `soccer_eval_gate_run`
exits nonzero on REJECT and a pipefail'd grep swallowed it).

| arm | record | GF/GA | payoff vs analytic | Wilson | decision |
|---|---|---|---|---|---|
| NEURAL_PASS_SPACE=0 | 31W-46D-43L | 71/96 | 0.45 | 0.364 | REJECT |
| NEURAL_PASS_SPACE=1 | 31W-46D-43L | 71/96 | 0.45 | 0.364 | REJECT |

**Byte-identical across all 120 games.** Toggling the gate changed nothing.

## Mechanism (traced, not guessed)

- **NOT a selection/execution severance.** The winning MCTS node carries its plan
  (`world.rs:~14438`, `plan: candidate.plan.clone()`), so a chosen space `target_point` *would* reach
  execution. My first hypothesis (label collapse) is falsified.
- My **first** guess (candidate never created at source) was **REFUTED by Codex in-repo** — see the
  correction box below. The byte-identical *data* stands; the *mechanism* is at execution, not source.

### CORRECTION (Codex, repo-grounded 2026-07-08 22:5x) — the null is at EXECUTION, not source

- The source **works**: the estimator (`world.rs:61773`) returns a real led point (forward/wide/
  velocity lead, capped by run distance), and train-path diagnostics already recorded **8000
  expansions, `Some` 100%, distinct `>1yd` 99.5%** (how-to-climb-codex-conversation.md:582). So the
  distinct space candidate **is** created and scored. My "never created / estimator collapses to feet"
  read was wrong.
- The candidate dies at **lowering/execution**: `PlayerAgent::action_from_learned_plan`
  (`player.rs:~13306`) lowers a pass plan to `SoccerAction::Pass { target_player, power, flight }` —
  **the action enum carries no `target_point`**. The plan's `target_point` is used only as a fallback
  to pick the nearest visible teammate, then **dropped**; the launch path (`world.rs:~25620`) sees
  only `target_player/power/flight`. So a feet candidate and a same-receiver *lead* candidate execute
  **identically** even though the winning MCTS node carried `plan.target_point`. That is the exact
  cause of the byte-identical 120-game result.

## Consequence + the fix (agreed with Codex)

The **spatial-falsification arm is null-by-construction until `target_point` survives execution.**
Minimal fix (Codex owns it on the canonical tree): make the selected `plan.target_point` survive
lowering into the actual pass **aim point** (extend/route through `SoccerAction::Pass`), then a
**hard preflight** asserting nonzero distinct rate not just at the source guard (`world.rs:~14229` in
Codex's tree) but **after scoring AND at executed launch**. The *truer* strategic fix for "POMDP owns
open-space target" is the receiverless path behind `DD_SOCCER_ENABLE_LEARNED_PASS_RECEIVER`
(`soccer.rs:12289`, space selection `world.rs:16354`) — the current gate still binds
`target_player: Some`, so it is not real open-space ownership. Only after the executed distinct rate
is provably `>0` does a
pass-space train/eval carry information.

## Second flag — MCTS selection noise (operator concern, open for Codex r14)

`mcts_enabled` is `true` in ~all blends (`world.rs` 7773/7806/7921/8293…; only 7906 false).
`neural_mcts_action_from_candidates` runs a real stochastic PUCT search
(`exploration·prior·√N/(1+visits)` + a rank-draw / selection-floor ranks), so the **executed** action
on both the training-sample and eval paths is a noisy PUCT pick, not the critic argmax — and the
critic bootstraps off noisy rollout values (`world.rs:~15592`). Open: does this contaminate on-policy
targets? Cheap probe = a clean `mcts_enabled` on/off A/B vs analytic (greedy target-generation:
exploration=0, no rank-draw) to see whether it reduces the draw-heavy stalemate or removes needed
exploration. Codex round-14 (`/tmp/codex_r14_prompt.txt`) carries both questions; queued because the
LAN bridge was returning HTTP 500 (codex-side exec code 1) on every call.

Did NOT edit `world.rs` (other track / Codex may be mid-edit; this tree is clean on `main`).
