# Pass-space is INERT AT SOURCE — recovered A/B finding (2026-07-08 pm, recovery-A/B track)

Companion to [how-to-climb-codex-conversation.md](how-to-climb-codex-conversation.md). This is the
recovered pass-space verdict (the r13 "lost verdict"). It **directly tests the r16 pass-space
preflight** ("validate by `target_point` differing from receiver feet, not a new label") and the
answer is: **the preflight fails today — the distinct candidate is essentially never created.**

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
- For 120 games to be byte-identical — adding even one scored candidate perturbs PUCT priors/
  normalization and cascades over ~162k ticks — the pass-space candidate must be **never created**.
- Its only creation guard is `world.rs:14028-14041`: needs `anticipated_pass_reception_point(...)` =
  `Some` **and** `space_point.distance(receiver_feet) > 1.0`. That precondition appears to essentially
  never fire ⇒ the second (space) candidate is never pushed ⇒ **gate inert at SOURCE, not sink.**

## Consequence for the locked r16 run order

The **spatial-falsification arm risks being null-by-construction**: training WITH a gate that never
emits a distinct candidate tests nothing. Make the r16 preflight a **hard gate before that run**:
instrument `anticipated_pass_reception_point` / the guard at `world.rs:14034` and prove it emits a
`>1.0`-yd-from-feet point at a **nonzero rate** in real play. If it doesn't, the minimal fix is in the
**reception-point estimator** (why is the anticipated lead point collapsing onto the receiver's
feet?), not in the MCTS or the reward. Only after the candidate is provably distinct does a
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
