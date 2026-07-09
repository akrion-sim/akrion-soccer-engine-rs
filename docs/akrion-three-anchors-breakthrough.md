---
name: akrion-three-anchors-breakthrough
description: Root-cause of the akrion-soccer parity ceiling — three imitation anchors + train/eval objective mismatch — and the ordered breakthrough plan
metadata: 
  node_type: memory
  type: project
  originSessionId: eaafcbd1-fa79-40a0-bba8-b6eaa59eed68
---

Verified from code 2026-07-09 (4 Explore agents). WHY the neural soccer policy is pinned at
parity-below-analytic: it's a CEILING built into the architecture, not a score being lost. THREE
independent anchors each cap the net at "reproduce analytic" (any one alone => parity; fixing one
leaves the others binding — which is why ~9 single levers in [[akrion-climb-state]] all failed:
they tuned WHICH of analytic's options gets picked, never the anchors).

ANCHOR 1 — re-ranks analytic's OWN candidate set. Neural-authoritative `Q_eff=λ·V_net+1e-6·Q_tab`;
MCTS off by default, depth=1 (soccer.rs:18533-18567), no rollout (visit-count re-rank of ~4 cands);
MPC is a feasibility FILTER not a planner. Ceiling = analytic candidate GENERATION.
ANCHOR 2 — critic IMITATES aliased tabular Q. `target=r+γ·best_value_hierarchical(s')`
(world.rs:18178-18205) => regresses onto tabular-Q imitation of analytic, can't exceed it
(dp-critic-target-plan.md; MC-critic + self-bootstrap both falsified). Value head also collapsed to
constant (low-var targets); fixed by DD_SOCCER_ENABLE_TARGET_STANDARDIZATION but still 0.478.
ANCHOR 3 — only dense long-horizon reward is a PROPER PBRS potential over a HAND-BUILT xT that nets
to ZERO. Φ=expected_threat=`fwd³+0.8·fwd⁴·centrality` (pitch_value.rs:144), applied as
F=γΦ(s')−Φ(s) (reward_shaping.rs:40; live soccer.rs:27783). Telescopes to 0 over any unconverted
possession => "territory without conversion" BY CONSTRUCTION; only non-cancelling signal is sparse
goal(±160,~1/g) which Anchor 2 can't propagate. => every territory/reward-shaping arm was
policy-invariant/flat.

OBJECTIVE MISMATCH (decisive, independent): TRAINING NEVER PLAYS ANALYTIC. League optimizes "beat
frozen neural past-selves + fresh random nets" (soccer_league_train.rs:1149); analytic only at EVAL
(SOCCER_EVAL_ANALYTIC_FIELD). The flag that puts analytic in TRAIN —
`SOCCER_LEAGUE_ANALYTIC_OPPONENTS=1` (soccer_league_train.rs:1163) — is set NOWHERE. 0.53-vs-field /
0.35-vs-analytic split = textbook train/eval objective mismatch. No asymmetric neural-vs-analytic
eval ladder exists (reframe-july/current-learning-state).

BREAKTHROUGH PLAN (cut anchors TOGETHER, ordered; most pieces already BUILT but unwired):
- MOVE 0 (free, diagnostic, first): SOCCER_LEAGUE_ANALYTIC_OPPONENTS=1 + build asymmetric eval ladder.
- MOVE 1 (core): 1a swap expected_threat→learned_epv INSIDE the potential (team_expected_threat/
  territorial_advantage/soccer_replay_pitch_value_potential pitch_value.rs:353/418). Fitted EPV grid
  ALREADY exists end-to-end (scripts/fit_epv_grid.py: goal1.0/SOT.30/turnover−.15; learned_epv
  pitch_value.rs:205; gate DD_SOCCER_ENABLE_LEARNED_EPV default-off) but wired ONLY into a per-pass
  bonus (world.rs:22879) — move it into the potential. 1b break tabular-Q bootstrap via
  DD_SOCCER_ENABLE_DP_CRITIC_TARGET (V_DP(bucket) anchor) + target-standardization (Tier-3 of
  adp-can-break-the-ceiling.md — DP backbone + neural residual, genuinely untried combo).
- MOVE 2 (representation): distill set-transformer/GNN encoder into live head (offline prototype
  showed ~47% held-out gain; DD_SOCCER_ENABLE_OFFLINE_DISTILLED_HEAD) within 66ms/15Hz budget.
- MOVE 3 (conversion, the bottleneck 0-1 expose): Shoot carries ONLY power => give actor shot
  PLACEMENT (real gap, unlike redundant fwd-pass action); + real shallow MCTS (depth>1) with the
  now-outcome-grounded value as leaf. Only works AFTER 1-2 (why loop's action/search-first levers failed).

CAVEAT: analytic may be near-optimal for this sim; realistic prize = small clean margin (Wilson
LB>0.5), not domination. But stuck-AT-parity (not near it) is the anchors; none was in the
falsified-lever set. Deferred-pass-credit (DD_SOCCER_ENABLE_DEFERRED_PASS_CREDIT) = only confirmed
structural win (fwd passes 2×) but REJECTED on payoff => bottleneck = conversion/finishing.
