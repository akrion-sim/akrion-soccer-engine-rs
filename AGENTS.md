WORK ONLY ON `main`. Do NOT `git checkout` another branch, do NOT create new
branches, and do NOT create git worktrees on other branches. Stay in this one
working tree on `main` and commit directly to it. To incorporate another branch's
work, run `git merge <branch>` FROM main (a merge does not check the branch out).
Branches/worktrees are forbidden even as a "safe sandbox" — keep everything on main.
If a throwaway worktree is ever unavoidable, it MUST live under `tmp/worktrees/`
(`tmp/` is gitignored), e.g. `git worktree add tmp/worktrees/<name>`. NEVER create a
worktree in the repo root or any other in-tree path: `git add -A` then stages it as an
embedded gitlink/submodule (the `adding embedded git repository` warning). System `/tmp`
is also fine, but `tmp/worktrees/` is the convention here.
x is sideline-to-sideline (width) dimension, y is goal-to-goal (length) dimension
MDP = markov decision process
MPC = model predictive control
POMDP = partially observable markov decision process
LP = linear program (continuous variables, linear objective + constraints)
IP / IPM = interior-point / interior-point METHOD — the algorithm that solves the
  formation LP. IP here is NOT integer programming: there are NO integer/binary
  decision variables and NO MILP/branch-and-bound anywhere in the engine. The only
  mathematical program is the continuous formation LP, solved by Clarabel's IPM.
Clarabel = the Rust conic/LP solver used for the formation LP (`solve_lp_clarabel`),
  a sparse interior-point method; an internal simplex is the deterministic fallback
  (`SOCCER_FORMATION_LP_DETERMINISTIC`). It is the ONLY optimization solver in the
  engine — distance/cover checks, threshold gates, and "budgets" elsewhere are plain
  geometry, not LP/IP, so do not describe them as IP/IPM.

## Control architecture: who decides what

MPC is an INDIVIDUAL-player tool only. There is deliberately NO team/collective MPC
(no joint multi-player optimizer). New MPC work must control one actor's own
trajectory — the controlled player is the only point-mass state — never optimize
several players' motion jointly. (Audit Jun 2026; the reserved `tier1_team_enabled`
team-MPC flag was removed as it encoded this ruled-out design.)

Division of labor:
- TEAM FORMATION = LP / IPM. The *allocation* problem (which player -> which slot,
  optimal anchors, per tick). `SoccerFormationLpBrain::solve_tick` -> sparse
  interior-point (Clarabel, `solve_lp_clarabel`), with an optional internal simplex.
  Formation is static allocation + geometry (no time-dynamics), so LP/IPM is the
  right tool, NOT MPC.
- REPRESENTATION / RETRIEVAL = vectors. Config-vector similarity / embeddings,
  pitch value. Not a controller.
- INDIVIDUAL EXECUTION = MPC. Given one player's LP-assigned slot/target, plan THAT
  player's acceleration over a horizon to reach it under speed/accel limits, bending
  around moving bodies. The LP gives the destination point; MPC gives the
  dynamically-feasible path in time to it.

Rule of thumb: LP/IPM decides the shape; per-player MPC executes the move to it. Do
NOT add team-level MPC — it is redundant with the LP, far costlier (joint-state
blowup), and explicitly ruled out.

Every `PlanarPointMassMpc` solve is single-actor (all other players + the ball enter
only as `PlanarObstacle` soft keep-outs, never co-optimized state):
- tier2 per-player QP: `WorldSnapshot::mpc_desired_velocity` (warm-started cache
  `HashMap<usize, PlanarPointMassMpc>` keyed by player id)
- formation-local refinement of LP guidance: `soccer_local_mpc_refined_guidance`
  (one player; `apply_local_mpc_tick` loops players, each solved independently)
- pass execution (off by default): `mpc_predicted_receiver_path`
- keeper positioning

MPC defaults: base `MatchConfig` OFF (deterministic tests/learning); live-gameplay
preset ON (tier2 + field-aware + reconcile + latent + formation-local). Master live
toggle: `SOCCER_LIVE_MPC`.
