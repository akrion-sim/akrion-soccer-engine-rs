do not git checkout to switch branches stay on main branch
do not switch branches away from main
x is sideline-to-sideline (width) dimension, y is goal-to-goal (length) dimension
mdp = markov decision process
mpc = model predictive control
pomdp = partially observable markov decision process

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
