# Field-Vector Coverage — everything is a function of the field vector

The FIELD VECTOR = positions + velocities of all 10 players + the ball (the POMDP state).
Every component below reads it; nothing depends on a fixed constant where a field-vector
function is meaningful.

| Component | Field-vector dependence | Where |
|---|---|---|
| **POMDP** (actor obs) | egocentric ball/goal/teammate/opponent relatives, lane openness, pressure | `observe()` game.rs:509 |
| **MARL / MAPPO** (critic) | centralized global state over all players+ball | `global_state()` game.rs:1591 |
| **MDP reward/penalty** | `FieldRewardContext::from_world` derives 7 field quantities — `pass_value, safe_outlet_value, turnover_risk, dribble_pressure, goalside_score, burst_score, return_stale` — captured **pre- and post-action** and used to modulate every family (pass, turnover, dribble, goalside, burst, loop). Rewards use the field-vector **delta**. | train.rs:159/169, 437–494 |
| **MPC** (finishing) | shot-aim optimizer enumerates mouth points scored by keeper gap + defender lane coverage | shot aim in `apply_on_ball` |
| **MPC** (get-open) | `mpc_open_lane_target` picks the relocation maximizing lane clearness + width + space vs nearest defender | game.rs |
| **Passing** | target scored by forward progress + receiver openness; reward ∝ `pass_value`/`safe_outlet_value` field delta | `pass_candidates`, train.rs:461–471 |
| **Dribbling** | shielded direction vs nearest opponent; reward turnover ∝ `dribble_pressure` | `shielded_dribble_dir`, train.rs:490–492 |
| **Shooting** | reward = `(base + q·placement)·xG`, xG = f(distance,angle) over shot position, placement = f(keeper,lane) | train.rs shot block; `last_shot_xg_a` game.rs |
| **Formation / LP-IPM** | 5-a-side deliberately has NO LP/IPM formation machinery; its replacement — possession-conditioned off-ball positioning (`open_space_target`, `goalside_recovery_target`, advance/width/goalside) — is entirely field-vector-driven | game.rs, train.rs offense/defense blocks |

**Learnable:** each field-vector quantity is scaled by an env-tunable weight (`Rw`/`wenv`:
`pass_credit, field_pass, turnover, bad_pass_turnover, dribble_turnover, goalside, goalside_run,
burst_gear, ahead, make_run, recycle, return_pass, return_stale, shot_base, shot_q, …`), each with
sane defaults + a tiny non-zero lower clamp, so a search can tune *how much* each field-vector
signal counts without any of them being a fixed constant or fully dead.
