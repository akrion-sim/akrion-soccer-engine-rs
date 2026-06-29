# Decision-test red triage (pre-existing at `main`)

A full `cargo test --lib` on `main` (tip `c61eb7a`, after the crash-box + slip-break +
pass-risk-by-role + upstream merges) has **19 deterministic decision-test failures** that
are unrelated to the flank crash-the-box feature (its own 10 tests pass; see
[crash-box-plan.md](crash-box-plan.md) §5). They were verified pre-existing by running the
same tests in a clean `HEAD` worktree.

These are **behavioral flips introduced by recently merged concurrent features** —
forward-pass-option recognition, pass-risk-by-role, slip-break-the-offside-trap, and
blindside-steal. For most, deciding *update the stale expectation* vs *fix a regression*
requires the **intent of the commit that changed the behavior**, so each is routed to that
owner rather than rewritten blind. Two smell like genuine ownerless regressions and are
called out first.

How to read a row: the assertion delta is the exact `left`/`right` or `got` value from the
panic. "Call" is a best-guess starting hypothesis, **not** a verdict — confirm against the
owning commit before editing.

---

## Priority 0 — likely genuine regressions (investigate, don't just update the test)

| Test | Delta | Why it smells like a bug |
|---|---|---|
| `fatigue_accumulates_from_repeated_sprints_and_recovers_at_rest` | `high_after_sprints > 0.0` is false (fatigue stayed `0`) | Fatigue **not accumulating at all** after repeated sprints is almost certainly broken plumbing, not an intended behavior change. Check whether a merge dropped the fatigue accumulation path. |
| `blocked_goal_pressure_preempts_recycling_with_single_killer_pass` | `got 0/120` — killer pass **never** chosen | The killer pass going from "chosen under blocked goal pressure" to "never" is a categorical loss of a finishing option. Likely the pass-risk-by-role work over-penalised it. Confirm it's intended; if not, it's a real attacking regression. |

---

## Priority 1 — categorical decision flips (route to the owning feature)

| Test | Delta (`left` = now, `right` = expected) | Likely owning change |
|---|---|---|
| `exploit_space_strategy_sends_runner_to_receivable_pocket` | `"run-in-behind"` vs `"exploit-space-run"` | forward-pass / run recognition biasing in-behind runs |
| `exploit_space_strategy_can_choose_front_of_line_pocket` | `"run-in-behind"` vs `"exploit-space-run"` | same |
| `xavi_turn_is_chosen_by_a_skilled_boxed_in_carrier` | order leads `"hold-up-flank"` vs `"xavi-turn"` | possession-option re-ranking (pass-risk / forward-pass) |
| `support_operation_order_can_prioritize_check_to_ball` | `0` vs `160` (run-in-behind wins) | run recognition out-prioritising check-to-ball |
| `blocked_final_third_attacker_falls_back_to_killer_pass` | flight `OverTop` vs `Floor` | aerial/over-top pass selection change |
| `through_ball_strategy_fires_offside_breaking_runs_off_cadence` | run fires off-cadence when it "should not fire yet" | slip-break / through-ball cadence change |
| `pass_concede_decision_respects_passer_perception` | blindside marker now vetoes a concession it shouldn't | blindside-steal perception interaction |
| `aerial_pass_targeting_can_bypass_blocked_floor_lane` | ranked aerial targets no longer `contains(&target)` | aerial target ranking change |

## Priority 2 — threshold / margin drift (smallest risk; likely stale expectation)

| Test | Delta | Note |
|---|---|---|
| `attacking_midfielder_roams_and_overlaps_when_striker_has_ball` | overlap share `0.317` vs "≥ one third" (`0.333`) | 1.6% under the bar — likely a tuning drift; confirm the floor is still the intent |
| `defender_in_possession_keeps_three_to_four_yards_and_releases_as_gap_closes` | `open=0.2286` vs `soft_gap=0.2238` | razor-thin (~0.005) inversion |
| `open_support_outlets_raise_pressured_release_over_stale_carry` | `open=0.787` vs `closed=0.722` | open *is* higher — verify the assert's required margin/direction |
| `pressured_holder_without_good_outlet_prefers_escape_over_panic_pass` | `escape=7.25` vs `pass=14.45` | now prefers the pass — pass-risk change; confirm whether retain-by-shielding is still intended |
| `low_pressure_marked_receiver_favors_carry_over_forced_pass` | "test should expose a visible pass target" (setup yields none) | perception/visibility change broke the *fixture*, not necessarily the behavior |

## Priority 3 — defensive shape / observation (route to slip-break + back-line owners)

| Test | Delta | Likely owning change |
|---|---|---|
| `defenders_drop_when_ball_carrier_threatens_back_line_break` | target `y=15` vs retreat band `6` after the 40yd ball-gap cap | back-line retreat / offside-trap change |
| `defenders_press_opponent_midfielders_and_mids_press_next_pass_focus` | `mark.distance <= 8.5` violated | marking distance change |
| `defensive_recovery_effort_includes_in_possession_back_line_consistency` | sprint consistency urgency `0` (expected sprint-level) | back-line consistency change |
| `first_touch_observation_prices_field_shape_before_one_touch_release` | POMDP open vs blocked one-touch picture no longer distinguished | first-touch / perception observation change |

---

## Process recommendation

1. **P0 first** — root-cause the two suspected regressions. If `fatigue` accumulation or
   the killer-pass option were dropped by a merge, that's a real gameplay/learning loss and
   should be fixed in code, not papered over by editing the test.
2. **P1/P3** — the feature owners (forward-pass recognition, pass-risk-by-role, slip-break,
   blindside) each confirm whether the new behavior is intended; if so, update the assertion
   in the **same** commit/PR as the behavior change so the suite never carries silent reds.
3. **P2** — almost certainly expectation drift; safe to relax once the owner confirms the
   new tuned value is the target.
4. Until then these are **known reds**, not regressions from any single in-flight feature —
   but do not let them become permanent: a green `cargo test --lib` is the only way the next
   change (yours or anyone's) can tell it didn't break something.

> Why this wasn't auto-fixed here: rewriting 19 other features' behavioral expectations
> without their authors' intent would risk masking the P0 regressions and overwriting
> deliberate P1 changes — in a tree those authors are actively committing to. Triage and
> route; let owners land the fix beside the behavior.
