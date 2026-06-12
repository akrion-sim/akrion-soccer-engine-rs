# TODOs

## Deferred — re-include in soccer learning Q-key

- [ ] **Re-include the 22 per-player skill bins + 6 pitch-surface physics bins in `SoccerQStateKey`** — GitHub issue [#1](https://github.com/ORESoftware/soccer-sim-game-engine.rs/issues/1).
  - Removed 2026-06-12 in the discrete state-space reduction (they fragmented the tabular Q-key for ~zero in-game signal; still live as neural features at dim 154).
  - **Last commit with the original code:** `f41b098f24168e7948ac0db00dbab7dabcff41f9`
  - **File:** `src/des/general/soccer.rs`
  - **Restore sites (all in soccer.rs):** `struct SoccerQStateKey` field decls; `SoccerQStateKey::from_parts` assignments (`skill_*_bin: skill_bucket(observation.skill_*)`, `ball_surface_*_bin: distance_bucket(observation.…, &[…])`); `matches_learning_context` comparisons. Exact code: `git show f41b098:src/des/general/soccer.rs`.
  - **Skill bins (22):** top_speed, acceleration, stamina, strength, height, weight, dribbling, aggression, right_foot_shot, left_foot_shot, passing_completion, passing, first_touch, flair_passing, vision, crossing, crossing_left, crossing_right, goalkeeping, defending, shooting, defensive_tracking.
  - **Physics bins (6):** ball_surface_drag, ball_surface_air_resistance, ball_surface_grass_resistance, ball_surface_stop_speed, ball_resistance_total_loss, ball_resistance_rolling_contact.
  - **Design note:** don't straight-revert — re-add as a shared/coarser representation (fewer buckets / role-relative tiers / separate skill-conditioned table) so rows accumulate visits, else the tabular path gains nothing over the neural net.

## Permanently removed (no re-inclusion)

- **4 player pitch-grid cell IDs** (`player_fine/tactical/macro/root_cell_id`) — removed 2026-06-12 from `SoccerQStateKey` and the spatial-index hierarchy. Player position now lives only in the neural feature vector. Not to be re-included.
