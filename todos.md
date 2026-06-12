# soccer-sim-game-engine.rs — TODOs

## Potentially re-add to the MDP/POMDP tabular state key (`SoccerQStateKey`)

On 2026-06-12 the tabular Q-state was cut from **161 → 129 dimensions** to fix
unbounded memory growth (the table grew to multi-GB; visits/entry ≈ 1.0 — it never
converged) and the mid-game freezes. The cut fields are listed below. They are NOT lost —
they remain available to the **neural** value/policy heads as features (skills are sourced
from the observation in the neural feature vector). Re-add to the *tabular key* only if a
future design needs them there (e.g. a much coarser, deliberately low-cardinality
encoding, or a per-archetype skill bucket instead of per-skill bins).

### 🔴 Pitch-grid cell IDs (4) — huge cardinality
- `player_fine_cell_id`, `player_tactical_cell_id`, `player_macro_cell_id`, `player_root_cell_id`
- `fine_cell_id` alone is hundreds of values; this made nearly every state unique.
- `ball_zone_x` / `ball_zone_y` were KEPT as a coarse location signal.
- If re-added: use ONE coarse cell (not 4 resolutions), or keep position only in the neural net.

### 🔴 Per-player skill bins (22) — constant per player, fragmented the table
- `skill_top_speed/acceleration/stamina/strength/height/weight/dribbling/aggression`
- `skill_right_foot_shot/left_foot_shot/passing_completion/passing/first_touch/flair_passing`
- `skill_vision/crossing/crossing_left/crossing_right/goalkeeping/defending/shooting/defensive_tracking`
- These never change in a match, so a striker's tabular rows never shared with another striker.
- If re-added: a single coarse "player archetype" bucket, not 22 per-skill bins.

### 🟠 Near-constant pitch-surface physics (6) — ~zero information
- `ball_surface_drag/air_resistance/grass_resistance/stop_speed`, `ball_resistance_total_loss/rolling_contact`
- The pitch doesn't change mid-match. Only worth re-adding if variable pitch/weather is introduced.

Removal sites touched (for reference when re-adding): the `SoccerQStateKey` struct, its
constructor (`from_parts`), `matches_learning_context` / `matches_relaxed_learning_context`,
the dead grid spatial fns (`matches_spatial_cell`, `spatial_index_state`,
`relaxed_cell_id_for_state`), the Postgres row builder, and the neural feature vector
(`soccer_neural_transition_features` — skills re-sourced from the observation, surface/cell
zeroed to preserve input dim).

## Other open items
- **Scooped pass** — needs a new `PassFlight::Scoop` variant: aerial speed floor is 19mph so a
  ~10mph loft isn't expressible today, and loft must decouple from speed (slow but high).
- **Touch model** — verify/extend `DribbleTouchDecision` so the first touch matters most.
- **Grow the neural net** — currently tiny (3,745 params). It should be bigger and ideally consume
  richer RAW inputs (all-22 player kinematics) so it discovers latent structure we didn't hand-bin.
- **Self-play accelerator** — built but disabled (RSS); re-enable routed through the neural net (fixed size) rather than the unbounded tabular table.
- **Long aerial passes** — 20–40% too long (overhit out / to keeper).
