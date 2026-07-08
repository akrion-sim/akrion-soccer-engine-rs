# Learned Pass-Receiver Head (`DD_SOCCER_ENABLE_LEARNED_PASS_RECEIVER`)

> Status: in progress. Default-OFF, byte-identical parity when off. Opt-in / kill via
> `DD_SOCCER_ENABLE_LEARNED_PASS_RECEIVER=1|0`.

## Problem

Today the POMDP/tabular-Q head decides *whether* to pass and emits a coarse **grid-zone**
preference (`SoccerQTargetKey` keyed on pitch-grid cell ids). The specific **receiver** is then
chosen by a heuristic MPC-blend in `best_pass_target_player_for_snapshot`
(`soccer.rs`): learned grid value (weight 1.0 if any) + `expected_completion*0.82 +
mpc_receipt_probability*0.52 + stride_fit*0.22 + receiver_openness*0.20`. Because the learned
term is a grid value shared by every teammate in a cell, the receiver is effectively picked by
the heuristics, and "pass into open space / to no particular teammate" is unrepresentable
(candidates are always visible teammates). MPC only refines aim/speed for the already-chosen
receiver; when a pass is MPC-infeasible, `mpc_reconciled_learned_plan` replans at the
**action** level (pass→dribble→hold), never trying a *different receiver*.

## Design

The POMDP head decides **who** (or open space), and MPC feasibility is a **legality mask** that
kicks an infeasible receiver back to the head for its next choice.

### Receiver descriptor (folded into the learned target key)

A compact, attack-relative discretization of the receiver, computed identically at trace time
(training credit) and inference time (query) via one shared
`WorldSnapshot::pass_receiver_descriptor`:

| axis        | buckets                                   |
|-------------|-------------------------------------------|
| kind        | teammate (0) / space (1) / nobody (2)     |
| role        | GK 0 / DEF 1 / MID 2 / FWD 3 (teammate)   |
| lane        | left 0 / central 1 / right 2 (attack-rel) |
| progression | backward 0 / square 1 / forward 2         |
| openness    | tight 0 / contested 1 / open 2            |

Packed: `((kind*4 + role)*3 + lane)*3 + progression)*3 + openness` ∈ `0..=323`.
`-1` = **Unspecified** (parity/legacy: gate off, pre-migration rows, non-pass entries).

### Key / entry / persistence

- `SoccerQTargetKey` + `SoccerQTargetEntry` gain `receiver_descriptor: i32` (serde default -1).
- `AgentActionTargetTrace` gains `receiver_descriptor: Option<i32>` (serde default None),
  set in `learned_plan_action_target_trace` when the gate is on.
- RDS: `des_soccer_learning_policy_entries` and `des_soccer_learning_run_deltas` gain
  `receiver_descriptor integer default -1 not null`, added to their unique indexes + ON CONFLICT
  clauses (so two receivers in one grid cell are distinct rows, not a collision). Migration +
  polyglot regen in `k8s-cluster/remote/libs/pg-defs`.

### Training

`update_with_reward`: when gate on and the transition is a pass carrying a receiver descriptor,
credit `target_values[(state, action, grid, receiver_descriptor)]`. Gate off ⇒ descriptor is
Unspecified(-1) ⇒ key identical to today ⇒ parity.

### Inference (`best_pass_target_player_for_snapshot`, gate on)

1. For each visible-teammate candidate, compute its descriptor and look up the learned
   per-(grid,descriptor) value. The **learned value dominates**; the heuristic quality terms
   become tie-breakers only.
2. Synthesize a **ToSpace** candidate from the head's preferred target grid
   (`best_target_grid_for_state_action`) when that grid is not occupied by a teammate — the
   existing grid head becomes the "pass to space" mechanism.
3. **MPC kickback:** rank candidates by head value; walk the ranking, and skip any whose MPC
   pass-execution probability is below the impossible threshold (mask). Commit the first
   MPC-feasible one. Only if *no* receiver (incl. space) is feasible does control fall through to
   the existing action-level `mpc_reconciled_learned_plan`.

Gate off ⇒ the legacy heuristic path runs unchanged.

## RDS migration

`schema.sql` in `k8s-cluster/remote/libs/pg-defs` is authoritative; migrations are diff-generated
into `tmp/migrations/<env>/pg-defs-diff.sql` and human-applied. The diff tool auto-emits the
`add column` + check constraint but **not** the unique-index recreation (adding a key column
changes the index), so apply this reviewed, idempotent SQL to both tables:

```sql
-- des_soccer_learning_policy_entries  (repeat verbatim for des_soccer_learning_run_deltas)
alter table des_soccer_learning_policy_entries
  add column if not exists receiver_descriptor integer default -1 not null;
alter table des_soccer_learning_policy_entries
  add constraint des_soccer_learning_policy_entries_receiver_descriptor_chk
  check (receiver_descriptor >= -1) not valid;
alter table des_soccer_learning_policy_entries
  validate constraint des_soccer_learning_policy_entries_receiver_descriptor_chk;
drop index if exists des_soccer_learning_policy_entries_key_uq;
create unique index if not exists des_soccer_learning_policy_entries_key_uq
  on des_soccer_learning_policy_entries (
    policy_version_id, team, entry_kind, state_hash, action,
    target_fine_cell_id, target_tactical_cell_id, target_macro_cell_id, target_root_cell_id,
    receiver_descriptor
  );
```

Existing rows default to `-1` (Unspecified) — byte-compatible with the grid-only entries. Safe to
apply before the code lands (the gate is default-OFF, so nothing writes a non-`-1` descriptor
until an operator opts in).

</content>
</invoke>
