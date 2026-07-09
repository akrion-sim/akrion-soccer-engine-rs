# Rollout PoC harness — implementation spec (Codex-refined)

Go/no-go on the premise: **does the analytic base wrongly suppress useful forward passes** — i.e. does
forcing a good forward pass (evaluated by short rollout under the analytic base) beat what the analytic
naturally does, on an EPV-aware leaf? Positive ⇒ build the full offline rollout teacher. Negative ⇒
premise fails or fork-fidelity is off (see gating parity control).

Builds on the landed primitive: `SoccerMatch::fork_for_rollout(seed)` + `rollout_forced_action` (commit
a093a38). Codex review folded in: parity control (gating), event-aligned termination, best-of-k arms,
EPV-aware leaf, explicit/stratified sampling.

## Visibility constraint
The bin is a separate crate → only `pub` items of `soccer_engine` are reachable. `rollout_forced_action`
and `pending_pass` are `pub(crate)`. So ALL rollout/leaf/event logic lives as **`pub` methods on
`SoccerMatch`** (they can see internals); the bin is a thin driver over pub methods + pub fields
(`ball.holder`, `players`, `stats`, `score_home/away`, `tick`, `clock_seconds`, `config`).

## LIB additions (new `pub` methods in an `impl SoccerMatch` block in world.rs — all additive/inert)

Add a pub result struct (near the other pub structs in world.rs):

```rust
#[derive(Clone, Debug, Default)]
pub struct PocArmResult {
    pub leaf: f64,               // EPV-aware value (see formula)
    pub epv_end: f64,            // expected_threat at the ball's ending position
    pub possession_retained: bool,
    pub turnover: bool,
    pub completed_fwd_delta: i64,// carrier team's completed_forward_pass count gained over the roll (diagnostic)
    pub ticks_run: u32,
    pub forced_action_fired: bool, // did the forced action actually execute (guards silent no-ops)
}
```

1. Carrier detection:
```rust
/// The on-ball ball carrier if it's a FIELD player (not a keeper), else None.
pub fn poc_on_ball_field_carrier(&self) -> Option<usize> {
    let id = self.ball.holder?;
    let p = self.players.iter().find(|p| p.id == id)?;
    if p.is_goalkeeper() { None } else { Some(id) }   // use the real GK predicate; grep player.rs for role==Goalkeeper
}
```

2. Forward-open teammate targets (best-of-k), ranked by `expected_threat`:
```rust
/// Up to `k` forward, open teammate positions for `carrier`, ranked by expected_threat desc.
/// "forward" = strictly higher expected_threat than the carrier's own cell (+ small margin);
/// "open" = nearest opponent farther than `open_yds`.
pub fn poc_forward_pass_targets(&self, carrier: usize, k: usize, open_yds: f64) -> Vec<Vec2> {
    // team + carrier pos from self.players; field dims from self.config.field_width_yards/length_yards.
    // et(p) = crate::des::general::soccer::pitch_value::expected_threat(team, p, w, l)
    // filter teammates (same team, != carrier, not GK) with et(tm) > et(carrier)+MARGIN and
    // min distance to any opponent > open_yds; sort by et desc; take k positions.
}
```

3. The rollout arm (fork → optional force → event-aligned roll → EPV leaf):
```rust
/// Fork, optionally force `(action_label, target)` on the carrier's next decision, roll under the
/// analytic base until the play's immediate outcome settles (event-aligned) or `max_h` ticks, and
/// return the EPV-aware leaf. `forced == None` = the NATURAL arm (analytic plays its own pick).
pub fn poc_rollout_arm(&self, carrier: usize, forced: Option<(String, Option<Vec2>)>,
                       seed: u64, max_h: u32) -> PocArmResult {
    let mut w = self.fork_for_rollout(seed);
    let team = self.players.iter().find(|p| p.id == carrier).map(|p| p.team);
    let team = match team { Some(t) => t, None => return PocArmResult::default() };
    let (w_ratio, l) = (self.config.field_width_yards, self.config.field_length_yards);
    let fwd0 = w.stats /* carrier team's completed_forward_pass count at start */;
    let mut had_pending = false; // a pass/shot went in flight during the roll
    if let Some((action, target)) = forced {
        w.set_rollout_forced_action(carrier, action, target); // new pub setter (below)
    }
    let mut ticks = 0u32;
    let forced_fired_probe = /* see note */;
    loop {
        let pending_before = w.poc_has_pending_ball_event(); // pub helper: pending_pass||pending_shot||pending_rebound is Some
        w.run_time_step();
        ticks += 1;
        if pending_before { had_pending = true; }
        let holder_team = w.ball.holder.and_then(|h| w.players.iter().find(|p| p.id==h).map(|p| p.team));
        let settled_other = holder_team == Some(team.other()); // turnover: other team controls
        let resolved = had_pending && !w.poc_has_pending_ball_event() && w.ball.holder.is_some(); // pass/shot resolved to a holder
        if settled_other || resolved || ticks >= max_h { break; }
    }
    let epv_end = pitch_value::expected_threat(team, w.ball.position(), w_ratio, l);
    let holder_team_end = w.ball.holder.and_then(|h| w.players.iter().find(|p| p.id==h).map(|p| p.team));
    let possession_retained = holder_team_end == Some(team);
    let turnover = holder_team_end == Some(team.other());
    let fwd1 = /* carrier team's completed_forward_pass count now */;
    let leaf = epv_end
        + POSSESSION_BONUS * (possession_retained as i32 as f64)
        - TURNOVER_PENALTY * (turnover as i32 as f64);
    PocArmResult { leaf, epv_end, possession_retained, turnover,
                   completed_fwd_delta: (fwd1 - fwd0) as i64, ticks_run: ticks,
                   forced_action_fired: /* set via a fork probe or by comparing action; see note */ }
}
```
Constants (tune, but start): `POSSESSION_BONUS = 0.30`, `TURNOVER_PENALTY = 0.60`, `MARGIN = 0.05`,
`open_yds = 3.0`. `completed_forward` fields: `stats.passes_completed_forward_home/away` keyed by team.

4. Small pub helpers used above (encapsulate pub(crate) fields):
```rust
pub fn poc_has_pending_ball_event(&self) -> bool {
    self.pending_pass.is_some() || self.pending_shot.is_some() || self.pending_rebound.is_some()
}
pub fn set_rollout_forced_action(&mut self, player_id: usize, action: String, target: Option<Vec2>) {
    self.rollout_forced_action = Some((player_id, action, target));
}
```
(`ball.position()` — use the real accessor; grep BallAgent for the position getter, likely `self.ball.state.position` or a `position()` method.)

5. Parity control (gating): natural's own action re-forced must ≈ natural.
```rust
/// If the carrier launches a PASS on the natural next tick, return its target (from pending_pass),
/// so the driver can force ("pass", target) and confirm it reproduces natural within CRN noise.
pub fn poc_natural_pass_target(&self, carrier: usize, seed: u64) -> Option<Vec2> {
    let mut w = self.fork_for_rollout(seed);
    w.run_time_step();
    // if a pass just launched from carrier, return its target point (pending_pass target/destination).
    w.poc_pending_pass_target_for(carrier) // pub helper reading pending_pass fields
}
```

## BIN: `src/bin/rollout_poc.rs`
Driver only; env-config; walks ONE long analytic self-play, evaluates arms inline at sampled on-ball
states, aggregates.

- Build match: `MatchConfig::default()` with `learning_enabled=false`, neural OFF (analytic base).
  Construct via the same public constructor the other bins use (grep `SoccerMatch::` in
  src/bin/measure_offense.rs for the canonical headless setup).
- Env: `POC_STATES` (default 200), `POC_SEEDS` (CRN for selection, default 4), `POC_HOLDOUT` (disjoint
  eval seeds, default 4), `POC_MAX_H` (default 45 = 3s cap), `POC_K` (default 3), `POC_SAMPLE_EVERY`
  (ticks between samples during possession, default 10).
- Loop: step the live match; each `POC_SAMPLE_EVERY` ticks, if `poc_on_ball_field_carrier()` is Some:
  - `total_on_ball += 1`. targets = `poc_forward_pass_targets(carrier, K, 3.0)`.
  - If targets empty → record stratum "no-forward-option", continue (still counts in denominator).
  - `with_forward_option += 1`.
  - SELECTION seeds (POC_SEEDS): natural arm = `poc_rollout_arm(carrier, None, s, MAX_H)`; fwd arms =
    `poc_rollout_arm(carrier, Some(("pass".into(), Some(t))), s, MAX_H)` for each t; dribble arm =
    forced `("dribble", Some(carrier_pos + fwd_unit*6.0))`. Q_hat(arm) = mean leaf over seeds.
  - best_fwd = argmax Q_hat over fwd arms.
  - HELD-OUT seeds (POC_HOLDOUT, disjoint): re-roll natural + best_fwd + dribble; record held-out leaf.
  - PARITY: `poc_natural_pass_target(carrier, s0)`; if Some(t_nat), force ("pass", t_nat) on holdout
    seeds and record `parity_delta = mean|leaf_forced_nat - leaf_natural|`.
  - Stratify: zone (thirds by ball y vs attack dir), whether natural arm's result shows nat already
    played forward (epv gain>0 & retained), pressure bucket if available.
  - Continue the LIVE match (unforced) so sampling walks real self-play.
- After N states, REPORT (stdout, one block):
  - PARITY GATE: mean parity_delta and its 90% CI — must be ≈0 (state the pass/second so it's clear).
  - Denominator: total_on_ball, with_forward_option (%).
  - HELD-OUT paired deltas: mean(best_fwd_leaf − natural_leaf) with bootstrap 95% CI; fraction of
    states best_fwd > natural; fraction where fwd is the global argmax (incl natural/dribble).
  - Diagnostics: mean completed_fwd_delta per arm; mean turnover rate per arm; mean ticks_run.
  - Stratified: the held-out delta split by zone + nat-already-forward.
  - VERDICT line: POSITIVE iff parity gate passed AND held-out mean(best_fwd − natural) > 0 with 95%
    CI excluding 0; else INCONCLUSIVE/NEGATIVE (and note: a single-arm loss with parity-failing does
    NOT falsify the premise).

## Build + run
```
cd tmp/worktrees/rollout-poc
CARGO_TARGET_DIR=/Users/maca5/codes/akrion-sim/akrion-soccer-engine.rs/target cargo check --lib 2>&1 | tail -40
# then the bin (release — debug is ~10-50x too slow to run the experiment):
CARGO_TARGET_DIR=.../target cargo build --release --bin rollout_poc 2>&1 | tail -20
POC_STATES=200 POC_SEEDS=4 POC_HOLDOUT=4 POC_MAX_H=45 POC_K=3 \
  CARGO_TARGET_DIR=.../target ./... /release/rollout_poc 2>&1 | tail -60
```
Register the bin in Cargo.toml if the `[[bin]]` list is explicit (it is — add a `[[bin]] name="rollout_poc" path="src/bin/rollout_poc.rs"` entry).
