# Soccer Sim — MDP / POMDP Decision Schema

> **Source of truth for the agent decision space.** Discretizing and categorizing
> every decision is what turns the simulator into a well-posed POMDP: an
> enumerable, maskable action space **A** that a neural value/policy can
> approximate, with intra-bucket randomization restoring continuous,
> natural-looking execution.

## 0. Why discretize

- Continuous multi-agent control is intractable to learn directly and impossible
  to enumerate for value-based selection.
- Discretizing each decision into labeled **buckets** gives: (a) an enumerable,
  scorable action set per decision; (b) clean credit assignment — the *bucket* is
  the action key; (c) legality **masking**; (d) a stable tabular fallback the
  neural net blends with.
- **Dither** (small in-bucket randomization at execution) recovers continuity so
  play looks natural, and doubles as exploration. The learner always credits the
  **bucket**, never the dithered value. Disabled in deterministic/eval mode.
- Neural function approximation (the value/critic net, `output_dim:1`,
  ConfidenceGated-blended with tabular Q) is what makes the resulting
  high-dimensional **A** tractable.
- **Bucket counts, ranges, and speeds below are calibrated to real football**
  (elite drives ~80–90 mph, hard outfield passes ~50–60 mph, backward balls
  physically slow), not to any single literal example — refined from soccer
  domain knowledge as the model matures.

## 1. The POMDP tuple

| Symbol | Meaning | Status / where |
|---|---|---|
| **S** | Full Markov sim state: player pos/vel/facing/fatigue(W′), ball pos/vel/altitude/spin, score, clock, set-piece phase. | exists (`world.rs`) |
| **O** | Per-agent **partial** observation: ~129-dim feature vector + legality mask. Includes own `facing_yaw`, so both direction frames (absolute + facing-relative) and the facing-dependent speed cap are derivable. Hidden: opponents' chosen actions/intent, exact teammate future trajectories (estimated via run/lead prediction), info beyond perception; plus noise. | exists (feature builder), **POMDP** |
| **A** | Factored, discretized decision (this doc, §3). | **new — the work** |
| **T** | One engine step; semi-deterministic physics + execution dither. Multi-agent: all 22 agents act per decision tick. | exists (`world.rs`) |
| **R** | Goals (primary), possession/turnover (`deferred_reward_transitions`), shaping (territory, openness, beat-defender, shot quality). | exists (`soccer.rs`) |
| **γ** | Discount over the decision-tick horizon. | exists |

**Decision cadence.** Agents emit a `PlayerIntent { action, sprint }`. Macro
actions commit for a short window (discrete argmax + commit), with trajectory
sampling capped by `SOCCER_TRAJECTORY_MAX_DECISION_GAP_TICKS` for credit
assignment. On-ball agent has the rich choice below; off-ball agents choose from
the movement/defending families.

## 2. Selection architecture (factored / hierarchical)

1. **Macro head** picks the family: `pass | shoot | dribble | clear | hold` (on
   ball) or `move | run | press | …` (off ball). Uses the existing label-based
   Q/critic (`SoccerQActionKey.action: String`, `best_action -> String`,
   `soccer.rs:6212`).
2. **Family sub-policy** enumerates that family's **masked** bucket candidates and
   scores each with the value net (ConfidenceGated blend with tabular Q). For
   passing, candidates are **intent-aware** (generated per targeting mode, §3.2).
3. **Dither + lower**: the winning bucket tuple is randomized within-bucket (§4)
   and lowered to physics via one shared `kick_release` (pass/shoot) or carry
   primitive (dribble).
4. **Realtime guard**: for the live server (15 Hz), a cheap geometric prefilter
   caps candidates (~30–60) before net scoring.

**Action-key encoding** (tabular Q + neural (state,action) features); macro label
stays the prefix so existing macro Q is unchanged:

```
pass|m:lead|tgt:7|spd:7|dir:135|face:open|crv:R|fl:aerial
shot|plc:5hole|pow:8|crv:L|elev:chip
drib|mv:left-cut|dir:b3|face:hard-turn|tch:close|spd:burst|shld:2
```

## 3. Action families (the factored A)

Buckets are the discrete choice; ranges/realized values come from dither + skill
scaling. **Status**: `exists` = primitive already in engine, `new` = to add,
`partial` = present but heuristic-only.

### 3.1 Shared "kick" core (Pass + Shoot lower through one `kick_release`)

`kick_release(origin, speed_yps, abs_dir, curve, elevation, skills)` →
`(ball.velocity, curl accel, loft)`. Reuses existing curl
(`bend_yards` → `MAX_BALL_CURL_YPS2`, `soccer.rs:40915`) and aerial/altitude
physics. Both the new learned path and legacy heuristic/set-piece paths call it.

**Direction is two-framed.** Every kick/carry direction is chosen in the
**absolute field frame** (where the ball/player goes — the value/target frame) but
is always evaluated against the player's current **body facing**. The
`facing offset = abs_dir − facing_yaw` decides *feasibility, forced technique, and
cost*: certain passes/dribbles are simply unavailable given where the player is
facing (you can't drive a ball or carry sharply behind you without first turning).
Both frames are modeled — absolute in the action *target*, facing-relative in the
*mask/technique/cost* — and the facing zone is carried in the action key so
"same target, different body shape" are distinct actions. This generalizes the
existing `pass_facing_outcome` gate (≤30° full → 90–120° hard-turn → >120°
backheel-only) and is bounded by `MAX_BODY_YAW_RATE_RAD_S` for carries. Applies to
passing and dribbling directly; shooting inherits it via this shared core.

### 3.2 Passing

| Axis | Buckets | Status | Mask / notes |
|---|---|---|---|
| target mode | feet / space / lead / clear (4) | new | drives candidate gen; **in key** |
| target ref | per-teammate / space-cell | new | metadata for reward + gates |
| lead amount | short / med / long onto run (3) | reuse lead-point geom | only in `lead` mode |
| direction (target) | 36 absolute (10°), derived from target then quantized | new | where the ball goes — value frame |
| facing offset | in-stride ≤30° / open 30–90° / hard-turn 90–120° / behind >120° (4) | partial (`pass_facing_outcome`) | `abs_dir − facing_yaw`; drives mask + forced technique (backheel) + **speed cap**; in key |
| speed / weight | 10, masked by **distance _and_ facing** | replaces continuous `power` | ceiling falls as the angle leaves facing: ~driven 50–55 mph in-stride → ~30–35 square → ~15–20 behind → ~≤12 backheel |
| curve | none / L / R (3) | curl exists | masked off < ~10 yd / low speed |
| flight | floor / aerial / scoop (3) | `PassFlight` exists (`soccer.rs:2622`) | scoop = slow+high dink |
| technique | inside / outside / laces / no-look-disguise | new | gates curve + disguise |
| touch | one-touch / settle-then-pass (2) | partial (control-touch) | timing |

Targeting resolves to `PassIntent { ToPlayer(id) | ToSpace(Vec2) | ToLead(id,amt) | ToNobody }`.

**Forward-option recognition (anti–backward-recycle).** A "good forward option" is
**not** defined by the receiver's nearest-opponent openness at their current spot
alone: an advanced runner on a clear, completable lane — with a trailing or
already-beaten defender nominally close — reads as "not open" under raw openness, so
the carrier exhausts its forward search and recycles the ball backward even though a
genuine forward ball existed. The POMDP observation `O` therefore also carries
`best_forward_pass_option_quality` (≥ `best_forward_pass_receiver_openness` by
construction): `forward_pass_option_quality(receiver_openness, expected_completion)`
blends openness with the lane-aware completion so recognition can only **add**, never
remove, options. It drives two decisions: (a) the **forward-pass-first release** floors
`pass1` and damps the hold/dribble family when a forward option clears the bar — sooner
the longer the carrier has dwelt; (b) **pass-target ranking** demotes a backward/square
target once a genuinely good forward option exists. The forward-option *count*
(`visible_forward_pass_options`) follows every playable forward visible target — it is
**not** gated on an uncontested straight floor lane, so a contested-lane ball to an open
runner still registers as a recognised option. Gate
`DD_SOCCER_ENABLE_FORWARD_OPTION_RECOGNITION` (default-ON in prod, kill-switch `=0`;
default-OFF under test ⇒ byte-identical option-scoring parity). Not part of the neural
feature encoder, so `FEATURE_DIM` is unchanged.

### 3.3 Shooting

| Axis | Buckets | Status | Mask / notes |
|---|---|---|---|
| placement | bL bR tL tR, center-low/high, near/far post, **5-hole/keeper-legs** (~9) | new | the "direction"; maps to an absolute angle |
| power | 10, masked (finesse↔power tradeoff) | replaces `Shoot{power}` | |
| curve | none / L / R (3) | curl exists | bend round the keeper |
| elevation | driven-floor / low / lofted / **chip** / scoop (5) | partial | chip = dink the keeper |
| technique | laces(power) / inside(finesse) / outside(trivela) / volley / half-volley / header | new | volley/header gated by ball altitude |
| foot | left / right | skills exist (`skill_*_foot_shot_power`) | weak-foot penalty |

### 3.4 Dribbling

| Axis | Buckets | Status | Mask / notes |
|---|---|---|---|
| move / feint | carry / cut L,R / out L,R / nutmeg / fake-cut L,R / xavi-turn / stepover / … | `DribbleMoveKind` exists | extend set |
| direction (target) | 36 absolute (10°) | change from attack-relative | where the carry goes |
| facing offset | in-stride / openable / hard-turn / behind (4) | partial | `abs_dir − facing_yaw`, bounded by `MAX_BODY_YAW_RATE_RAD_S`; **couples with move/feint** (cuts are *how* facing changes) + caps carry speed; in key |
| touch distance | close-control ↔ knock-and-chase (3–4) | `DribbleTouchDecision.distance` exists | |
| speed / burst | jog / run / sprint / explosive-burst (4) | sprint flag exists | change of pace |
| **shield level** | 1 light screen / 2 full shield / 3 back-to-goal hold-up | `ProtectBall` partial | graded retention |
| body orientation | which side to screen / open up | facing exists | |

### 3.5–3.8 Remaining families — *proposed, pending sign-off*

Same machinery (factored → mask → dither). Bucket counts tentative.

- **Receiving / first touch**: touch direction (quantized), settle vs redirect vs
  let-it-run/dummy (3), shield-on-reception (0–2), first-time-release (bool).
- **Off-ball movement**: make-run / hold / check-to-ball / overlap / underlap /
  drop (≈6), run timing (now/delay), target lane.
- **Defending**: press intensity 1–3, jockey/contain vs commit-tackle (2),
  intercept-lane (bool), tactical-foul (bool, gated), blindside-steal (bool,
  gated `DD_SOCCER_ENABLE_BLINDSIDE_STEAL`): creep into a slow forward-dribbler's
  blind arc and nick it from behind *only when the steal is believed catchable*
  (closing-speed margin). The carrier's countermeasure is a side-glance head-scan
  (`blindside_threat_from_behind` obs at a control-drift cost) that drives a
  break-away escape.
- **Goalkeeping**: set-position, come/stay (2), save-style, distribution
  (throw/roll/short/long × the kick core).

## 4. Masking & dither

**Mask (legality)** — illegal candidates pruned *before* scoring:
- **facing feasibility (primary)**: directions far from current `facing_yaw` are
  illegal this tick, force turn-first/backheel, and **cap the legal speed buckets**
  (you can drive it forward, only roll it backward) — bounded by
  `MAX_BODY_YAW_RATE_RAD_S` (carries) and `pass_facing_outcome` zones (kicks);
- curve requires distance ≥ ~10 yd and speed ≥ threshold;
- volley/half-volley/header require matching ball altitude;
- weak-foot / skill caps bound speed & technique buckets;
- other safety gates: lane veto (led-OR-feet), shot placement must cross the goal mouth.

**Dither (execution only)** — bucket → realized value:
- speed: bucket-center ± ½-bucket Gaussian; angle: ± ~5°; curl magnitude
  skill-scaled with small jitter;
- never crosses into a neighbor bucket's meaning; **off in eval/deterministic mode**;
- learner credits the bucket.

## 5. Migration / rollout (behavior-neutral first)

1. Shared bucket scaffold + `kick_release` lowering — unify physics, **no behavior
   change**. Legacy heuristic `Pass`/`Shoot` lower through it.
2. Add `PassKick` / `ShotKick` variants + masks + dither + intent classifier.
3. Intent-aware candidate generation + factored sub-policy scoring.
4. Action-key encoding + reward/gate masks; dribble axes; then phase-2 families.
5. Tests at each step; parity vs the ~60-fail `dt=1/15` baseline (build in the
   main dir, not a worktree). Legacy heuristic passing stays as a safety-gated
   fallback during rollout (same flags as the neural blend).

## 6. Open / to confirm

- ~~Direction frame: absolute vs attack-relative~~ **Resolved:** two-framed for
  passing *and* dribbling — absolute field angle as the action target, plus a
  facing-relative zone that drives feasibility/technique/cost **and the speed cap**
  (§3.1, §4). Dribbling moves off attack-relative to this model.
- Are phase-2 families (§3.5–3.8) in the first build batch, or after on-ball?
- Any axes still missing (give-and-go intent, shape-to-shoot feint, set-piece
  variants)?
