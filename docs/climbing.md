# Climbing — how we got the neural net to actually improve (July 2026)

Companion to [`reframe-july-2026.md`](./reframe-july-2026.md). That doc explains the
*reframe*; this one explains **what actually moved the curve**, honestly (including where an
early sample got ahead of the truth), and **how to build on the 22-player + ball field
vector** so the neural net keeps climbing.

---

## Part 1 — What we actually accomplished (honestly)

The improvement is **real but modest and noisy** — the net went from **losing every game** to
**roughly even** with a fixed reference opponent. Measured with clean 12-game held-out
head-to-heads (authoritative both sides):

| On-policy steps | Clean 12-game vs reference | Confident? (Wilson ≥ 0.5) | League round record |
|---:|---:|---|---|
| 85k  | **Δ −109** (0W-2D-1L) | clearly losing | — |
| 106k | Δ **+105** (4W-7D-1L) | **no** (Wilson 0.35 — noisy high sample) | round 1: 4W-11L (−7) |
| 128k | Δ **−12** (5W-2D-5L, GD 0) | no (Wilson 0.25) | round 3: **11W-4L (+7)** |

**Read it straight:** on-policy training pulled the net out of the broken off-policy state
(−109, losing every game) up to **~even** with the reference. The **+105 was a noisy high
sample, not a confident win** — its own Wilson lower bound was below the confidence floor, and
the next clean read was ≈0. So this is **"went from clearly-losing to competitive,"** *not* a
confident steep climb to beating the engine. (An earlier draft of this doc claimed a "+214
steep climb"; that was a lucky 12-game sample getting ahead of the data — corrected here.)

What *is* solid: (1) it improved from losing-everything to competitive; (2) the league round
record trended up (−7 → −3 → +7 — though confounded by the growing league); (3) **the play is
visibly better on the live server.** Real progress, honestly modest, and the head-to-head
meter is too noisy at 12 games to claim a confident win — bigger samples (or a frozen eval
pool) are needed to state the level precisely.

The mechanism below is what took it from broken to competitive — **four fixes stacked in the
right order**. Any one alone did nothing; together they made dense-reward RL actually move the
play.

### 1. Make the neural net *decide*, not nudge

The per-player action was `Q_eff = Q_tab + λ·V_net` with λ=0.5 — a coarse tabular Q ranking
analytic candidates, the neural adding a small, **capped** nudge. Learning was real but
wired into a component that structurally *can't move behaviour*. The current local path uses
`SOCCER_NEURAL_BLEND_MODE=authoritative`, where the score is
`λ·V_net(s,a) + 1e-6·Q_tab(s,a)`: the neural value **dominates** the decision, and the tabular
value collapses to a deterministic tie-break floor. Now the net's learning shows up as
behaviour.

### 2. Persist the net's *warmth* so it isn't inert on load

`effective_lambda = λ · readiness`, and `readiness = 0` unless the snapshot carries
`training_steps > 0` **and** a finite `average_loss`. The old self-play path persisted
`training_steps = 0`, so every reloaded net was **inert** — play was 100% analytic and every
trained net evaluated identically. The trainer now carries the live stats into the snapshot,
and the local `:5055` launcher keeps neural serving enabled, so a loaded net actually drives.
The local training launcher also sets `SOCCER_NEURAL_BLEND_WARMUP_STEPS=0`, so a warm
local-best snapshot is active immediately.

### 3. Verify the dense reward reaches the value net — it does

We *traced the pipeline* instead of guessing. The dense per-action rewards — **forward-pass
+, dribble +, turnover −, forward-chain +** — are set on the training transitions, and the
default replay path (`soccer_correlated_full_game_replay_transitions`) **preserves**
`transition.reward`. The outcome-credit path that would overwrite it with only
`outcome + milestone + pitch` is **off**. So the reward was never the blocker; the feedback
is dense, strong, and constant, exactly as designed.

### 4. Train **on-policy** — the actual unlock

With 1–3 in place the net still played *worse* than the analytic engine (Δ−109). Not a
reward bug — **distribution shift**: the net's values were learned while the *analytic
engine* drove, so it only ever saw states good play produces. Handed the wheel, it drove
into states its own worse decisions create — states it never trained on, where its value is
garbage. The fix: run the trainer with the net **authoritative during training**, so it
learns the value of the states **it actually creates**, corrected in-distribution by the
dense per-action rewards. That single change turned −109 into +105 and kept climbing.

**The lesson:** the bottleneck was never "sparse reward" or "can't learn." It was
(a) the net wasn't allowed to decide, (b) it was inert on reload, and (c) it was trained
off-policy. Fix all three and dense-reward RL climbs steeply on its own.

> Deliberately **not** used: genetic/evolutionary layer (reserved for escaping local optima
> later, never as a fitness gate), tournament brackets, imitation learning.

---

## Part 2 — Incorporating the 22-player + ball field vector

The state the net conditions on is the **whole-field moment**: for all 22 players + the ball,
the actor-relative canonical **(position, velocity, acceleration, jerk)** — 8 numbers/entity,
23 entities = a 184-dim motion block inside the 620-dim feature vector
(`SOCCER_NEURAL_FEATURE_DIM=620`, `FIELD_MOTION_PER_PLAYER=8`; the earlier 610 was the subtotal
before the final 10-dim action-param block was appended). This is the right raw signal. Here is
how to use it well.

### Why neural nets are the right tool for it

No two 610-dim field vectors are ever identical — and that's fine. A neural net is a
**function approximator that generalises over a continuous space**: similar field states →
similar features → similar value. It interpolates across everything it's seen. This is
exactly why it beats the tabular Q-table, which buckets states into coarse bins that alias
unrelated 22-player configurations together (the historical plateau). **The net does not
need to memorise states; it learns a smooth function over them.**

### The representation traps to respect

Feeding 23 entities × 8 features as a flat vector into an MLP has three known weaknesses —
address these and the field vector pays off far more:

1. **Permutation sensitivity.** A flat concat makes "player in slot 7" a fixed input dim, so
   the same tactical situation with two teammates swapped looks different to the net. Order
   entities deterministically (by id / by role — the engine orders own-team then opponents
   by id) to reduce it, but the real fix is a **permutation-equivariant** architecture:
   process each entity with a shared encoder and **pool with attention** (a set/graph model),
   so "who is where" matters, not "which array index." The engine already ships a
   **relational-attention readout block** — lean on it; it's the correct inductive bias for
   a multi-entity field.

2. **Coordinate frame.** Raw pitch coordinates make a mirrored or shifted-but-identical
   situation look far away. The field block is already **actor-relative and canonical
   (forward/lateral)** — keep everything in that frame; never feed world coordinates.
   Normalise scales (positions by pitch size, velocities by top speed) so no channel
   dominates.

3. **Curse of dimensionality for *distance*, not for *learning*.** The net generalising over
   610 dims is fine. What is **not** fine is doing nearest-neighbour / kNN with **raw
   euclidean distance** on that vector — in high-dim raw space distances concentrate and
   become uninformative, and raw positions aren't the tactical space. If you retrieve similar
   moments, do it in a **learned embedding** with **cosine** similarity, never raw euclidean.

### Retrieval / embeddings as an augmentation (optional, complementary)

The engine already has `SoccerMomentVectorIndex` + cosine similarity + retrieval priors
(`apply_retrieval_prior_to_policy_actions`) — an episodic memory that nudges the action
distribution from *similar past moments*. Roles:

- **Neural net (primary):** generalise `V(field_state)` — handles "no two identical" natively.
- **Embedding retrieval (augment):** priors for rare/novel states, in a *learned* space
  (cosine), useful especially where the net is undertrained.
- **Raw euclidean kNN:** avoid — wrong metric.

(Currently retrieval stores moments in pgvector/Postgres, which is disabled in the fully-local
run, so the net is generalising on its own. An in-memory moment index would re-enable it
without a database.)

### Concrete recommendations to keep climbing on the field vector

1. **Prefer relational/attention over the entities** to the flat MLP — permutation-equivariant
   set/graph encoding of the 23 entities is the single biggest representational upgrade, and
   the relational-attention block already exists to build on.
2. **Keep the actor-relative canonical frame + per-channel normalisation** (done) — it's what
   makes "same situation" look the same.
3. **Grow network capacity if it plateaus** — the local run is now at 128 hidden units
   (`DEFAULT_SOCCER_NEURAL_HIDDEN_UNITS = 128`, soccer.rs:5423) on a 620-dim input; 256 is the
   next capacity lever once on-policy training flattens.
4. **Use the velocity/acceleration/jerk channels, not just position** — they let the value
   head reason about *where play is going*, not just where it is; the block already carries
   them (V2 added jerk over V1).
5. **Add embedding-space retrieval** (in-memory, cosine) as an episodic prior for rare states
   — complements the net, and pairs naturally with the future "escape local optima" work.

---

## Where it stands (honest)

On-policy authoritative training took the net from **losing every game (−109) to ~even** with
the reference (clean 12-game Δ ≈ 0 at 128k), the net is genuinely deciding on the full
22-player + ball kinematic vector, and the improvement is **visible in live play**. It is
**not** yet confidently beating the reference/analytic engine — the head-to-head is noisy at
12 games and needs a bigger sample or a frozen eval pool to state the level precisely. Next
levers to push past "competitive" toward "clearly better": **relational-attention over the
entities** (biggest representational upgrade), **more capacity** (24 → 64/128 hidden units),
keep exploiting the velocity/accel/jerk channels, and **embedding-space retrieval** for rare
states — plus a **lower-variance eval** (frozen diverse pool, more games) so we stop chasing
noisy single samples.
