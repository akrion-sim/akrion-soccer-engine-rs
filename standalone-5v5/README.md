# standalone 5-a-side — a learning existence proof

A **fully self-contained** 5-a-side soccer RL demo, deliberately stripped down to
answer one question: *can a policy actually learn to play?* Yes — and you can
watch it happen.

It shares **no code** with `soccer_engine` / `des_engine` — own PRNG, own MLP +
backprop, own PPO. Zero external crates. An empty `[workspace]` table detaches it
from the surrounding cargo workspace so it builds in complete isolation.

## What's deliberately NOT here
- **No formation LP / IPM / shape machinery.** Off-ball movement is pure
  *possession + open-space*: run upfield to attack, drop to defend, drift to the
  most open point on the pitch. Possession is tracked (`World.owner`).
- **No POMDP feature stack, no DB, no server.** A ~34-dim egocentric observation
  in the player's attack frame, and a 13-action masked macro space.

## What IS here
- **5 v 5 with goalkeepers** and a realistic ~7 m net on a small (42×28) pitch.
- Macro actions mirroring the real engine's vocabulary: on-ball
  `shoot · pass_a · pass_b · dribble{fwd,left,right} · clear · hold`; off-ball
  `chase · support · spread · mark · stay` — with legality masking.
- **MPC-lite finishing.** `shoot` runs a short-horizon aim optimizer that
  enumerates aim points across the mouth and places the ball at the corner
  farthest from the keeper (no MPC in the *decision*, only in the *finish*).
- The **4 outfielders** are a shared-weight policy trained with **PPO (clipped)
  + GAE**; the keeper is a fixed rule. Opponent = a scripted "analytic-lite"
  baseline (also the benchmark to beat). Reward = goals + **potential-based
  shaping** (forward progress) + small progressive-pass credit.
- **keep-best** snapshotting: PPO over-trains and collapses past its peak, so we
  evaluate each checkpoint and showcase the best (winning + passing) one.

## Run it
```bash
cargo run --release -- sanity        # scripted-vs-scripted symmetry check (~0.0)
cargo run --release -- train 80      # train; writes out/learning_curve.csv + traces
./viz/serve.sh                       # assemble the page + serve on localhost:5060
```

## Result (representative)
- Untrained policy: loses, never keeps the ball.
- Trained (best checkpoint, ~iter 40, ~3 min): **goal-diff ≈ +0.85, ~71% win**,
  ~2 goals/game, **~11 completed passes/game** — it presses to win the ball,
  passes to teammates in space, carries, and places shots past the keeper.

## The honest frontier
The emergent style is *direct-ish* — dominant, coordinated multi-pass build-up is
harder to force from shared-weight PPO vs a scripted bot and is sensitive to the
defense's strength and reward scale. Levers to push it further: a stronger
positional defense, self-play, or an explicit possession objective.

## Layout
- `src/rng.rs` — xorshift PRNG (seeded, reproducible)
- `src/nn.rs` — MLP + manual backprop + Adam, masked softmax
- `src/game.rs` — sim: physics, actions, keeper, finishing, scripted baseline, reward potential
- `src/train.rs` — PPO + GAE, rollouts, evaluation
- `src/main.rs` — CLI, best-checkpoint tracking, trace recording
- `viz/` — self-contained before/after + learning-curve page (`assemble.py`, `serve.sh`)
