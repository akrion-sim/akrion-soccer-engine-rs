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
- **No DB, no server, and none of the engine's ~129-dim POMDP feature stack.**
  It *is* still a POMDP — each outfielder acts on a **partial, egocentric 71-dim
  observation** in its own attack frame (ball/goal/teammate/opponent relatives,
  pass-candidate openness, shot-lane clearness, pressure), hidden from it are
  opponent intent and teammates' exact future runs. The policy is **reactive /
  memoryless** (no belief-state or RNN) — the standard approximation, not the
  full feature-builder + masking + tabular-Q-blend machinery of the big engine.

## What IS here
- **5 v 5 with goalkeepers** and a realistic ~7 m net on a small (42×28) pitch.
- Macro actions mirroring the real engine's vocabulary: on-ball
  `shoot · pass_a · pass_b · pass_c · dribble{fwd,left,right} · clear · hold`; off-ball
  `chase · support · spread · mark · stay` — with legality masking.
- **MPC-lite finishing.** `shoot` runs a short-horizon aim optimizer that
  enumerates aim points across the mouth and places the ball at the corner
  farthest from the keeper (no MPC in the *decision*, only in the *finish*).
- The **4 outfielders** are a shared-weight policy warm-started by scripted
  behavior cloning, then trained with **PPO (clipped) + GAE** over threaded
  rollout batches. The keeper is a fixed rule. Opponent = a scripted
  "analytic-lite" baseline plus light noisy-scripted variants during training;
  clean evaluation is still against the scripted benchmark. Reward = goals +
  **potential-based shaping** (forward progress) + small progressive-pass credit.
- **keep-best** snapshotting: PPO over-trains and collapses past its peak, so we
  evaluate each checkpoint and showcase the best (winning + passing) one.

## Run it
```bash
cargo run --release -- sanity        # scripted-vs-scripted symmetry check (~0.0)
cargo run --release -- train 80      # writes out/learning_curve.csv, run_manifest.json, model + traces
./viz/serve.sh                       # assemble the page + serve on localhost:8080
```

Useful reproducibility knobs:
```bash
cargo run --release -- train 80 --seed 20260710 --games-per-iter 12 --bc-games 12 --bc-epochs 2 --eval-games 80 --final-games 300
FIVEASIDE_ROLLOUT_THREADS=2 cargo run --release -- train 80
SPACING_W=0.003 cargo run --release -- train 80
PORT=5060 ./viz/serve.sh
```

## Result (representative)
- Untrained policy: typically loses and struggles to keep the ball.
- Trained policy: use `out/run_manifest.json` and the generated page as the
  source of truth. The selected checkpoint records whether it cleared the
  behavior hardening gates (positive goal diff, nonzero shots/conversion,
  completed passes, possession floor, and bunching cap).

## Hardening
- `cargo test --locked` covers deterministic PRNG replay, masked softmax,
  model save/load validation, observation/global-state finiteness, legal-action
  masking, full pass-target action coverage, behavior-cloning warm start, the
  2-pass shot gate, goal validity, and evaluation metric bounds.
- `cargo clippy --locked --all-targets -- -D warnings` is intended to stay clean.
- Generated model/viz artifacts are tied together by `out/run_manifest.json`
  with git commit, seed/config/env, selected checkpoint, final holdout stats,
  and checksums for the saved actor/critic/CSV/traces.

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
