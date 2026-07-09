NOTE: Use `~/codes/akrion-sim/akrion-soccer-engine.rs` (the current/canonical
soccer engine) for this work.

WORK ONLY ON `main`. Do NOT `git checkout` another branch, do NOT create new
branches, and do NOT create git worktrees on other branches. Stay in this one
working tree on `main` and commit directly to it. To incorporate another branch's
work, run `git merge <branch>` FROM main (a merge does not check the branch out).
Branches/worktrees are forbidden even as a "safe sandbox" — keep everything on main.
If a throwaway worktree is ever unavoidable, it MUST live under `tmp/worktrees/`
(`tmp/` is gitignored), e.g. `git worktree add tmp/worktrees/<name>`. NEVER create a
worktree in the repo root or any other in-tree path: `git add -A` then stages it as an
embedded gitlink/submodule (the `adding embedded git repository` warning). System `/tmp`
is also fine, but `tmp/worktrees/` is the convention here.

NEVER use `git stash`. It silently hides work off to the side, and when multiple agents
(or an autosaver) touch the same tree, a stash/pop races and clobbers in-progress edits —
a `git stash pop` landing mid-merge can drop conflict resolutions, and a background
autosaver stashing your working tree makes any merge non-deterministic. Keep changes
either committed on `main` or live in the working tree — never stashed. To set work aside,
COMMIT it (a `wip:` commit is fine): durable, visible in history, and rebase/merge-safe.
Do not add tooling that runs `git stash` (no "autosaver"). If you find a stash, reconstruct
it into a commit and drop it; do not leave work hidden in the stash list. Past stash misuse
also leaves ORPHANED stash commits that no longer show in `git stash list` but linger as
unreachable objects (`git fsck --unreachable | grep commit`, subjects like `WIP on ...`,
`index on ...`, `On <branch>: ...`). Treat these the same way: before resurrecting one, run
the semantic-containment check below — most are snapshots of work that already landed. Only
reconstruct+merge the genuinely-unmerged remainder; drop the rest (let git gc reclaim them),
do NOT re-merge already-landed work (that creates duplicates).

MERGE CONCEPTUALLY — never merely pick a side. When reconciling divergent work (branches,
worktrees, reconstructed stashes) into `main`, integrate the IDEAS: keep every distinct
concept from both sides. Resolve a conflict wholesale to one side ONLY after verifying that
side is a strict superset (the other is an older snapshot, a pure reformat, or already
contained) — and STATE that justification in the commit message. If both sides carry a real,
different idea (e.g. two implementations of one feature, or one has a test/doc/insight the
other lacks), COMBINE them — take the spec-faithful implementation AND the other's test and
documentation. A blind "take theirs / take ours" without that superset check is a bug, not a
merge.

CHECK CONTAINMENT BY CONCEPT, NOT BY TEXT. Before merging any branch/worktree/reconstructed
stash, verify whether its IDEAS are already in `main` — semantically, not by literal diff or
symbol name. A textual diff or reverse-apply (`git apply -R --check`) is unreliable here:
context drift on an old base makes already-present code look "unmerged," and a later RENAME or
REFACTOR in `main` hides an already-landed feature (e.g. `quick_forward_release_bonus` was
merged and then evolved into `quick_forward_release_bonus_value` + `_opportunity_reward`; a
grep for the old names falsely reports it MISSING). So: read the actual added code, find the
concept in `main` by behavior (constants/gates/env knob/call site/test), and only merge the
part that is truly absent. If the whole thing is already present in evolved form, merge NOTHING
and drop the source — re-adding it would duplicate a better implementation.

x is sideline-to-sideline (width) dimension, y is goal-to-goal (length) dimension
MDP = markov decision process
MPC = model predictive control
POMDP = partially observable markov decision process
LP = linear program (continuous variables, linear objective + constraints)
IP / IPM = interior-point / interior-point METHOD — the algorithm that solves the
  formation LP. IP here is NOT integer programming: there are NO integer/binary
  decision variables and NO MILP/branch-and-bound anywhere in the engine. The only
  mathematical program is the continuous formation LP, solved by Clarabel's IPM.
Clarabel = the Rust conic/LP solver used for the formation LP (`solve_lp_clarabel`),
  a sparse interior-point method; an internal simplex is the deterministic fallback
  (`SOCCER_FORMATION_LP_DETERMINISTIC`). It is the ONLY optimization solver in the
  engine — distance/cover checks, threshold gates, and "budgets" elsewhere are plain
  geometry, not LP/IP, so do not describe them as IP/IPM.

## Control architecture: who decides what

MPC is an INDIVIDUAL-player tool only. There is deliberately NO team/collective MPC
(no joint multi-player optimizer). New MPC work must control one actor's own
trajectory — the controlled player is the only point-mass state — never optimize
several players' motion jointly. (Audit Jun 2026; the reserved `tier1_team_enabled`
team-MPC flag was removed as it encoded this ruled-out design.)

Division of labor:
- TEAM FORMATION = LP / IPM. The *allocation* problem (which player -> which slot,
  optimal anchors, per tick). `SoccerFormationLpBrain::solve_tick` -> sparse
  interior-point (Clarabel, `solve_lp_clarabel`), with an optional internal simplex.
  Formation is static allocation + geometry (no time-dynamics), so LP/IPM is the
  right tool, NOT MPC.
- REPRESENTATION / RETRIEVAL = vectors. Config-vector similarity / embeddings,
  pitch value. Not a controller.
- INDIVIDUAL EXECUTION = MPC. Given one player's LP-assigned slot/target, plan THAT
  player's acceleration over a horizon to reach it under speed/accel limits, bending
  around moving bodies. The LP gives the destination point; MPC gives the
  dynamically-feasible path in time to it.

Rule of thumb: LP/IPM decides the shape; per-player MPC executes the move to it. Do
NOT add team-level MPC — it is redundant with the LP, far costlier (joint-state
blowup), and explicitly ruled out.

Every `PlanarPointMassMpc` solve is single-actor (all other players + the ball enter
only as `PlanarObstacle` soft keep-outs, never co-optimized state):
- tier2 per-player QP: `WorldSnapshot::mpc_desired_velocity` (warm-started cache
  `HashMap<usize, PlanarPointMassMpc>` keyed by player id)
- formation-local refinement of LP guidance: `soccer_local_mpc_refined_guidance`
  (one player; `apply_local_mpc_tick` loops players, each solved independently)
- pass execution (off by default): `mpc_predicted_receiver_path`
- keeper positioning

MPC defaults: base `MatchConfig` OFF (deterministic tests/learning); live-gameplay
preset ON (tier2 + field-aware + reconcile + latent + formation-local). Master live
toggle: `SOCCER_LIVE_MPC`.

## TRAINING DEPLOY — both clusters (Hetzner + AWS EC2 k8s). This WORKS; don't claim AWS is blocked.

Learners build from **ORESoftware/upstream** (NOT akrion/origin): deploy
`dd-soccer-learning-rds-continuous` clones `SOCCER_SOURCE_REPO`
(github.com/ORESoftware/soccer-sim-game-engine.rs) at `SOCCER_SOURCE_REF=learning`
+ the DES repo as a sibling, `cargo build --release`, then loops `main_soccer_learning_run`
writing policy gens to RDS experiment `soccer-self-play-k8s-overnight` (what the live :5055
reads). So: push your work to `upstream learning` (and `upstream main`), then restart the
learner pods. `upstream/learning` is usually a strict ancestor of `main` ⇒ `git push upstream
HEAD:learning` is a clean fast-forward (verify with `git rev-list --left-right --count
upstream/learning...HEAD`). My default-on gates need no all-gates-on.sh edit, but add them
there anyway for an unambiguous lineage. (Engine changes that DON'T touch FEATURE_DIM are safe
to CONTINUE training from the current gen.)

Hetzner (use the ~/.ssh alias): `ssh hetzner-k8s-bastion` runs `kubectl` (k3s; transient
`TLS handshake timeout` to one API VIP — just retry). `kubectl scale deploy
dd-soccer-learning-rds-continuous --replicas=1` then `kubectl rollout restart deploy/...`.
If Pending with `Insufficient memory` (requests, not usage), lower it:
`kubectl set resources deploy/dd-soccer-learning-rds-continuous --requests=memory=8Gi
--limits=memory=60Gi`, and `kubectl scale deploy dd-soccer-learning-all-gates-on --replicas=0`
(that one chronically CrashLoopBackOffs and reserves 16Gi). Confirm code: `kubectl logs <pod> |
grep source_commit_actual` (must equal your pushed SHA).

AWS EC2 k8s — the API (https://98.90.186.114:6443, context `dd-ec2-runtime`) is firewalled
from the laptop, BUT it IS reachable via an **SSM port-forward** (aws CLI = user `my-cli-user`,
acct 710156900967, us-east-1; the lone SSM-Online instance is `i-0cc2461a55d491af6`). The only
missing piece is `session-manager-plugin`; install it USER-SPACE (no sudo):
```
ARCH=$(uname -m)  # arm64 on this Mac
curl -so smb.zip https://s3.amazonaws.com/session-manager-downloads/plugin/latest/mac_arm64/sessionmanager-bundle.zip
unzip -oq smb.zip && xattr -dr com.apple.quarantine sessionmanager-bundle/bin/session-manager-plugin
export PATH="$PWD/sessionmanager-bundle/bin:$PATH"
nohup aws ssm start-session --target i-0cc2461a55d491af6 \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["6443"],"localPortNumber":["16443"]}' >/tmp/ssm_pf.log 2>&1 &
kubectl --context dd-ec2-runtime --server https://localhost:16443 --insecure-skip-tls-verify get deploy -A
```
Then it's normal kubectl (same `dd-soccer-learning-rds-continuous` learner, same `learning` ref,
same RDS experiment; a commit-watcher there auto-redeploys on new pushes). `aws ssm send-command`
does NOT work (no AWS-RunShellCommand doc on this acct) — use the port-forward, not send-command.

## Command safety — STRICT (all agents MUST follow)

Never run destructive or irreversible shell commands. To remove or move files,
**always go through git** so the change is tracked and recoverable.

**Blacklisted — do NOT run:**
- `rm`, `rm -rf`, `rmdir`, `unlink` — never delete via raw `rm`.
- raw `mv` of tracked files; truncating a tracked file with `>`.
- `git reset --hard`, `git clean -fdx`, `git checkout -- .` / `git restore .` mass-discard.
- `git push --force` / history rewrites on shared branches (esp. `main`).
- `git stash` — banned outright (see "NEVER use `git stash`" above); it hides work and races.
- `dd`, `mkfs`, `shred`, `find … -delete`, recursive `chmod -R`/`chown -R` on broad paths, fork bombs.

**Whitelisted — safe, prefer these:**
- `git rm` / `git rm --cached` — remove files through git (recoverable via history).
- `git mv` — rename/move through git.
- `git restore <path>` (single file), `git revert` — reversible. (NOT `git stash`.)
- Editing via the editor tools, `git add`, `git commit`.
  (Do NOT `git switch -c` / create branches — this repo is main-only; see the top of this file.)

If a genuinely destructive action seems unavoidable, **STOP and ask the operator
first** — do not improvise around this rule.
