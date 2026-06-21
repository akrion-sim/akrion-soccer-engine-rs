# Soccer continuous-learning + tournament runbook

How overnight/continuous learning works, what fires on a push, and the manual levers.
**TL;DR: to train on new code, just `git push origin learning` (and `main`). Everything else is
automatic within ~120s** — both clusters rebuild the learner from source and kick off tournaments.

## The push → learn → tournament loop (automatic)

`dd-soccer-commit-watcher` (a Deployment running on **both** AWS and Hetzner) polls
`origin/learning` every `WATCH_POLL_SECONDS` (120s). On a new commit it:

1. `kubectl rollout restart deployment/dd-soccer-learning-rds-continuous` — the learner
   re-clones + `cargo build`s the branch from source (~4–5 min cold build) and resumes self-play.
2. Launches a **push-tournament** Job from the nightly cronjob template with `SOURCE_REF=learning`
   (24 teams, group size 3, 1 advancer, `THREADS=2`, 10-min matches), labelled
   `dd-soccer/push-tournament=true`, capped at `MAX_PARALLEL_PUSH_TOURNAMENTS=5` per cluster.
3. Records the SHA in configmap `dd-soccer-commit-watcher-state` (`.data.lastSha`).

So **every learning push redeploys the learner AND runs fresh tournaments on both clusters** — no
manual step. Verify: `kubectl get cm dd-soccer-commit-watcher-state -o jsonpath='{.data.lastSha}'`
should equal `git rev-parse origin/learning`.

## Components

| Thing | AWS (default ns) | Hetzner (dd-k8s) | Role |
|---|---|---|---|
| `dd-soccer-learning-rds-continuous` (Deploy) | **running**, `SOCCER_PARALLEL_GAMES=1` (PINNED — see below) | **scaled 0** (32 GB nodes too small; learner peaks ~70 GB) | continuous self-play; writes every game to RDS; promotes gen-N to slug `soccer-self-play-k8s-overnight` |
| `dd-soccer-commit-watcher` (Deploy) | running | running | the push→learn→tournament loop above |
| `dd-soccer-tournament-nightly` (CronJob) | `0 2 * * *` America/Chicago (128-team full run) | **`*/5 * * * *`** — continuous tournament **farm** (learner is off here) | champions + top-16 team-brains → GA elite pool |

AWS is the single learner (123 GB / 14 vCPU node). Hetzner is a tournament farm cranking a bracket
every ~12 min plus per-push tournaments. **Net: ≥3 parallel games at all times** = the AWS learner (1)
+ the always-running Hetzner tournament (`THREADS` matches) + any AWS tournament. Parallelism beyond
the single self-play game comes from TOURNAMENTS, not the learner.

## Parallelism / "maximize learning" knobs

- **The continuous learner is PINNED to `SOCCER_PARALLEL_GAMES=1`** — the RDS-continuous gradient path
  is serial and the binary hard-fails any other value (`continuous_config_invalid ... expected=1` →
  CrashLoopBackOff). Do NOT change it. (Parallel self-play would need the queue learner
  `main_soccer_learning_queue` with `SOCCER_QUEUE_PARALLEL_GAMES`, a separate deployment — not wired.)
- Parallel games therefore scale via TOURNAMENTS:
  - Push-tournament size/parallelism: env on the `dd-soccer-commit-watcher` Deployment
    (`PUSH_TOURNAMENT_TEAMS/THREADS/MATCH_SECONDS/...`, `MAX_PARALLEL_PUSH_TOURNAMENTS`) — `THREADS`
    is the number of concurrent matches per tournament.
  - Hetzner farm cadence: the `*/5` schedule on its `dd-soccer-tournament-nightly` cronjob; raise its
    `SOCCER_TOURNAMENT_THREADS` (and the matching cpu limit) for more concurrent matches.
  - AWS has ~12 idle vCPU between push-tournaments (learner uses ~1–2); a continuous AWS tournament
    farm (a `*/5` cronjob like Hetzner's, `THREADS=4`, ~22 Gi alongside the ~70 GB learner) would use
    them for more continuous parallel games — not yet deployed.

## Manual ops

Kick a tournament now (e.g. a full 128-team run, or extra throughput):
```
kubectl create job dd-soccer-tournament-manual-$(date +%s) --from=cronjob/dd-soccer-tournament-nightly \
  --dry-run=client -o json | jq '<override env: SOCCER_SOURCE_REF=learning,
  SOCCER_TOURNAMENT_ALLOW_NONCANONICAL_LOCK=true, unique SOCCER_TOURNAMENT_LOCK_KEY, TEAMS/THREADS/...>' \
  | kubectl create -f -
```
(`SOCCER_TOURNAMENT_ALLOW_NONCANONICAL_LOCK=true` is REQUIRED for a manual lock key, else the run
self-skips with `noncanonical_lock_key`.) See `/tmp/launch_t2.sh` for the exact jq.

Verify learning is flowing (RDS):
```
select pv.generation, pv.status, pv.created_at from des_soccer_learning_policy_versions pv
  join des_soccer_learning_experiments ex on ex.id=pv.experiment_id
  where ex.slug='soccer-self-play-k8s-overnight' order by pv.generation desc limit 3;
select id,status,champion_team_id,finished_at from des_soccer_tournaments order by id desc limit 5;
```

## Cluster access

- **AWS** via SSM: instance `i-0cc2461a55d491af6`, `sudo kubectl --kubeconfig=/etc/kubernetes/admin.conf`
  (helper: `/tmp/ssmrun.sh <script-file>`; `AWS_PROFILE=dd-codex AWS_REGION=us-east-1`).
- **Hetzner** via ssh: `ssh -i ~/.ssh/id_hetzner root@167.233.100.88` (fsn1 only),
  `export KUBECONFIG=/etc/kubernetes/admin.conf`.

## Gotchas

- RDS presents an Amazon CA the containers don't trust → every PG-touching job needs
  `SOCCER_PG_TLS_INSECURE=1` (encrypted, unauthenticated). Without it: dies at PG connect after 5 retries.
- Tournament build cache is a per-pod `emptyDir` (NOT a node hostPath) so it works on any node in a
  multi-node cluster → a cold clone+build (~4–5 min) every run; acceptable.
- Learner memory ceiling: ~70 GB steady-state at the 90 Gi limit on the 123 GB AWS node; a too-high
  `SOCCER_PARALLEL_GAMES` or memory limit risks OOM or starving co-scheduled tournaments.
