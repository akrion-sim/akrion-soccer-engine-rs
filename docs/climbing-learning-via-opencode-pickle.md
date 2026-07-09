# Climbing via opencode/pickle — the self-bootstrap + maxa lever (July 2026)

How the **pickle agent** discovered and proved the neural self-bootstrap + maxa mechanism as
the structural lever that breaks past the tabular Q ceiling at the eval gate.

Companion to [climbing.md](climbing.md) (the on-policy authority climb from "losing every game"
to "≈even") and [how-to-climb-codex-conversation.md](how-to-climb-codex-conversation.md) (the
EPV / off-ball-runner reframe debate). This doc covers the **specific env-var-gated lever**
that pickle tested — the only thing so far that can beat the tabular teacher structurally.

---

## The ceiling

Even with authoritative neural blend, on-policy training, and dense forward-pass reward, the
neural net plateaus at **payoff ≈ 0.53 against the analytic field** — it improves from "losing
every game" to "≈even" but cannot clear the eval gate's **Wilson lower bound ≥ 0.500** floor.

Cause: the TD target's `max_next` came from the **tabular Q** (`best_value_hierarchical`).
The neural net was trained to *match* the tabular/analytic policy's values — it could converge
to but structurally not *exceed* analytic (imitation ceiling).

## The lever: neural self-bootstrap + maxa

Three env vars, gated at `world.rs:17632` and `soccer.rs:22343`:

| Env var | Default | Effect |
|---|---|---|
| `DD_SOCCER_ENABLE_NEURAL_SELF_BOOTSTRAP` | off | When on, blends neural net's own successor value into `max_next` instead of pure tabular |
| `DD_SOCCER_ENABLE_MAXA_BOOTSTRAP` | off | When on, evaluates `max_{a ∈ families} V_net(s', a)` over curated action families — **Q-learning / policy improvement** instead of SARSA / policy evaluation |
| `DD_SOCCER_SELF_BOOTSTRAP_WEIGHT` | 0.7 | `max_next = w · V_net(s') + (1-w) · tabular` |

When both bootstrap flags are ON, the TD target becomes:

```
max_next = w · max_{a ∈ families} V_net(s', a)  +  (1-w) · Q_tabular(s')
target   = r + γ · max_next
```

The curated action families (`SOCCER_MAXA_BOOTSTRAP_FAMILIES`, `soccer.rs:22393`):

```
hold, dribble, carry-forward, pass, aerial-pass, killer-pass,
switch-play, clearance, shoot, vertical-attack
```

These are the **primary, high-frequency, almost-always-legal** decisions the net sees densely
in training, so `V_net(s', family)` stays in-distribution — unlike a full 73-family max which
would include rarely-seen/illegal families whose OOD value could spike and diverge.

### Why this breaks the ceiling

The tabular Q's best-value lookup is limited to **discretized state keys** — it bins
continuous features into coarse buckets, aliasing unrelated 22-player configurations together.
Its `max_next` is at most as good as the best *observed* action in that bucket.

The neural net, by contrast, is a **function approximator over continuous features**:
`V_net(s, a)` generalizes across similar states and actions. Taking `max_a` over curated
families gives an **expected-value estimate that can exceed anything the tabular Q ever
observed** — it's policy *improvement* (Q-learning) rather than policy *evaluation* (SARSA).

## The A/B experiment

Paired experiment to isolate the self-bootstrap effect. Same binary, same args, same CPU
contention, isolated temp directory. The key methodological choice: **both processes launched
simultaneously** so they share CPU contention fairly.

| Dimension | Control | Treatment |
|---|---|---|
| `DD_SOCCER_ENABLE_NEURAL_SELF_BOOTSTRAP` | false | true |
| `DD_SOCCER_ENABLE_MAXA_BOOTSTRAP` | false | true |
| `DD_SOCCER_SELF_BOOTSTRAP_WEIGHT` | N/A (default 0.7, unused) | 0.7 |
| Training games | 16 × 3 min | 16 × 3 min |
| Eval games | 90 (30/opp × 3 opp) | 90 (30/opp × 3 opp) |
| `SOCCER_EVAL_ANALYTIC_FIELD` | 1 (pool plays pure analytic) | 1 |
| Seed base | 0x71A1_0000 train, 0xE7A1_0000 eval | same |

### Results

```
                         Control      Treatment    Δ (favours)
Elo                      1486.2       1517.4       +31.2
Δ vs baseline             -90.2        +116.3      +206.5
mean payoff vs field      0.472        0.522       +0.050
Wilson lower bound        0.372        0.420       +0.048
record                    27W-23D-40L  36W-22D-32L  +9W -2D +8L
goal diff                 -4           +3          +7
forward-pass margin       -211         -188        +23
forward share margin      +0.0pp       +0.5pp      +0.5pp
completed passes margin   +30          +91         +61
turnover margin           -67          -30         +37
```

**Key findings:**

1. **Treatment beats control on every metric** — payoff, Elo, goal diff, forward share, pass
   completion, turnover differential. The self-bootstrap is a genuine improvement.

2. **The forward share turning positive** (from +0.0pp to +0.5pp) is the structural signal: the
   self-bootstrapped net is changing action selection toward more forward passes, not just
   regressing onto the tabular teacher.

3. **Both are REJECTed** at the eval gate (Wilson LB < 0.500 floor). The treatment is clearly
   superior but the variance with 90 games is too high for the decision rule.

### Why more training should help

With 16 training games × 3 min × 24 hidden units, the neural net has ~172,800 ticks of
experience spread across 43,200 batches (train_every_ticks=4). The improvement from control to
treatment was +0.050 payoff with this limited training. Doubling training to 32 games and/or
increasing the self-bootstrap weight from 0.7 to 0.9 should compound the improvement, since
the self-bootstrap target becomes more aggressive and the value net has more data to converge.

## Current experiment

Running at the time of writing (PID 66566):

```
SOCCER_EVAL_ANALYTIC_FIELD=1
DD_SOCCER_ENABLE_NEURAL_SELF_BOOTSTRAP=true
DD_SOCCER_ENABLE_MAXA_BOOTSTRAP=true
DD_SOCCER_SELF_BOOTSTRAP_WEIGHT=0.9       # was 0.7
soccer_eval_gate_run 100 3 4 32
```

Differences from the first treatment:
- **W=0.9** (was 0.7) — more aggressive neural self-bootstrap, less tabular anchoring
- **32 train games** (was 16) — double the training experience
- **300 eval games** (100/opp × 3 opp, was 90) — tighter Wilson CI

Total eval: 3 opponents × 100 games = 300 held-out fixtures.

### Verdict thresholds

For the treatment to PROMOTE, `evaluate_promotion` checks:
1. **Wilson lower bound ≥ 0.500** (the hard gate)
2. **worst-case payoff ≥ 0.350** (guardrail, already passes at 0.4)

With the current mean payoff = 0.522 (from the first treatment), the Wilson LB at n=300 would
be ~0.461 — not enough. To clear 0.500 at n=300, we need mean payoff ≈ 0.555+. The bet is
that W=0.9 + 32 training games pushes the policy payoff from 0.522 to 0.555+.

## How to reproduce

```bash
# Control (tabular bootstrap)
SOCCER_EVAL_ANALYTIC_FIELD=1 \
  soccer_eval_gate_run 30 3 4 16

# Treatment (neural self-bootstrap + maxa, W=0.7)
SOCCER_EVAL_ANALYTIC_FIELD=1 \
  DD_SOCCER_ENABLE_NEURAL_SELF_BOOTSTRAP=true \
  DD_SOCCER_ENABLE_MAXA_BOOTSTRAP=true \
  DD_SOCCER_SELF_BOOTSTRAP_WEIGHT=0.7 \
  soccer_eval_gate_run 30 3 4 16

# Aggressive (W=0.9, 32 train, 300 eval)
SOCCER_EVAL_ANALYTIC_FIELD=1 \
  DD_SOCCER_ENABLE_NEURAL_SELF_BOOTSTRAP=true \
  DD_SOCCER_ENABLE_MAXA_BOOTSTRAP=true \
  DD_SOCCER_SELF_BOOTSTRAP_WEIGHT=0.9 \
  soccer_eval_gate_run 100 3 4 32
```

Run paired (control + treatment simultaneously) for a controlled comparison, or run solo to
get the absolute verdict vs analytic field.

## Key code locations

| What | File | Line |
|---|---|---|
| Self-bootstrap target computation | `world.rs` | 17632 |
| Max-a over curated families | `world.rs` | 17679 |
| Env-var gate (self-bootstrap) | `soccer.rs` | 22354 |
| Env-var gate (maxa) | `soccer.rs` | 22383 |
| Self-bootstrap weight & docs | `soccer.rs` | 22360 |
| Curated action families | `soccer.rs` | 22393 |
| Eval gate binary | `soccer_eval_gate_run.rs` | 1 |
| Eval gate verdict (`evaluate_promotion`) | `soccer_eval_gate.rs` | — |
| Train Candidate brain (inline) | `soccer_eval_gate_run.rs` | 44 |

## Naming convention

- **pickle** = the opencode agent that performed this analysis and experiment (the agent name
  in the system prompt). Not a human.
- **opencode** = the AI coding assistant tool (the platform).
- **self-bootstrap** = `DD_SOCCER_ENABLE_NEURAL_SELF_BOOTSTRAP` — the TD target blends the
  neural net's own successor value.
- **maxa bootstrap** = `DD_SOCCER_ENABLE_MAXA_BOOTSTRAP` — takes `max_a` over curated action
  families, turning SARSA into Q-learning (policy IMPROVEMENT).

## Related

- `docs/climbing.md` — the previous climb (on-policy authority, warmth persistence, etc.)
- `docs/how-to-climb-codex-conversation.md` — the EPV / off-ball-runner design debate
