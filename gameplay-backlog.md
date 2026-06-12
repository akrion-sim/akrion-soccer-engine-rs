# Soccer gameplay backlog — batch 2026-06-12 (b)

Captured from a direct request. Each item: implement in engine + add an automated test +
clean commit. Several refine/supersede earlier requests (noted).

## Passing / decision quality (overlaps the killer/threaded-pass cluster currently being fixed)
1. **Penalize passes to nobody** — a pass not received by a teammate must be penalized; MDP/POMDP
   (+ neural) should select these less over time and learn the scenarios that cause them.
2. **Reward chains** — a pass received by a teammate that leads to a *subsequent* received pass OR
   a shot on the goalframe OR a goal should be rewarded more. **Reward scale: goal = 100 pts,
   shot on goal = 40 pts.** (Refines the earlier "reward forward passes" items with concrete numbers.)
3. **Stop distance shooting** — 25yd ok, 30yd really not, 40yd not at all — UNLESS the GK is badly
   out of position. (Refines the earlier "less reward for far shots / penalty off-target" item.)
4. **Final-third pass-to-nobody is exceptionally bad** — if attacking team in the final third plays
   the ball forward with NO teammate within 10yd of the ball trajectory, or no player ahead of the
   passer at all when the ball was played, that must not happen (hard penalty / illegal).
5. **Communicate + prefer short** — a carrier under LESS pressure should direct teammates within 20yd
   to open space (esp. wide/flanks); when "in communication" with 1–2 players, pass to one of them
   more often. Short passes (<20yd) preferred over longer ones more strongly.

## Positioning / shape / lane affinity
6. **Wingback lane affinity** — LWB/RWB drift out of position and sometimes SWITCH SIDES on corners /
   goal-kicks. They must keep to their lane (quadrant + tertile) and return to it — lane affinity.
7. **Less midfield/defense symmetry** — outside backs (wingbacks) may push forward more often than
   central defenders; CBs stay back and must NOT dribble forward if a defender is within 5yd ahead.
8. **Defenders off the goal-line** — a defender should virtually never go deeper than the 6yd line
   unless the ball is between the end-line and (their) end-line; more typically hold 8–10yd off the
   goal-line. Push defenders away from the goal-line overall (unless the ball is behind them).
9. **Track-back / man-mark urgency** — if the ball carrier is running/sprinting/making a run upfield
   into space, the nearest midfielder/striker should feel urgency to track (man-mark) that player for
   3–8 seconds even if it pulls them out of position.

## Body rotation / first touch / receiving
10. **Rotation soft-limits** — players rotate/turn too fast; add soft limits on how far they can rotate
    within ~5s. They can only pass/dribble the way they face, EXCEPT a backheel pass — allowed only in
    the opponent's half, never their own half.
11. **Attack the ball on reception** — on a floor pass, receivers wait for the ball to nearly stop
    instead of taking control earlier. Receiver should move toward / attack the ball to control it
    earlier — UNLESS letting it roll takes them away from pressure.
12. **First-touch into space** — audit/harden first touch: a player may take a longer first touch into
    space if it takes them away from the closest marking defender's movement/momentum/trajectory.

## Positioning / passing (cont.)
13. **Play-out-from-own-box under pressure** — defending team in possession inside their OWN 18yd
    box under pressure (1+ opponent within 3yd): 1–2yd passes are bad. Prefer passes that are
    lateral and/or 3+ yards, and **to the flanks is best** (get the ball wide out of the box).
    **Rarely play across the goal** (square ball across the own-goal face) unless it's from one
    side of the box to the other AND highly uncontested. *(Implemented: own-box play-out pass
    scoring — short-pass penalty, flank reward, across-goal penalty waived when the lane is clear.)*

14. **Wingers get open on the flanks (CRITICAL)** — wide attackers must run/sprint to open space
    on the wings far more; off-ball flank runs into space when their team has possession. And the
    engine should **prioritize playing the ball forward to the most-open player, almost always**
    (strong forward × openness bias in pass selection). *(In progress.)*

15. **Don't crowd the ball carrier** — off-ball teammates must not run directly at a team-mate
    who is dribbling. General 3yd-outside / 2yd-inside teammate padding is the existing SOFT
    barrier (TEAMMATE_SPACING clocks). The HARD rule on top: when the carrier is **unpressured**,
    no teammate may run at them and get within **2yd** (outside the 18yd box) — ever.
    *(Implemented: dribble_carrier_standoff_adjusted_target, hard 2yd floor when carrier is
    unpressured + outside box; chained into discipline_intent_against_bunchball.)*

## Sequencing note
These land AFTER the test-suite green-up (foundation that lets each be verified), then are
implemented one-at-a-time with tests + commits alongside the earlier ~50-item pass. Items 1–5
share machinery with the killer/threaded-pass cluster being fixed now.
