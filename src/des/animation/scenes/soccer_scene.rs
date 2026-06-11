//! Pitch + bench animation for the soccer rotation showcase.
//!
//! Renders a vertical pitch with the on-field players laid out in a real
//! **formation** (rows from the keeper up to the attack), a bench column of the
//! resting players, a scoreboard/affinity panel, and a substitution log. Each
//! player is a numbered jersey with their `PlayerN` name; at every period
//! boundary the players who come **on** glide from the bench onto their slot
//! (highlighted green) and the players who come **off** glide to the bench
//! (highlighted red) — the substitution motion is produced by the caller
//! feeding a handful of interpolation frames (`transition` 0→1) before the
//! settled period frames.
//!
//! PORT NOTE: only the subset of `crate::des::general::soccer_rotation::
//! SoccerProblem` the scene reads is mirrored locally in [`SoccerProblem`],
//! plus the `formation` row sizes used for the pitch layout.

#![allow(dead_code)]

use crate::des::animation::types::{
    js_num, to_fixed, Anchor, ChartSeries, ChartSpec, CircleShape, FontWeight, FrameParts,
    LineShape, RectShape, Shape, TextShape,
};

pub const STAGE_W: f64 = 1200.0;
pub const STAGE_H: f64 = 760.0;

// Left column: scoreboard / affinity (top) + substitution log (bottom).
const META_X: f64 = 24.0;
const META_Y: f64 = 56.0;
const META_W: f64 = 250.0;
const META_H: f64 = 300.0;

const SUB_X: f64 = 24.0;
const SUB_Y: f64 = 372.0;
const SUB_W: f64 = 250.0;
const SUB_H: f64 = 300.0;

// Centre: the pitch (vertical, our goal at the bottom).
const PITCH_X: f64 = 300.0;
const PITCH_Y: f64 = 56.0;
const PITCH_W: f64 = 540.0;
const PITCH_H: f64 = 616.0;

// Right: the bench (resting players).
const BENCH_X: f64 = 864.0;
const BENCH_Y: f64 = 56.0;
const BENCH_W: f64 = 312.0;
const BENCH_H: f64 = 616.0;

const COLOR_PITCH: &str = "#15803d"; // grass
const COLOR_PITCH_STRIPE: &str = "#16a34a";
const COLOR_LINE: &str = "#e8fff0";
const COLOR_PLAYER_FIELD: &str = "#2563eb";
const COLOR_PLAYER_GK: &str = "#f59e0b";
const COLOR_PLAYER_BENCH: &str = "#64748b";
const COLOR_IN: &str = "#22c55e"; // subbed on
const COLOR_OUT: &str = "#ef4444"; // subbed off
const COLOR_GOAL_US: &str = "#facc15";
const COLOR_GOAL_THEM: &str = "#dc2626";

// PORT NOTE: local mirror of the soccer model (subset used by the scene).
#[derive(Clone, Debug, Default)]
pub struct SoccerProblem {
    pub num_positions: usize,
    pub num_periods: f64,
    /// Full formation row sizes, GK row first (`[1, 3, 2, 1]`); sums to
    /// `num_positions`. Empty falls back to a single centred line.
    pub formation: Vec<usize>,
    pub player_names: Option<Vec<String>>,
    pub position_names: Option<Vec<String>>,
}

/// `'us' | 'them'` goal flash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoalSide {
    Us,
    Them,
}

pub struct SoccerFrameInput<'a> {
    /// Game minute (integer).
    pub t: f64,
    /// Period index (0-based).
    pub period: f64,
    /// Total game minutes (for the minute display).
    pub total_minutes: f64,
    /// `positions[pos] = playerId` for the current period.
    pub positions: Vec<usize>,
    /// `bench[i] = playerId` for the current period.
    pub bench: Vec<usize>,
    /// The previous arrangement, used as the start point for substitution
    /// motion. Equal to the current arrangement on settled frames.
    pub prev_positions: Vec<usize>,
    pub prev_bench: Vec<usize>,
    /// 0 → at the previous arrangement, 1 → settled at the current one.
    pub transition: f64,
    /// Player ids that came on at this period's boundary (highlight green).
    pub entered: Vec<usize>,
    /// Player ids that went off at this period's boundary (highlight red).
    pub left: Vec<usize>,
    /// `(out, in)` swap pairs at the most recent boundary, for the sub log.
    pub recent_subs: Vec<(usize, usize)>,
    pub goals_for: f64,
    pub goals_against: f64,
    /// Average affinity in [0, 1].
    pub affinity_now: f64,
    /// Optional flash for "this minute had a goal event".
    pub goal_this_tick: Option<GoalSide>,
    pub problem: &'a SoccerProblem,
}

fn player_name(names: &Option<Vec<String>>, player_id: usize) -> String {
    names
        .as_ref()
        .and_then(|n| n.get(player_id))
        .cloned()
        .unwrap_or_else(|| format!("Player{}", player_id + 1))
}

fn lerp(a: f64, b: f64, alpha: f64) -> f64 {
    a + (b - a) * alpha
}

/// Absolute pitch coordinate for position index `pos` under `formation`
/// (GK row first). GK sits near the bottom touchline, the attack near the top.
fn field_slot(formation: &[usize], pos: usize) -> (f64, f64) {
    if formation.is_empty() {
        return (PITCH_X + PITCH_W * 0.5, PITCH_Y + PITCH_H * 0.5);
    }
    let mut idx = pos;
    let mut row = 0usize;
    let mut col = 0usize;
    let mut row_len = formation[0];
    for (r, &n) in formation.iter().enumerate() {
        if idx < n {
            row = r;
            col = idx;
            row_len = n;
            break;
        }
        idx -= n;
    }
    let rows = formation.len();
    let yf = if rows <= 1 {
        0.5
    } else {
        0.90 - (row as f64 / (rows as f64 - 1.0)) * 0.80
    };
    let xf = (col as f64 + 1.0) / (row_len as f64 + 1.0);
    (PITCH_X + xf * PITCH_W, PITCH_Y + yf * PITCH_H)
}

/// Absolute coordinate for the `i`-th bench player when `count` are benched.
/// Up to six in one column, more spill into a second column.
fn bench_slot(i: usize, count: usize) -> (f64, f64) {
    let count = count.max(1);
    let cols = if count > 6 { 2 } else { 1 };
    let per_col = (count + cols - 1) / cols;
    let per_col = per_col.max(1);
    let col = i / per_col;
    let row = i % per_col;
    let cell_w = BENCH_W / cols as f64;
    let x = BENCH_X + cell_w * col as f64 + cell_w * 0.5 - 36.0;
    let top = BENCH_Y + 50.0;
    let avail = BENCH_H - 64.0;
    let step = avail / per_col as f64;
    let y = top + step * row as f64 + step * 0.5;
    (x, y)
}

/// `coords[playerId] = (x, y, is_on_field)` for an arrangement.
fn arrangement_coords(
    formation: &[usize],
    positions: &[usize],
    bench: &[usize],
) -> Vec<(f64, f64, bool)> {
    let total = positions.len() + bench.len();
    let mut out = vec![(PITCH_X + PITCH_W * 0.5, PITCH_Y + PITCH_H * 0.5, false); total];
    for (pos, &pid) in positions.iter().enumerate() {
        if pid < total {
            let (x, y) = field_slot(formation, pos);
            out[pid] = (x, y, true);
        }
    }
    let bn = bench.len();
    for (i, &pid) in bench.iter().enumerate() {
        if pid < total {
            let (x, y) = bench_slot(i, bn);
            out[pid] = (x, y, false);
        }
    }
    out
}

fn text(x: f64, y: f64, s: impl Into<String>, size: f64, fill: &str, anchor: Anchor) -> Shape {
    Shape::Text(TextShape {
        x,
        y,
        text: s.into(),
        font_size: Some(size),
        fill: Some(fill.to_string()),
        anchor: Some(anchor),
        ..Default::default()
    })
}

fn text_bold(x: f64, y: f64, s: impl Into<String>, size: f64, fill: &str, anchor: Anchor) -> Shape {
    Shape::Text(TextShape {
        x,
        y,
        text: s.into(),
        font_size: Some(size),
        fill: Some(fill.to_string()),
        anchor: Some(anchor),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    })
}

pub fn build_soccer_frame(_t: f64, _tick: f64, input: &SoccerFrameInput) -> FrameParts {
    let mut shapes: Vec<Shape> = Vec::new();
    let player_names = &input.problem.player_names;
    let position_names = &input.problem.position_names;
    let formation = if input.problem.formation.is_empty() {
        vec![input.problem.num_positions]
    } else {
        input.problem.formation.clone()
    };

    // ── Left panel: scoreboard / period / affinity. ─────────────────────────
    shapes.push(Shape::Rect(RectShape {
        x: META_X,
        y: META_Y,
        w: META_W,
        h: META_H,
        fill: "#0f172a".to_string(),
        stroke: Some("#334155".to_string()),
        stroke_width: Some(1.0),
        rx: Some(8.0),
        ..Default::default()
    }));
    shapes.push(text_bold(
        META_X + META_W / 2.0,
        META_Y + 34.0,
        format!("{}-a-side", input.problem.num_positions),
        22.0,
        "#f1f5f9",
        Anchor::Middle,
    ));
    shapes.push(text(
        META_X + META_W / 2.0,
        META_Y + 62.0,
        format!(
            "Period {} / {}",
            js_num(input.period + 1.0),
            js_num(input.problem.num_periods)
        ),
        15.0,
        "#cbd5e1",
        Anchor::Middle,
    ));
    shapes.push(text(
        META_X + META_W / 2.0,
        META_Y + 84.0,
        format!(
            "Minute {} / {}",
            js_num(input.t),
            js_num(input.total_minutes)
        ),
        13.0,
        "#94a3b8",
        Anchor::Middle,
    ));
    // Score.
    shapes.push(text_bold(
        META_X + META_W / 2.0,
        META_Y + 134.0,
        format!(
            "Us  {}  \u{2014}  {}  Them",
            js_num(input.goals_for),
            js_num(input.goals_against)
        ),
        24.0,
        "#fde68a",
        Anchor::Middle,
    ));
    if input.goal_this_tick == Some(GoalSide::Us) {
        shapes.push(Shape::Rect(RectShape {
            x: META_X + 40.0,
            y: META_Y + 150.0,
            w: META_W - 80.0,
            h: 28.0,
            fill: COLOR_GOAL_US.to_string(),
            opacity: Some(0.95),
            rx: Some(6.0),
            ..Default::default()
        }));
        shapes.push(text_bold(
            META_X + META_W / 2.0,
            META_Y + 170.0,
            "GOAL!",
            17.0,
            "#1f2937",
            Anchor::Middle,
        ));
    } else if input.goal_this_tick == Some(GoalSide::Them) {
        shapes.push(Shape::Rect(RectShape {
            x: META_X + 40.0,
            y: META_Y + 150.0,
            w: META_W - 80.0,
            h: 28.0,
            fill: COLOR_GOAL_THEM.to_string(),
            opacity: Some(0.95),
            rx: Some(6.0),
            ..Default::default()
        }));
        shapes.push(text_bold(
            META_X + META_W / 2.0,
            META_Y + 170.0,
            "CONCEDED",
            15.0,
            "#fff",
            Anchor::Middle,
        ));
    }
    // Affinity bar.
    shapes.push(text(
        META_X + 16.0,
        META_Y + 214.0,
        "On-field avg affinity",
        12.0,
        "#94a3b8",
        Anchor::Start,
    ));
    shapes.push(Shape::Rect(RectShape {
        x: META_X + 16.0,
        y: META_Y + 224.0,
        w: META_W - 32.0,
        h: 16.0,
        fill: "#334155".to_string(),
        stroke: Some("#475569".to_string()),
        stroke_width: Some(1.0),
        rx: Some(3.0),
        ..Default::default()
    }));
    shapes.push(Shape::Rect(RectShape {
        x: META_X + 16.0,
        y: META_Y + 224.0,
        w: (META_W - 32.0) * input.affinity_now.clamp(0.0, 1.0),
        h: 16.0,
        fill: "#22d3ee".to_string(),
        rx: Some(3.0),
        ..Default::default()
    }));
    shapes.push(text(
        META_X + META_W - 16.0,
        META_Y + 262.0,
        format!("{}%", to_fixed(input.affinity_now * 100.0, 0)),
        12.0,
        "#cbd5e1",
        Anchor::End,
    ));

    // ── Left panel (bottom): substitution log. ──────────────────────────────
    shapes.push(Shape::Rect(RectShape {
        x: SUB_X,
        y: SUB_Y,
        w: SUB_W,
        h: SUB_H,
        fill: "#0f172a".to_string(),
        stroke: Some("#334155".to_string()),
        stroke_width: Some(1.0),
        rx: Some(8.0),
        ..Default::default()
    }));
    shapes.push(text_bold(
        SUB_X + SUB_W / 2.0,
        SUB_Y + 28.0,
        "SUBSTITUTIONS",
        14.0,
        "#fde68a",
        Anchor::Middle,
    ));
    shapes.push(text(
        SUB_X + SUB_W / 2.0,
        SUB_Y + 48.0,
        format!("period {}", js_num(input.period + 1.0)),
        12.0,
        "#94a3b8",
        Anchor::Middle,
    ));
    if input.recent_subs.is_empty() {
        shapes.push(text(
            SUB_X + SUB_W / 2.0,
            SUB_Y + 96.0,
            "kick-off lineup",
            13.0,
            "#cbd5e1",
            Anchor::Middle,
        ));
    } else {
        for (i, &(out_p, in_p)) in input.recent_subs.iter().take(8).enumerate() {
            let y = SUB_Y + 78.0 + i as f64 * 26.0;
            shapes.push(text(
                SUB_X + 16.0,
                y,
                player_name(player_names, out_p),
                12.0,
                COLOR_OUT,
                Anchor::Start,
            ));
            shapes.push(text(
                SUB_X + SUB_W / 2.0 + 6.0,
                y,
                "\u{2192}",
                12.0,
                "#cbd5e1",
                Anchor::Middle,
            ));
            shapes.push(text(
                SUB_X + SUB_W - 16.0,
                y,
                player_name(player_names, in_p),
                12.0,
                COLOR_IN,
                Anchor::End,
            ));
        }
    }

    // ── Pitch. ───────────────────────────────────────────────────────────────
    shapes.push(Shape::Rect(RectShape {
        x: PITCH_X,
        y: PITCH_Y,
        w: PITCH_W,
        h: PITCH_H,
        fill: COLOR_PITCH.to_string(),
        stroke: Some(COLOR_LINE.to_string()),
        stroke_width: Some(2.0),
        ..Default::default()
    }));
    // Mowing stripes.
    let stripes = 8;
    for s in 0..stripes {
        if s % 2 == 1 {
            shapes.push(Shape::Rect(RectShape {
                x: PITCH_X,
                y: PITCH_Y + PITCH_H / stripes as f64 * s as f64,
                w: PITCH_W,
                h: PITCH_H / stripes as f64,
                fill: COLOR_PITCH_STRIPE.to_string(),
                opacity: Some(0.55),
                ..Default::default()
            }));
        }
    }
    // Halfway line + centre circle.
    shapes.push(Shape::Line(LineShape {
        x1: PITCH_X,
        y1: PITCH_Y + PITCH_H / 2.0,
        x2: PITCH_X + PITCH_W,
        y2: PITCH_Y + PITCH_H / 2.0,
        stroke: COLOR_LINE.to_string(),
        stroke_width: Some(2.0),
        ..Default::default()
    }));
    shapes.push(Shape::Circle(CircleShape {
        x: PITCH_X + PITCH_W / 2.0,
        y: PITCH_Y + PITCH_H / 2.0,
        r: 54.0,
        fill: "none".to_string(),
        stroke: Some(COLOR_LINE.to_string()),
        stroke_width: Some(2.0),
        ..Default::default()
    }));
    // Penalty boxes + goals (top = opponent, bottom = ours).
    for (by, gy) in [
        (PITCH_Y, PITCH_Y - 8.0),
        (PITCH_Y + PITCH_H - 80.0, PITCH_Y + PITCH_H),
    ] {
        shapes.push(Shape::Rect(RectShape {
            x: PITCH_X + PITCH_W / 2.0 - 100.0,
            y: by,
            w: 200.0,
            h: 80.0,
            fill: "none".to_string(),
            stroke: Some(COLOR_LINE.to_string()),
            stroke_width: Some(2.0),
            ..Default::default()
        }));
        shapes.push(Shape::Rect(RectShape {
            x: PITCH_X + PITCH_W / 2.0 - 34.0,
            y: gy,
            w: 68.0,
            h: 8.0,
            fill: COLOR_LINE.to_string(),
            ..Default::default()
        }));
    }

    // ── Bench zone. ───────────────────────────────────────────────────────────
    shapes.push(Shape::Rect(RectShape {
        x: BENCH_X,
        y: BENCH_Y,
        w: BENCH_W,
        h: BENCH_H,
        fill: "#0f172a".to_string(),
        stroke: Some("#334155".to_string()),
        stroke_width: Some(1.0),
        rx: Some(8.0),
        ..Default::default()
    }));
    shapes.push(text_bold(
        BENCH_X + BENCH_W / 2.0,
        BENCH_Y + 28.0,
        format!("BENCH  ({} resting)", input.bench.len()),
        14.0,
        "#fde68a",
        Anchor::Middle,
    ));

    // ── Players (one circle each, interpolated between arrangements). ─────────
    let alpha = input.transition.clamp(0.0, 1.0);
    let cur = arrangement_coords(&formation, &input.positions, &input.bench);
    let prev = if alpha >= 1.0 {
        cur.clone()
    } else {
        arrangement_coords(&formation, &input.prev_positions, &input.prev_bench)
    };
    let gk_player = input.positions.first().copied();
    let total = cur.len();

    for pid in 0..total {
        let (cx, cy, on_field) = cur[pid];
        let (px, py, _) = prev.get(pid).copied().unwrap_or((cx, cy, on_field));
        let x = lerp(px, cx, alpha);
        let y = lerp(py, cy, alpha);
        let is_entering = input.entered.contains(&pid);
        let is_leaving = input.left.contains(&pid);

        let (base_fill, r) = if on_field {
            if gk_player == Some(pid) {
                (COLOR_PLAYER_GK, 21.0)
            } else {
                (COLOR_PLAYER_FIELD, 21.0)
            }
        } else {
            (COLOR_PLAYER_BENCH, 16.0)
        };
        let (stroke, sw) = if is_entering {
            (COLOR_IN, 4.0)
        } else if is_leaving {
            (COLOR_OUT, 4.0)
        } else {
            ("#ffffff", 2.0)
        };

        shapes.push(Shape::Circle(CircleShape {
            x,
            y,
            r,
            fill: base_fill.to_string(),
            stroke: Some(stroke.to_string()),
            stroke_width: Some(sw),
            title: Some(player_name(player_names, pid)),
            ..Default::default()
        }));
        // Jersey number.
        shapes.push(text_bold(
            x,
            y + 5.0,
            format!("{}", pid + 1),
            if on_field { 15.0 } else { 12.0 },
            "#ffffff",
            Anchor::Middle,
        ));

        if on_field {
            // Position label above, name below.
            let pos_idx = input.positions.iter().position(|&q| q == pid);
            if let Some(pi) = pos_idx {
                let label = position_names
                    .as_ref()
                    .and_then(|p| p.get(pi))
                    .cloned()
                    .unwrap_or_else(|| pi.to_string());
                shapes.push(text_bold(
                    x,
                    y - r - 6.0,
                    label,
                    11.0,
                    "#fde68a",
                    Anchor::Middle,
                ));
            }
            shapes.push(text_bold(
                x,
                y + r + 14.0,
                player_name(player_names, pid),
                10.0,
                "#f8fafc",
                Anchor::Middle,
            ));
            if is_entering {
                shapes.push(text_bold(
                    x + r + 4.0,
                    y - r,
                    "\u{25B2} IN",
                    10.0,
                    COLOR_IN,
                    Anchor::Start,
                ));
            }
        } else {
            // Name to the right of the bench jersey.
            shapes.push(text(
                x + r + 8.0,
                y - 2.0,
                player_name(player_names, pid),
                12.0,
                "#e2e8f0",
                Anchor::Start,
            ));
            if is_leaving {
                shapes.push(text_bold(
                    x + r + 8.0,
                    y + 13.0,
                    "OUT",
                    10.0,
                    COLOR_OUT,
                    Anchor::Start,
                ));
            }
        }
    }

    let caption = format!(
        "t={}min  P{}  Us {}-{} Them  affinity {}%",
        js_num(input.t),
        js_num(input.period + 1.0),
        js_num(input.goals_for),
        js_num(input.goals_against),
        to_fixed(input.affinity_now * 100.0, 0)
    );
    FrameParts::with_caption(shapes, caption)
}

/// Build companion charts: per-tick affinity and cumulative goal differential,
/// laid out along the bottom of the stage.
pub fn build_soccer_charts(
    ts: &[f64],
    affinity: &[f64],
    goals_for: &[f64],
    goals_against: &[f64],
) -> Vec<ChartSpec> {
    let chart_y = 688.0;
    let chart_h = 60.0;
    vec![
        ChartSpec {
            x: META_X,
            y: chart_y,
            w: 560.0,
            h: chart_h,
            title: Some("On-field avg affinity".to_string()),
            y_min: Some(0.0),
            y_max: Some(1.0),
            series: vec![ChartSeries {
                label: "affinity".to_string(),
                color: "#22d3ee".to_string(),
                t: ts.to_vec(),
                y: affinity.to_vec(),
            }],
            ..Default::default()
        },
        ChartSpec {
            x: 616.0,
            y: chart_y,
            w: 560.0,
            h: chart_h,
            title: Some("Cumulative goals".to_string()),
            series: vec![
                ChartSeries {
                    label: "us".to_string(),
                    color: COLOR_GOAL_US.to_string(),
                    t: ts.to_vec(),
                    y: goals_for.to_vec(),
                },
                ChartSeries {
                    label: "them".to_string(),
                    color: COLOR_GOAL_THEM.to_string(),
                    t: ts.to_vec(),
                    y: goals_against.to_vec(),
                },
            ],
            ..Default::default()
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem() -> SoccerProblem {
        SoccerProblem {
            num_positions: 7,
            num_periods: 4.0,
            formation: vec![1, 2, 3, 1],
            player_names: Some((0..13).map(|i| format!("Player{}", i + 1)).collect()),
            position_names: Some(
                ["GK", "LB", "RB", "LM", "CM", "RM", "ST"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
        }
    }

    #[test]
    fn frame_caption_shows_score_and_affinity() {
        let input = SoccerFrameInput {
            t: 20.0,
            period: 0.0,
            total_minutes: 40.0,
            positions: vec![0, 1, 2, 3, 4, 5, 6],
            bench: vec![7, 8, 9, 10, 11, 12],
            prev_positions: vec![0, 1, 2, 3, 4, 5, 6],
            prev_bench: vec![7, 8, 9, 10, 11, 12],
            transition: 1.0,
            entered: vec![],
            left: vec![],
            recent_subs: vec![],
            goals_for: 2.0,
            goals_against: 1.0,
            affinity_now: 0.75,
            goal_this_tick: Some(GoalSide::Us),
            problem: &problem(),
        };
        let fp = build_soccer_frame(0.0, 0.0, &input);
        assert_eq!(
            fp.caption.as_deref(),
            Some("t=20min  P1  Us 2-1 Them  affinity 75%")
        );
        assert!(fp
            .shapes
            .iter()
            .any(|s| matches!(s, Shape::Text(t) if t.text == "GOAL!")));
        // Every player (on field + bench) is drawn with their name.
        assert!(fp
            .shapes
            .iter()
            .any(|s| matches!(s, Shape::Text(t) if t.text == "Player1")));
        assert!(fp
            .shapes
            .iter()
            .any(|s| matches!(s, Shape::Text(t) if t.text == "Player13")));
    }

    #[test]
    fn entering_player_is_highlighted_green() {
        let p = problem();
        let input = SoccerFrameInput {
            t: 10.0,
            period: 1.0,
            total_minutes: 40.0,
            positions: vec![0, 7, 2, 3, 4, 5, 6],
            bench: vec![1, 8, 9, 10, 11, 12],
            prev_positions: vec![0, 1, 2, 3, 4, 5, 6],
            prev_bench: vec![7, 8, 9, 10, 11, 12],
            transition: 0.5,
            entered: vec![7],
            left: vec![1],
            recent_subs: vec![(1, 7)],
            goals_for: 0.0,
            goals_against: 0.0,
            affinity_now: 0.5,
            goal_this_tick: None,
            problem: &p,
        };
        let fp = build_soccer_frame(0.0, 0.0, &input);
        // The incoming player gets a green-stroked jersey.
        assert!(fp
            .shapes
            .iter()
            .any(|s| matches!(s, Shape::Circle(c) if c.stroke.as_deref() == Some(COLOR_IN))));
        // The sub log shows the swap arrow.
        assert!(fp
            .shapes
            .iter()
            .any(|s| matches!(s, Shape::Text(t) if t.text == "\u{2192}")));
    }
}
