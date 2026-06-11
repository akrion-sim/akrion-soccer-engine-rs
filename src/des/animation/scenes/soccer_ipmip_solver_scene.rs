//! Port of `src/des/animation/scenes/soccer-ipmip-solver-scene.ts`.
//!
//! Shows the optimization side of the 7v7 rotation example as a DES network: a
//! model source emits branch/cut subproblem tokens, stationary solver entities
//! process LP relaxations and candidates, and the schedule sink receives the
//! best feasible incumbent.
//!
//! ## Conversion notes
//!
//! * `NODE_BY_ID` (a `Map` over `NODES`) → [`node_by_id`] linear lookup.
//! * `edgeKey(a, b)` → a `"a->b"` `String` key.
//! * PORT NOTE: only the subset of
//!   `crate::des::general::ip_mip_des::{IPMIPSolution, IPMIPTraceEvent}` the
//!   scene reads is mirrored locally below.

#![allow(dead_code)]

use crate::des::animation::types::{
    js_num, to_fixed, Anchor, ChartSeries, ChartSpec, CircleShape, FontWeight, FrameParts,
    LineShape, PathShape, RectShape, Shape, TextShape,
};

pub const SOCCER_IPMIP_SOLVER_W: f64 = 1100.0;
pub const SOCCER_IPMIP_SOLVER_H: f64 = 680.0;
pub const SOLVER_FRAMES_PER_EVENT: i64 = 7;

// PORT NOTE: local mirror of the IP/MIP solution + trace event (subset used).
#[derive(Clone, Debug, Default)]
pub struct IPMIPTraceEvent {
    pub node_id: String,
    pub depth: f64,
    pub action: String,
    pub lp_z: Option<f64>,
    pub fractional: Vec<f64>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct IPMIPSolution {
    pub trace: Vec<IPMIPTraceEvent>,
    pub status: String,
    pub z: f64,
    pub gap: f64,
    pub best_bound: f64,
    pub lp_algorithm: String,
    pub lp_solves: f64,
    pub elapsed_ms: f64,
    pub nodes_explored: f64,
    pub cuts_added: f64,
    pub candidates_tried: f64,
    /// `Object.entries(...)` order is preserved as a `Vec`.
    pub lp_algorithm_usage: Vec<(String, f64)>,
    pub incumbent_source: Option<String>,
}

struct NodeBox {
    id: &'static str,
    label: &'static str,
    role: &'static str,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    kind: &'static str,
}

struct Edge {
    from: &'static str,
    to: &'static str,
    label: &'static str,
}

const NODES: &[NodeBox] = &[
    NodeBox {
        id: "source",
        label: "Model Source",
        role: "emits soccer IP/MIP root problem",
        x: 36.0,
        y: 105.0,
        w: 135.0,
        h: 58.0,
        kind: "source",
    },
    NodeBox {
        id: "ip-search-controller",
        label: "Search Controller",
        role: "frontier of branch/cut subproblems",
        x: 220.0,
        y: 105.0,
        w: 150.0,
        h: 58.0,
        kind: "station",
    },
    NodeBox {
        id: "ip-lp-relaxation",
        label: "LP Relaxation",
        role: "internal simplex relaxation station",
        x: 430.0,
        y: 105.0,
        w: 150.0,
        h: 58.0,
        kind: "station",
    },
    NodeBox {
        id: "ip-rounding-repair",
        label: "Rounding Repair",
        role: "candidate movable repair",
        x: 655.0,
        y: 52.0,
        w: 155.0,
        h: 58.0,
        kind: "station",
    },
    NodeBox {
        id: "ip-cut-generator",
        label: "Cut Generator",
        role: "valid inequality station",
        x: 655.0,
        y: 158.0,
        w: 155.0,
        h: 58.0,
        kind: "station",
    },
    NodeBox {
        id: "ip-incumbent",
        label: "Incumbent",
        role: "best feasible solution anchor",
        x: 882.0,
        y: 52.0,
        w: 150.0,
        h: 58.0,
        kind: "sink",
    },
    NodeBox {
        id: "ip-node-decision",
        label: "Node Decision",
        role: "prune, cut, branch, or accept",
        x: 882.0,
        y: 158.0,
        w: 150.0,
        h: 58.0,
        kind: "station",
    },
    NodeBox {
        id: "sink",
        label: "Schedule Sink",
        role: "feasible lineup schedule",
        x: 882.0,
        y: 296.0,
        w: 150.0,
        h: 58.0,
        kind: "sink",
    },
];

const EDGES: &[Edge] = &[
    Edge {
        from: "source",
        to: "ip-search-controller",
        label: "root",
    },
    Edge {
        from: "ip-search-controller",
        to: "ip-lp-relaxation",
        label: "node",
    },
    Edge {
        from: "ip-lp-relaxation",
        to: "ip-rounding-repair",
        label: "relax",
    },
    Edge {
        from: "ip-rounding-repair",
        to: "ip-incumbent",
        label: "candidate",
    },
    Edge {
        from: "ip-lp-relaxation",
        to: "ip-cut-generator",
        label: "relax",
    },
    Edge {
        from: "ip-cut-generator",
        to: "ip-node-decision",
        label: "cut",
    },
    Edge {
        from: "ip-lp-relaxation",
        to: "ip-node-decision",
        label: "relax",
    },
    Edge {
        from: "ip-node-decision",
        to: "ip-search-controller",
        label: "children",
    },
    Edge {
        from: "ip-node-decision",
        to: "sink",
        label: "complete",
    },
    Edge {
        from: "ip-incumbent",
        to: "sink",
        label: "incumbent",
    },
];

fn node_by_id(id: &str) -> &'static NodeBox {
    NODES.iter().find(|n| n.id == id).expect("node id present")
}

fn center(n: &NodeBox) -> (f64, f64) {
    (n.x + n.w / 2.0, n.y + n.h / 2.0)
}

fn interpolate(a: (f64, f64), b: (f64, f64), u: f64) -> (f64, f64) {
    (a.0 + (b.0 - a.0) * u, a.1 + (b.1 - a.1) * u)
}

fn fmt(x: f64, digits: usize) -> String {
    if x.is_finite() {
        to_fixed(x, digits)
    } else {
        js_num(x)
    }
}

fn pct(x: f64) -> String {
    if x.is_finite() {
        format!("{}%", to_fixed(100.0 * x, 1))
    } else {
        "n/a".to_string()
    }
}

fn trunc(s: Option<&str>, n: usize) -> String {
    match s {
        None => String::new(),
        Some(s) if s.is_empty() => String::new(),
        Some(s) => {
            if s.chars().count() <= n {
                s.to_string()
            } else {
                let take = n.saturating_sub(1);
                let head: String = s.chars().take(take).collect();
                format!("{head}...")
            }
        }
    }
}

fn trace_event(solution: &IPMIPSolution, event_index: i64) -> Option<&IPMIPTraceEvent> {
    if solution.trace.is_empty() {
        return None;
    }
    let idx = event_index.clamp(0, solution.trace.len() as i64 - 1) as usize;
    Some(&solution.trace[idx])
}

fn event_path(ev: Option<&IPMIPTraceEvent>) -> Vec<&'static str> {
    let ev = match ev {
        None => {
            return vec![
                "source",
                "ip-search-controller",
                "ip-lp-relaxation",
                "ip-rounding-repair",
                "ip-incumbent",
                "sink",
            ]
        }
        Some(e) => e,
    };
    match ev.action.as_str() {
        "cut" => vec![
            "source",
            "ip-search-controller",
            "ip-lp-relaxation",
            "ip-cut-generator",
            "ip-node-decision",
            "ip-search-controller",
        ],
        "branch" => vec![
            "source",
            "ip-search-controller",
            "ip-lp-relaxation",
            "ip-node-decision",
            "ip-search-controller",
        ],
        "incumbent" => vec![
            "source",
            "ip-search-controller",
            "ip-lp-relaxation",
            "ip-node-decision",
            "ip-incumbent",
            "sink",
        ],
        "prune"
            if ev
                .reason
                .as_deref()
                .is_some_and(|r| r.contains("incumbent")) =>
        {
            vec![
                "source",
                "ip-search-controller",
                "ip-lp-relaxation",
                "ip-rounding-repair",
                "ip-incumbent",
                "ip-node-decision",
                "sink",
            ]
        }
        _ => vec![
            "source",
            "ip-search-controller",
            "ip-lp-relaxation",
            "ip-node-decision",
            "sink",
        ],
    }
}

struct Segment {
    from: &'static str,
    to: &'static str,
    u: f64,
}

fn active_segment(path: &[&'static str], subframe: i64) -> Segment {
    let segments = 1.max(path.len() as i64 - 1);
    let progress = (subframe as f64 / 1.0_f64.max(SOLVER_FRAMES_PER_EVENT as f64 - 1.0))
        .max(0.0)
        .min(0.999);
    let raw = progress * segments as f64;
    let idx = (segments - 1).min(raw.floor() as i64).max(0) as usize;
    Segment {
        from: path[idx],
        to: path[idx + 1],
        u: raw - idx as f64,
    }
}

fn edge_key(a: &str, b: &str) -> String {
    format!("{a}->{b}")
}

fn node_fill(n: &NodeBox, active: bool) -> String {
    if active {
        return "#fde68a".to_string();
    }
    match n.kind {
        "source" => "#dbeafe".to_string(),
        "sink" => "#dcfce7".to_string(),
        _ => "#f8fafc".to_string(),
    }
}

fn draw_arrow(
    shapes: &mut Vec<Shape>,
    a: (f64, f64),
    b: (f64, f64),
    color: &str,
    width: f64,
    opacity: f64,
) {
    shapes.push(Shape::Line(LineShape {
        x1: a.0,
        y1: a.1,
        x2: b.0,
        y2: b.1,
        stroke: color.to_string(),
        stroke_width: Some(width),
        opacity: Some(opacity),
        ..Default::default()
    }));
    let angle = (b.1 - a.1).atan2(b.0 - a.0);
    let size = 8.0 + width;
    let p1 = (
        b.0 - (angle - 0.45).cos() * size,
        b.1 - (angle - 0.45).sin() * size,
    );
    let p2 = (
        b.0 - (angle + 0.45).cos() * size,
        b.1 - (angle + 0.45).sin() * size,
    );
    shapes.push(Shape::Path(PathShape {
        d: format!(
            "M {} {} L {} {} L {} {} Z",
            js_num(b.0),
            js_num(b.1),
            js_num(p1.0),
            js_num(p1.1),
            js_num(p2.0),
            js_num(p2.1)
        ),
        fill: Some(color.to_string()),
        opacity: Some(opacity),
        ..Default::default()
    }));
}

fn draw_node(shapes: &mut Vec<Shape>, n: &NodeBox, active: bool, used: bool) {
    let stroke = if active {
        "#92400e"
    } else if used {
        "#2563eb"
    } else {
        "#64748b"
    };
    let width = if active {
        3.0
    } else if used {
        2.0
    } else {
        1.2
    };
    shapes.push(Shape::Rect(RectShape {
        x: n.x,
        y: n.y,
        w: n.w,
        h: n.h,
        rx: Some(8.0),
        fill: node_fill(n, active),
        stroke: Some(stroke.to_string()),
        stroke_width: Some(width),
        title: Some(n.role.to_string()),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: n.x + n.w / 2.0,
        y: n.y + 24.0,
        text: n.label.to_string(),
        font_size: Some(13.0),
        fill: Some("#0f172a".to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: n.x + n.w / 2.0,
        y: n.y + 43.0,
        text: n.kind.to_string(),
        font_size: Some(10.0),
        fill: Some("#475569".to_string()),
        anchor: Some(Anchor::Middle),
        ..Default::default()
    }));
}

fn draw_metric(shapes: &mut Vec<Shape>, x: f64, y: f64, label: &str, value: &str, color: &str) {
    shapes.push(Shape::Text(TextShape {
        x,
        y,
        text: label.to_string(),
        font_size: Some(11.0),
        fill: Some("#64748b".to_string()),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: x + 158.0,
        y,
        text: value.to_string(),
        font_size: Some(12.0),
        fill: Some(color.to_string()),
        anchor: Some(Anchor::End),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
}

fn draw_side_panel(
    shapes: &mut Vec<Shape>,
    solution: &IPMIPSolution,
    ev: Option<&IPMIPTraceEvent>,
    frame_index: i64,
) {
    let x = 36.0;
    let y = 390.0;
    let w = 996.0;
    shapes.push(Shape::Rect(RectShape {
        x,
        y,
        w,
        h: 106.0,
        rx: Some(8.0),
        fill: "#ffffff".to_string(),
        stroke: Some("#cbd5e1".to_string()),
        stroke_width: Some(1.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: x + 18.0,
        y: y + 24.0,
        text: "Solver status".to_string(),
        font_size: Some(15.0),
        fill: Some("#0f172a".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    draw_metric(
        shapes,
        x + 18.0,
        y + 48.0,
        "status",
        &solution.status,
        if solution.status == "optimal" {
            "#15803d"
        } else {
            "#b45309"
        },
    );
    draw_metric(
        shapes,
        x + 18.0,
        y + 70.0,
        "objective",
        &fmt(solution.z, 4),
        "#0f172a",
    );
    draw_metric(
        shapes,
        x + 18.0,
        y + 92.0,
        "gap",
        &pct(solution.gap),
        "#0f172a",
    );
    draw_metric(
        shapes,
        x + 238.0,
        y + 48.0,
        "LP backend",
        &solution.lp_algorithm,
        "#0f172a",
    );
    draw_metric(
        shapes,
        x + 238.0,
        y + 70.0,
        "LP solves",
        &js_num(solution.lp_solves),
        "#0f172a",
    );
    draw_metric(
        shapes,
        x + 238.0,
        y + 92.0,
        "elapsed",
        &format!("{}ms", js_num(solution.elapsed_ms)),
        "#0f172a",
    );
    draw_metric(
        shapes,
        x + 458.0,
        y + 48.0,
        "nodes",
        &js_num(solution.nodes_explored),
        "#0f172a",
    );
    draw_metric(
        shapes,
        x + 458.0,
        y + 70.0,
        "cuts",
        &js_num(solution.cuts_added),
        "#0f172a",
    );
    draw_metric(
        shapes,
        x + 458.0,
        y + 92.0,
        "candidates",
        &js_num(solution.candidates_tried),
        "#0f172a",
    );

    let ex = x + 690.0;
    shapes.push(Shape::Text(TextShape {
        x: ex,
        y: y + 24.0,
        text: format!(
            "Event {} / {}",
            frame_index + 1,
            1.max(solution.trace.len() as i64)
        ),
        font_size: Some(15.0),
        fill: Some("#0f172a".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    if let Some(ev) = ev {
        shapes.push(Shape::Text(TextShape {
            x: ex,
            y: y + 48.0,
            text: format!(
                "node {}, depth {}, action {}",
                ev.node_id,
                js_num(ev.depth),
                ev.action
            ),
            font_size: Some(12.0),
            fill: Some("#334155".to_string()),
            ..Default::default()
        }));
        let lp_z = match ev.lp_z {
            None => "n/a".to_string(),
            Some(v) => fmt(v, 4),
        };
        shapes.push(Shape::Text(TextShape {
            x: ex,
            y: y + 70.0,
            text: format!("LP z {}, fractional vars {}", lp_z, ev.fractional.len()),
            font_size: Some(12.0),
            fill: Some("#334155".to_string()),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: ex,
            y: y + 92.0,
            text: trunc(ev.reason.as_deref(), 44),
            font_size: Some(12.0),
            fill: Some("#64748b".to_string()),
            ..Default::default()
        }));
    } else {
        shapes.push(Shape::Text(TextShape {
            x: ex,
            y: y + 55.0,
            text: "root model solved without branch trace events".to_string(),
            font_size: Some(12.0),
            fill: Some("#334155".to_string()),
            ..Default::default()
        }));
    }
}

pub fn soccer_ipmip_solver_frame_count(solution: &IPMIPSolution) -> i64 {
    1.max(solution.trace.len() as i64) * SOLVER_FRAMES_PER_EVENT
}

pub fn build_soccer_ipmip_solver_frame(solution: &IPMIPSolution, frame_index: i64) -> FrameParts {
    let event_count = 1.max(solution.trace.len() as i64);
    let event_index = (event_count - 1).min(frame_index / SOLVER_FRAMES_PER_EVENT);
    let subframe = frame_index % SOLVER_FRAMES_PER_EVENT;
    let ev = trace_event(solution, event_index);
    let path = event_path(ev);
    let seg = active_segment(&path, subframe);
    let active_edge = edge_key(seg.from, seg.to);
    let mut shapes: Vec<Shape> = Vec::new();

    shapes.push(Shape::Rect(RectShape {
        x: 0.0,
        y: 0.0,
        w: SOCCER_IPMIP_SOLVER_W,
        h: SOCCER_IPMIP_SOLVER_H,
        fill: "#f8fafc".to_string(),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: 36.0,
        y: 36.0,
        text: "7v7 IP/MIP solver entities".to_string(),
        font_size: Some(22.0),
        fill: Some("#0f172a".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: 36.0,
        y: 60.0,
        text: "Internal LP relaxation station + DES branch/cut token flow".to_string(),
        font_size: Some(13.0),
        fill: Some("#475569".to_string()),
        ..Default::default()
    }));

    shapes.push(Shape::Rect(RectShape {
        x: 24.0,
        y: 78.0,
        w: 1020.0,
        h: 292.0,
        rx: Some(8.0),
        fill: "#eef2ff".to_string(),
        stroke: Some("#c7d2fe".to_string()),
        stroke_width: Some(1.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: 42.0,
        y: 96.0,
        text: "station graph".to_string(),
        font_size: Some(11.0),
        fill: Some("#4338ca".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));

    for e in EDGES {
        let a = node_by_id(e.from);
        let b = node_by_id(e.to);
        let ca = center(a);
        let cb = center(b);
        let is_active = edge_key(e.from, e.to) == active_edge;
        draw_arrow(
            &mut shapes,
            ca,
            cb,
            if is_active { "#2563eb" } else { "#94a3b8" },
            if is_active { 4.0 } else { 1.6 },
            if is_active { 0.95 } else { 0.55 },
        );
        let mid = interpolate(ca, cb, 0.52);
        shapes.push(Shape::Text(TextShape {
            x: mid.0,
            y: mid.1 - 6.0,
            text: e.label.to_string(),
            font_size: Some(9.0),
            fill: Some(if is_active {
                "#1d4ed8".to_string()
            } else {
                "#64748b".to_string()
            }),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
    }

    for n in NODES {
        draw_node(
            &mut shapes,
            n,
            n.id == seg.from || n.id == seg.to,
            path.contains(&n.id),
        );
    }

    let a = node_by_id(seg.from);
    let b = node_by_id(seg.to);
    let p = interpolate(center(a), center(b), seg.u);
    shapes.push(Shape::Circle(CircleShape {
        x: p.0,
        y: p.1,
        r: 13.0,
        fill: "#f97316".to_string(),
        stroke: Some("#fff7ed".to_string()),
        stroke_width: Some(3.0),
        title: Some("movable solver token".to_string()),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: p.0,
        y: p.1 + 4.0,
        text: "x".to_string(),
        font_size: Some(11.0),
        fill: Some("#ffffff".to_string()),
        anchor: Some(Anchor::Middle),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));

    draw_side_panel(&mut shapes, solution, ev, event_index);

    let usage_joined = solution
        .lp_algorithm_usage
        .iter()
        .map(|(k, v)| format!("{k}={}", js_num(*v)))
        .collect::<Vec<_>>()
        .join(", ");
    let usage = if usage_joined.is_empty() {
        "none".to_string()
    } else {
        usage_joined
    };
    shapes.push(Shape::Text(TextShape {
        x: 36.0,
        y: 526.0,
        text: format!("LP usage: {usage}"),
        font_size: Some(12.0),
        fill: Some("#334155".to_string()),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: 36.0,
        y: 546.0,
        text: format!(
            "Incumbent source: {}",
            solution.incumbent_source.as_deref().unwrap_or("none")
        ),
        font_size: Some(12.0),
        fill: Some("#334155".to_string()),
        ..Default::default()
    }));

    let caption = match ev {
        Some(ev) => format!(
            "event={}/{} node={} action={} status={}",
            event_index + 1,
            solution.trace.len(),
            ev.node_id,
            ev.action,
            solution.status
        ),
        None => format!(
            "status={} objective={}",
            solution.status,
            fmt(solution.z, 4)
        ),
    };
    FrameParts::with_caption(shapes, caption)
}

pub fn build_soccer_ipmip_solver_charts(solution: &IPMIPSolution) -> Vec<ChartSpec> {
    // `events = trace.length ? trace : [null]`.
    let count = if solution.trace.is_empty() {
        1
    } else {
        solution.trace.len()
    };
    let t: Vec<f64> = (0..count).map(|i| i as f64).collect();
    let lp_bound: Vec<f64> = (0..count)
        .map(|i| {
            solution
                .trace
                .get(i)
                .and_then(|ev| ev.lp_z)
                .unwrap_or(solution.best_bound)
        })
        .collect();
    let fractional: Vec<f64> = (0..count)
        .map(|i| {
            solution
                .trace
                .get(i)
                .map_or(0.0, |ev| ev.fractional.len() as f64)
        })
        .collect();
    let incumbent: Vec<f64> = (0..count).map(|_| solution.z).collect();
    vec![
        ChartSpec {
            x: 650.0,
            y: 520.0,
            w: 180.0,
            h: 110.0,
            title: Some("LP bound / incumbent".to_string()),
            series: vec![
                ChartSeries {
                    label: "lp bound".to_string(),
                    color: "#2563eb".to_string(),
                    t: t.clone(),
                    y: lp_bound,
                },
                ChartSeries {
                    label: "incumbent".to_string(),
                    color: "#16a34a".to_string(),
                    t: t.clone(),
                    y: incumbent,
                },
            ],
            ..Default::default()
        },
        ChartSpec {
            x: 852.0,
            y: 520.0,
            w: 180.0,
            h: 110.0,
            title: Some("fractional vars".to_string()),
            y_min: Some(0.0),
            series: vec![ChartSeries {
                label: "fractional".to_string(),
                color: "#f97316".to_string(),
                t,
                y: fractional,
            }],
            ..Default::default()
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_contains_text(frame: &FrameParts, needle: &str) -> bool {
        frame
            .caption
            .as_deref()
            .is_some_and(|caption| caption.contains(needle))
            || frame.shapes.iter().any(|shape| match shape {
                Shape::Text(text) => text.text.contains(needle),
                Shape::Rect(rect) => rect
                    .label
                    .as_deref()
                    .is_some_and(|label| label.contains(needle)),
                Shape::Circle(circle) => circle
                    .label
                    .as_deref()
                    .is_some_and(|label| label.contains(needle)),
                _ => false,
            })
    }

    #[test]
    fn event_path_for_cut_action() {
        let ev = IPMIPTraceEvent {
            action: "cut".to_string(),
            ..Default::default()
        };
        let path = event_path(Some(&ev));
        assert_eq!(path.last(), Some(&"ip-search-controller"));
    }

    #[test]
    fn frame_count_and_caption() {
        let solution = IPMIPSolution {
            trace: vec![IPMIPTraceEvent {
                node_id: "n0".to_string(),
                depth: 0.0,
                action: "branch".to_string(),
                lp_z: Some(1.5),
                fractional: vec![0.5],
                reason: None,
            }],
            status: "optimal".to_string(),
            z: 2.0,
            ..Default::default()
        };
        assert_eq!(
            soccer_ipmip_solver_frame_count(&solution),
            SOLVER_FRAMES_PER_EVENT
        );
        let fp = build_soccer_ipmip_solver_frame(&solution, 0);
        assert_eq!(
            fp.caption.as_deref(),
            Some("event=1/1 node=n0 action=branch status=optimal")
        );
    }

    #[test]
    fn solver_frame_renders_visual_blocks_and_internal_backend() {
        let solution = IPMIPSolution {
            trace: vec![IPMIPTraceEvent {
                node_id: "0".to_string(),
                depth: 0.0,
                action: "incumbent".to_string(),
                lp_z: Some(4.0),
                fractional: Vec::new(),
                reason: Some("integer feasible".to_string()),
            }],
            status: "optimal".to_string(),
            z: 4.0,
            gap: 0.0,
            best_bound: 4.0,
            lp_algorithm: "internal-simplex".to_string(),
            lp_solves: 2.0,
            elapsed_ms: 12.0,
            nodes_explored: 2.0,
            ..Default::default()
        };

        let fp = build_soccer_ipmip_solver_frame(&solution, 0);

        assert!(frame_contains_text(&fp, "7v7 IP/MIP solver entities"));
        assert!(frame_contains_text(&fp, "station graph"));
        assert!(frame_contains_text(&fp, "Search Controller"));
        assert!(frame_contains_text(&fp, "LP Relaxation"));
        assert!(frame_contains_text(&fp, "internal-simplex"));
        assert!(frame_contains_text(&fp, "integer feasible"));
    }
}
