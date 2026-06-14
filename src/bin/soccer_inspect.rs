//! `soccer_inspect` — external live-state inspector for the running soccer
//! server. Maps the shared-memory ring file the server publishes (enable it with
//! `SOCCER_INSPECT=1` on the server) and reads frames **zero-copy**, with no
//! Python and no ptrace — the process just maps the same file the server wrote.
//!
//! Usage:
//!   soccer_inspect [PATH] [--player ID] [--fields a,b,c] [--history N]
//!                  [--watch] [--raw OFF LEN] [--json]
//!
//! PATH defaults to $SOCCER_INSPECT_PATH, else the newest
//! `soccer_inspect_*.shm` in the temp dir.
//!
//! Examples:
//!   soccer_inspect --player 18 --fields facing_yaw,yaw_rate,action_facing
//!   soccer_inspect --player 18 --fields facing_yaw --history 45   # last 3s @15Hz
//!   soccer_inspect --watch --player 18 --fields facing_yaw,yaw_rate
//!   soccer_inspect --raw 8 8     # random-access read: header.latest_index bytes

use soccer_engine::des::general::soccer::{
    facing_label, gait_label, name_str, role_label, FrameInspect, InspectReader, PlayerInspect,
};

struct Args {
    path: Option<String>,
    player: Option<u32>,
    fields: Vec<String>,
    history: usize,
    watch: bool,
    json: bool,
    raw: Option<(usize, usize)>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        path: None,
        player: None,
        fields: Vec::new(),
        history: 0,
        watch: false,
        json: false,
        raw: None,
    };
    let mut it = std::env::args().skip(1).peekable();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--player" | "-p" => {
                args.player = Some(
                    it.next()
                        .ok_or("--player needs an id")?
                        .parse()
                        .map_err(|_| "--player id must be a number")?,
                );
            }
            "--fields" | "--field" | "-f" => {
                let raw = it.next().ok_or("--fields needs a value")?;
                args.fields
                    .extend(raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()));
            }
            "--history" | "-n" => {
                args.history = it
                    .next()
                    .ok_or("--history needs a count")?
                    .parse()
                    .map_err(|_| "--history must be a number")?;
            }
            "--watch" | "-w" => args.watch = true,
            "--json" => args.json = true,
            "--raw" => {
                let off = it.next().ok_or("--raw needs OFFSET LEN")?;
                let len = it.next().ok_or("--raw needs OFFSET LEN")?;
                args.raw = Some((
                    off.parse().map_err(|_| "raw OFFSET must be a number")?,
                    len.parse().map_err(|_| "raw LEN must be a number")?,
                ));
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other if !other.starts_with('-') && args.path.is_none() => {
                args.path = Some(other.to_string());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(args)
}

fn print_help() {
    eprintln!(
        "soccer_inspect — external zero-copy live-state inspector\n\
\n\
USAGE:\n\
  soccer_inspect [PATH] [--player ID] [--fields a,b,c] [--history N] [--watch] [--raw OFF LEN] [--json]\n\
\n\
PATH defaults to $SOCCER_INSPECT_PATH, else the newest soccer_inspect_*.shm in the temp dir.\n\
Server must run with SOCCER_INSPECT=1 to publish the ring."
    );
}

/// Pick the inspector file: explicit arg → $SOCCER_INSPECT_PATH → newest
/// `soccer_inspect_*.shm` in the temp dir.
fn resolve_path(explicit: Option<String>) -> Result<std::path::PathBuf, String> {
    if let Some(p) = explicit {
        return Ok(p.into());
    }
    if let Ok(p) = std::env::var("SOCCER_INSPECT_PATH") {
        return Ok(p.into());
    }
    let dir = std::env::temp_dir();
    let mut newest: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
    let entries = std::fs::read_dir(&dir).map_err(|e| format!("read temp dir: {e}"))?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("soccer_inspect_") && name.ends_with(".shm") {
            if let Ok(meta) = entry.metadata() {
                let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                if newest.as_ref().map(|(t, _)| mtime > *t).unwrap_or(true) {
                    newest = Some((mtime, entry.path()));
                }
            }
        }
    }
    newest.map(|(_, p)| p).ok_or_else(|| {
        format!(
            "no soccer_inspect_*.shm found in {} — run the server with SOCCER_INSPECT=1 or pass PATH",
            dir.display()
        )
    })
}

fn field_value(p: &PlayerInspect, field: &str) -> String {
    match field.to_ascii_lowercase().as_str() {
        "facing_yaw" | "facingyaw" => format!("{:.4}", p.facing_yaw),
        "yaw_rate" | "yawrate" => format!("{:.4}", p.yaw_rate),
        "action_facing" | "actionfacing" => facing_label(p.action_facing).to_string(),
        "receive_facing" | "receivefacing" => facing_label(p.receive_facing).to_string(),
        "gait" => gait_label(p.gait).to_string(),
        "role" => role_label(p.role).to_string(),
        "dizziness" => format!("{:.4}", p.dizziness),
        "fatigue" => format!("{:.4}", p.fatigue),
        "anaerobic_load" | "anaerobicload" => format!("{:.4}", p.anaerobic_load),
        "decision_confidence" | "decisionconfidence" => format!("{:.4}", p.decision_confidence),
        "position" => format!("({:.2},{:.2})", p.position.x, p.position.y),
        "velocity" => format!("({:.2},{:.2})", p.velocity.x, p.velocity.y),
        "speed" => format!("{:.2}", (p.velocity.x.powi(2) + p.velocity.y.powi(2)).sqrt()),
        other => format!("<unknown:{other}>"),
    }
}

fn find_player(frame: &FrameInspect, id: u32) -> Option<&PlayerInspect> {
    // `players_slice` clamps player_count to the array bound — the file is
    // externally written, so a corrupt count must not panic the reader.
    frame.players_slice().iter().find(|p| p.id == id)
}

fn print_player_line(frame: &FrameInspect, p: &PlayerInspect, fields: &[String]) {
    let mut parts = vec![
        format!("t={}", frame.tick),
        format!("clk={:.2}", frame.clock_seconds),
        format!("#{} {}", p.id, name_str(&p.name)),
    ];
    if fields.is_empty() {
        parts.push(format!("facing_yaw={:.4}", p.facing_yaw));
        parts.push(format!("yaw_rate={:.4}", p.yaw_rate));
        parts.push(format!("action_facing={}", facing_label(p.action_facing)));
        parts.push(format!("gait={}", gait_label(p.gait)));
    } else {
        for f in fields {
            parts.push(format!("{}={}", f, field_value(p, f)));
        }
    }
    println!("{}", parts.join("  "));
}

fn print_frame_summary(frame: &FrameInspect) {
    println!(
        "tick={} clock={:.2} score={}-{} players={}",
        frame.tick, frame.clock_seconds, frame.score_home, frame.score_away, frame.player_count
    );
    let b = &frame.ball;
    println!(
        "  ball: holder={} pos=({:.2},{:.2}) vel=({:.2},{:.2}) jerk=({:.0},{:.0}) orbit[rad={:.3} r={:.2} swept={:.3}]",
        if b.holder < 0 { "none".to_string() } else { b.holder.to_string() },
        b.position.x, b.position.y, b.velocity.x, b.velocity.y, b.jerk.x, b.jerk.y,
        b.carry_orbit_world_rad, b.carry_orbit_radius_yards, b.carry_orbit_swept_rad
    );
    for p in frame.players_slice() {
        println!(
            "  #{:<2} {:<10} {:<3} yaw={:+.3} yaw_rate={:+.3} face={:<3} gait={:<9} pos=({:.1},{:.1})",
            p.id,
            name_str(&p.name),
            role_label(p.role),
            p.facing_yaw,
            p.yaw_rate,
            facing_label(p.action_facing),
            gait_label(p.gait),
            p.position.x,
            p.position.y,
        );
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let path = resolve_path(args.path.clone())?;

    // Random-access raw byte read ("read this one memory location"): map the file
    // and dump LEN bytes at OFF as hex without interpreting the layout.
    if let Some((off, len)) = args.raw {
        let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let end = off.saturating_add(len).min(bytes.len());
        if off >= bytes.len() {
            return Err(format!("offset {off} past end ({} bytes)", bytes.len()));
        }
        let slice = &bytes[off..end];
        let hex: Vec<String> = slice.iter().map(|b| format!("{b:02x}")).collect();
        println!("@{off}+{}: {}", slice.len(), hex.join(" "));
        if len == 8 && slice.len() == 8 {
            let mut a = [0u8; 8];
            a.copy_from_slice(slice);
            println!("  as u64 LE: {}", u64::from_le_bytes(a));
            println!("  as f64 LE: {}", f64::from_le_bytes(a));
        }
        return Ok(());
    }

    let reader = InspectReader::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let header = reader.header();
    eprintln!(
        "[soccer_inspect] {} (pid {}, v{}, {} slots, frame {} bytes)",
        path.display(),
        header.pid,
        header.version,
        header.ring_slots,
        header.frame_size
    );

    let render = |reader: &InspectReader| -> Result<(), String> {
        if args.history > 0 {
            let frames = reader.history(args.history);
            if frames.is_empty() {
                return Err("ring is empty (no frames published yet)".into());
            }
            if let Some(id) = args.player {
                for f in &frames {
                    if let Some(p) = find_player(f, id) {
                        print_player_line(f, p, &args.fields);
                    }
                }
            } else if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &frames.iter().map(frame_to_json).collect::<Vec<_>>()
                    )
                    .map_err(|e| e.to_string())?
                );
            } else {
                for f in &frames {
                    print_frame_summary(f);
                }
            }
            return Ok(());
        }

        let frame = reader
            .latest()
            .ok_or("ring is empty (no frames published yet)")?;
        match args.player {
            Some(id) => {
                let p = find_player(&frame, id)
                    .ok_or_else(|| format!("no player with id {id} in latest frame"))?;
                if args.json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&player_to_json(&frame, p))
                            .map_err(|e| e.to_string())?
                    );
                } else {
                    print_player_line(&frame, p, &args.fields);
                }
            }
            None => {
                if args.json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&frame_to_json(&frame))
                            .map_err(|e| e.to_string())?
                    );
                } else {
                    print_frame_summary(&frame);
                }
            }
        }
        Ok(())
    };

    if args.watch {
        let mut last = u64::MAX;
        loop {
            let idx = reader.latest_index();
            if idx != last {
                last = idx;
                if let Err(e) = render(&reader) {
                    eprintln!("[soccer_inspect] {e}");
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(33));
        }
    } else {
        render(&reader)?;
    }
    Ok(())
}

fn player_to_json(frame: &FrameInspect, p: &PlayerInspect) -> serde_json::Value {
    serde_json::json!({
        "tick": frame.tick,
        "clockSeconds": frame.clock_seconds,
        "id": p.id,
        "name": name_str(&p.name),
        "role": role_label(p.role),
        "facingYaw": p.facing_yaw,
        "yawRate": p.yaw_rate,
        "actionFacing": facing_label(p.action_facing),
        "gait": gait_label(p.gait),
        "dizziness": p.dizziness,
        "fatigue": p.fatigue,
        "position": { "x": p.position.x, "y": p.position.y },
        "velocity": { "x": p.velocity.x, "y": p.velocity.y },
    })
}

fn frame_to_json(frame: &FrameInspect) -> serde_json::Value {
    let players: Vec<serde_json::Value> = frame
        .players_slice()
        .iter()
        .map(|p| player_to_json(frame, p))
        .collect();
    serde_json::json!({
        "tick": frame.tick,
        "clockSeconds": frame.clock_seconds,
        "scoreHome": frame.score_home,
        "scoreAway": frame.score_away,
        "ball": {
            "holder": if frame.ball.holder < 0 { serde_json::Value::Null } else { serde_json::json!(frame.ball.holder) },
            "position": { "x": frame.ball.position.x, "y": frame.ball.position.y },
            "velocity": { "x": frame.ball.velocity.x, "y": frame.ball.velocity.y },
            "carryOrbitWorldRad": frame.ball.carry_orbit_world_rad,
            "carryOrbitSweptRad": frame.ball.carry_orbit_swept_rad,
        },
        "players": players,
    })
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
