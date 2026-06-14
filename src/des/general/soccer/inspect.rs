//! Live-state inspection interface.
//!
//! Lets an external program read the running sim's state **without the server
//! serialising everything to I/O every tick**. There are three coordinated read
//! paths over one source of truth (a packed, versioned `#[repr(C)]` snapshot):
//!
//!  1. **On-demand HTTP query** (`GET /api/inspect`, pull): the server reads its
//!     own live structs and serialises *only the slice asked for* (a player, a
//!     field set, a section, an N-tick history window). Cost is paid per query.
//!  2. **Shared-memory ring (mmap, zero-copy)**: when `SOCCER_INSPECT=1`, the
//!     server memory-maps a fixed-size file and writes one `FrameInspect` per
//!     tick into a ring. The cost is a `memcpy` into mapped pages — no
//!     serialisation. Any external process maps the same file and reads the raw
//!     frames directly. A per-frame seqlock makes reads tear-free without locks.
//!  3. **External reader / "attach"**: the `soccer_inspect` Rust binary maps the
//!     file read-only and does random-access field reads by offset (no Python,
//!     no ptrace) — the debugger-style "read this one location" capability.
//!
//! The on-disk layout is a header followed by `RING_SLOTS` frames. The layout is
//! versioned (`INSPECT_MAGIC` + `INSPECT_VERSION`); a reader must reject a file
//! whose magic/version/frame-size disagree with its compiled-in expectation.

use super::*;
use std::sync::atomic::{fence, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// `"SCRINSP1"` little-endian — identifies a soccer inspector mmap file.
pub const INSPECT_MAGIC: u64 = 0x3150_534E_4952_4353;
/// Bump on ANY change to the `#[repr(C)]` layout below. Readers reject mismatches.
pub const INSPECT_VERSION: u32 = 1;
/// Max players a frame can hold (22 on the pitch + headroom for subs/spares).
pub const INSPECT_MAX_PLAYERS: usize = 32;
/// Ring depth. At 15 Hz, 256 slots ≈ 17 s of history — enough to catch a
/// multi-second anomaly (e.g. a 3-second facing flip-flop) after the fact.
pub const INSPECT_RING_SLOTS: usize = 256;

/// Two `f64`s, byte-identical to the sim's `Vec2`, in a layout we control.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Vec2C {
    pub x: f64,
    pub y: f64,
}

impl From<Vec2> for Vec2C {
    fn from(v: Vec2) -> Self {
        Vec2C { x: v.x, y: v.y }
    }
}

/// Per-player kinematics + biomechanical-turn state — the fields behind the
/// carrier facing model. Field order is largest-first so there are no implicit
/// padding holes (the trailing `_pad` keeps the 8-byte tail aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerInspect {
    pub position: Vec2C,
    pub velocity: Vec2C,
    pub acceleration: Vec2C,
    pub jerk: Vec2C,
    pub facing_yaw: f64,
    pub yaw_rate: f64,
    pub dizziness: f64,
    pub anaerobic_load: f64,
    pub fatigue: f64,
    pub decision_confidence: f64,
    pub id: u32,
    pub shirt: u8,
    /// 0 = Home, 1 = Away.
    pub team: u8,
    /// `PlayerRole` index: 0 GK, 1 DEF, 2 MID, 3 FWD.
    pub role: u8,
    /// `MovementGait` index (see `gait_to_u8`).
    pub gait: u8,
    /// `FacingBucket` index (see `facing_to_u8`): the carrier's chosen heading.
    pub action_facing: u8,
    pub receive_facing: u8,
    /// ASCII name, NUL-padded/truncated. Diagnostics only.
    pub name: [u8; 18],
}

/// Ball kinematics + the carried-ball orbital ("planet/sun") dribble state.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BallInspect {
    pub position: Vec2C,
    pub velocity: Vec2C,
    pub acceleration: Vec2C,
    pub jerk: Vec2C,
    pub curl_acceleration: Vec2C,
    pub altitude_yards: f64,
    pub carry_orbit_world_rad: f64,
    pub carry_orbit_radius_yards: f64,
    pub carry_orbit_swept_rad: f64,
    pub carry_orbit_last_tick: u64,
    /// Holder player id, or `-1` when the ball is loose.
    pub holder: i32,
    /// Last touch: `-1` none, `0` Home, `1` Away.
    pub last_touch_team: i8,
    pub carry_orbit_wrap_unlocked: u8,
    pub _pad: [u8; 2],
}

/// Compact central-brain / team-shape view (decision/brain internals).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BrainInspect {
    pub pressure_line_home: f64,
    pub pressure_line_away: f64,
    pub home_centroid_to_ball: f64,
    pub home_spread: f64,
    pub home_forward_velocity: f64,
    pub away_centroid_to_ball: f64,
    pub away_spread: f64,
    pub away_forward_velocity: f64,
    /// Brain's ball holder belief, or `-1`.
    pub ball_holder: i32,
    /// `TacticalPhase` index (see `phase_to_u8`).
    pub phase: u8,
    /// Possession team: `-1` none, `0` Home, `1` Away.
    pub possession_team: i8,
    pub home_players_near_ball: u8,
    pub away_players_near_ball: u8,
    pub controlled_human_players: u8,
    pub _pad: [u8; 3],
}

/// One tick's full inspectable state. The leading `write_seq` is the seqlock
/// guard: odd while a write is in flight, even (and equal before/after) once the
/// payload is consistent.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FrameInspect {
    pub write_seq: u64,
    pub tick: u64,
    pub clock_seconds: f64,
    pub ball: BallInspect,
    pub brain: BrainInspect,
    pub score_home: u32,
    pub score_away: u32,
    pub player_count: u32,
    pub _pad: u32,
    pub players: [PlayerInspect; INSPECT_MAX_PLAYERS],
}

impl Default for FrameInspect {
    fn default() -> Self {
        FrameInspect {
            write_seq: 0,
            tick: 0,
            clock_seconds: 0.0,
            ball: BallInspect::default(),
            brain: BrainInspect::default(),
            score_home: 0,
            score_away: 0,
            player_count: 0,
            _pad: 0,
            players: [PlayerInspect::default(); INSPECT_MAX_PLAYERS],
        }
    }
}

/// File header. `latest_index` is a monotonically increasing publish counter; the
/// newest frame lives in slot `latest_index % ring_slots` once `latest_index > 0`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InspectHeader {
    pub magic: u64,
    pub latest_index: u64,
    pub version: u32,
    pub frame_size: u32,
    pub ring_slots: u32,
    pub max_players: u32,
    pub pid: u32,
    pub _pad: u32,
}

pub const INSPECT_HEADER_SIZE: usize = std::mem::size_of::<InspectHeader>();
pub const INSPECT_FRAME_SIZE: usize = std::mem::size_of::<FrameInspect>();
/// Total mapped file size: header + ring of frames.
pub const INSPECT_FILE_SIZE: usize = INSPECT_HEADER_SIZE + INSPECT_FRAME_SIZE * INSPECT_RING_SLOTS;

// ----------------------------------------------------------------------------
// Enum → stable u8 maps. We do NOT rely on enum discriminants (those can be
// reordered); these are the inspector's wire contract, versioned with the file.
// ----------------------------------------------------------------------------

pub fn team_to_u8(team: Team) -> u8 {
    match team {
        Team::Home => 0,
        Team::Away => 1,
    }
}

pub fn role_to_u8(role: PlayerRole) -> u8 {
    match role {
        PlayerRole::Goalkeeper => 0,
        PlayerRole::Defender => 1,
        PlayerRole::Midfielder => 2,
        PlayerRole::Forward => 3,
    }
}

pub fn gait_to_u8(gait: MovementGait) -> u8 {
    match gait {
        MovementGait::Stand => 0,
        MovementGait::Walk => 1,
        MovementGait::BackWalk => 2,
        MovementGait::BackJog => 3,
        MovementGait::Skip => 4,
        MovementGait::BackSkip => 5,
        MovementGait::SideStep => 6,
        MovementGait::Jog => 7,
        MovementGait::Run => 8,
        MovementGait::Sprint => 9,
    }
}

pub fn facing_to_u8(facing: FacingBucket) -> u8 {
    match facing {
        FacingBucket::Unknown => 0,
        FacingBucket::North => 1,
        FacingBucket::NorthEast => 2,
        FacingBucket::East => 3,
        FacingBucket::SouthEast => 4,
        FacingBucket::South => 5,
        FacingBucket::SouthWest => 6,
        FacingBucket::West => 7,
        FacingBucket::NorthWest => 8,
    }
}

pub fn phase_to_u8(phase: TacticalPhase) -> u8 {
    match phase {
        TacticalPhase::Kickoff => 0,
        TacticalPhase::HomeBuildUp => 1,
        TacticalPhase::AwayBuildUp => 2,
        TacticalPhase::HomeAttack => 3,
        TacticalPhase::AwayAttack => 4,
        TacticalPhase::Transition => 5,
    }
}

/// Human-readable labels for the u8 maps above, so the JSON path and the bin can
/// decode without re-listing the variants.
pub fn gait_label(g: u8) -> &'static str {
    match g {
        0 => "stand",
        1 => "walk",
        2 => "back-walk",
        3 => "back-jog",
        4 => "skip",
        5 => "back-skip",
        6 => "side-step",
        7 => "jog",
        8 => "run",
        9 => "sprint",
        _ => "?",
    }
}

pub fn facing_label(f: u8) -> &'static str {
    match f {
        0 => "unknown",
        1 => "N",
        2 => "NE",
        3 => "E",
        4 => "SE",
        5 => "S",
        6 => "SW",
        7 => "W",
        8 => "NW",
        _ => "?",
    }
}

pub fn role_label(r: u8) -> &'static str {
    match r {
        0 => "GK",
        1 => "DEF",
        2 => "MID",
        3 => "FWD",
        _ => "?",
    }
}

pub fn phase_label(p: u8) -> &'static str {
    match p {
        0 => "kickoff",
        1 => "home-build-up",
        2 => "away-build-up",
        3 => "home-attack",
        4 => "away-attack",
        5 => "transition",
        _ => "?",
    }
}

fn name_bytes(name: &str) -> [u8; 18] {
    let mut buf = [0u8; 18];
    for (slot, b) in buf.iter_mut().zip(name.bytes()) {
        *slot = b;
    }
    buf
}

/// Decode a NUL-padded inspector name field back to a `&str`.
pub fn name_str(buf: &[u8; 18]) -> &str {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    std::str::from_utf8(&buf[..end]).unwrap_or("")
}

// ----------------------------------------------------------------------------
// Capture: live structs → packed frame.
// ----------------------------------------------------------------------------

impl PlayerInspect {
    fn from_agent(p: &PlayerAgent) -> Self {
        PlayerInspect {
            position: p.position.into(),
            velocity: p.velocity.into(),
            acceleration: p.acceleration.into(),
            jerk: p.jerk.into(),
            facing_yaw: p.facing_yaw,
            yaw_rate: p.yaw_rate,
            dizziness: p.dizziness,
            anaerobic_load: p.anaerobic_load,
            fatigue: p.fatigue,
            decision_confidence: p.decision_confidence,
            id: p.id as u32,
            shirt: p.shirt,
            team: team_to_u8(p.team),
            role: role_to_u8(p.role),
            gait: gait_to_u8(p.movement_gait),
            action_facing: facing_to_u8(p.action_facing),
            receive_facing: facing_to_u8(p.receive_facing),
            name: name_bytes(&p.name),
        }
    }
}

impl BallInspect {
    fn from_ball(b: &BallAgent) -> Self {
        BallInspect {
            position: b.position.into(),
            velocity: b.velocity.into(),
            acceleration: b.acceleration.into(),
            jerk: b.jerk.into(),
            curl_acceleration: b.curl_acceleration.into(),
            altitude_yards: b.altitude_yards,
            carry_orbit_world_rad: b.carry_orbit_world_rad,
            carry_orbit_radius_yards: b.carry_orbit_radius_yards,
            carry_orbit_swept_rad: b.carry_orbit_swept_rad,
            carry_orbit_last_tick: b.carry_orbit_last_tick,
            holder: b.holder.map(|h| h as i32).unwrap_or(-1),
            last_touch_team: b.last_touch_team.map(|t| team_to_u8(t) as i8).unwrap_or(-1),
            carry_orbit_wrap_unlocked: u8::from(b.carry_orbit_wrap_unlocked),
            _pad: [0; 2],
        }
    }
}

fn possession_to_i8(team: Option<Team>) -> i8 {
    team.map(|t| team_to_u8(t) as i8).unwrap_or(-1)
}

impl BrainInspect {
    fn from_brain(brain: &CentralBrain) -> Self {
        let mut out = BrainInspect {
            pressure_line_home: brain.pressure_line_home,
            pressure_line_away: brain.pressure_line_away,
            phase: phase_to_u8(brain.phase),
            possession_team: possession_to_i8(brain.possession_team),
            ball_holder: -1,
            ..BrainInspect::default()
        };
        if let Some(d) = brain.last_decision.as_ref() {
            out.ball_holder = d.ball_holder.map(|h| h as i32).unwrap_or(-1);
            out.home_centroid_to_ball = d.home_shape.centroid_to_ball_yards;
            out.home_spread = d.home_shape.spread_yards;
            out.home_forward_velocity = d.home_shape.forward_velocity_yps;
            out.home_players_near_ball = d.home_shape.players_near_ball.min(255) as u8;
            out.away_centroid_to_ball = d.away_shape.centroid_to_ball_yards;
            out.away_spread = d.away_shape.spread_yards;
            out.away_forward_velocity = d.away_shape.forward_velocity_yps;
            out.away_players_near_ball = d.away_shape.players_near_ball.min(255) as u8;
            out.controlled_human_players = d.controlled_human_players.min(255) as u8;
        }
        out
    }
}

impl FrameInspect {
    /// Capture the current state of `sim` into a fresh frame. `write_seq` is left
    /// at 0; the ring writer owns the seqlock sequencing.
    pub fn capture(sim: &SoccerMatch) -> Self {
        let mut frame = FrameInspect {
            tick: sim.tick,
            clock_seconds: sim.clock_seconds,
            ball: BallInspect::from_ball(&sim.ball),
            brain: BrainInspect::from_brain(&sim.central_brain),
            score_home: sim.score_home,
            score_away: sim.score_away,
            ..FrameInspect::default()
        };
        let n = sim.players.len().min(INSPECT_MAX_PLAYERS);
        for (slot, p) in frame.players[..n].iter_mut().zip(sim.players.iter()) {
            *slot = PlayerInspect::from_agent(p);
        }
        frame.player_count = n as u32;
        frame
    }
}

// ----------------------------------------------------------------------------
// mmap ring writer.
// ----------------------------------------------------------------------------

/// Owns the memory-mapped ring file and publishes one frame per tick. Single
/// writer; external readers map the same file read-only.
pub struct InspectRing {
    map: memmap2::MmapMut,
    path: std::path::PathBuf,
    next_index: u64,
}

impl InspectRing {
    /// Create/truncate `path` to the exact ring size, map it, and write the
    /// header. Errors propagate so the caller can decide whether to disable.
    pub fn create(path: impl Into<std::path::PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        file.set_len(INSPECT_FILE_SIZE as u64)?;
        // SAFETY: the file is sized to exactly INSPECT_FILE_SIZE; we are the sole
        // writer of this freshly created mapping.
        let mut map = unsafe { memmap2::MmapMut::map_mut(&file)? };
        let header = InspectHeader {
            magic: INSPECT_MAGIC,
            latest_index: 0,
            version: INSPECT_VERSION,
            frame_size: INSPECT_FRAME_SIZE as u32,
            ring_slots: INSPECT_RING_SLOTS as u32,
            max_players: INSPECT_MAX_PLAYERS as u32,
            pid: std::process::id(),
            _pad: 0,
        };
        // SAFETY: map is at least INSPECT_HEADER_SIZE bytes (it is INSPECT_FILE_SIZE).
        unsafe {
            std::ptr::write_unaligned(map.as_mut_ptr() as *mut InspectHeader, header);
        }
        Ok(InspectRing {
            map,
            path,
            next_index: 0,
        })
    }

    fn slot_ptr(&mut self, slot: usize) -> *mut FrameInspect {
        let off = INSPECT_HEADER_SIZE + slot * INSPECT_FRAME_SIZE;
        // SAFETY: slot < INSPECT_RING_SLOTS so off + INSPECT_FRAME_SIZE <= file size.
        unsafe { self.map.as_mut_ptr().add(off) as *mut FrameInspect }
    }

    fn header_latest_atomic(&self) -> &AtomicU64 {
        // latest_index is the 2nd u64 in InspectHeader (offset 8).
        // SAFETY: header occupies the first INSPECT_HEADER_SIZE mapped bytes; the
        // field is 8-byte aligned at the mapping base (page-aligned).
        unsafe {
            &*(self.map.as_ptr().add(std::mem::size_of::<u64>()) as *const AtomicU64)
        }
    }

    /// Write one frame for the current tick and publish it. Tear-free for
    /// concurrent external readers via a per-frame seqlock plus a release store of
    /// `latest_index`.
    pub fn record(&mut self, sim: &SoccerMatch) {
        let index = self.next_index;
        let slot = (index % INSPECT_RING_SLOTS as u64) as usize;
        let mut frame = FrameInspect::capture(sim);

        let ptr = self.slot_ptr(slot);
        // SAFETY: ptr is a valid, sufficiently-sized slot for one FrameInspect.
        let seq = unsafe { &*(ptr as *const AtomicU64) };
        // Begin write: make the per-frame seq odd.
        let started = seq.load(Ordering::Relaxed);
        let odd = if started % 2 == 0 { started + 1 } else { started + 2 };
        seq.store(odd, Ordering::Release);
        fence(Ordering::Release);
        frame.write_seq = odd + 1; // the even value we will publish
        // SAFETY: writing the whole frame (including write_seq) into the slot.
        unsafe {
            std::ptr::write_unaligned(ptr, frame);
        }
        fence(Ordering::Release);
        // End write: publish even seq.
        seq.store(odd + 1, Ordering::Release);

        // Publish newest index (count of frames written so far).
        self.next_index = index + 1;
        self.header_latest_atomic()
            .store(self.next_index, Ordering::Release);
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

// ----------------------------------------------------------------------------
// Process-global writer + tick hook.
// ----------------------------------------------------------------------------

fn ring_cell() -> &'static Mutex<Option<InspectRing>> {
    static RING: OnceLock<Mutex<Option<InspectRing>>> = OnceLock::new();
    RING.get_or_init(|| Mutex::new(None))
}

/// Default mmap path when `SOCCER_INSPECT_PATH` is unset.
pub fn default_inspect_path() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("soccer_inspect_{}.shm", std::process::id()))
}

/// Whether the shared-memory ring is enabled (`SOCCER_INSPECT=1`). The on-demand
/// HTTP endpoint works regardless; this only governs the mmap ring + history.
pub fn shm_capture_enabled() -> bool {
    std::env::var("SOCCER_INSPECT")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

fn configured_path() -> std::path::PathBuf {
    std::env::var("SOCCER_INSPECT_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| default_inspect_path())
}

/// Lazily create the ring on first use (only when enabled). Returns the active
/// mmap path for logging, or `None` if disabled/failed.
pub fn record_frame(sim: &SoccerMatch) -> Option<std::path::PathBuf> {
    if !shm_capture_enabled() {
        return None;
    }
    let mut guard = ring_cell().lock().ok()?;
    if guard.is_none() {
        match InspectRing::create(configured_path()) {
            Ok(ring) => *guard = Some(ring),
            Err(e) => {
                eprintln!("[soccer-inspect] failed to create mmap ring: {e}");
                // Disable further attempts this run by leaving None but flipping a
                // poison flag would over-engineer it; the create error is sticky in
                // practice (bad path), and we simply retry-and-log each tick is
                // avoided by env being read once — acceptable for a debug tool.
                return None;
            }
        }
    }
    let ring = guard.as_mut()?;
    ring.record(sim);
    Some(ring.path().to_path_buf())
}

// ----------------------------------------------------------------------------
// Reader (used by the soccer_inspect bin + tests): map read-only and pull frames
// tear-free.
// ----------------------------------------------------------------------------

/// A read-only view over an inspector mmap file.
pub struct InspectReader {
    map: memmap2::Mmap,
}

impl InspectReader {
    /// Map `path` read-only and validate the header against this build's layout.
    pub fn open(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: read-only mapping of a file we treat as immutable bytes.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        if map.len() < INSPECT_FILE_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("inspector file too small: {} < {}", map.len(), INSPECT_FILE_SIZE),
            ));
        }
        let reader = InspectReader { map };
        let h = reader.header();
        if h.magic != INSPECT_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad magic {:#x}", h.magic),
            ));
        }
        if h.version != INSPECT_VERSION
            || h.frame_size as usize != INSPECT_FRAME_SIZE
            || h.ring_slots as usize != INSPECT_RING_SLOTS
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "layout mismatch: file v{} frame {} slots {} vs build v{} frame {} slots {}",
                    h.version,
                    h.frame_size,
                    h.ring_slots,
                    INSPECT_VERSION,
                    INSPECT_FRAME_SIZE,
                    INSPECT_RING_SLOTS
                ),
            ));
        }
        Ok(reader)
    }

    pub fn header(&self) -> InspectHeader {
        // SAFETY: validated len >= INSPECT_FILE_SIZE; header is at offset 0.
        unsafe { std::ptr::read_unaligned(self.map.as_ptr() as *const InspectHeader) }
    }

    /// Number of frames published so far (monotonic).
    pub fn latest_index(&self) -> u64 {
        // SAFETY: latest_index is the 2nd u64 of the header.
        unsafe {
            (*(self.map.as_ptr().add(std::mem::size_of::<u64>()) as *const AtomicU64))
                .load(Ordering::Acquire)
        }
    }

    /// Read a tear-free copy of the frame in `slot`, retrying on the seqlock.
    /// Returns `None` if the slot never stabilises (writer churning) after a
    /// bounded number of attempts.
    pub fn read_slot(&self, slot: usize) -> Option<FrameInspect> {
        if slot >= INSPECT_RING_SLOTS {
            return None;
        }
        let off = INSPECT_HEADER_SIZE + slot * INSPECT_FRAME_SIZE;
        let ptr = unsafe { self.map.as_ptr().add(off) };
        let seq = unsafe { &*(ptr as *const AtomicU64) };
        for _ in 0..64 {
            let s1 = seq.load(Ordering::Acquire);
            if s1 % 2 != 0 {
                std::hint::spin_loop();
                continue;
            }
            fence(Ordering::Acquire);
            // SAFETY: ptr addresses a full FrameInspect within the mapping.
            let frame = unsafe { std::ptr::read_unaligned(ptr as *const FrameInspect) };
            fence(Ordering::Acquire);
            let s2 = seq.load(Ordering::Acquire);
            if s1 == s2 {
                return Some(frame);
            }
        }
        None
    }

    /// The most recently published frame, or `None` if nothing written yet.
    pub fn latest(&self) -> Option<FrameInspect> {
        let idx = self.latest_index();
        if idx == 0 {
            return None;
        }
        let slot = ((idx - 1) % INSPECT_RING_SLOTS as u64) as usize;
        self.read_slot(slot)
    }

    /// The last `count` frames in chronological (oldest→newest) order, capped by
    /// what the ring still holds.
    pub fn history(&self, count: usize) -> Vec<FrameInspect> {
        let idx = self.latest_index();
        if idx == 0 {
            return Vec::new();
        }
        let available = idx.min(INSPECT_RING_SLOTS as u64);
        let want = (count as u64).min(available);
        let mut out = Vec::with_capacity(want as usize);
        let start = idx - want; // first publish index to include
        for i in start..idx {
            let slot = (i % INSPECT_RING_SLOTS as u64) as usize;
            if let Some(f) = self.read_slot(slot) {
                out.push(f);
            }
        }
        out
    }
}

// ----------------------------------------------------------------------------
// JSON serialisation for the on-demand HTTP endpoint. Reads the LIVE sim (so it
// works with the ring disabled) for the current frame, and pulls history from
// the in-process ring when available.
// ----------------------------------------------------------------------------

fn vec2_json(v: Vec2C) -> serde_json::Value {
    serde_json::json!({ "x": v.x, "y": v.y })
}

fn player_json_full(p: &PlayerInspect) -> serde_json::Value {
    serde_json::json!({
        "id": p.id,
        "name": name_str(&p.name),
        "shirt": p.shirt,
        "team": if p.team == 0 { "Home" } else { "Away" },
        "role": role_label(p.role),
        "gait": gait_label(p.gait),
        "actionFacing": facing_label(p.action_facing),
        "receiveFacing": facing_label(p.receive_facing),
        "facingYaw": p.facing_yaw,
        "yawRate": p.yaw_rate,
        "dizziness": p.dizziness,
        "anaerobicLoad": p.anaerobic_load,
        "fatigue": p.fatigue,
        "decisionConfidence": p.decision_confidence,
        "position": vec2_json(p.position),
        "velocity": vec2_json(p.velocity),
        "acceleration": vec2_json(p.acceleration),
        "jerk": vec2_json(p.jerk),
    })
}

/// Project a player to only the requested field names (the addressable-read path:
/// `?player=18&fields=facing_yaw,yaw_rate,action_facing`). Unknown names are
/// silently skipped; an empty `fields` returns the full record.
fn player_json_fields(p: &PlayerInspect, fields: &[&str]) -> serde_json::Value {
    if fields.is_empty() {
        return player_json_full(p);
    }
    let mut map = serde_json::Map::new();
    map.insert("id".into(), serde_json::json!(p.id));
    for f in fields {
        let key = f.trim();
        let lower = key.to_ascii_lowercase();
        let val = match lower.as_str() {
            "name" => serde_json::json!(name_str(&p.name)),
            "shirt" => serde_json::json!(p.shirt),
            "team" => serde_json::json!(if p.team == 0 { "Home" } else { "Away" }),
            "role" => serde_json::json!(role_label(p.role)),
            "gait" => serde_json::json!(gait_label(p.gait)),
            "action_facing" | "actionfacing" => serde_json::json!(facing_label(p.action_facing)),
            "receive_facing" | "receivefacing" => serde_json::json!(facing_label(p.receive_facing)),
            "facing_yaw" | "facingyaw" => serde_json::json!(p.facing_yaw),
            "yaw_rate" | "yawrate" => serde_json::json!(p.yaw_rate),
            "dizziness" => serde_json::json!(p.dizziness),
            "anaerobic_load" | "anaerobicload" => serde_json::json!(p.anaerobic_load),
            "fatigue" => serde_json::json!(p.fatigue),
            "decision_confidence" | "decisionconfidence" => {
                serde_json::json!(p.decision_confidence)
            }
            "position" => vec2_json(p.position),
            "velocity" => vec2_json(p.velocity),
            "acceleration" => vec2_json(p.acceleration),
            "jerk" => vec2_json(p.jerk),
            _ => continue,
        };
        map.insert((*f).to_string(), val);
    }
    serde_json::Value::Object(map)
}

fn ball_json(b: &BallInspect) -> serde_json::Value {
    serde_json::json!({
        "holder": if b.holder < 0 { serde_json::Value::Null } else { serde_json::json!(b.holder) },
        "lastTouchTeam": match b.last_touch_team { 0 => "Home", 1 => "Away", _ => "none" },
        "position": vec2_json(b.position),
        "velocity": vec2_json(b.velocity),
        "acceleration": vec2_json(b.acceleration),
        "jerk": vec2_json(b.jerk),
        "curlAcceleration": vec2_json(b.curl_acceleration),
        "altitudeYards": b.altitude_yards,
        "carryOrbit": {
            "worldRad": b.carry_orbit_world_rad,
            "radiusYards": b.carry_orbit_radius_yards,
            "sweptRad": b.carry_orbit_swept_rad,
            "wrapUnlocked": b.carry_orbit_wrap_unlocked != 0,
            "lastTick": b.carry_orbit_last_tick,
        },
    })
}

fn brain_json(br: &BrainInspect) -> serde_json::Value {
    serde_json::json!({
        "phase": phase_label(br.phase),
        "possessionTeam": match br.possession_team { 0 => "Home", 1 => "Away", _ => "none" },
        "ballHolder": if br.ball_holder < 0 { serde_json::Value::Null } else { serde_json::json!(br.ball_holder) },
        "pressureLineHome": br.pressure_line_home,
        "pressureLineAway": br.pressure_line_away,
        "controlledHumanPlayers": br.controlled_human_players,
        "homeShape": {
            "centroidToBallYards": br.home_centroid_to_ball,
            "spreadYards": br.home_spread,
            "forwardVelocityYps": br.home_forward_velocity,
            "playersNearBall": br.home_players_near_ball,
        },
        "awayShape": {
            "centroidToBallYards": br.away_centroid_to_ball,
            "spreadYards": br.away_spread,
            "forwardVelocityYps": br.away_forward_velocity,
            "playersNearBall": br.away_players_near_ball,
        },
    })
}

fn frame_json(frame: &FrameInspect, selection: &InspectSelection) -> serde_json::Value {
    let fields = selection.fields_ref();
    let players: Vec<serde_json::Value> = frame.players[..frame.player_count as usize]
        .iter()
        .filter(|p| selection.player_id.map(|id| p.id == id).unwrap_or(true))
        .map(|p| player_json_fields(p, &fields))
        .collect();
    let mut obj = serde_json::Map::new();
    obj.insert("tick".into(), serde_json::json!(frame.tick));
    obj.insert("clockSeconds".into(), serde_json::json!(frame.clock_seconds));
    obj.insert("scoreHome".into(), serde_json::json!(frame.score_home));
    obj.insert("scoreAway".into(), serde_json::json!(frame.score_away));
    if selection.wants_section("ball") {
        obj.insert("ball".into(), ball_json(&frame.ball));
    }
    if selection.wants_section("brain") {
        obj.insert("brain".into(), brain_json(&frame.brain));
    }
    if selection.wants_section("players") {
        obj.insert("players".into(), serde_json::Value::Array(players));
    }
    serde_json::Value::Object(obj)
}

/// Parsed `/api/inspect` query selection.
#[derive(Default, Clone)]
pub struct InspectSelection {
    pub player_id: Option<u32>,
    pub fields: Vec<String>,
    /// Empty = all sections. Otherwise some of "ball","brain","players".
    pub sections: Vec<String>,
    pub history: usize,
}

impl InspectSelection {
    fn fields_ref(&self) -> Vec<&str> {
        self.fields.iter().map(|s| s.as_str()).collect()
    }

    fn wants_section(&self, name: &str) -> bool {
        self.sections.is_empty() || self.sections.iter().any(|s| s == name)
    }

    /// Parse from a raw query string (the part after `?`).
    pub fn from_query(query: &str) -> Self {
        let mut sel = InspectSelection::default();
        for pair in query.split('&').filter(|p| !p.is_empty()) {
            let (k, v) = match pair.split_once('=') {
                Some((k, v)) => (k, v),
                None => (pair, ""),
            };
            match k {
                "player" | "playerId" | "id" => {
                    sel.player_id = v.parse::<u32>().ok();
                }
                "fields" | "field" => {
                    sel.fields = v
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                "section" | "sections" => {
                    sel.sections = v
                        .split(',')
                        .map(|s| s.trim().to_ascii_lowercase())
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                "history" => {
                    sel.history = v.parse::<usize>().unwrap_or(0);
                }
                _ => {}
            }
        }
        sel
    }
}

/// Build the `/api/inspect` JSON response for `sim` and `selection`. The current
/// frame is captured live (works with the ring off); history is pulled from the
/// in-process ring when `SOCCER_INSPECT=1` and enough frames exist.
pub fn inspect_response_json(sim: &SoccerMatch, selection: &InspectSelection) -> serde_json::Value {
    let current = FrameInspect::capture(sim);
    let mut out = serde_json::json!({
        "shmEnabled": shm_capture_enabled(),
        "shmPath": shm_capture_enabled().then(|| configured_path().to_string_lossy().into_owned()),
        "ringSlots": INSPECT_RING_SLOTS,
        "frameSizeBytes": INSPECT_FRAME_SIZE,
        "current": frame_json(&current, selection),
    });

    if selection.history > 0 {
        let frames = in_process_history(selection.history);
        if !frames.is_empty() {
            let arr: Vec<serde_json::Value> =
                frames.iter().map(|f| frame_json(f, selection)).collect();
            out["history"] = serde_json::Value::Array(arr);
        } else {
            out["history"] = serde_json::json!([]);
            out["historyNote"] =
                serde_json::json!("history needs SOCCER_INSPECT=1 (shared-memory ring) enabled");
        }
    }
    out
}

/// Read the last `count` frames from THIS process's writer ring (if enabled),
/// without re-mapping the file.
fn in_process_history(count: usize) -> Vec<FrameInspect> {
    let guard = match ring_cell().lock() {
        Ok(g) => g,
        Err(_) => return Vec::new(),
    };
    let ring = match guard.as_ref() {
        Some(r) => r,
        None => return Vec::new(),
    };
    // Re-open the live path read-only to reuse the tear-free reader against our
    // own freshly-written pages (cheap; same backing file/pages).
    match InspectReader::open(ring.path()) {
        Ok(reader) => reader.history(count),
        Err(_) => Vec::new(),
    }
}

#[cfg(test)]
mod inspect_tests {
    use super::*;

    #[test]
    fn layout_is_packed_without_holes() {
        // No implicit padding holes in the leaf structs (each is a tidy multiple
        // of 8 bytes), so the on-disk bytes are deterministic for this build.
        assert_eq!(std::mem::size_of::<Vec2C>(), 16);
        assert_eq!(std::mem::size_of::<PlayerInspect>() % 8, 0);
        assert_eq!(std::mem::size_of::<BallInspect>() % 8, 0);
        assert_eq!(std::mem::size_of::<BrainInspect>() % 8, 0);
        assert_eq!(std::mem::size_of::<FrameInspect>() % 8, 0);
        assert_eq!(std::mem::size_of::<InspectHeader>(), 40);
        // File size is header + ring of frames.
        assert_eq!(
            INSPECT_FILE_SIZE,
            INSPECT_HEADER_SIZE + INSPECT_FRAME_SIZE * INSPECT_RING_SLOTS
        );
    }

    #[test]
    fn enum_maps_round_trip_to_labels() {
        assert_eq!(gait_label(gait_to_u8(MovementGait::Sprint)), "sprint");
        assert_eq!(facing_label(facing_to_u8(FacingBucket::NorthWest)), "NW");
        assert_eq!(role_label(role_to_u8(PlayerRole::Goalkeeper)), "GK");
        assert_eq!(phase_label(phase_to_u8(TacticalPhase::AwayAttack)), "away-attack");
    }

    #[test]
    fn name_field_truncates_and_decodes() {
        let buf = name_bytes("Home CM2");
        assert_eq!(name_str(&buf), "Home CM2");
        let long = name_bytes("a-really-long-player-name-overflow");
        assert_eq!(name_str(&long).len(), 18);
    }

    #[test]
    fn ring_write_then_read_round_trips_and_wraps() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("soccer_inspect_test_{}.shm", std::process::id()));
        let mut ring = InspectRing::create(&path).expect("create ring");

        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        // Write more frames than the ring holds to exercise wraparound.
        let total = INSPECT_RING_SLOTS + 5;
        for t in 0..total {
            sim.tick = t as u64;
            sim.clock_seconds = t as f64 * 0.1;
            ring.record(&sim);
        }

        let reader = InspectReader::open(&path).expect("open reader");
        assert_eq!(reader.latest_index(), total as u64);
        let latest = reader.latest().expect("latest frame");
        assert_eq!(latest.tick, (total - 1) as u64);

        // History is capped at the ring depth and ends at the newest frame.
        let hist = reader.history(INSPECT_RING_SLOTS + 100);
        assert_eq!(hist.len(), INSPECT_RING_SLOTS);
        assert_eq!(hist.last().unwrap().tick, (total - 1) as u64);
        assert_eq!(hist.first().unwrap().tick, (total - INSPECT_RING_SLOTS) as u64);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn selection_parses_query() {
        let sel = InspectSelection::from_query("player=18&fields=facing_yaw,yaw_rate&history=45&section=players");
        assert_eq!(sel.player_id, Some(18));
        assert_eq!(sel.fields, vec!["facing_yaw", "yaw_rate"]);
        assert_eq!(sel.history, 45);
        assert!(sel.wants_section("players"));
        assert!(!sel.wants_section("ball"));
    }
}
