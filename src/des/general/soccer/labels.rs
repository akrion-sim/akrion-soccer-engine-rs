//! Canonical soccer *decision-label* vocabulary.
//!
//! The MDP/POMDP policy, the learning store (Postgres), the neural action
//! hashing and the frame-diff trackers all describe a chosen action with a
//! short string label (`"pass"`, `"slide-tackle"`, `"recycle-reset"`, ...).
//! Historically those strings were sprinkled through `soccer.rs` as bare
//! literals, and a sprawling `normalize_soccer_action_label` match folded the
//! many aliases (`"give-and-go"`, `"one_two_pass"`, ...) onto a canonical form.
//!
//! [`SoccerActionLabel`] makes that vocabulary a real type while preserving the
//! exact on-the-wire strings, so stored policies and traces are byte-for-byte
//! compatible:
//!
//! * [`SoccerActionLabel::as_str`] yields the canonical string.
//! * [`SoccerActionLabel::canonical_str`] is the alias→canonical table and is a
//!   drop-in for the old `normalize_soccer_action_label` (alloc-free; unknown
//!   labels pass straight through, identical to the historical `other => other`
//!   arm).
//! * [`SoccerActionLabel::from_label`] parses any alias/canonical string into a
//!   variant; genuinely unknown labels are preserved losslessly in
//!   [`SoccerActionLabel::Other`].
//! * `serde` serializes/deserializes as the canonical string.
//!
//! The vocabulary is intentionally *open*: `Other` guarantees correctness even
//! for labels not yet promoted to a named variant, so producers can be migrated
//! to the enum incrementally without risk of dropping information.

use std::fmt;
use std::str::FromStr;

use serde::de::Deserializer;
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};

/// Frequently shared canonical action labels. Keep these next to the action
/// vocabulary so hot-path callers do not need their own local string copies.
pub const ACTION_LABEL_RECOVER: &str = "recover";
pub const FIFTY_FIFTY_DUEL_ACTION_LABEL: &str = "fifty-fifty-duel";
pub const DUMMY_CLEAR_LANE_ACTION_LABEL: &str = "dummy-clear-lane";
pub const DUMMY_LET_RUN_ACTION_LABEL: &str = "dummy-let-run";
pub const OPEN_PASS_LANE_ACTION_LABEL: &str = "open-pass-lane";
pub const ROUND_GOALKEEPER_ACTION_LABEL: &str = "round-goalkeeper";

/// Shared operation/trace labels that are not policy actions themselves.
pub const TRACE_REACTIVE_GROUND_PASS: &str = "reactive-ground-pass";
pub const TRACE_TRAP_CONTROLLABLE_TRAJECTORY: &str = "trap-controllable-trajectory";

/// A soccer decision-label: the canonical vocabulary the policy/learning/trace
/// layers speak in. See the module docs for the round-trip guarantees.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SoccerActionLabel {
    // --- on-ball: passing family ---
    Pass,
    AerialPass,
    WallPass,
    KillerPass,
    SurprisePass,
    ScoopPass,
    SwitchPlay,
    RecycleReset,
    FlickOn,
    FirstTimePass,
    CornerFlagCross,
    FlankLowCross,
    FlankHighCross,
    RouteOne,
    LongBallFlight,
    Clearance,
    // --- on-ball: shooting / first-touch ---
    Shoot,
    FirstTimeShot,
    FirstTimeHeader,
    ControlTouch,
    // --- on-ball: dribbling family ---
    Dribble,
    CarryForward,
    CarryOutLeft,
    CarryOutRight,
    SideStep,
    HoldUpFlank,
    ProtectBall,
    LeftCut,
    RightCut,
    Nutmeg,
    FakeLeftCutRight,
    FakeRightCutLeft,
    /// The "Xavi turn": a ~280-300° shielded pirouette that keeps the ball on the far
    /// side of the defender and wheels the long way around to turn them.
    XaviTurn,
    /// A short, quick (sprinting) 1-4yd carry to shift a blocked passing lane off the
    /// opponent in it and open the pass to a teammate. See
    /// `WorldSnapshot::dribble_to_open_passing_lane_for`.
    OpenPassingLane,
    VerticalAttack,
    TurnoverBurst,
    /// On-ball: the carrier deliberately keeps the ball and *signals* teammates to
    /// improve their position before releasing (delay-before-pass). See
    /// `WorldSnapshot::hold_for_support_option_for`.
    WaitForSupport,
    /// On-ball: a short carrier dribble to change the passing angle and open a
    /// currently-blocked teammate lane.
    OpenPassLane,
    /// On-ball: a short, quick carry around the goalkeeper — closer to goal and off the
    /// keeper's covered angle — to open a clear shot, instead of a long shot the keeper
    /// saves. Unifies the two convergent "round the keeper" carries onto one label /
    /// implementation (`round_goalkeeper_plan` + `RoundGoalkeeperDribblePlan`).
    RoundGoalkeeper,
    /// On-ball: an outside/wing carrier knocks the ball past the immediate defender
    /// and sprints around to re-collect when there is no close covering defender.
    RunaroundDribble,
    // --- off-ball: support / movement ---
    Space,
    SupportShape,
    SupportRoam,
    SupportScreen,
    CheckToBall,
    RunInBehind,
    OneTwoRun,
    ExploitSpaceRun,
    VacateSpace,
    /// Off-ball: a same-team middle player recognises it is blocking the longer
    /// lane between two teammates and moves into space so the longer pass stays on.
    DummyClearLane,
    /// Off-ball/in-flight: the same middle-player read, but expressed as a dummy
    /// through decision — do not trap the pass; let it continue to the outer runner.
    DummyLetRun,
    WideOutlet,
    ShotCreationRun,
    OverlapRun,
    SetPlayRun,
    // --- defensive / utility ---
    Defend,
    DefendShape,
    DefendRoam,
    Tackle,
    SlideTackle,
    PressCover,
    Recover,
    Hold,
    /// Any label not (yet) promoted to a named variant. Preserves the exact
    /// string so round-tripping through the enum is lossless — mirrors the
    /// historical `normalize_soccer_action_label` `other => other` arm.
    Other(Box<str>),
}

impl SoccerActionLabel {
    /// The canonical wire string for this label. Stable across versions: this is
    /// what gets stored in Postgres and emitted in traces.
    pub fn as_str(&self) -> &str {
        match self {
            SoccerActionLabel::Pass => "pass",
            SoccerActionLabel::AerialPass => "aerial-pass",
            SoccerActionLabel::WallPass => "wall-pass",
            SoccerActionLabel::KillerPass => "killer-pass",
            SoccerActionLabel::SurprisePass => "surprise-pass",
            SoccerActionLabel::ScoopPass => "scoop-pass",
            SoccerActionLabel::SwitchPlay => "switch-play",
            SoccerActionLabel::RecycleReset => "recycle-reset",
            SoccerActionLabel::FlickOn => "flick-on",
            SoccerActionLabel::FirstTimePass => "first-time-pass",
            SoccerActionLabel::CornerFlagCross => "corner-flag-cross",
            SoccerActionLabel::FlankLowCross => "flank-low-cross",
            SoccerActionLabel::FlankHighCross => "flank-high-cross",
            SoccerActionLabel::RouteOne => "route-one",
            SoccerActionLabel::LongBallFlight => "long-ball-flight",
            SoccerActionLabel::Clearance => "clearance",
            SoccerActionLabel::Shoot => "shoot",
            SoccerActionLabel::FirstTimeShot => "first-time-shot",
            SoccerActionLabel::FirstTimeHeader => "first-time-header",
            SoccerActionLabel::ControlTouch => "control-touch",
            SoccerActionLabel::Dribble => "dribble",
            SoccerActionLabel::CarryForward => "carry-forward",
            SoccerActionLabel::CarryOutLeft => "carry-out-left",
            SoccerActionLabel::CarryOutRight => "carry-out-right",
            SoccerActionLabel::SideStep => "side-step",
            SoccerActionLabel::HoldUpFlank => "hold-up-flank",
            SoccerActionLabel::ProtectBall => "protect-ball",
            SoccerActionLabel::LeftCut => "left-cut",
            SoccerActionLabel::RightCut => "right-cut",
            SoccerActionLabel::Nutmeg => "nutmeg",
            SoccerActionLabel::XaviTurn => "xavi-turn",
            SoccerActionLabel::OpenPassingLane => "open-passing-lane",
            SoccerActionLabel::FakeLeftCutRight => "fake-left-cut-right",
            SoccerActionLabel::FakeRightCutLeft => "fake-right-cut-left",
            SoccerActionLabel::VerticalAttack => "vertical-attack",
            SoccerActionLabel::TurnoverBurst => "turnover-burst",
            SoccerActionLabel::WaitForSupport => "wait-for-support",
            SoccerActionLabel::OpenPassLane => "open-pass-lane",
            SoccerActionLabel::RoundGoalkeeper => "round-goalkeeper",
            SoccerActionLabel::RunaroundDribble => "runaround-dribble",
            SoccerActionLabel::Space => "space",
            SoccerActionLabel::SupportShape => "support-shape",
            SoccerActionLabel::SupportRoam => "support-roam",
            SoccerActionLabel::SupportScreen => "support-screen",
            SoccerActionLabel::CheckToBall => "check-to-ball",
            SoccerActionLabel::RunInBehind => "run-in-behind",
            SoccerActionLabel::OneTwoRun => "one-two-run",
            SoccerActionLabel::ExploitSpaceRun => "exploit-space-run",
            SoccerActionLabel::VacateSpace => "vacate-space",
            SoccerActionLabel::DummyClearLane => "dummy-clear-lane",
            SoccerActionLabel::DummyLetRun => "dummy-let-run",
            SoccerActionLabel::WideOutlet => "wide-outlet",
            SoccerActionLabel::ShotCreationRun => "shot-creation-run",
            SoccerActionLabel::OverlapRun => "overlap-run",
            SoccerActionLabel::SetPlayRun => "set-play-run",
            SoccerActionLabel::Defend => "defend",
            SoccerActionLabel::DefendShape => "defend-shape",
            SoccerActionLabel::DefendRoam => "defend-roam",
            SoccerActionLabel::Tackle => "tackle",
            SoccerActionLabel::SlideTackle => "slide-tackle",
            SoccerActionLabel::PressCover => "press-cover",
            SoccerActionLabel::Recover => "recover",
            SoccerActionLabel::Hold => "hold",
            SoccerActionLabel::Other(s) => s,
        }
    }

    /// Map a canonical string back to its variant. `None` if the string is not a
    /// recognised canonical label (callers fall back to [`SoccerActionLabel::Other`]).
    fn from_canonical_str(canonical: &str) -> Option<SoccerActionLabel> {
        let label = match canonical {
            "pass" => SoccerActionLabel::Pass,
            "aerial-pass" => SoccerActionLabel::AerialPass,
            "wall-pass" => SoccerActionLabel::WallPass,
            "killer-pass" => SoccerActionLabel::KillerPass,
            "surprise-pass" => SoccerActionLabel::SurprisePass,
            "scoop-pass" => SoccerActionLabel::ScoopPass,
            "switch-play" => SoccerActionLabel::SwitchPlay,
            "recycle-reset" => SoccerActionLabel::RecycleReset,
            "flick-on" => SoccerActionLabel::FlickOn,
            "first-time-pass" => SoccerActionLabel::FirstTimePass,
            "corner-flag-cross" => SoccerActionLabel::CornerFlagCross,
            "flank-low-cross" => SoccerActionLabel::FlankLowCross,
            "flank-high-cross" => SoccerActionLabel::FlankHighCross,
            "route-one" => SoccerActionLabel::RouteOne,
            "long-ball-flight" => SoccerActionLabel::LongBallFlight,
            "clearance" => SoccerActionLabel::Clearance,
            "shoot" => SoccerActionLabel::Shoot,
            "first-time-shot" => SoccerActionLabel::FirstTimeShot,
            "first-time-header" => SoccerActionLabel::FirstTimeHeader,
            "control-touch" => SoccerActionLabel::ControlTouch,
            "dribble" => SoccerActionLabel::Dribble,
            "carry-forward" => SoccerActionLabel::CarryForward,
            "carry-out-left" => SoccerActionLabel::CarryOutLeft,
            "carry-out-right" => SoccerActionLabel::CarryOutRight,
            "side-step" => SoccerActionLabel::SideStep,
            "hold-up-flank" => SoccerActionLabel::HoldUpFlank,
            "protect-ball" => SoccerActionLabel::ProtectBall,
            "left-cut" => SoccerActionLabel::LeftCut,
            "right-cut" => SoccerActionLabel::RightCut,
            "nutmeg" => SoccerActionLabel::Nutmeg,
            "fake-left-cut-right" => SoccerActionLabel::FakeLeftCutRight,
            "fake-right-cut-left" => SoccerActionLabel::FakeRightCutLeft,
            "xavi-turn" => SoccerActionLabel::XaviTurn,
            "open-passing-lane" => SoccerActionLabel::OpenPassingLane,
            "vertical-attack" => SoccerActionLabel::VerticalAttack,
            "turnover-burst" => SoccerActionLabel::TurnoverBurst,
            "wait-for-support" => SoccerActionLabel::WaitForSupport,
            "open-pass-lane" => SoccerActionLabel::OpenPassLane,
            "round-goalkeeper" => SoccerActionLabel::RoundGoalkeeper,
            "runaround-dribble" => SoccerActionLabel::RunaroundDribble,
            "space" => SoccerActionLabel::Space,
            "support-shape" => SoccerActionLabel::SupportShape,
            "support-roam" => SoccerActionLabel::SupportRoam,
            "support-screen" => SoccerActionLabel::SupportScreen,
            "check-to-ball" => SoccerActionLabel::CheckToBall,
            "run-in-behind" => SoccerActionLabel::RunInBehind,
            "one-two-run" => SoccerActionLabel::OneTwoRun,
            "exploit-space-run" => SoccerActionLabel::ExploitSpaceRun,
            "vacate-space" => SoccerActionLabel::VacateSpace,
            "dummy-clear-lane" => SoccerActionLabel::DummyClearLane,
            "dummy-let-run" => SoccerActionLabel::DummyLetRun,
            "wide-outlet" => SoccerActionLabel::WideOutlet,
            "shot-creation-run" => SoccerActionLabel::ShotCreationRun,
            "overlap-run" => SoccerActionLabel::OverlapRun,
            "set-play-run" => SoccerActionLabel::SetPlayRun,
            "defend" => SoccerActionLabel::Defend,
            "defend-shape" => SoccerActionLabel::DefendShape,
            "defend-roam" => SoccerActionLabel::DefendRoam,
            "tackle" => SoccerActionLabel::Tackle,
            "slide-tackle" => SoccerActionLabel::SlideTackle,
            "press-cover" => SoccerActionLabel::PressCover,
            "recover" => SoccerActionLabel::Recover,
            "hold" => SoccerActionLabel::Hold,
            _ => return None,
        };
        Some(label)
    }

    /// Alias→canonical normalizer. This is the single source of truth for the
    /// label aliasing that used to live in `normalize_soccer_action_label`.
    ///
    /// Returns `Some(canonical)` (a `'static` string) when `action` is a known
    /// alias *or* canonical label, and `None` when it is unrecognised — letting
    /// callers pass the original `&str` straight through with no allocation,
    /// exactly as the historical `other => other` arm did.
    pub fn canonical_str(action: &str) -> Option<&'static str> {
        let canonical = match action {
            "move" => "space",
            "pass1" | "pass2" | "pass3" => "pass",
            "aerial-pass1" | "aerial-pass2" | "aerial-pass3" => "aerial-pass",
            "wall-pass" | "wall_pass" | "wallpass" | "give-and-go" | "give_and_go" | "giveandgo"
            | "one-two-pass" | "one_two_pass" | "onetwopass" => "wall-pass",
            "corner-flag-cross"
            | "corner_flag_cross"
            | "cornerflagcross"
            | "byline-cross"
            | "byline_cross"
            | "bylinecross"
            | "penalty-spot-cross"
            | "penalty_spot_cross"
            | "penaltyspotcross" => "corner-flag-cross",
            "vertical-attack"
            | "vertical_attack"
            | "verticalattack"
            | "support-push-up"
            | "supportpushup"
            | "support_push_up"
            | "pushup"
            | "push-up"
            | "attack-vertical"
            | "attack_vertical"
            | "drive-forward"
            | "drive_forward"
            | "driveforward" => "vertical-attack",
            "vacate-space"
            | "vacate_space"
            | "vacatespace"
            | "create-space-run"
            | "create_space_run"
            | "createspacerun"
            | "decoy-run"
            | "decoy_run"
            | "decoyrun" => "vacate-space",
            "surprise-pass"
            | "surprise_pass"
            | "surprisepass"
            | "backheel-pass"
            | "backheel_pass"
            | "backheelpass"
            | "backheel" => "surprise-pass",
            "scoop-pass"
            | "scoop_pass"
            | "scooppass"
            | "lob"
            | "lob-pass"
            | "lob_pass"
            | "lobpass"
            | "chip-pass"
            | "chip_pass"
            | "chippass" => "scoop-pass",
            "flick-on" | "flick_on" | "flickon" | "header-flick" | "header_flick" => "flick-on",
            "wait-for-support"
            | "wait_for_support"
            | "waitforsupport"
            | "hold-and-summon"
            | "hold_and_summon"
            | "holdandsummon"
            | "summon-support"
            | "summon_support"
            | "delay-pass"
            | "delay_pass"
            | "delaypass" => "wait-for-support",
            "open-pass-lane"
            | "open_pass_lane"
            | "openpasslane"
            | "pass-lane-dribble"
            | "pass_lane_dribble"
            | "passlanedribble"
            | "dribble-to-pass"
            | "dribble_to_pass"
            | "dribbletopass" => "open-pass-lane",
            // Unified vocabulary: both teams' alias spellings (ours "round-the-keeper" et al.
            // and theirs "round-goalkeeper" et al.) normalize to the single canonical label so
            // any persisted/external label from either side still resolves.
            "round-goalkeeper"
            | "round_goalkeeper"
            | "roundgoalkeeper"
            | "round-the-goalkeeper"
            | "round_the_goalkeeper"
            | "roundthegoalkeeper"
            | "round-gk"
            | "round_gk"
            | "beat-keeper"
            | "beat_keeper"
            | "dribble-around-keeper"
            | "dribble_around_keeper"
            | "round-the-keeper"
            | "round_the_keeper"
            | "roundthekeeper"
            | "round-the-gk"
            | "round-keeper"
            | "round_keeper"
            | "rounding-the-keeper"
            | "dribble-round-keeper"
            | "beat-the-keeper" => "round-goalkeeper",
            "runaround-dribble"
            | "runaround_dribble"
            | "runarounddribble"
            | "run-around-dribble"
            | "run_around_dribble"
            | "runaround"
            | "outside-mid-attack-defender"
            | "outside_mid_attack_defender"
            | "outsidemidattackdefender"
            | "wing-attack-defender"
            | "wing_attack_defender"
            | "wingattackdefender"
            | "attack-defender"
            | "attack_defender"
            | "attackdefender" => "runaround-dribble",
            "turnover-burst"
            | "turnover_burst"
            | "turnoverburst"
            | "transition-burst"
            | "transition_burst"
            | "counter-burst"
            | "counter_burst"
            | "counterburst" => "turnover-burst",
            "switch"
            | "switch_play"
            | "switchplay"
            | "side-switch"
            | "side_switch"
            | "cross-field-pass"
            | "cross_field_pass"
            | "crossfieldpass" => "switch-play",
            "recycle"
            | "recycle_reset"
            | "recyclereset"
            | "reset-pass"
            | "reset_pass"
            | "short-reset"
            | "short_reset"
            | "backward-reset"
            | "backward_reset" => "recycle-reset",
            "press-cover" | "press_cover" | "presscover" | "cover-press" | "cover_press" => {
                "press-cover"
            }
            "slide-tackle" | "slide_tackle" | "slidetackle" | "slide" | "sliding-tackle"
            | "sliding_tackle" | "slide-challenge" => "slide-tackle",
            "killer" | "killer_pass" | "killerpass" | "through-pass" | "through_pass"
            | "throughpass" | "threaded" | "threaded-pass" | "threaded_pass" | "threadedpass" => {
                "killer-pass"
            }
            "low-cross"
            | "low_cross"
            | "lowcross"
            | "flank-low-cross"
            | "flank_low_cross"
            | "flanklowcross"
            | "down-flank-low-cross"
            | "down_flank_low_cross"
            | "downflanklowcross"
            | "play-down-flank-low-cross"
            | "play_down_flank_low_cross"
            | "playdownflanklowcross" => "flank-low-cross",
            "high-cross"
            | "high_cross"
            | "highcross"
            | "cross-for-header"
            | "cross_for_header"
            | "crossforheader"
            | "high-cross-header"
            | "high_cross_header"
            | "highcrossheader"
            | "flank-high-cross"
            | "flank_high_cross"
            | "flankhighcross"
            | "flank-high-cross-header"
            | "flank_high_cross_header"
            | "flankhighcrossheader"
            | "down-flank-high-cross"
            | "down_flank_high_cross"
            | "downflankhighcross"
            | "play-down-flank-high-cross"
            | "play_down_flank_high_cross"
            | "playdownflankhighcross" => "flank-high-cross",
            "route1" | "route-1" | "long-ball" | "longball" => "route-one",
            "hoof" | "hoofed-clearance" => "clearance",
            "header" => "first-time-header",
            "one-touch-shot" | "one_touch_shot" | "onetouchshot" | "1-touch-shot" | "1touchshot"
            | "first-touch-shot" | "first_touch_shot" | "firsttouchshot" => "first-time-shot",
            "one-touch-pass" | "one_touch_pass" | "onetouchpass" | "1-touch-pass" | "1touchpass"
            | "first-touch-pass" | "first_touch_pass" | "firsttouchpass" => "first-time-pass",
            "chest-control" => "control-touch",
            "sidestep" | "side_step" | "side-step-dribble" | "sidestep-dribble" => "side-step",
            "hold-up" | "holdup" | "hold_up" | "hold-up-dribble" | "hold-up-flank-dribble" => {
                "hold-up-flank"
            }
            "carry"
            | "carryforward"
            | "carry_forward"
            | "carry-forward-dribble"
            | "dribble-forward" => "carry-forward",
            "carry-out" | "carryout" | "carry_out" | "carry-out-dribble" | "dribble-out" => {
                "carry-out-left"
            }
            "carryoutleft" | "carry_out_left" | "carry-left" | "carry_left" => "carry-out-left",
            "carryoutright" | "carry_out_right" | "carry-right" | "carry_right" => "carry-out-right",
            "protect" | "protectball" | "protect_ball" | "shield" | "shield-ball" | "shield_ball"
            | "body-shield" => "protect-ball",
            "leftcut" | "left_cut" | "left-cut-dribble" | "cut-left" => "left-cut",
            "rightcut" | "right_cut" | "right-cut-dribble" | "cut-right" => "right-cut",
            "nut-meg" | "nut_meg" | "meg" => "nutmeg",
            "xavi"
            | "xaviturn"
            | "xavi_turn"
            | "xavi-turn"
            | "xavi-turn-dribble"
            | "xavi_turn_dribble"
            | "la-pelopina"
            | "la_pelopina"
            | "shielded-turn"
            | "shielded_turn"
            | "turn-and-shield" => "xavi-turn",
            "open_passing_lane"
            | "open-lane"
            | "open_lane"
            | "open-passing-lane-dribble"
            | "create-passing-lane"
            | "create-passing-angle"
            | "dribble-open-lane" => "open-passing-lane",
            "fake_left_cut_right" | "fake-left-right" | "fake-left-cut-right-dribble" => {
                "fake-left-cut-right"
            }
            "fake_right_cut_left" | "fake-right-left" | "fake-right-cut-left-dribble" => {
                "fake-right-cut-left"
            }
            "supportshape" | "support_shape" => "support-shape",
            "supportroam" | "support_roam" => "support-roam",
            "checktoball" | "check_to_ball" | "check-ball" | "checkrun" => "check-to-ball",
            "runinbehind" | "run_in_behind" | "inbehind" | "in-behind" => "run-in-behind",
            "exploitspace"
            | "exploit_space"
            | "exploit-space"
            | "exploitspacerun"
            | "exploit_space_run"
            | "space-run"
            | "space_run" => "exploit-space-run",
            "wideoutlet" | "wide_outlet" | "touchlineoutlet" | "touchline-outlet" => "wide-outlet",
            "shotcreationrun" | "shot_creation_run" | "shootingrun" | "shooting-run" => {
                "shot-creation-run"
            }
            "overlap" | "overlaprun" | "overlap_run" => "overlap-run",
            "supportscreen" | "support_screen" | "screenrun" | "screen-run" => "support-screen",
            _ => return None,
        };
        Some(canonical)
    }

    /// Classify a label string into a *known* variant, allocation-free.
    /// Returns `None` for an unrecognised label (so hot-path `match` sites can
    /// fall through to a default without constructing an owned
    /// [`SoccerActionLabel::Other`]). This is the right entry point for
    /// classifier predicates that used to `match normalize_soccer_action_label(..)`.
    pub fn classify(action: &str) -> Option<SoccerActionLabel> {
        let canonical = SoccerActionLabel::canonical_str(action).unwrap_or(action);
        SoccerActionLabel::from_canonical_str(canonical)
    }

    /// Parse any alias or canonical label string into a variant. Unrecognised
    /// labels are preserved verbatim in [`SoccerActionLabel::Other`].
    pub fn from_label(action: &str) -> SoccerActionLabel {
        let canonical = SoccerActionLabel::canonical_str(action).unwrap_or(action);
        SoccerActionLabel::classify(action)
            .unwrap_or_else(|| SoccerActionLabel::Other(canonical.into()))
    }
}

impl fmt::Display for SoccerActionLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SoccerActionLabel {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(SoccerActionLabel::from_label(s))
    }
}

impl Serialize for SoccerActionLabel {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for SoccerActionLabel {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Ok(SoccerActionLabel::from_label(&raw))
    }
}

/// The restart / set-play event vocabulary. This is a **separate** vocabulary
/// from [`SoccerActionLabel`]: the strings (`"throw-in"`, `"goal-kick"`,
/// `"free-kick"`, ...) describe how play is restarted, not a chosen on/off-ball
/// action, and they are matched independently across the engine. The canonical
/// strings are stable (stored/compared as-is).
///
/// Note the engine historically treats the free-kick variants inconsistently
/// (some sites recognise only `free-kick`, others all three), so conversions
/// stay 1:1 per call site rather than collapsing the variants — see
/// [`RestartLabel::is_free_kick_family`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RestartLabel {
    Kickoff,
    ThrowIn,
    GoalKick,
    CornerKick,
    FreeKick,
    DirectFreeKick,
    IndirectFreeKick,
    Offside,
}

impl RestartLabel {
    /// The canonical wire string for this restart label.
    pub fn as_str(&self) -> &'static str {
        match self {
            RestartLabel::Kickoff => "kickoff",
            RestartLabel::ThrowIn => "throw-in",
            RestartLabel::GoalKick => "goal-kick",
            RestartLabel::CornerKick => "corner-kick",
            RestartLabel::FreeKick => "free-kick",
            RestartLabel::DirectFreeKick => "direct-free-kick",
            RestartLabel::IndirectFreeKick => "indirect-free-kick",
            RestartLabel::Offside => "offside",
        }
    }

    /// Parse an exact canonical restart string into a variant. Returns `None`
    /// for anything else (no alias folding — restart labels have no aliases in
    /// the action vocabulary, so this matches the historical exact-string
    /// behaviour). CSV-header aliasing stays in `canonical_set_play_restart_label`.
    pub fn from_canonical(label: &str) -> Option<RestartLabel> {
        let restart = match label {
            "kickoff" => RestartLabel::Kickoff,
            "throw-in" => RestartLabel::ThrowIn,
            "goal-kick" => RestartLabel::GoalKick,
            "corner-kick" => RestartLabel::CornerKick,
            "free-kick" => RestartLabel::FreeKick,
            "direct-free-kick" => RestartLabel::DirectFreeKick,
            "indirect-free-kick" => RestartLabel::IndirectFreeKick,
            "offside" => RestartLabel::Offside,
            _ => return None,
        };
        Some(restart)
    }

    /// Whether this is any of the free-kick variants (direct/indirect/plain).
    pub fn is_free_kick_family(&self) -> bool {
        matches!(
            self,
            RestartLabel::FreeKick | RestartLabel::DirectFreeKick | RestartLabel::IndirectFreeKick
        )
    }
}

impl fmt::Display for RestartLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_round_trips_through_enum() {
        for canonical in [
            "pass",
            "slide-tackle",
            "recycle-reset",
            "flank-high-cross",
            "exploit-space-run",
            "dummy-clear-lane",
            "dummy-let-run",
            "hold",
            "defend",
            "space",
            "xavi-turn",
        ] {
            let label = SoccerActionLabel::from_label(canonical);
            assert_eq!(label.as_str(), canonical, "round-trip for {canonical}");
        }
    }

    #[test]
    fn aliases_fold_onto_canonical() {
        assert_eq!(SoccerActionLabel::from_label("give-and-go").as_str(), "wall-pass");
        assert_eq!(SoccerActionLabel::from_label("hoof").as_str(), "clearance");
        assert_eq!(SoccerActionLabel::from_label("move").as_str(), "space");
        assert_eq!(SoccerActionLabel::from_label("through-pass").as_str(), "killer-pass");
        assert_eq!(SoccerActionLabel::from_label("xavi_turn").as_str(), "xavi-turn");
    }

    #[test]
    fn unknown_labels_are_preserved_losslessly() {
        let label = SoccerActionLabel::from_label("some-future-label");
        assert_eq!(label, SoccerActionLabel::Other("some-future-label".into()));
        assert_eq!(label.as_str(), "some-future-label");
    }

    #[test]
    fn canonical_str_matches_passthrough_semantics() {
        // Known alias -> canonical 'static string.
        assert_eq!(SoccerActionLabel::canonical_str("giveandgo"), Some("wall-pass"));
        // Unknown -> None, so the historical `unwrap_or(input)` passes it through.
        assert_eq!(SoccerActionLabel::canonical_str("brand-new"), None);
        // Bare canonical that had no explicit alias arm also passes through.
        assert_eq!(SoccerActionLabel::canonical_str("hold"), None);
    }
}
