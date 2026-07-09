# shellcheck shell=bash
# all-gates-on.sh — turn EVERY DD_SOCCER_* feature gate ON.
#
#   source scripts/all-gates-on.sh      # exports the vars into your shell
#
# "On" = set every DD_SOCCER_ENABLE_* gate (off-by-default features) plus the two
# bare boolean gates, and leave every DD_SOCCER_DISABLE_* gate UNSET (those guard
# features that are already ON by default — setting them would turn features OFF,
# the opposite of what we want).
#
# ⚠ READ BEFORE USING IN TRAINING/SERVING ⚠
#  1. RETRAIN REQUIRED. Several gates change the neural FEATURE ENCODING
#     (ball-zone scale, player x/y, assigned-position embedding, field-numbers) or
#     what the policy OBSERVES (perception noise / occlusion), and many change the
#     REWARD. A net trained without them will mismatch its inputs/objective — these
#     are for a FRESH/continued training lineage, not a frozen-inference server.
#  2. SERVING MUST MATCH. Once weights are trained with these on, the live/serving
#     process must export the SAME encoding/perception gates or it feeds the net
#     differently than it was trained.
#  3. KITCHEN-SINK CAVEAT. These ~48 gates are independent experiments meant for
#     one-at-a-time A/B against Elo. All-on at once is an UNVALIDATED combination
#     and may regress play; measure Elo/win-rate, and prefer enabling in waves.
#  4. NEW SLUG. Point a training run at a NEW SOCCER_EXPERIMENT_SLUG so this does
#     not clobber an existing trained lineage (e.g. soccer-self-play-k8s-overnight).
#  5. Some new gates only exist on code that includes this session's changes; on
#     older builds the unknown vars are harmless no-ops.

# --- Observation / perception / encoding (retrain-critical) ---
export DD_SOCCER_ENABLE_BALL_ZONE_TACTICAL_SCALE=true
export DD_SOCCER_ENABLE_PLAYER_GRID_XY_FEATURES=true
export DD_SOCCER_ENABLE_ASSIGNED_POSITION_EMBEDDING=true
export DD_SOCCER_ENABLE_PERCEPTION_NOISE=true
export DD_SOCCER_ENABLE_OCCLUSION=true
export DD_SOCCER_ENABLE_HEAD_SCAN=true
export DD_SOCCER_OPPONENT_BELIEF=true

# --- Reward shaping / training objective ---
export DD_SOCCER_ENABLE_SHAPING_DISCIPLINE=true
export DD_SOCCER_ENABLE_PITCH_VALUE_REWARD=true
export DD_SOCCER_ENABLE_MATCH_OUTCOME_REWARD=true
export DD_SOCCER_OUTCOME_CREDIT=true
export DD_SOCCER_ENABLE_ADVANTAGE_NORMALIZATION=true
export DD_SOCCER_ENABLE_XT_TERMINAL_COST=true
export DD_SOCCER_ENABLE_WASTED_ENERGY_PENALTY=true
export DD_SOCCER_ENABLE_KEEPER_SAVE_REWARD=true
export DD_SOCCER_ENABLE_SUSTAINED_EFFORT_NO_OUTCOME_PENALTY=true
export DD_SOCCER_ENABLE_OVERLOAD_WEIGHTED_PROGRESSION=true
export DD_SOCCER_ENABLE_MARL_BALANCED_TEAM_COMPONENT=true
export DD_SOCCER_ENABLE_SPECIALIST_CURRICULUM=true
export SOCCER_NEURAL_TARGET_POPART=true
export DD_SOCCER_ENABLE_TARGET_STANDARDIZATION=true
export DD_SOCCER_ENABLE_MC_CRITIC_TARGET=true
export DD_SOCCER_ENABLE_NEURAL_SELF_BOOTSTRAP=true
export DD_SOCCER_ENABLE_MAXA_BOOTSTRAP=true
export DD_SOCCER_ENABLE_NOVELTY_BONUS=true
export DD_SOCCER_FORWARD_PASS_CLIMB_CURRICULUM=true
export SOCCER_EVAL_REQUIRE_FORWARD_PASS_CLIMB=true
export SOCCER_EVAL_MIN_FORWARD_PASS_MARGIN=0
export SOCCER_EVAL_MIN_NET_FORWARD_PASS_MARGIN=0
export SOCCER_EVAL_MIN_FORWARD_PASS_RATE_MARGIN=0.0
export SOCCER_NEURAL_POPULATION_REQUIRE_FORWARD_PASS_CLIMB=true
export SOCCER_NEURAL_POPULATION_MIN_FORWARD_PASS_MARGIN=0
export SOCCER_NEURAL_POPULATION_MIN_NET_FORWARD_PASS_MARGIN=0
export SOCCER_NEURAL_POPULATION_MIN_FORWARD_PASS_RATE_MARGIN=0.0
export SOCCER_LEARNING_ANALYTIC_OPPONENT=true
export SOCCER_ANCHOR_PROMOTION_GATE_ENABLED=true
export SOCCER_NEURAL_ACTOR_CRITIC=true
export SOCCER_ENABLE_ACTOR_CRITIC=true
export SOCCER_NEURAL_LP_COUPLING_ENABLED=true

# --- Policy selection / learned heads ---
export DD_SOCCER_ENABLE_STOCHASTIC_POLICY_TOPK=true
export DD_SOCCER_ENABLE_FULL_ACTION_MPC_COVERAGE=true
export DD_SOCCER_ENABLE_KEEPER_POLICY_HEAD=true
export DD_SOCCER_ENABLE_SKILL_POLICY_HEADS=true
export DD_SOCCER_ENABLE_LEARNED_PASS_COMPLETION=true
export DD_SOCCER_ENABLE_LEARNED_BEAT_DEFENDER=true
export DD_SOCCER_ENABLE_LEARNED_AERIAL_RECEPTION=true
export DD_SOCCER_ENABLE_LEARNED_LONG_PASS_RUN=true
export DD_SOCCER_ENABLE_LEARNED_SPACING_TARGET=true
export DD_SOCCER_ENABLE_LEARNED_SUPPORT_SCORER=true
export DD_SOCCER_ENABLE_LEARNED_MPC_OBJECTIVE=true
export DD_SOCCER_ENABLE_LEARNED_PASS_RECEIVER=true
export DD_SOCCER_ENABLE_NEURAL_PASS_SPACE=true
export DD_SOCCER_ENABLE_DISCRETIZED_KICK=true
export DD_SOCCER_ENABLE_ACTION_PARAM_FEATURES=true
export DD_SOCCER_ENABLE_LOOSE_BALL_COMMIT_MODEL=true
export DD_SOCCER_ENABLE_RECEIVE_APPROACH_MODEL=true
export DD_SOCCER_ENABLE_MULTIMODAL_RUN_PREDICTION=true
export DD_SOCCER_ENABLE_SCORED_SHOT_PLACEMENT=true

# --- Line / unit models ---
export DD_SOCCER_ENABLE_BACK_FOUR_LINE_MODEL=true
export DD_SOCCER_ENABLE_BACK_FOUR_LINE_DEPTH_V2=true
export DD_SOCCER_ENABLE_MIDFIELD_LINE_MODEL=true

# --- Positioning / tactics / movement ---
export DD_SOCCER_ENABLE_GENOME_ANCHOR_HOME_POSITIONS=true
export DD_SOCCER_ENABLE_LANE_DISCIPLINE_V2=true
export DD_SOCCER_ENABLE_DEFENSIVE_SHEPHERD=true
export DD_SOCCER_ENABLE_PRESS_COVER=true
export DD_SOCCER_ENABLE_OFF_BALL_SPACE_DISCIPLINE=true
export DD_SOCCER_ENABLE_OUTSIDE_MID_ATTACK_DEFENDER=true
export DD_SOCCER_ENABLE_PASS_LANE_YIELD=true
export DD_SOCCER_ENABLE_QUICK_FORWARD_PASS=true
export DD_SOCCER_ENABLE_AERIAL_PASS_OOB_DISCIPLINE=true
export DD_SOCCER_ENABLE_FAR_OFFBALL_ENERGY_CONSERVATION=true
export DD_SOCCER_ENABLE_AEROBIC_ANAEROBIC_SPEED_SPLIT=true
export DD_SOCCER_ENABLE_DECISION_REFRACTORY=true
export DD_SOCCER_ENABLE_OBSTACLE_AWARE_INTERCEPT=true
export DD_SOCCER_ENABLE_MPC_PASS=true

# Jun 2026 offense/defense + reward + pass-physics batch (these are gate_default_on, i.e. already
# ON in a release build; set explicitly so the training lineage is unambiguous).
export DD_SOCCER_ENABLE_TEAM_ADVANCE_UPFIELD=true
export DD_SOCCER_ENABLE_PROGRESSIVE_CARRY_REWARD=true
export DD_SOCCER_ENABLE_BACKWARD_PASS_DISCIPLINE=true
export DD_SOCCER_ENABLE_MPC_PASS_WEIGHT=true
export DD_SOCCER_ENABLE_BUILDUP_CHAIN_CREDIT=true
export DD_SOCCER_ENABLE_BELLMAN_TERMINALS=true
export DD_SOCCER_ENABLE_NUMBERS_UP_PRESS=true
export DD_SOCCER_ENABLE_STATIONARY_HOLDER_PRESS=true
export DD_SOCCER_ENABLE_TERRIBLE_PASS_VETO=true
export DD_SOCCER_ENABLE_GROUND_PASS_SPEED_FLOOR=true
export DD_SOCCER_ENABLE_IN_STRIDE_PASS_MARGIN=true
export DD_SOCCER_ENABLE_CONTINUE_RUN_AFTER_PASS=true
export DD_SOCCER_ENABLE_STRATEGY_PERSIST=true

echo "all-gates-on: exported $(env | grep -cE '^DD_SOCCER_(ENABLE_|OPPONENT_BELIEF|OUTCOME_CREDIT)') soccer feature gates = on" >&2
