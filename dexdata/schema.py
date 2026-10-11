"""Dataset schema shared by all XHand trajectory collectors.

Conventions (apply to every field):
- Control step: one row per policy step (20 Hz in dexscrew and dexgen). ``step_dt`` is stored in the manifest.
- Joint order: ``JOINT_NAMES`` (alphabetical; identical to the dexscrew/IsaacGym and dexgen order).
- Quaternions: ``wxyz`` (Isaac Lab convention), always unit-norm, sign not canonicalized.
- Wrist frame: pose of the hand root link ``left_hand_link``. ``*_wrist`` fields are expressed in it.
- Units: metres, radians, seconds, newtons (torque in N*m).
- Row ``t`` holds the state observed *before* the action of step ``t`` together with that action, so the
  next state of row ``t`` is row ``t + 1`` of the same episode. Episodes are stored contiguously.
- Missing quantities (e.g. a joint DOF for a free object) are ``NaN`` for floats and ``False`` for flags.
"""

SCHEMA_VERSION = "1.0"

JOINT_NAMES = (
    "left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2",
    "left_hand_mid_joint1", "left_hand_mid_joint2", "left_hand_pinky_joint1", "left_hand_pinky_joint2",
    "left_hand_ring_joint1", "left_hand_ring_joint2", "left_hand_thumb_bend_joint",
    "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2",
)
WRIST_BODY = "left_hand_link"

# Every finger link is stored so that the keypoint set (K) can be chosen at training time.
KEYPOINT_BODIES = (
    "left_hand_index_bend_link", "left_hand_index_rota_link1", "left_hand_index_rota_link2", "left_hand_index_rota_tip",
    "left_hand_mid_link1", "left_hand_mid_link2", "left_hand_mid_tip",
    "left_hand_ring_link1", "left_hand_ring_link2", "left_hand_ring_tip",
    "left_hand_pinky_link1", "left_hand_pinky_link2", "left_hand_pinky_tip",
    "left_hand_thumb_bend_link", "left_hand_thumb_rota_link1", "left_hand_thumb_rota_link2", "left_hand_thumb_rota_tip",
)
# Suggested default keypoint subsets (indices into KEYPOINT_BODIES).
KEYPOINT_SETS = {
    "tips": [3, 6, 9, 12, 16],                                # K=5, fingertips
    "tips_mid": [2, 3, 5, 6, 8, 9, 11, 12, 15, 16],           # K=10, distal + middle phalanx (paper-style PIP+tip)
    "all": list(range(len(KEYPOINT_BODIES))),                 # K=17
}
FINGER_ORDER = ("index", "mid", "ring", "pinky", "thumb")

# Episode end reasons (canonical across tasks).
END_REASONS = {
    "timeout": 0,      # reached the episode length limit
    "drop": 1,         # object left the hand / fingers too far / fell
    "low_contact": 2,  # contact with the object was lost for too long
    "stagnation": 3,   # object stopped moving (task-specific; object still in the hand)
    "other": 4,        # any other termination
    "truncated": 5,    # collection ended while the episode was running (only if partial episodes are kept)
}
# Paper (DexterityGen, App. A): if a rollout ends with dropping the object, discard its last 2 s.
DEFAULT_DROP_TAIL_REASONS = ("drop", "low_contact")
DEFAULT_DROP_TAIL_SECONDS = 2.0

N_J = len(JOINT_NAMES)
N_K = len(KEYPOINT_BODIES)

# name: (shape, dtype, description)
STEP_FIELDS = {
    # --- proprioception (DexGen state inputs)
    "joint_pos": ((N_J,), "float32", "measured joint position q_t"),
    "joint_vel": ((N_J,), "float32", "measured joint velocity (simulator)"),
    "joint_target": ((N_J,), "float32", "joint target in effect when q_t was observed (= applied_target[t-1])"),
    "control_error": ((N_J,), "float32", "joint_target - joint_pos"),
    "keypoint_pos_wrist": ((N_K, 3), "float32", "finger link origins in the wrist frame (KEYPOINT_BODIES order)"),
    "keypoint_quat_wrist": ((N_K, 4), "float32", "finger link orientations in the wrist frame, wxyz"),
    # --- wrist
    "wrist_pos_env": ((3,), "float32", "wrist position in the env-local world frame"),
    "wrist_quat_world": ((4,), "float32", "wrist orientation in the world frame, wxyz (gravity is -z)"),
    "gravity_wrist": ((3,), "float32", "unit gravity direction expressed in the wrist frame"),
    # --- action
    "action": ((N_J,), "float32", "raw policy output a_t (after clipping to [-1, 1])"),
    "applied_target": ((N_J,), "float32", "joint target commanded at step t (incl. env action noise/limits)"),
    "applied_torque": ((N_J,), "float32", "joint torque applied in the last physics substep of step t"),
    # --- object (privileged; for analysis, critics, filtering)
    "object_pos_wrist": ((3,), "float32", "object (moving body) position in the wrist frame"),
    "object_quat_wrist": ((4,), "float32", "object orientation in the wrist frame, wxyz"),
    "object_linvel_wrist": ((3,), "float32", "object COM linear velocity, rotated into the wrist frame"),
    "object_angvel_wrist": ((3,), "float32", "object angular velocity, rotated into the wrist frame"),
    "object_dof_pos": ((1,), "float32", "object joint angle (nut on bolt / screwdriver); NaN for free objects"),
    "object_dof_vel": ((1,), "float32", "object joint velocity; NaN for free objects"),
    "object_contact_force": ((1,), "float32", "norm of the net contact force on the object body"),
    "finger_contact_force": ((5,), "float32", "per-finger contact force with the object (FINGER_ORDER); NaN if unavailable"),
    "goal_pos_wrist": ((3,), "float32", "task goal position in the wrist frame; NaN if the task has no goal"),
    "goal_quat_wrist": ((4,), "float32", "task goal orientation in the wrist frame, wxyz; NaN if none"),
    # --- task signals
    "reward": ((1,), "float32", "task reward of step t"),
    "success_event": ((1,), "bool", "task-defined success event fired at step t"),
    "goal_reached_event": ((1,), "bool", "goal reached at step t (goal-conditioned tasks)"),
}

# Per-episode table. Physical parameters are those in effect during the episode.
EPISODE_FIELDS = {
    "episode_id": ((), "int64", "unique id within the collection run"),
    "env_id": ((), "int32", "parallel environment index"),
    "task_id": ((), "int32", "index into manifest['tasks']"),
    "policy_id": ((), "int32", "index into manifest['policies']"),
    "start": ((), "int64", "first row of the episode inside its shard"),
    "length": ((), "int32", "number of stored rows (after filtering)"),
    "raw_length": ((), "int32", "number of policy steps the episode actually ran"),
    "end_reason": ((), "int8", "END_REASONS code"),
    "success": ((), "bool", "any success_event within the stored rows"),
    "object_mass": ((), "float32", "object (moving body) mass [kg]"),
    "object_scale": ((), "float32", "uniform object scale; NaN if unknown"),
    "object_com_offset": ((3,), "float32", "object COM in its body frame [m]"),
    "friction": ((), "float32", "object material static friction (hand uses the same value in both envs)"),
    "restitution": ((), "float32", "object material restitution"),
    "p_gain": ((N_J,), "float32", "PD controller P gain per joint"),
    "d_gain": ((N_J,), "float32", "PD controller D gain per joint"),
}


def field_numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n
