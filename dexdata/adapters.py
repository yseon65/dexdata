"""Adapters that read the canonical per-step fields (schema.STEP_FIELDS) from a running Isaac Lab env.

An adapter wraps one environment instance and exposes:
    read_state()            -> dict of (N, ...) tensors observed before the action of the current step
    step(action)            -> (obs, post: dict, done, reason)  post holds the action-related step fields
    episode_params(env_ids) -> dict of numpy arrays (schema.EPISODE_FIELDS physical parameters)

Supported envs:
    DexscrewAdapter  dexscrew.tasks.xhand_hora_lab.XHandHoraLab (nut-bolt / screwdriver, VecTask-style API)
    DexgenAdapter    dexgen NutSpinEnv / NutGoalEnv (Isaac Lab DirectRLEnv, stepped through RslRlVecEnvWrapper)
"""

import math

import numpy as np
import torch

from . import frames, schema

NAN = float("nan")


class _XHandAdapter:
    def __init__(self, env, hand):
        self.env = env
        self.hand = hand
        self.device = hand.device
        self.num_envs = hand.num_instances
        self.joint_ids = hand.find_joints(list(schema.JOINT_NAMES), preserve_order=True)[0]
        assert len(self.joint_ids) == schema.N_J, self.joint_ids
        self.kp_ids = hand.find_bodies(list(schema.KEYPOINT_BODIES), preserve_order=True)[0]
        assert len(self.kp_ids) == schema.N_K, self.kp_ids
        wrist = hand.find_bodies([schema.WRIST_BODY])[0]
        assert len(wrist) == 1
        self.wrist_id = wrist[0]
        self.jt = torch.as_tensor(self.joint_ids, device=self.device)
        self.kt = torch.as_tensor(self.kp_ids, device=self.device)
        self._gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)

    def _nan(self, *shape):
        return torch.full((self.num_envs, *shape), NAN, device=self.device)

    # --- to be provided by subclasses
    def current_target(self): raise NotImplementedError
    def object_state_w(self): raise NotImplementedError       # pos (N,3), quat (N,4), linvel, angvel (world)
    def object_dof(self): return self._nan(1), self._nan(1)
    def object_contact(self): raise NotImplementedError       # (N,1)
    def finger_contacts(self): return self._nan(5)
    def goal_pose_w(self): return None
    def env_origins(self): raise NotImplementedError

    def read_state(self):
        d = self.hand.data
        wrist_pos = d.body_link_pos_w[:, self.wrist_id]
        wrist_quat = d.body_link_quat_w[:, self.wrist_id]
        q = d.joint_pos[:, self.jt]
        target = self.current_target()
        kp_pos, kp_quat = frames.to_frame(d.body_link_pos_w[:, self.kt], d.body_link_quat_w[:, self.kt],
                                          wrist_pos[:, None], wrist_quat[:, None].expand(-1, schema.N_K, -1))
        o_pos, o_quat, o_lin, o_ang = self.object_state_w()
        o_pos_w, o_quat_w = frames.to_frame(o_pos, o_quat, wrist_pos, wrist_quat)
        dof_pos, dof_vel = self.object_dof()
        goal = self.goal_pose_w()
        if goal is None:
            g_pos, g_quat = self._nan(3), self._nan(4)
        else:
            g_pos, g_quat = frames.to_frame(goal[0], goal[1], wrist_pos, wrist_quat)
        return {
            "joint_pos": q,
            "joint_vel": d.joint_vel[:, self.jt],
            "joint_target": target,
            "control_error": target - q,
            "keypoint_pos_wrist": kp_pos,
            "keypoint_quat_wrist": kp_quat,
            "wrist_pos_env": wrist_pos - self.env_origins(),
            "wrist_quat_world": wrist_quat,
            "gravity_wrist": frames.rotate_into(self._gravity, wrist_quat),
            "object_pos_wrist": o_pos_w,
            "object_quat_wrist": o_quat_w,
            "object_linvel_wrist": frames.rotate_into(o_lin, wrist_quat),
            "object_angvel_wrist": frames.rotate_into(o_ang, wrist_quat),
            "object_dof_pos": dof_pos,
            "object_dof_vel": dof_vel,
            "object_contact_force": self.object_contact(),
            "finger_contact_force": self.finger_contacts(),
            "goal_pos_wrist": g_pos,
            "goal_quat_wrist": g_quat,
        }

    @staticmethod
    def _material(view, env_ids):
        mat = view.get_material_properties()[torch.as_tensor(env_ids).long()]
        return mat[:, :, 0].mean(-1).numpy(), mat[:, :, 2].mean(-1).numpy()


# ---------------------------------------------------------------------------------------------------- dexscrew
class DexscrewAdapter(_XHandAdapter):
    """dexscrew XHandHoraLab. Its internal buffers are xyzw/env-local, so everything is read from Isaac Lab data."""

    def __init__(self, env):
        super().__init__(env, env.hand)
        self.obj = env.object
        self.nut_body = env.screw_nut_rb_handle
        self._applied_target = env.cur_targets[:, :schema.N_J].clone()
        self._term_flags = None
        orig_pre = env.pre_physics_step

        def pre_physics_step(actions):
            orig_pre(actions)
            self._applied_target = env.cur_targets[:, :schema.N_J].clone()
        env.pre_physics_step = pre_physics_step

        orig_check = env.check_termination

        def check_termination(object_pos):
            th, ix = env._thumb_index_pos()
            far = (torch.norm(th - env.nut_pos, dim=-1) > env.reset_dist_threshold) | \
                  (torch.norm(ix - env.nut_pos, dim=-1) > env.reset_dist_threshold)
            filled = env.progress_buf >= env.nut_termination_history_len
            stagnant = (torch.var(env.nut_dof_pos_history, dim=1).squeeze(-1) < env.nut_stagnation_eps) & filled
            no_contact = torch.all(env.nut_contact_history <= 1e-3, dim=1).squeeze(-1) & filled
            limit = env.nut_dof_pos[:, 0] > env.screw_upper_limit - 5.0
            self._term_flags = (far, no_contact, limit, stagnant)
            return orig_check(object_pos)
        env.check_termination = check_termination

    def env_origins(self):
        return self.env.env_origins

    def current_target(self):
        return self.env.cur_targets[:, :schema.N_J].clone()

    def object_state_w(self):
        d = self.obj.data
        b = self.nut_body
        return d.body_link_pos_w[:, b], d.body_link_quat_w[:, b], d.body_com_lin_vel_w[:, b], d.body_com_ang_vel_w[:, b]

    def object_dof(self):
        d = self.obj.data
        return d.joint_pos[:, 0:1], d.joint_vel[:, 0:1]

    def object_contact(self):
        return self.env.nut_contact.clone()

    def step(self, action):
        env = self.env
        obs, rew, done, info = env.step(action)
        torque = env.torques[:, -1, :schema.N_J]
        torque = torch.max(torch.min(torque, env.xhand_effort_limits), -env.xhand_effort_limits)
        far, no_contact, limit, stagnant = self._term_flags
        timeout = info["time_outs"].bool()
        R = schema.END_REASONS
        reason = torch.full_like(done, R["other"], dtype=torch.long)
        for flag, code in ((stagnant, R["stagnation"]), (limit, R["other"]), (no_contact, R["low_contact"]),
                           (far, R["drop"])):  # later entries take priority
            reason = torch.where(flag, code, reason)
        reason = torch.where(timeout & ~(far | no_contact | limit | stagnant), R["timeout"], reason)
        post = {
            "action": action.clamp(-1.0, 1.0),
            "applied_target": self._applied_target,
            "applied_torque": torque.clone(),
            "reward": rew[:, None].float(),
            "success_event": torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device),
            "goal_reached_event": torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device),
        }
        return obs, post, done.bool(), reason

    def episode_params(self, env_ids):
        env = self.env
        ids = torch.as_tensor(env_ids, device=self.device).long()
        view = self.obj.root_physx_view
        cpu = torch.as_tensor(env_ids).long()
        friction, restitution = self._material(view, env_ids)
        n_assets = len(env.object_asset_meta)
        return {
            "object_mass": view.get_masses()[cpu, self.nut_body].numpy(),
            "object_scale": np.array([env.object_asset_meta[int(i) % n_assets][1] for i in env_ids], dtype=np.float32),
            "object_com_offset": view.get_coms()[cpu, self.nut_body, :3].numpy(),
            "friction": friction,
            "restitution": restitution,
            "p_gain": env.p_gain[ids].cpu().numpy(),
            "d_gain": env.d_gain[ids].cpu().numpy(),
        }


# ---------------------------------------------------------------------------------------------------- dexgen
class DexgenAdapter(_XHandAdapter):
    """dexgen NutSpinEnv / NutGoalEnv. `wrapped` is the RslRlVecEnvWrapper used to step the env."""

    def __init__(self, raw_env, wrapped, object_name="nut"):
        super().__init__(raw_env, raw_env.scene["hand"])
        self.wrapped = wrapped
        self.obj = raw_env.scene[object_name]
        self._applied_target = raw_env.targets.clone()
        orig_pre = raw_env._pre_physics_step

        def _pre_physics_step(actions):
            orig_pre(actions)
            self._applied_target = raw_env.targets.clone()
        raw_env._pre_physics_step = _pre_physics_step

    def env_origins(self):
        return self.env.scene.env_origins

    def current_target(self):
        return self.env.targets.clone()

    def object_state_w(self):
        d = self.obj.data
        return d.root_link_pos_w, d.root_link_quat_w, d.root_com_lin_vel_w, d.root_com_ang_vel_w

    def object_contact(self):
        return torch.linalg.vector_norm(self.env.nut_contact.data.net_forces_w[:, 0], dim=-1, keepdim=True)

    def finger_contacts(self):
        sensors = getattr(self.env, "finger_contacts", None)
        if not sensors:
            return self._nan(5)
        names = ("index", "mid", "pinky", "ring", "thumb")  # dexgen FINGER_NAMES order
        forces = {n: torch.linalg.vector_norm(s.data.force_matrix_w[:, 0], dim=-1).sum(-1) for n, s in zip(names, sensors)}
        return torch.stack([forces[n] for n in schema.FINGER_ORDER], dim=-1)

    def goal_pose_w(self):
        if hasattr(self.env, "goal_pos") and hasattr(self.env, "goal_quat"):
            return self.env.goal_pos, self.env.goal_quat
        return None

    def step(self, action):
        env = self.env
        obs, rew, done, info = self.wrapped.step(action)
        timeout = info.get("time_outs", torch.zeros_like(done)).bool()
        R = schema.END_REASONS
        # extras["task/terminal_reason"] is cloned before the auto-reset; env._terminal_reason may already be cleared
        tr = info.get("task/terminal_reason", getattr(env, "_terminal_reason", None))
        if tr is None:
            reason = torch.full_like(done.long(), R["other"])
        else:  # dexgen codes: 1 dropped, 2 lost grasp, 3 low contact, 4 tipped over (tilt limit), 0 none
            reason = torch.full_like(tr, R["other"])
            reason = torch.where((tr == 1) | (tr == 2) | (tr == 4), R["drop"], reason)
            reason = torch.where(tr == 3, R["low_contact"], reason)
        reason = torch.where(timeout & (tr == 0 if tr is not None else True), R["timeout"], reason)
        zeros = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        succ = info.get("task/success")
        goal = info.get("task/goal_reached")
        post = {
            "action": action.clamp(-1.0, 1.0),
            "applied_target": self._applied_target,
            "applied_torque": env.effort.clone(),
            "reward": rew.reshape(-1, 1).float(),
            "success_event": zeros if succ is None else succ.reshape(-1, 1).bool(),
            "goal_reached_event": zeros if goal is None else goal.reshape(-1, 1).bool(),
        }
        return obs, post, done.bool(), reason

    def episode_params(self, env_ids):
        env = self.env
        ids = torch.as_tensor(env_ids, device=self.device).long()
        cpu = torch.as_tensor(env_ids).long()
        view = self.obj.root_physx_view
        friction, restitution = self._material(view, env_ids)
        p, d = env.cfg.p_gain, env.cfg.d_gain
        pf = env.p_factor[ids] if hasattr(env, "p_factor") else torch.ones(len(ids), device=self.device)
        df = env.d_factor[ids] if hasattr(env, "d_factor") else torch.ones(len(ids), device=self.device)
        ones = torch.ones((len(ids), schema.N_J), device=self.device)
        scale = env.scale[ids].cpu().numpy() if hasattr(env, "scale") else np.full(len(ids), math.nan, np.float32)
        masses, coms = view.get_masses(), view.get_coms()  # RigidObject views: (N, 1) / (N, 7)
        masses = masses.reshape(masses.shape[0], -1)[cpu, 0]
        coms = coms.reshape(coms.shape[0], -1, 7)[cpu, 0, :3]
        return {
            "object_mass": masses.numpy(),
            "object_scale": scale,
            "object_com_offset": coms.numpy(),
            "friction": friction,
            "restitution": restitution,
            "p_gain": (p * pf[:, None] * ones).cpu().numpy(),
            "d_gain": (d * df[:, None] * ones).cpu().numpy(),
        }
