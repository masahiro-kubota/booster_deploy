from __future__ import annotations
from dataclasses import MISSING
import json
import os
import inspect
import re
import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
    PolicyCfg
)
from booster_deploy.robots.booster import K1_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.motion_loader import MotionLoader


def _resolve_task_file(task_path: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(task_path, path)


def _default_manifest_path(checkpoint_path: str) -> str:
    stem, _ = os.path.splitext(checkpoint_path)
    return f"{stem}.manifest.json"


class BeyondMimicPolicy(Policy):
    def __init__(self, cfg: BeyondMimicPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        checkpoint_path = _resolve_task_file(self.task_path, self.cfg.checkpoint_path)
        self._model: torch.jit.ScriptModule = torch.jit.load(checkpoint_path)
        self._model.to(self.cfg.device).eval()

        self.robot = controller.robot

        self.action_scale = (
            0.25 * self.robot.effort_limit / self.robot.joint_stiffness
        ).to(self.cfg.device)
        self._check_manifest(checkpoint_path)

        self.robot.data.to(self.cfg.device)

        motion_file = _resolve_task_file(self.task_path, self.cfg.motion_path)
        self.motion = MotionLoader(
            motion_file=motion_file,
            track_body_names=[self.cfg.anchor_body_name],
            track_joint_names=self.robot.cfg.sim_joint_names,
            default_motion_body_names=self.robot.cfg.sim_body_names,
            default_motion_joint_names=self.robot.cfg.sim_joint_names,
            align_to_first_frame=True,
            device=self.cfg.device
        )

        self.default_joint_pos = self.robot.default_joint_pos.to(
            self.cfg.device)

    def _check_manifest(self, checkpoint_path: str) -> None:
        manifest_path = self.cfg.manifest_path
        if manifest_path is None:
            manifest_path = _default_manifest_path(checkpoint_path)
        else:
            manifest_path = _resolve_task_file(self.task_path, manifest_path)

        if not os.path.isfile(manifest_path):
            return

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        self._check_manifest_action_scale(manifest, manifest_path)
        self._check_manifest_motion(manifest, manifest_path)

    def _check_manifest_action_scale(self, manifest: dict, manifest_path: str) -> None:
        manifest_scale = manifest.get("action_scale")
        if not isinstance(manifest_scale, dict):
            return

        expected_scale = []
        for joint_name in self.robot.cfg.joint_names:
            scale = None
            for joint_pattern, value in manifest_scale.items():
                if re.fullmatch(joint_pattern, joint_name):
                    scale = float(value)
                    break
            if scale is None:
                return
            expected_scale.append(scale)

        expected_scale_tensor = torch.tensor(expected_scale, dtype=torch.float32, device=self.cfg.device)
        scale_error = torch.abs(expected_scale_tensor - self.action_scale)
        max_error = float(torch.max(scale_error).item())
        if max_error <= 1.0e-4:
            return

        worst_index = int(torch.argmax(scale_error).item())
        print(
            "[WARN] Policy manifest action_scale differs from deploy config: "
            f"manifest={manifest_path}, joint={self.robot.cfg.joint_names[worst_index]}, "
            f"manifest_scale={float(expected_scale_tensor[worst_index].item()):.6f}, "
            f"deploy_scale={float(self.action_scale[worst_index].item()):.6f}, "
            f"max_abs_error={max_error:.6f}"
        )

    def _check_manifest_motion(self, manifest: dict, manifest_path: str) -> None:
        manifest_motion = manifest.get("motion_file")
        if not isinstance(manifest_motion, str):
            return

        manifest_motion_name = os.path.basename(manifest_motion)
        deploy_motion_name = os.path.basename(self.cfg.motion_path)
        if manifest_motion_name == deploy_motion_name:
            return

        print(
            "[WARN] Policy manifest motion file differs from deploy config: "
            f"manifest={manifest_path}, manifest_motion={manifest_motion_name}, "
            f"deploy_motion={deploy_motion_name}"
        )

    def reset(self) -> None:
        self.init_root_yaw_quat_w_inv = lab_math.quat_inv(
            lab_math.yaw_quat(self.robot.data.root_quat_w))
        self.anchor_index = self.motion.track_body_names.index(
            self.cfg.anchor_body_name)
        self.current_frame = 0
        self.last_action = torch.zeros(
            self.robot.num_joints,
            dtype=torch.float32, device=self.cfg.device)
        self.motion.to(self.cfg.device)

    def _set_command(self):
        if self.cfg.loop_motion:
            row_ids = self.current_frame % self.motion.time_step_total
        else:
            row_ids = min(self.current_frame, self.motion.time_step_total - 1)

        self.cmd_dof_pos = self.motion.joint_pos[row_ids]
        self.cmd_dof_vel = self.motion.joint_vel[row_ids]

        self.cmd_root_pos_w = self.motion.body_pos_w[
            row_ids, self.anchor_index]
        self.cmd_root_quat_w = self.motion.body_quat_w[
            row_ids, self.anchor_index]

    def compute_observation(self) -> torch.Tensor:
        """Computes observations"""
        self._set_command()

        command = torch.cat([self.cmd_dof_pos, self.cmd_dof_vel], dim=0)
        cur_root_quat_w = lab_math.quat_mul(
            self.init_root_yaw_quat_w_inv, self.robot.data.root_quat_w)

        pos, ori = lab_math.subtract_frame_transforms(
            self.robot.data.root_pos_w,
            cur_root_quat_w,
            self.cmd_root_pos_w,
            self.cmd_root_quat_w,
        )

        motion_anchor_pos_b = pos  # noqa: F841

        motion_anchor_ori_b = lab_math.matrix_from_quat(ori)[..., :2].flatten()

        real2sim_map = self.robot.data.real2sim_joint_indexes
        dof_pos = self.robot.data.joint_pos[real2sim_map]
        joint_pos = dof_pos - self.default_joint_pos[real2sim_map]
        joint_vel = self.robot.data.joint_vel[real2sim_map]

        obs = torch.cat(
            (
                command,
                # motion_anchor_pos_b,    # linear states
                motion_anchor_ori_b,
                # self.robot.data.root_lin_vel_b,           # linear states
                self.robot.data.root_ang_vel_b,
                joint_pos,
                joint_vel,
                self.last_action,
            ),
            dim=-1,
        )
        return obs.reshape(1, -1)

    def inference(self) -> torch.Tensor:
        """Called by the controller each step to obtain the action tensor.

        Reads `robot.data` and velocity commands from the controller,
        runs the underlying model's inference, and returns an action
        as a `torch.Tensor`.
        """

        with torch.no_grad():
            obs = self.compute_observation()
            action = self._model(obs).flatten()

        # for motion visualization in Mujoco controller
        if hasattr(self.controller, "set_reference_qpos"):
            joint_pos = self.cmd_dof_pos[self.robot.data.sim2real_joint_indexes]
            ref_qpos = torch.cat(
                [self.cmd_root_pos_w, self.cmd_root_quat_w, joint_pos],
                dim=0,
            )
            self.controller.set_reference_qpos(ref_qpos)    # type: ignore

        self.current_frame += 1
        self.last_action = action

        if action is None:
            raise RuntimeError("Underlying model returned None from inference")

        if self.cfg.enable_safety_fallback:
            # monitor policy state validity during execution.
            gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32,
                                     device=self.cfg.device)
            projected_gravity = lab_math.quat_apply_inverse(
                self.robot.data.root_quat_w, gravity_w)
            motion_projected_gravity = lab_math.quat_apply_inverse(
                self.cmd_root_quat_w, gravity_w)

            if torch.dot(projected_gravity, motion_projected_gravity) < 0.5:
                print("\nLarge root tracking error is detected, stopping policy"
                      " for safety. You can disable safety fallback by setting "
                      f"{self.cfg.__class__.__name__}.enable_safety_fallback "
                      "to False.")
                self.controller.stop()

        sim2real_map = self.robot.data.sim2real_joint_indexes
        return (
            action[sim2real_map] * self.action_scale
            + self.default_joint_pos
        )


@configclass
class BeyondMimicPolicyCfg(PolicyCfg):
    constructor = BeyondMimicPolicy
    checkpoint_path: str = MISSING
    motion_path: str = MISSING
    manifest_path: str | None = None
    loop_motion: bool = False

    anchor_body_name: str = "Trunk"


@configclass
class K1BeyondMimicControllerCfg(ControllerCfg):
    robot = K1_CFG.replace(     # type: ignore
        joint_stiffness=[
            4.0, 4.0,
            4.0, 4.0, 4.0, 4.0,
            4.0, 4.0, 4.0, 4.0,
            80., 80.0, 80., 80., 30., 30.,
            80., 80.0, 80., 80., 30., 30.,
        ],
        joint_damping=[
            1., 1.,
            1., 1., 1., 1.,
            1., 1., 1., 1.,
            2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2.,
        ]
    )
    enable_velocity_commands = False
    policy: BeyondMimicPolicyCfg = BeyondMimicPolicyCfg()
    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.57],
        visualize_reference_ghost=True,
    )
