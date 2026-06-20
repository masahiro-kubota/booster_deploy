from __future__ import annotations

import json
import os
import time

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import PolicyCfg
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.isaaclab import math as lab_math
from tasks.locomotion import K1WalkControllerCfg1


_K1_WALK_REFERENCE_CFG = K1WalkControllerCfg1()
_K1_WALK_ACTION_JOINT_NAMES = list(_K1_WALK_REFERENCE_CFG.policy.policy_joint_names)
_K1_WALK_ACTION_SCALE = float(_K1_WALK_REFERENCE_CFG.policy.action_scale)


def _resolve_task_path(task_path: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(task_path, path)


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class _DeployDiffusionSchedule:
    def __init__(self, num_steps: int, device: torch.device | str) -> None:
        betas = torch.linspace(1.0e-4, 0.02, num_steps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat((torch.ones(1, dtype=torch.float32, device=device), alpha_bars[:-1]), dim=0)
        self.num_steps = int(num_steps)
        self.posterior_variance = torch.clamp(betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars), min=1.0e-20)
        self.posterior_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)
        self.posterior_mean_coef2 = (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars)

    def p_sample(self, noisy: torch.Tensor, clean_pred: torch.Tensor, step: int, noise_scale: float = 1.0) -> torch.Tensor:
        step_tensor = torch.tensor([step], dtype=torch.long, device=noisy.device)
        shape = (noisy.shape[0],) + (1,) * (noisy.dim() - 1)
        mean = (
            self.posterior_mean_coef1[step_tensor].view(shape) * clean_pred
            + self.posterior_mean_coef2[step_tensor].view(shape) * noisy
        )
        if step == 0 or noise_scale <= 0.0:
            return mean
        variance = self.posterior_variance[step_tensor].view(shape)
        return mean + float(noise_scale) * torch.sqrt(variance) * torch.randn_like(noisy)


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _yaw_rotation_matrix(yaw: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)
    return torch.stack(
        (
            torch.stack((cos_yaw, -sin_yaw, zeros), dim=-1),
            torch.stack((sin_yaw, cos_yaw, zeros), dim=-1),
            torch.stack((zeros, zeros, ones), dim=-1),
        ),
        dim=-2,
    )


def _matrix_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = torch.nn.functional.normalize(quat, dim=-1)
    qw, qx, qy, qz = quat.unbind(dim=-1)
    two_s = 2.0
    return torch.stack(
        (
            1.0 - two_s * (qy * qy + qz * qz),
            two_s * (qx * qy - qz * qw),
            two_s * (qx * qz + qy * qw),
            two_s * (qx * qy + qz * qw),
            1.0 - two_s * (qx * qx + qz * qz),
            two_s * (qy * qz - qx * qw),
            two_s * (qx * qz - qy * qw),
            two_s * (qy * qz + qx * qw),
            1.0 - two_s * (qx * qx + qy * qy),
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _rotation_6d_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2].reshape(matrix.shape[:-2] + (6,))


def _build_character_yaw_state(
    *,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    joint_pos_rel: torch.Tensor,
    joint_vel: torch.Tensor,
    last_action: torch.Tensor,
) -> torch.Tensor:
    root_pos_w = root_pos_w.view(1, 3)
    root_quat_w = root_quat_w.view(1, 4)
    root_lin_vel_w = root_lin_vel_w.view(1, 3)
    root_ang_vel_w = root_ang_vel_w.view(1, 3)
    body_pos_w = body_pos_w.view(1, body_pos_w.shape[0], 3)
    body_lin_vel_w = body_lin_vel_w.view(1, body_lin_vel_w.shape[0], 3)
    joint_pos_rel = joint_pos_rel.view(1, -1)
    joint_vel = joint_vel.view(1, -1)
    last_action = last_action.view(1, -1)

    yaw = _yaw_from_quat_wxyz(root_quat_w)
    yaw_rot_w = _yaw_rotation_matrix(yaw)
    root_rot_rel = torch.matmul(yaw_rot_w.transpose(-1, -2), _matrix_from_quat_wxyz(root_quat_w))
    root_lin_vel_rel = torch.matmul(root_lin_vel_w.unsqueeze(-2), yaw_rot_w).squeeze(-2)
    root_ang_vel_rel = torch.matmul(root_ang_vel_w.unsqueeze(-2), yaw_rot_w).squeeze(-2)
    body_pos_local = torch.matmul(body_pos_w - root_pos_w.unsqueeze(-2), yaw_rot_w).reshape(1, -1)
    body_vel_local = torch.matmul(body_lin_vel_w, yaw_rot_w).reshape(1, -1)
    gravity_w = torch.zeros_like(root_pos_w)
    gravity_w[..., 2] = -1.0
    projected_gravity = torch.matmul(gravity_w.unsqueeze(-2), yaw_rot_w).squeeze(-2)
    return torch.cat(
        (
            torch.zeros_like(root_pos_w),
            _rotation_6d_from_matrix(root_rot_rel),
            root_lin_vel_rel,
            root_ang_vel_rel,
            body_pos_local,
            body_vel_local,
            projected_gravity,
            joint_pos_rel,
            joint_vel,
            last_action,
        ),
        dim=-1,
    ).flatten()


class JoystickGuidanceCost:
    def __init__(
        self,
        *,
        state_layout: dict,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        state_pseudoinverse: torch.Tensor,
        current_index: int,
        command_weights: list[float],
    ) -> None:
        self.state_layout = state_layout
        self.state_mean = state_mean
        self.state_std = state_std
        self.state_pseudoinverse = state_pseudoinverse
        self.current_index = int(current_index)
        self.command_weights = command_weights

    def projected_to_physical_state(self, projected_state: torch.Tensor) -> torch.Tensor:
        raw_norm = torch.matmul(projected_state, self.state_pseudoinverse.t())
        return raw_norm * self.state_std + self.state_mean

    def __call__(self, projected_state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        physical_state = self.projected_to_physical_state(projected_state[:, self.current_index :])
        planar_start, planar_end = self.state_layout["planar_velocity"]
        yaw_start, yaw_end = self.state_layout["yaw_rate"]
        planar_velocity = physical_state[..., planar_start:planar_end]
        yaw_rate = physical_state[..., yaw_start:yaw_end].squeeze(-1)
        planar_error = planar_velocity - command[:2].view(1, 1, 2)
        yaw_error = yaw_rate - command[2]
        return (
            float(self.command_weights[0]) * torch.mean(planar_error[..., 0] ** 2)
            + float(self.command_weights[1]) * torch.mean(planar_error[..., 1] ** 2)
            + float(self.command_weights[2]) * torch.mean(yaw_error ** 2)
        )


class WaypointGuidanceCost:
    def __call__(self, root_position: torch.Tensor, goal_position: torch.Tensor, root_velocity: torch.Tensor) -> torch.Tensor:
        position_error = root_position - goal_position.view(1, 1, -1)
        near_goal = torch.exp(-torch.linalg.norm(position_error[..., :2], dim=-1))
        return torch.mean(position_error[..., :2] ** 2) + torch.mean(near_goal * torch.sum(root_velocity[..., :2] ** 2, dim=-1))


class ObstacleSdfGuidanceCost:
    def __init__(self, margin: float = 0.05) -> None:
        self.margin = float(margin)

    def __call__(self, sdf_values: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.relu(self.margin - sdf_values) ** 2)


class MotionInpaintingGuidanceCost:
    def __call__(self, predicted_state: torch.Tensor, target_state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.mean(((predicted_state - target_state) * mask) ** 2)


class BeyondMimicDiffusionJoystickPolicy(Policy):
    def __init__(self, cfg: "BeyondMimicDiffusionJoystickPolicyCfg", controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot
        self.model_dir = _resolve_task_path(self.task_path, self.cfg.model_dir)
        self._check_artifacts()

        self._decoder = torch.jit.load(os.path.join(self.model_dir, "vae_decoder.pt"), map_location=self.cfg.device)
        self._diffusion = torch.jit.load(os.path.join(self.model_dir, "diffusion.pt"), map_location=self.cfg.device)
        self._decoder.eval()
        self._diffusion.eval()
        self._teacher_prior = None
        if max(float(self.cfg.teacher_prior_blend), float(self.cfg.backward_teacher_prior_blend)) > 0.0:
            teacher_path = _resolve_task_path(self.task_path, self.cfg.teacher_policy_path)
            if not os.path.isfile(teacher_path):
                raise FileNotFoundError(
                    f"ablation teacher prior policy is missing: {teacher_path}. "
                    "The default paper-style path does not use a runtime teacher prior; set teacher_prior_blend=0.0 "
                    "or export/copy the K1 locomotion TorchScript policy for ablation."
                )
            self._teacher_prior = torch.jit.load(teacher_path, map_location=self.cfg.device)
            self._teacher_prior.eval()

        self.manifest = _load_json(os.path.join(self.model_dir, "manifest.json"))
        projection = torch.load(os.path.join(self.model_dir, "state_projection.pt"), map_location=self.cfg.device)
        normalizer = _load_json(os.path.join(self.model_dir, "normalizer.json"))
        self.state_mean = torch.tensor(normalizer["state_mean"], dtype=torch.float32, device=self.cfg.device)
        self.state_std = torch.tensor(normalizer["state_std"], dtype=torch.float32, device=self.cfg.device)
        self.action_mean = torch.tensor(normalizer["action_mean"], dtype=torch.float32, device=self.cfg.device)
        self.action_std = torch.tensor(normalizer["action_std"], dtype=torch.float32, device=self.cfg.device)
        self.state_projection = projection["projection"].to(device=self.cfg.device, dtype=torch.float32)
        self.state_pseudoinverse = projection["pseudoinverse"].to(device=self.cfg.device, dtype=torch.float32)

        self.raw_state_dim = int(self.manifest.get("raw_state_dim", projection["raw_state_dim"]))
        self.state_dim = int(self.manifest["state_dim"])
        self.projected_state_dim = int(self.manifest.get("projected_state_dim", self.state_dim))
        self.action_dim = int(self.manifest["action_dim"])
        self.latent_dim = int(self.manifest["latent_dim"])
        self.history_length = int(self.manifest.get("history_length", self.cfg.history_length))
        self.current_index = int(self.manifest.get("current_index", self.history_length))
        self.horizon = int(self.manifest.get("horizon", self.cfg.horizon))
        self.window_length = int(self.manifest.get("window_length", self.history_length + 1 + self.horizon))
        self.denoising_steps = int(self.manifest.get("denoising_steps", self.cfg.denoising_steps))
        self.state_layout = self.manifest["state_layout"]
        self.action_joint_names = list(self.manifest["action_joint_names"])
        if len(self.action_joint_names) != self.action_dim:
            raise ValueError("manifest action_joint_names length must match action_dim")
        if self.action_joint_names != list(self.cfg.action_joint_names):
            raise ValueError(
                "manifest action_joint_names must match the K1 locomotion joystick action order: "
                f"expected={self.cfg.action_joint_names}, got={self.action_joint_names}"
            )

        self.action_joint_ids = torch.tensor(
            [self.robot.cfg.joint_names.index(name) for name in self.action_joint_names],
            dtype=torch.long,
            device=self.cfg.device,
        )
        self.default_action_joint_pos = self.robot.default_joint_pos[self.action_joint_ids].to(self.cfg.device)
        self.robot.data.to(self.cfg.device)
        self.robot.default_joint_pos = self.robot.default_joint_pos.to(self.cfg.device)
        self.action_scale = torch.full((self.action_dim,), float(self.cfg.action_scale), dtype=torch.float32, device=self.cfg.device)
        self.schedule = _DeployDiffusionSchedule(self.denoising_steps, self.cfg.device)
        self.joystick_guidance_cost = JoystickGuidanceCost(
            state_layout=self.state_layout,
            state_mean=self.state_mean,
            state_std=self.state_std,
            state_pseudoinverse=self.state_pseudoinverse,
            current_index=self.current_index,
            command_weights=self.cfg.command_weights,
        )
        if self.state_mean.numel() != self.raw_state_dim:
            raise ValueError("normalizer state_mean length must match raw_state_dim")
        if self.state_projection.shape != (self.projected_state_dim, self.raw_state_dim):
            raise ValueError(
                "state_projection.pt shape mismatch: "
                f"got={tuple(self.state_projection.shape)}, expected={(self.projected_state_dim, self.raw_state_dim)}"
            )

    def _check_artifacts(self) -> None:
        required = ["vae_decoder.pt", "diffusion.pt", "normalizer.json", "state_projection.pt", "manifest.json"]
        missing = [name for name in required if not os.path.isfile(os.path.join(self.model_dir, name))]
        if not missing:
            return
        expected = (
            "\ncd /mnt/ssd2/booster/booster_train\n"
            "source /mnt/ssd2/booster/env_isaaclab/bin/activate\n"
            "export DISPLAY=:1\n"
            "python scripts/beyond_mimic_diffusion/export.py \\\n"
            "  --vae-checkpoint <vae.pt> \\\n"
            "  --diffusion-checkpoint <diffusion.pt> \\\n"
            "  --normalizer <normalizer.json> \\\n"
            f"  --output-dir {self.model_dir} \\\n"
            "  --policy-dt 0.04 \\\n"
            "  --num-bodies 23 \\\n"
            f"  --action-joint-names {' '.join(self.cfg.action_joint_names)}"
        )
        raise FileNotFoundError(
            f"BeyondMimic diffusion artifacts are missing from {self.model_dir}: {missing}. "
            f"Export them with:{expected}"
        )

    def reset(self) -> None:
        self.last_action = torch.zeros(self.action_dim, dtype=torch.float32, device=self.cfg.device)
        self.state_history: list[torch.Tensor] = []
        self.teacher_obs_history = None
        self.teacher_last_action = torch.zeros(self.action_dim, dtype=torch.float32, device=self.cfg.device)
        self.last_inference_ms = 0.0
        self.last_velocity_error = 0.0

    def _projected_gravity(self) -> torch.Tensor:
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.cfg.device)
        return lab_math.quat_apply_inverse(self.robot.data.root_quat_w, gravity_w)

    def _current_state(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos[self.action_joint_ids] - self.default_action_joint_pos
        joint_vel = self.robot.data.joint_vel[self.action_joint_ids]
        root_lin_vel_w = lab_math.quat_apply(self.robot.data.root_quat_w, self.robot.data.root_lin_vel_b)
        root_ang_vel_w = lab_math.quat_apply(self.robot.data.root_quat_w, self.robot.data.root_ang_vel_b)
        state = _build_character_yaw_state(
            root_pos_w=self.robot.data.root_pos_w,
            root_quat_w=self.robot.data.root_quat_w,
            root_lin_vel_w=root_lin_vel_w,
            root_ang_vel_w=root_ang_vel_w,
            body_pos_w=self.robot.data.body_pos_w,
            body_lin_vel_w=self.robot.data.body_lin_vel_w,
            joint_pos_rel=joint_pos,
            joint_vel=joint_vel,
            last_action=self.last_action,
        )
        if state.numel() != self.raw_state_dim:
            raise RuntimeError(f"state dim mismatch: built {state.numel()}, manifest expects raw_state_dim={self.raw_state_dim}")
        return state

    def _normalized_current_state(self) -> torch.Tensor:
        return (self._current_state() - self.state_mean) / self.state_std

    def _project_state(self, state_norm: torch.Tensor) -> torch.Tensor:
        return torch.matmul(state_norm, self.state_projection.t())

    def _teacher_prior_action(self, projected_gravity: torch.Tensor, command: torch.Tensor) -> torch.Tensor | None:
        if self._teacher_prior is None:
            return None

        joint_pos = self.robot.data.joint_pos[self.action_joint_ids] - self.default_action_joint_pos
        joint_vel = self.robot.data.joint_vel[self.action_joint_ids]
        obs = torch.cat(
            (
                self.robot.data.root_ang_vel_b,
                projected_gravity,
                command,
                joint_pos,
                joint_vel * float(self.cfg.teacher_obs_dof_vel_scale),
                self.teacher_last_action,
            ),
            dim=0,
        ).clamp(-100.0, 100.0)
        if self.teacher_obs_history is None:
            self.teacher_obs_history = torch.zeros(
                int(self.cfg.teacher_obs_history_length),
                obs.numel(),
                dtype=torch.float32,
                device=self.cfg.device,
            )
        self.teacher_obs_history = self.teacher_obs_history.roll(shifts=-1, dims=0)
        self.teacher_obs_history[-1] = obs
        with torch.inference_mode():
            action = self._teacher_prior(self.teacher_obs_history.flatten()).squeeze(0)
        return torch.clamp(action, -100.0, 100.0)

    def _command_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [
                self.controller.vel_command.lin_vel_x,
                self.controller.vel_command.lin_vel_y,
                self.controller.vel_command.ang_vel_yaw,
            ],
            dtype=torch.float32,
            device=self.cfg.device,
        )

    def _teacher_prior_blend(self, command: torch.Tensor) -> float:
        blend = float(self.cfg.teacher_prior_blend)
        pure_backward = (
            float(command[0]) < -float(self.cfg.backward_command_threshold)
            and abs(float(command[1])) < float(self.cfg.backward_lateral_tolerance)
            and abs(float(command[2])) < float(self.cfg.backward_yaw_tolerance)
        )
        if pure_backward:
            blend = max(blend, float(self.cfg.backward_teacher_prior_blend))
        return min(max(blend, 0.0), 1.0)

    def _guidance_cost(self, clean_pred: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        projected_state = clean_pred[:, :, : self.projected_state_dim]
        return self.joystick_guidance_cost(projected_state, command)

    def _guided_denoise(
        self,
        sample: torch.Tensor,
        step_tensor: torch.Tensor,
        command: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.guidance_scale <= 0.0:
            with torch.no_grad():
                return self._diffusion(sample, step_tensor), sample

        sample_var = sample.detach().clone().requires_grad_(True)
        clean_pred = self._diffusion(sample_var, step_tensor)
        cost = self._guidance_cost(clean_pred, command)
        grad = torch.autograd.grad(cost, sample_var)[0]
        guided_sample = sample - float(self.cfg.guidance_scale) * grad.detach()
        return clean_pred.detach(), guided_sample.detach()

    def _history_window(self, current_state_norm: torch.Tensor) -> torch.Tensor:
        self.state_history.append(current_state_norm.detach())
        observed_length = self.current_index + 1
        self.state_history = self.state_history[-observed_length:]
        if len(self.state_history) < observed_length:
            padding = [self.state_history[0]] * (observed_length - len(self.state_history))
            history = padding + self.state_history
        else:
            history = self.state_history
        raw_window = torch.stack(history, dim=0).view(1, observed_length, self.raw_state_dim)
        return torch.matmul(raw_window, self.state_projection.t())

    def _sample_latent_trajectory(self, state_history_norm: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        observed_length = self.current_index + 1
        trajectory_dim = self.projected_state_dim + self.latent_dim
        sample = torch.randn(1, self.window_length, trajectory_dim, dtype=torch.float32, device=self.cfg.device)
        sample[:, :observed_length, : self.projected_state_dim] = state_history_norm

        for step in reversed(range(self.denoising_steps)):
            step_tensor = torch.full((1, self.window_length, 2), step, dtype=torch.long, device=self.cfg.device)
            clean_pred, noisy_sample = self._guided_denoise(sample, step_tensor, command)
            clean_pred[:, :observed_length, : self.projected_state_dim] = state_history_norm
            noisy_sample[:, :observed_length, : self.projected_state_dim] = state_history_norm
            sample = self.schedule.p_sample(noisy_sample, clean_pred, step, self.cfg.sampling_noise_scale)
            sample[:, :observed_length, : self.projected_state_dim] = state_history_norm
        return sample

    def inference(self) -> torch.Tensor:
        start = time.perf_counter()
        projected_gravity = self._projected_gravity()
        current_state_norm = (self._current_state() - self.state_mean) / self.state_std
        command = self._command_tensor()
        teacher_action = self._teacher_prior_action(projected_gravity, command)
        state_history_norm = self._history_window(current_state_norm)
        trajectory = self._sample_latent_trajectory(state_history_norm, command)
        action_index = min(self.current_index, self.window_length - 1)
        latent = trajectory[:, action_index, self.projected_state_dim:]
        with torch.inference_mode():
            action_norm = self._decoder(current_state_norm.view(1, -1), latent).flatten()
        action = action_norm * self.action_std + self.action_mean
        action = torch.clamp(action, -float(self.cfg.max_action), float(self.cfg.max_action))
        if teacher_action is not None:
            blend = self._teacher_prior_blend(command)
            action = (1.0 - blend) * action + blend * teacher_action.to(self.cfg.device)
        if self.cfg.action_smoothing > 0.0:
            smoothing = min(max(float(self.cfg.action_smoothing), 0.0), 0.99)
            action = (1.0 - smoothing) * action + smoothing * self.last_action

        dof_targets = self.robot.default_joint_pos.clone()
        dof_targets[self.action_joint_ids] = self.default_action_joint_pos + action * self.action_scale
        self.last_action = action.detach()
        self.teacher_last_action = action.detach()
        self.last_inference_ms = (time.perf_counter() - start) * 1000.0
        current_velocity = torch.cat((self.robot.data.root_lin_vel_b[:2], self.robot.data.root_ang_vel_b[2:3]))
        self.last_velocity_error = float(torch.norm(current_velocity - command).detach().cpu())
        return dof_targets.detach()


@configclass
class BeyondMimicDiffusionJoystickPolicyCfg(PolicyCfg):
    constructor = BeyondMimicDiffusionJoystickPolicy
    checkpoint_path: str = ""
    model_dir: str = "models/k1_beyondmimic_joystick"
    action_joint_names: list[str] = _K1_WALK_ACTION_JOINT_NAMES
    denoising_steps: int = 20
    horizon: int = 16
    history_length: int = 4
    latent_dim: int = 32
    guidance_scale: float = 0.15
    command_weights: list[float] = [1.0, 1.0, 0.5]
    sampling_noise_scale: float = 1.0
    action_smoothing: float = 0.0
    teacher_prior_blend: float = 0.0
    backward_teacher_prior_blend: float = 0.0
    backward_command_threshold: float = 0.05
    backward_lateral_tolerance: float = 0.05
    backward_yaw_tolerance: float = 0.05
    teacher_policy_path: str = "../locomotion/models/k1_locomotion.pt"
    teacher_obs_history_length: int = 10
    teacher_obs_dof_vel_scale: float = 0.1
    action_scale: float = _K1_WALK_ACTION_SCALE
    max_action: float = 8.0


@configclass
class K1BeyondMimicDiffusionJoystickControllerCfg(K1WalkControllerCfg1):
    policy: BeyondMimicDiffusionJoystickPolicyCfg = BeyondMimicDiffusionJoystickPolicyCfg()

    def __post_init__(self):
        super().__post_init__()
        self.policy_dt = 0.04
        self.mujoco.physics_dt = self.policy_dt / self.mujoco.decimation
