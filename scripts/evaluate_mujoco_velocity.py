#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import pkgutil
import sys

import numpy as np
import torch

sys.path.append(".")


COMMAND_SCHEDULES = {
    "forward": [(0.0, (0.5, 0.0, 0.0))],
    "backward": [(0.0, (-0.35, 0.0, 0.0))],
    "lateral": [(0.0, (0.0, 0.35, 0.0))],
    "yaw": [(0.0, (0.0, 0.0, 0.45))],
    "mixed": [
        (0.0, (0.4, 0.0, 0.0)),
        (2.5, (0.0, 0.35, 0.0)),
        (5.0, (0.35, 0.0, 0.35)),
        (7.5, (-0.25, 0.0, -0.35)),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a velocity-commanded MuJoCo task.")
    parser.add_argument("--task", default="k1_beyondmimic_joystick")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--schedules", nargs="+", default=list(COMMAND_SCHEDULES))
    parser.add_argument("--min-root-z", type=float, default=0.32)
    parser.add_argument("--max-roll-pitch-deg", type=float, default=55.0)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--denoising-steps", type=int, default=None)
    parser.add_argument("--command-weights", type=float, nargs=3, default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--sampling-noise-scale", type=float, default=None)
    parser.add_argument("--action-smoothing", type=float, default=None)
    parser.add_argument("--action-scale", type=float, default=None)
    return parser.parse_args()


def import_tasks() -> None:
    import tasks as tasks_pkg

    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        __import__(mod_info.name)


def quat_to_roll_pitch(q: np.ndarray) -> tuple[float, float]:
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    return roll, pitch


def command_at(schedule: list[tuple[float, tuple[float, float, float]]], elapsed: float) -> tuple[float, float, float]:
    current = schedule[0][1]
    for start_time, command in schedule:
        if elapsed >= start_time:
            current = command
        else:
            break
    return current


def set_command(controller, command: tuple[float, float, float]) -> None:
    controller.vel_command.lin_vel_x = float(command[0])
    controller.vel_command.lin_vel_y = float(command[1])
    controller.vel_command.ang_vel_yaw = float(command[2])


def run_schedule(args: argparse.Namespace, base_cfg, schedule_name: str) -> dict[str, float | str | bool]:
    from booster_deploy.controllers.mujoco_controller import MujocoController

    cfg = copy.deepcopy(base_cfg)
    cfg.policy.device = args.device
    controller = MujocoController(cfg)
    controller.update_state()
    controller.start()
    schedule = COMMAND_SCHEDULES[schedule_name]
    total_steps = int(round(args.duration_s / cfg.policy_dt))
    velocity_errors = []
    yaw_errors = []
    action_abs = []
    inference_ms = []
    failure = None

    for step in range(total_steps):
        elapsed = step * cfg.policy_dt
        command = command_at(schedule, elapsed)
        set_command(controller, command)
        controller.update_state()
        dof_targets = controller.policy_step()
        controller.ctrl_step(dof_targets)

        root_quat = controller.mj_data.qpos.astype(np.float32)[3:7]
        roll, pitch = quat_to_roll_pitch(root_quat)
        root_z = float(controller.mj_data.qpos[2])
        if root_z < args.min_root_z:
            failure = "root_height"
        if max(abs(math.degrees(roll)), abs(math.degrees(pitch))) > args.max_roll_pitch_deg:
            failure = "roll_pitch"
        if not controller.is_running:
            failure = "safety_fallback"

        root_vel = controller.robot.data.root_lin_vel_b.detach().cpu().numpy()
        root_ang = controller.robot.data.root_ang_vel_b.detach().cpu().numpy()
        velocity_errors.append(float(np.linalg.norm(root_vel[:2] - np.asarray(command[:2], dtype=np.float32))))
        yaw_errors.append(float(abs(root_ang[2] - command[2])))
        last_action = getattr(controller.policy, "last_action", torch.zeros(1))
        action_abs.append(float(torch.mean(torch.abs(last_action)).detach().cpu()))
        inference_ms.append(float(getattr(controller.policy, "last_inference_ms", 0.0)))
        if failure is not None:
            break

    return {
        "schedule": schedule_name,
        "success": failure is None,
        "failure": failure or "",
        "steps": step + 1,
        "mean_planar_velocity_error": float(np.mean(velocity_errors)) if velocity_errors else float("nan"),
        "mean_yaw_rate_error": float(np.mean(yaw_errors)) if yaw_errors else float("nan"),
        "mean_abs_action": float(np.mean(action_abs)) if action_abs else float("nan"),
        "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else float("nan"),
        "p95_inference_ms": float(np.percentile(inference_ms, 95)) if inference_ms else float("nan"),
    }


def main() -> None:
    args = parse_args()
    import_tasks()
    from booster_deploy.utils.registry import get_task

    base_cfg = get_task(args.task)
    if args.guidance_scale is not None and hasattr(base_cfg.policy, "guidance_scale"):
        base_cfg.policy.guidance_scale = float(args.guidance_scale)
    if args.denoising_steps is not None and hasattr(base_cfg.policy, "denoising_steps"):
        base_cfg.policy.denoising_steps = int(args.denoising_steps)
    if args.command_weights is not None and hasattr(base_cfg.policy, "command_weights"):
        base_cfg.policy.command_weights = [float(value) for value in args.command_weights]
    if args.model_dir is not None and hasattr(base_cfg.policy, "model_dir"):
        base_cfg.policy.model_dir = args.model_dir
    if args.sampling_noise_scale is not None and hasattr(base_cfg.policy, "sampling_noise_scale"):
        base_cfg.policy.sampling_noise_scale = float(args.sampling_noise_scale)
    if args.action_smoothing is not None and hasattr(base_cfg.policy, "action_smoothing"):
        base_cfg.policy.action_smoothing = float(args.action_smoothing)
    if args.action_scale is not None and hasattr(base_cfg.policy, "action_scale"):
        base_cfg.policy.action_scale = float(args.action_scale)
    unknown = [name for name in args.schedules if name not in COMMAND_SCHEDULES]
    if unknown:
        raise ValueError(f"Unknown schedules: {unknown}")

    rows = [run_schedule(args, base_cfg, name) for name in args.schedules]
    if args.output_dir is None:
        output_dir = Path("logs") / "mujoco_velocity_eval" / f"{args.task}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "runs.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "task": args.task,
        "duration_s": args.duration_s,
        "success_rate": sum(1 for row in rows if row["success"]) / max(len(rows), 1),
        "runs": rows,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
