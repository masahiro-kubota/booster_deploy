from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import pkgutil
import sys
from time import sleep

import mujoco
import mujoco.viewer
import numpy as np
import torch

sys.path.append(".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run headless MuJoCo Sim2Sim evaluation for a Booster Deploy task."
    )
    parser.add_argument("--task", required=True, help="Registered Booster Deploy task name.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional TorchScript policy path. Overrides the task policy checkpoint_path.",
    )
    parser.add_argument("--device", default="cpu", help="Policy device, for example 'cpu' or 'cuda:0'.")
    parser.add_argument("--num-runs", type=int, default=20, help="Number of randomized evaluation runs.")
    parser.add_argument("--max-time-s", type=float, default=20.0, help="Maximum duration per run.")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed.")
    parser.add_argument("--output-dir", default=None, help="Directory for runs.csv and summary.json.")
    parser.add_argument("--render", action="store_true", help="Open the MuJoCo viewer while evaluating.")
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sleep for policy_dt between rendered steps.",
    )
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="Keep running until the target time after recording the first failure.",
    )

    parser.add_argument("--yaw-range-deg", type=float, default=10.0, help="Initial yaw perturbation range.")
    parser.add_argument("--xy-offset-range", type=float, default=0.03, help="Initial x/y offset range in meters.")
    parser.add_argument(
        "--root-height-offset-range", type=float, default=0.01, help="Initial root height offset range in meters."
    )
    parser.add_argument("--friction-min", type=float, default=0.6, help="Minimum ground friction.")
    parser.add_argument("--friction-max", type=float, default=1.2, help="Maximum ground friction.")
    parser.add_argument(
        "--fixed-motion-start-frame",
        type=int,
        default=None,
        help="Use a fixed motion start frame. By default a valid start frame is sampled per run.",
    )

    parser.add_argument("--min-root-z", type=float, default=0.35, help="Failure threshold for root height.")
    parser.add_argument(
        "--max-roll-pitch-deg", type=float, default=45.0, help="Failure threshold for absolute roll or pitch."
    )
    parser.add_argument(
        "--max-anchor-pos-error", type=float, default=0.50, help="Failure threshold for anchor position error."
    )
    parser.add_argument(
        "--max-anchor-rot-error-deg", type=float, default=60.0, help="Failure threshold for anchor rotation error."
    )
    return parser.parse_args()


def import_tasks() -> None:
    import tasks as tasks_pkg

    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        __import__(mod_info.name)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def yaw_quat(yaw: float) -> np.ndarray:
    half_yaw = 0.5 * yaw
    return np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=np.float64)


def quat_to_roll_pitch(q: np.ndarray) -> tuple[float, float]:
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    return roll, pitch


def quat_angle_error(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(abs(np.dot(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def set_ground_friction(controller, friction: float) -> None:
    ground_id = mujoco.mj_name2id(controller.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if ground_id >= 0:
        controller.mj_model.geom_friction[ground_id, 0] = friction


def reset_to_motion_frame(controller, frame: int, yaw: float, xy_offset: np.ndarray, root_height_offset: float) -> None:
    policy = controller.policy
    motion = policy.motion
    frame = int(np.clip(frame, 0, motion.time_step_total - 1))
    anchor_index = motion.track_body_names.index(policy.cfg.anchor_body_name)

    root_pos = tensor_to_numpy(motion.body_pos_w[frame, anchor_index]).astype(np.float64)
    root_quat = tensor_to_numpy(motion.body_quat_w[frame, anchor_index]).astype(np.float64)
    joint_pos_sim = tensor_to_numpy(motion.joint_pos[frame]).astype(np.float64)
    joint_vel_sim = tensor_to_numpy(motion.joint_vel[frame]).astype(np.float64)

    joint_pos_real = joint_pos_sim[controller.robot.data.sim2real_joint_indexes]
    joint_vel_real = joint_vel_sim[controller.robot.data.sim2real_joint_indexes]

    root_pos[:2] += xy_offset
    root_pos[2] += root_height_offset
    root_quat = quat_mul(yaw_quat(yaw), root_quat)
    root_quat = root_quat / np.linalg.norm(root_quat)

    controller.mj_data.qpos[:] = np.concatenate([root_pos, root_quat, joint_pos_real])
    controller.mj_data.qvel[:] = 0.0
    controller.mj_data.qvel[6:] = joint_vel_real
    mujoco.mj_forward(controller.mj_model, controller.mj_data)
    controller.update_state()


def sample_start_frame(rng: np.random.Generator, motion_steps: int, eval_steps: int, fixed_frame: int | None) -> int:
    if fixed_frame is not None:
        return int(np.clip(fixed_frame, 0, max(0, motion_steps - 1)))
    max_start = max(0, motion_steps - eval_steps - 1)
    if max_start == 0:
        return 0
    return int(rng.integers(0, max_start + 1))


def check_failure(args: argparse.Namespace, metrics: dict[str, float], is_running: bool) -> str | None:
    if not is_running:
        return "safety_fallback"
    if metrics["root_z"] < args.min_root_z:
        return "root_height"
    if max(abs(metrics["roll_deg"]), abs(metrics["pitch_deg"])) > args.max_roll_pitch_deg:
        return "roll_pitch"
    if metrics["anchor_pos_error"] > args.max_anchor_pos_error:
        return "anchor_pos"
    if metrics["anchor_rot_error_deg"] > args.max_anchor_rot_error_deg:
        return "anchor_rot"
    return None


def update_viewer(controller, viewer) -> None:
    if controller.cfg.mujoco.visualize_reference_ghost:
        controller.render_reference_robot(
            viewer,
            rgba=controller._ghost_rgba,
        )
    viewer.cam.lookat[:] = controller.mj_data.qpos.astype(np.float32)[0:3]
    viewer.sync()


def collect_step_metrics(controller, reference_qpos) -> tuple[dict[str, float], float, float]:
    root_pos = controller.mj_data.qpos[:3].copy()
    root_quat = controller.mj_data.qpos[3:7].copy()
    roll, pitch = quat_to_roll_pitch(root_quat)
    dof_pos = controller.mj_data.qpos[7:].copy()

    if reference_qpos is None:
        ref_root_pos = root_pos
        ref_root_quat = root_quat
        ref_dof_pos = dof_pos
    else:
        ref_root_pos = reference_qpos[:3]
        ref_root_quat = reference_qpos[3:7]
        ref_dof_pos = reference_qpos[7:]

    metrics = {
        "root_z": float(root_pos[2]),
        "roll_deg": math.degrees(roll),
        "pitch_deg": math.degrees(pitch),
        "anchor_pos_error": float(np.linalg.norm(root_pos - ref_root_pos)),
        "anchor_rot_error_deg": math.degrees(quat_angle_error(root_quat, ref_root_quat)),
    }
    joint_pos_error = float(np.sqrt(np.mean(np.square(dof_pos - ref_dof_pos))))
    last_action = getattr(controller.policy, "last_action", torch.zeros(controller.robot.num_joints))
    action_abs = np.abs(tensor_to_numpy(last_action)).astype(np.float64)
    return metrics, joint_pos_error, action_abs


def run_one(
    args: argparse.Namespace, base_cfg, run_index: int, rng: np.random.Generator
) -> dict[str, float | int | str | bool | None]:
    from booster_deploy.controllers.mujoco_controller import MujocoController

    cfg = copy.deepcopy(base_cfg)
    cfg.policy.device = args.device
    if args.checkpoint is not None:
        cfg.policy.checkpoint_path = args.checkpoint

    controller = MujocoController(cfg)
    set_ground_friction(controller, float(rng.uniform(args.friction_min, args.friction_max)))

    requested_eval_steps = int(round(args.max_time_s / cfg.policy_dt))
    motion_steps = int(controller.policy.motion.time_step_total)
    motion_start_frame = sample_start_frame(rng, motion_steps, requested_eval_steps, args.fixed_motion_start_frame)
    eval_steps = max(1, min(requested_eval_steps, motion_steps - motion_start_frame))
    yaw = math.radians(float(rng.uniform(-args.yaw_range_deg, args.yaw_range_deg)))
    xy_offset = rng.uniform(-args.xy_offset_range, args.xy_offset_range, size=2)
    root_height_offset = float(rng.uniform(-args.root_height_offset_range, args.root_height_offset_range))

    reset_to_motion_frame(controller, motion_start_frame, yaw, xy_offset, root_height_offset)
    controller.start()
    controller.policy.current_frame = motion_start_frame

    root_z_values: list[float] = []
    roll_abs_values: list[float] = []
    pitch_abs_values: list[float] = []
    anchor_pos_errors: list[float] = []
    anchor_rot_errors: list[float] = []
    joint_pos_errors: list[float] = []
    action_abs_values: list[float] = []
    failure_reason = None
    completed_steps = 0

    viewer_context = mujoco.viewer.launch_passive(controller.mj_model, controller.mj_data) if args.render else None
    viewer = None
    if viewer_context is not None:
        viewer = viewer_context.__enter__()
        controller.viewer = viewer
        viewer.cam.elevation = -20
        update_viewer(controller, viewer)

    for step in range(eval_steps):
        if viewer is not None and not viewer.is_running():
            failure_reason = "viewer_closed"
            break
        controller.update_state()
        dof_targets = controller.policy_step()

        reference_qpos = controller._reference_qpos
        controller.ctrl_step(dof_targets)
        controller.update_state()

        metrics, joint_pos_error, action_abs = collect_step_metrics(controller, reference_qpos)

        root_z_values.append(metrics["root_z"])
        roll_abs_values.append(abs(metrics["roll_deg"]))
        pitch_abs_values.append(abs(metrics["pitch_deg"]))
        anchor_pos_errors.append(metrics["anchor_pos_error"])
        anchor_rot_errors.append(metrics["anchor_rot_error_deg"])
        joint_pos_errors.append(joint_pos_error)
        action_abs_values.extend(action_abs.tolist())

        completed_steps = step + 1
        step_failure_reason = check_failure(args, metrics, controller.is_running)
        if failure_reason is None:
            failure_reason = step_failure_reason
        if step_failure_reason is not None and not args.continue_after_failure:
            break

        if viewer is not None:
            update_viewer(controller, viewer)
            if args.realtime:
                sleep(controller.cfg.policy_dt)

    if viewer_context is not None:
        controller.viewer = None
        viewer = None
        viewer_context.__exit__(None, None, None)
    controller.stop()
    success = failure_reason is None and completed_steps >= eval_steps

    return {
        "run_index": run_index,
        "seed": args.seed + run_index,
        "task": args.task,
        "checkpoint": cfg.policy.checkpoint_path,
        "success": success,
        "failure_reason": failure_reason,
        "target_time_s": eval_steps * cfg.policy_dt,
        "survival_time_s": completed_steps * cfg.policy_dt,
        "completed_steps": completed_steps,
        "motion_start_frame": motion_start_frame,
        "initial_yaw_deg": math.degrees(yaw),
        "initial_x_offset": float(xy_offset[0]),
        "initial_y_offset": float(xy_offset[1]),
        "root_height_offset": root_height_offset,
        "ground_friction": float(controller.mj_model.geom_friction[0, 0]),
        "min_root_z": min(root_z_values) if root_z_values else None,
        "max_abs_roll_deg": max(roll_abs_values) if roll_abs_values else None,
        "max_abs_pitch_deg": max(pitch_abs_values) if pitch_abs_values else None,
        "mean_anchor_pos_error": float(np.mean(anchor_pos_errors)) if anchor_pos_errors else None,
        "max_anchor_pos_error": max(anchor_pos_errors) if anchor_pos_errors else None,
        "mean_anchor_rot_error_deg": float(np.mean(anchor_rot_errors)) if anchor_rot_errors else None,
        "max_anchor_rot_error_deg": max(anchor_rot_errors) if anchor_rot_errors else None,
        "mean_joint_pos_error": float(np.mean(joint_pos_errors)) if joint_pos_errors else None,
        "max_joint_pos_error": max(joint_pos_errors) if joint_pos_errors else None,
        "mean_action_abs": float(np.mean(action_abs_values)) if action_abs_values else None,
        "max_action_abs": max(action_abs_values) if action_abs_values else None,
    }


def percentile(values: list[float], q: float) -> float | None:
    clean_values = [v for v in values if v is not None and not math.isnan(v)]
    if not clean_values:
        return None
    return float(np.percentile(clean_values, q))


def summarize(rows: list[dict]) -> dict:
    num_runs = len(rows)
    successes = [row for row in rows if row["success"]]

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    survival_times = values("survival_time_s")
    min_root_z = values("min_root_z")
    anchor_pos = values("max_anchor_pos_error")
    anchor_rot = values("max_anchor_rot_error_deg")
    joint_pos = values("max_joint_pos_error")

    return {
        "num_runs": num_runs,
        "success_count": len(successes),
        "success_rate": len(successes) / num_runs if num_runs else 0.0,
        "mean_survival_time_s": float(np.mean(survival_times)) if survival_times else None,
        "p10_survival_time_s": percentile(survival_times, 10.0),
        "mean_min_root_z": float(np.mean(min_root_z)) if min_root_z else None,
        "worst_min_root_z": min(min_root_z) if min_root_z else None,
        "p95_anchor_pos_error": percentile(anchor_pos, 95.0),
        "p95_anchor_rot_error_deg": percentile(anchor_rot, 95.0),
        "p95_joint_pos_error": percentile(joint_pos, 95.0),
        "failure_counts": {
            reason: sum(1 for row in rows if row["failure_reason"] == reason)
            for reason in sorted({row["failure_reason"] for row in rows if row["failure_reason"] is not None})
        },
    }


def write_outputs(output_dir: Path, rows: list[dict], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "runs.csv"
    summary_path = output_dir / "summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[INFO] Wrote per-run results: {csv_path}")
    print(f"[INFO] Wrote summary: {summary_path}")


def main() -> None:
    args = parse_args()
    import_tasks()

    from booster_deploy.utils.registry import get_task

    base_cfg = get_task(args.task)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("logs") / "mujoco_eval" / f"{args.task}_{timestamp}"
    else:
        output_dir = Path(args.output_dir)

    rows = []
    for run_index in range(args.num_runs):
        rng = np.random.default_rng(args.seed + run_index)
        row = run_one(args, base_cfg, run_index, rng)
        rows.append(row)
        status = "success" if row["success"] else f"failed:{row['failure_reason']}"
        print(
            f"[{run_index + 1}/{args.num_runs}] {status} "
            f"survival={row['survival_time_s']:.2f}s min_root_z={row['min_root_z']:.3f}"
        )

    summary = {
        "task": args.task,
        "checkpoint": args.checkpoint or base_cfg.policy.checkpoint_path,
        "device": args.device,
        "max_time_s": args.max_time_s,
        "thresholds": {
            "min_root_z": args.min_root_z,
            "max_roll_pitch_deg": args.max_roll_pitch_deg,
            "max_anchor_pos_error": args.max_anchor_pos_error,
            "max_anchor_rot_error_deg": args.max_anchor_rot_error_deg,
        },
        **summarize(rows),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    write_outputs(output_dir, rows, summary)


if __name__ == "__main__":
    main()
