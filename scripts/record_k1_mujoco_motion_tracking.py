#!/usr/bin/env python3
"""Record MuJoCo motion tracking with a reference ghost and error evidence.

This script is intended to run from the root of the booster_deploy checkout so
that its task registry and Python packages are importable.  It accepts an
absolute motion NPZ path and TorchScript policy path, avoiding a task-specific
copy of either artifact.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import pkgutil
import platform
import subprocess
import sys
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

sys.path.append(".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record MuJoCo actual/reference overlay and tracking errors."
    )
    parser.add_argument("--task", default="k1_pikachu_idle")
    parser.add_argument("--checkpoint", required=True, help="TorchScript policy path.")
    parser.add_argument("--motion", required=True, help="Reference motion NPZ path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--duration-s", type=float, default=13.0)
    parser.add_argument("--motion-start-frame", type=int, default=0)
    parser.add_argument("--render-fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=368)
    parser.add_argument("--camera-distance", type=float, default=2.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-15.0)
    parser.add_argument(
        "--ghost-rgba",
        type=float,
        nargs=4,
        default=[0.12, 1.0, 0.22, 0.32],
        metavar=("R", "G", "B", "A"),
    )
    parser.add_argument(
        "--continue-after-safety-stop",
        action="store_true",
        help="Continue applying the last target if the policy safety fallback stops.",
    )
    return parser.parse_args()


def import_tasks() -> None:
    import tasks as tasks_pkg

    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        __import__(mod_info.name)


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quat_angle_error_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def quat_rotate_inverse(q: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate vectors by inverse unit quaternion q in MuJoCo wxyz order."""
    q = q / np.linalg.norm(q)
    w = q[0]
    xyz = q[1:]
    # q^-1 * v * q, vectorized.  Equivalent to the standard rotation formula
    # with the vector part negated.
    inv_xyz = -xyz
    return (
        vectors * (2.0 * w * w - 1.0)
        + 2.0 * np.outer(vectors @ inv_xyz, inv_xyz)
        + 2.0 * w * np.cross(inv_xyz, vectors)
    )


def wrapped_joint_error(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    delta = actual - reference
    return np.arctan2(np.sin(delta), np.cos(delta))


def add_reference_geoms(renderer: mujoco.Renderer, controller, rgba: np.ndarray) -> int:
    """Append the controller's reference FK as translucent dynamic geoms."""
    first_reference_geom = int(renderer.scene.ngeom)
    perturb = mujoco.MjvPerturb()
    mujoco.mjv_defaultPerturb(perturb)
    mujoco.mjv_addGeoms(
        controller.mj_model,
        controller._ghost_mj_data,
        controller._ghost_scene_option,
        perturb,
        int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
        renderer.scene,
    )
    for geom_index in range(first_reference_geom, int(renderer.scene.ngeom)):
        renderer.scene.geoms[geom_index].rgba[:] = rgba
    return int(renderer.scene.ngeom) - first_reference_geom


def annotate_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    line_height = 14
    box_height = 12 + line_height * len(lines)
    box_width = min(image.width - 16, 360)
    draw.rounded_rectangle((8, 8, 8 + box_width, 8 + box_height), radius=5, fill=(0, 0, 0, 168))
    y = 14
    for index, line in enumerate(lines):
        fill = (95, 255, 115, 255) if index == 1 else (255, 255, 255, 255)
        draw.text((15, y), line, font=font, fill=fill)
        y += line_height
    return np.asarray(image)


def save_contact_sheet(frames: list[tuple[str, np.ndarray]], path: Path) -> None:
    if not frames:
        return
    thumb_width = 480
    aspect = frames[0][1].shape[0] / frames[0][1].shape[1]
    thumb_height = int(round(thumb_width * aspect))
    label_height = 28
    columns = 2
    rows = int(math.ceil(len(frames) / columns))
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, frame) in enumerate(frames):
        row, column = divmod(index, columns)
        image = Image.fromarray(frame).resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = column * thumb_width
        y = row * (thumb_height + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 8, y + thumb_height + 7), label, font=font, fill="black")
    sheet.save(path)


def array_stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "rmse": float(np.sqrt(np.mean(np.square(flat)))),
        "p95": float(np.percentile(flat, 95.0)),
        "max": float(np.max(flat)),
    }


def git_head(cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_step_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_named_stats(
    path: Path, names: list[str], errors: np.ndarray, *, scale: float, unit: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        stats = array_stats(np.abs(errors[:, index]) * scale)
        rows.append({"name": name, "unit": unit, **stats})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_errors(
    path: Path,
    time_s: np.ndarray,
    joint_errors: np.ndarray,
    root_pos_errors: np.ndarray,
    root_rot_errors_deg: np.ndarray,
    body_global_errors: np.ndarray,
    body_local_errors: np.ndarray,
) -> None:
    joint_rmse_deg = np.degrees(np.sqrt(np.mean(np.square(joint_errors), axis=1)))
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(time_s, joint_rmse_deg, color="#8e44ad", linewidth=1.2)
    axes[0].set_ylabel("Joint RMSE [deg]")
    axes[1].plot(time_s, root_pos_errors * 100.0, color="#d35400", linewidth=1.2)
    axes[1].set_ylabel("Root pos [cm]")
    axes[2].plot(time_s, root_rot_errors_deg, color="#2471a3", linewidth=1.2)
    axes[2].set_ylabel("Root rot [deg]")
    axes[3].plot(
        time_s,
        np.mean(body_global_errors, axis=1) * 100.0,
        label="global",
        color="#c0392b",
        linewidth=1.2,
    )
    axes[3].plot(
        time_s,
        np.mean(body_local_errors, axis=1) * 100.0,
        label="root-local",
        color="#16a085",
        linewidth=1.2,
    )
    axes[3].set_ylabel("Mean body pos [cm]")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(loc="upper right")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle("MuJoCo actual vs original reference motion")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_named_bars(
    path: Path,
    joint_names: list[str],
    joint_errors: np.ndarray,
    body_names: list[str],
    body_local_errors: np.ndarray,
) -> None:
    joint_mae = np.degrees(np.mean(np.abs(joint_errors), axis=0))
    body_mae_cm = np.mean(body_local_errors, axis=0) * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    joint_order = np.argsort(joint_mae)
    body_order = np.argsort(body_mae_cm)
    axes[0].barh(np.asarray(joint_names)[joint_order], joint_mae[joint_order], color="#8e44ad")
    axes[0].set_xlabel("Joint MAE [deg]")
    axes[0].grid(True, axis="x", alpha=0.25)
    axes[1].barh(np.asarray(body_names)[body_order], body_mae_cm[body_order], color="#16a085")
    axes[1].set_xlabel("Root-local body-position MAE [cm]")
    axes[1].grid(True, axis="x", alpha=0.25)
    fig.suptitle("Tracking error by joint and body")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    import_tasks()
    from booster_deploy.controllers.mujoco_controller import MujocoController
    from booster_deploy.utils.registry import get_task

    output_dir = Path(args.output_dir).resolve()
    video_dir = output_dir / "videos"
    frame_dir = output_dir / "frames"
    data_dir = output_dir / "data"
    plot_dir = output_dir / "plots"
    for directory in (video_dir, frame_dir, data_dir, plot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint).resolve()
    motion_path = Path(args.motion).resolve()
    cfg = copy.deepcopy(get_task(args.task))
    cfg.policy.device = args.device
    cfg.policy.checkpoint_path = str(checkpoint_path)
    cfg.policy.motion_path = str(motion_path)
    cfg.policy.loop_motion = True
    cfg.mujoco.visualize_reference_ghost = True
    cfg.mujoco.ghost_rgba = list(args.ghost_rgba)

    controller = MujocoController(cfg)
    motion = controller.policy.motion
    anchor_index = motion.track_body_names.index(controller.policy.cfg.anchor_body_name)
    start_frame = int(np.clip(args.motion_start_frame, 0, motion.time_step_total - 1))

    ref_root_pos = tensor_to_numpy(motion.body_pos_w[start_frame, anchor_index]).astype(np.float64)
    ref_root_quat = tensor_to_numpy(motion.body_quat_w[start_frame, anchor_index]).astype(np.float64)
    ref_joint_sim = tensor_to_numpy(motion.joint_pos[start_frame]).astype(np.float64)
    ref_joint_vel_sim = tensor_to_numpy(motion.joint_vel[start_frame]).astype(np.float64)
    sim2real = controller.robot.data.sim2real_joint_indexes
    controller.mj_data.qpos[:] = np.concatenate([ref_root_pos, ref_root_quat, ref_joint_sim[sim2real]])
    controller.mj_data.qvel[:] = 0.0
    controller.mj_data.qvel[6:] = ref_joint_vel_sim[sim2real]
    mujoco.mj_forward(controller.mj_model, controller.mj_data)
    controller.update_state()
    controller.start()
    controller.policy.current_frame = start_frame

    body_names = list(controller.robot.cfg.sim_body_names)
    motion_body_names = list(motion._body_names)
    motion_body_indexes = [motion_body_names.index(name) for name in body_names]
    joint_names_real = list(controller.robot.cfg.joint_names)

    policy_dt = float(cfg.policy_dt)
    total_steps = int(round(args.duration_s / policy_dt))
    render_stride = max(1, int(round(1.0 / (args.render_fps * policy_dt))))
    video_path = video_dir / "actual_vs_reference_ghost.mp4"

    renderer = mujoco.Renderer(controller.mj_model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    rgba = np.asarray(args.ghost_rgba, dtype=np.float32)

    time_values: list[float] = []
    frame_indexes: list[int] = []
    actual_qpos_values: list[np.ndarray] = []
    reference_qpos_values: list[np.ndarray] = []
    dof_target_values: list[np.ndarray] = []
    actual_body_values: list[np.ndarray] = []
    reference_body_values: list[np.ndarray] = []
    reference_fk_body_values: list[np.ndarray] = []
    joint_error_values: list[np.ndarray] = []
    root_pos_error_values: list[float] = []
    root_rot_error_values_deg: list[float] = []
    body_global_error_values: list[np.ndarray] = []
    body_local_error_values: list[np.ndarray] = []
    reference_fk_motion_error_values: list[np.ndarray] = []
    step_rows: list[dict[str, Any]] = []

    selected_frames: dict[str, tuple[int, np.ndarray]] = {}
    worst_joint = (-1.0, -1, None)
    worst_body = (-1.0, -1, None)
    rendered_frames = 0
    reference_geom_count: int | None = None
    safety_stop_step: int | None = None
    last_targets: torch.Tensor | None = None

    writer = imageio.get_writer(
        video_path, fps=args.render_fps, codec="libx264", quality=8, macro_block_size=16
    )
    try:
        for step in range(total_steps):
            controller.update_state()
            if controller.is_running:
                dof_targets = controller.policy_step()
                last_targets = dof_targets
            elif args.continue_after_safety_stop and last_targets is not None:
                dof_targets = last_targets
            else:
                break

            reference_qpos = controller._reference_qpos.copy()
            reference_frame = int((start_frame + step) % motion.time_step_total)
            controller.ctrl_step(dof_targets)
            controller.update_state()
            if not controller.is_running and safety_stop_step is None:
                safety_stop_step = step

            actual_qpos = controller.mj_data.qpos.copy()
            actual_body = controller.mj_data.xpos[controller._body_ids].copy()
            reference_fk_body = controller._ghost_mj_data.xpos[controller._body_ids].copy()
            reference_body = tensor_to_numpy(motion._body_pos_w[reference_frame, motion_body_indexes]).astype(
                np.float64
            )

            joint_error = wrapped_joint_error(actual_qpos[7:], reference_qpos[7:])
            root_pos_error = float(np.linalg.norm(actual_qpos[:3] - reference_qpos[:3]))
            root_rot_error_deg = math.degrees(
                quat_angle_error_rad(actual_qpos[3:7], reference_qpos[3:7])
            )
            body_global_error = np.linalg.norm(actual_body - reference_body, axis=1)

            actual_local = quat_rotate_inverse(
                actual_qpos[3:7], actual_body - actual_qpos[:3][None, :]
            )
            reference_local = quat_rotate_inverse(
                reference_qpos[3:7], reference_body - reference_qpos[:3][None, :]
            )
            body_local_error = np.linalg.norm(actual_local - reference_local, axis=1)
            reference_fk_motion_error = np.linalg.norm(reference_fk_body - reference_body, axis=1)

            time_s = (step + 1) * policy_dt
            joint_rmse_rad = float(np.sqrt(np.mean(np.square(joint_error))))
            joint_mae_rad = float(np.mean(np.abs(joint_error)))
            step_rows.append(
                {
                    "step": step,
                    "time_s": time_s,
                    "reference_frame": reference_frame,
                    "joint_rmse_deg": math.degrees(joint_rmse_rad),
                    "joint_mae_deg": math.degrees(joint_mae_rad),
                    "joint_max_abs_deg": math.degrees(float(np.max(np.abs(joint_error)))),
                    "root_pos_error_cm": root_pos_error * 100.0,
                    "root_rot_error_deg": root_rot_error_deg,
                    "body_global_mean_error_cm": float(np.mean(body_global_error)) * 100.0,
                    "body_global_max_error_cm": float(np.max(body_global_error)) * 100.0,
                    "body_local_mean_error_cm": float(np.mean(body_local_error)) * 100.0,
                    "body_local_max_error_cm": float(np.max(body_local_error)) * 100.0,
                    "reference_fk_motion_mean_error_cm": float(np.mean(reference_fk_motion_error)) * 100.0,
                    "root_z_m": float(actual_qpos[2]),
                    "policy_running": bool(controller.is_running),
                }
            )

            time_values.append(time_s)
            frame_indexes.append(reference_frame)
            actual_qpos_values.append(actual_qpos)
            reference_qpos_values.append(reference_qpos)
            dof_target_values.append(tensor_to_numpy(dof_targets).copy())
            actual_body_values.append(actual_body)
            reference_body_values.append(reference_body)
            reference_fk_body_values.append(reference_fk_body)
            joint_error_values.append(joint_error)
            root_pos_error_values.append(root_pos_error)
            root_rot_error_values_deg.append(root_rot_error_deg)
            body_global_error_values.append(body_global_error)
            body_local_error_values.append(body_local_error)
            reference_fk_motion_error_values.append(reference_fk_motion_error)

            if step % render_stride != 0:
                continue
            camera.lookat[:] = 0.5 * (actual_qpos[:3] + reference_qpos[:3])
            renderer.update_scene(controller.mj_data, camera=camera)
            added_geoms = add_reference_geoms(renderer, controller, rgba)
            if reference_geom_count is None:
                reference_geom_count = added_geoms
            rgb = renderer.render()
            rgb = annotate_frame(
                rgb,
                [
                    "Actual: opaque robot",
                    "Reference: green ghost",
                    f"t={time_s:5.2f}s  frame={reference_frame}",
                    f"joint RMSE={math.degrees(joint_rmse_rad):5.2f} deg  root={root_pos_error*100:5.2f} cm",
                    f"body mean: global={np.mean(body_global_error)*100:5.2f} cm  local={np.mean(body_local_error)*100:5.2f} cm",
                ],
            )
            writer.append_data(rgb)
            rendered_frames += 1

            if "start" not in selected_frames:
                selected_frames["start"] = (step, rgb.copy())
            if abs(step - total_steps // 2) <= render_stride // 2:
                selected_frames["middle"] = (step, rgb.copy())
            selected_frames["end"] = (step, rgb.copy())
            if joint_rmse_rad > worst_joint[0]:
                worst_joint = (joint_rmse_rad, step, rgb.copy())
            if float(np.mean(body_local_error)) > worst_body[0]:
                worst_body = (float(np.mean(body_local_error)), step, rgb.copy())
    finally:
        writer.close()
        renderer.close()
        controller.stop()

    if not step_rows:
        raise RuntimeError("No simulation steps completed.")

    if worst_joint[2] is not None:
        selected_frames["worst_joint"] = (int(worst_joint[1]), worst_joint[2])
    if worst_body[2] is not None:
        selected_frames["worst_body_local"] = (int(worst_body[1]), worst_body[2])
    contact_frames: list[tuple[str, np.ndarray]] = []
    for label in ("start", "middle", "end", "worst_joint", "worst_body_local"):
        if label not in selected_frames:
            continue
        step, frame = selected_frames[label]
        frame_path = frame_dir / f"{label}_step_{step:04d}.png"
        imageio.imwrite(frame_path, frame)
        contact_frames.append((f"{label} / step {step}", frame))
    save_contact_sheet(contact_frames, frame_dir / "contact_sheet.png")

    time_array = np.asarray(time_values)
    frame_array = np.asarray(frame_indexes, dtype=np.int32)
    actual_qpos_array = np.asarray(actual_qpos_values)
    reference_qpos_array = np.asarray(reference_qpos_values)
    dof_target_array = np.asarray(dof_target_values)
    actual_body_array = np.asarray(actual_body_values)
    reference_body_array = np.asarray(reference_body_values)
    reference_fk_body_array = np.asarray(reference_fk_body_values)
    joint_error_array = np.asarray(joint_error_values)
    root_pos_error_array = np.asarray(root_pos_error_values)
    root_rot_error_array_deg = np.asarray(root_rot_error_values_deg)
    body_global_error_array = np.asarray(body_global_error_values)
    body_local_error_array = np.asarray(body_local_error_values)
    reference_fk_motion_error_array = np.asarray(reference_fk_motion_error_values)

    np.savez_compressed(
        data_dir / "tracking_trajectory.npz",
        time_s=time_array,
        reference_frame=frame_array,
        actual_qpos=actual_qpos_array,
        reference_qpos=reference_qpos_array,
        dof_targets=dof_target_array,
        actual_body_pos_w=actual_body_array,
        reference_motion_body_pos_w=reference_body_array,
        reference_mujoco_fk_body_pos_w=reference_fk_body_array,
        joint_error_rad=joint_error_array,
        root_pos_error_m=root_pos_error_array,
        root_rot_error_deg=root_rot_error_array_deg,
        body_global_error_m=body_global_error_array,
        body_root_local_error_m=body_local_error_array,
        reference_fk_motion_error_m=reference_fk_motion_error_array,
        joint_names=np.asarray(joint_names_real),
        body_names=np.asarray(body_names),
    )
    write_step_csv(data_dir / "step_metrics.csv", step_rows)
    joint_rows = write_named_stats(
        data_dir / "per_joint_error.csv",
        joint_names_real,
        joint_error_array,
        scale=180.0 / math.pi,
        unit="deg",
    )
    body_global_rows = write_named_stats(
        data_dir / "per_body_global_error.csv",
        body_names,
        body_global_error_array,
        scale=100.0,
        unit="cm",
    )
    body_local_rows = write_named_stats(
        data_dir / "per_body_root_local_error.csv",
        body_names,
        body_local_error_array,
        scale=100.0,
        unit="cm",
    )

    plot_errors(
        plot_dir / "tracking_error_timeseries.png",
        time_array,
        joint_error_array,
        root_pos_error_array,
        root_rot_error_array_deg,
        body_global_error_array,
        body_local_error_array,
    )
    plot_named_bars(
        plot_dir / "tracking_error_by_joint_and_body.png",
        joint_names_real,
        joint_error_array,
        body_names,
        body_local_error_array,
    )

    joint_abs_deg = np.degrees(np.abs(joint_error_array))
    summary = {
        "task": args.task,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "motion": str(motion_path),
        "motion_sha256": sha256(motion_path),
        "motion_fps": float(np.asarray(motion.fps).reshape(-1)[0]),
        "motion_frames": int(motion.time_step_total),
        "motion_start_frame": start_frame,
        "policy_dt_s": policy_dt,
        "requested_duration_s": args.duration_s,
        "completed_steps": len(step_rows),
        "completed_duration_s": len(step_rows) * policy_dt,
        "render_fps": args.render_fps,
        "rendered_frames": rendered_frames,
        "resolution": [args.width, args.height],
        "reference_overlay": {
            "enabled": True,
            "description": "green translucent MuJoCo FK ghost from the original reference root/joint pose",
            "rgba": list(args.ghost_rgba),
            "reference_geom_count": reference_geom_count,
        },
        "alignment": {
            "initial_state": "actual MuJoCo qpos initialized exactly from aligned reference frame",
            "motion_loader": "global first-frame XY and yaw removed; original relative motion preserved",
            "comparison": "actual post-control state versus the reference command for that control step",
            "body_global": "actual MuJoCo body positions versus aligned original NPZ body positions",
            "body_root_local": "body positions expressed in each pose's own root frame",
        },
        "safety_stop_step": safety_stop_step,
        "joint_error_deg": array_stats(joint_abs_deg),
        "joint_step_rmse_deg": array_stats(
            np.degrees(np.sqrt(np.mean(np.square(joint_error_array), axis=1)))
        ),
        "root_position_error_cm": array_stats(root_pos_error_array * 100.0),
        "root_rotation_error_deg": array_stats(root_rot_error_array_deg),
        "body_global_position_error_cm": array_stats(body_global_error_array * 100.0),
        "body_root_local_position_error_cm": array_stats(body_local_error_array * 100.0),
        "reference_mujoco_fk_vs_motion_body_error_cm": array_stats(
            reference_fk_motion_error_array * 100.0
        ),
        "worst_joint_by_mae": max(joint_rows, key=lambda row: float(row["mean"])),
        "worst_body_global_by_mae": max(body_global_rows, key=lambda row: float(row["mean"])),
        "worst_body_root_local_by_mae": max(body_local_rows, key=lambda row: float(row["mean"])),
        "minimum_root_z_m": float(np.min(actual_qpos_array[:, 2])),
        "artifacts": {
            "video": str(video_path),
            "contact_sheet": str(frame_dir / "contact_sheet.png"),
            "timeseries_plot": str(plot_dir / "tracking_error_timeseries.png"),
            "named_error_plot": str(plot_dir / "tracking_error_by_joint_and_body.png"),
            "trajectory": str(data_dir / "tracking_trajectory.npz"),
            "step_metrics": str(data_dir / "step_metrics.csv"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "mujoco": mujoco.__version__,
            "torch": torch.__version__,
            "deploy_git_head": git_head(Path.cwd()),
            "command": [sys.executable, *sys.argv],
        },
    }
    summary_path = data_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
