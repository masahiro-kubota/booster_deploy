from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import pkgutil
import sys
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.append(".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record MuJoCo videos for Pikachu Idle deploy checkpoints.")
    parser.add_argument("--task", default="k1_pikachu_idle", help="Registered Booster Deploy task.")
    parser.add_argument("--checkpoint", required=True, help="TorchScript policy path, absolute or task-relative.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--summary", required=True, help="Output JSON summary path.")
    parser.add_argument("--frame", required=True, help="Representative PNG frame path.")
    parser.add_argument("--device", default="cpu", help="Policy device.")
    parser.add_argument("--duration-s", type=float, default=13.0, help="Video duration.")
    parser.add_argument("--render-fps", type=int, default=25, help="Recorded video FPS.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--camera-distance", type=float, default=2.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-15.0)
    parser.add_argument("--center-threshold-frac", type=float, default=0.18)
    return parser.parse_args()


def import_tasks() -> None:
    import tasks as tasks_pkg

    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        __import__(mod_info.name)


def robot_geom_ids(model: mujoco.MjModel, body_ids: list[int]) -> set[int]:
    body_id_set = set(body_ids)
    return {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_id_set
    }


def robot_mask_from_segmentation(segmentation: np.ndarray, geom_ids: set[int]) -> np.ndarray:
    # MuJoCo segmentation render stores object id in channel 0 and object type in channel 1.
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    return (segmentation[..., 1] == geom_type) & np.isin(segmentation[..., 0], list(geom_ids))


def mask_bbox(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return {
            "visible": False,
            "pixel_count": 0,
            "bbox": None,
            "center": None,
            "center_offset_px": None,
            "center_offset_frac": None,
        }
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    h, w = mask.shape
    center_offset_px = float(np.hypot(cx - 0.5 * (w - 1), cy - 0.5 * (h - 1)))
    center_offset_frac = center_offset_px / float(min(w, h))
    return {
        "visible": True,
        "pixel_count": int(xs.size),
        "bbox": [x0, y0, x1, y1],
        "center": [cx, cy],
        "center_offset_px": center_offset_px,
        "center_offset_frac": center_offset_frac,
    }


def main() -> None:
    args = parse_args()
    import_tasks()

    from booster_deploy.controllers.mujoco_controller import MujocoController
    from booster_deploy.utils.registry import get_task

    cfg = copy.deepcopy(get_task(args.task))
    cfg.policy.device = args.device
    cfg.policy.checkpoint_path = args.checkpoint

    controller = MujocoController(cfg)
    controller.update_state()
    controller.start()

    renderer = mujoco.Renderer(controller.mj_model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation

    geom_ids = robot_geom_ids(controller.mj_model, controller._body_ids)
    policy_dt = float(controller.cfg.policy_dt)
    total_steps = int(round(args.duration_s / policy_dt))
    render_stride = max(1, int(round(1.0 / (args.render_fps * policy_dt))))
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    frame_path = Path(args.frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.parent.mkdir(parents=True, exist_ok=True)

    center_checks: list[dict[str, Any]] = []
    representative_frame: np.ndarray | None = None
    representative_frame_metadata: dict[str, Any] | None = None
    representative_index = total_steps // 2
    rendered_frames = 0

    writer = imageio.get_writer(output_path, fps=args.render_fps, codec="libx264", quality=8, macro_block_size=16)
    try:
        for step in range(total_steps):
            controller.update_state()
            dof_targets = controller.policy_step()
            controller.ctrl_step(dof_targets)
            controller.update_state()

            if step % render_stride != 0:
                continue

            root_pos = controller.mj_data.qpos[:3].copy()
            camera.lookat[:] = root_pos

            renderer.disable_segmentation_rendering()
            renderer.update_scene(controller.mj_data, camera=camera)
            rgb = renderer.render()
            writer.append_data(rgb)
            rendered_frames += 1

            renderer.enable_segmentation_rendering()
            renderer.update_scene(controller.mj_data, camera=camera)
            segmentation = renderer.render()
            mask = robot_mask_from_segmentation(segmentation, geom_ids)
            check = mask_bbox(mask)
            check["step"] = step
            check["time_s"] = step * policy_dt
            center_checks.append(check)

            if representative_frame_metadata is None or abs(step - representative_index) < abs(
                int(representative_frame_metadata["step"]) - representative_index
            ):
                representative_frame = rgb.copy()
                representative_frame_metadata = check
    finally:
        writer.close()
        renderer.close()
        controller.stop()

    if representative_frame is not None:
        imageio.imwrite(frame_path, representative_frame)

    visible_checks = [check for check in center_checks if check["visible"]]
    max_center_offset_frac = max((check["center_offset_frac"] for check in visible_checks), default=None)
    max_center_offset_px = max((check["center_offset_px"] for check in visible_checks), default=None)
    center_pass = (
        bool(visible_checks)
        and max_center_offset_frac is not None
        and max_center_offset_frac <= args.center_threshold_frac
    )
    summary = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "output": str(output_path),
        "representative_frame": str(frame_path),
        "duration_s": args.duration_s,
        "policy_dt": policy_dt,
        "steps": total_steps,
        "render_fps": args.render_fps,
        "rendered_frames": rendered_frames,
        "resolution": [args.width, args.height],
        "camera": {
            "distance": args.camera_distance,
            "azimuth": args.camera_azimuth,
            "elevation": args.camera_elevation,
            "lookat": "root qpos xyz updated every rendered frame",
        },
        "center_check": {
            "pass": center_pass,
            "threshold_frac": args.center_threshold_frac,
            "visible_frame_count": len(visible_checks),
            "checked_frame_count": len(center_checks),
            "max_center_offset_px": max_center_offset_px,
            "max_center_offset_frac": max_center_offset_frac,
            "sample_first": center_checks[0] if center_checks else None,
            "sample_middle": center_checks[len(center_checks) // 2] if center_checks else None,
            "sample_last": center_checks[-1] if center_checks else None,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
