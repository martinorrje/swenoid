#!/usr/bin/env python3
"""Render a MuJoCo qpos CSV trajectory to a video."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mediapy
import mujoco
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

from swenoid.model_constants import SWENOID_XML


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=SWENOID_XML)
    parser.add_argument("--qpos-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", default="tracking")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--source-fps", type=float, default=120.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    return parser.parse_args()


def load_qpos(path: Path, nq: int) -> np.ndarray:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        qpos_columns = [f"qpos_{i}" for i in range(nq)]
        missing = [
            column for column in qpos_columns if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"{path} is missing qpos columns, first missing column: {missing[0]}"
            )
        return np.array(
            [[float(row[column]) for column in qpos_columns] for row in reader],
            dtype=np.float64,
        )


def main() -> None:
    args = parse_args()
    mediapy.set_ffmpeg(get_ffmpeg_exe())
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    qpos = load_qpos(args.qpos_csv, model.nq)
    if len(qpos) == 0:
        raise SystemExit(f"{args.qpos_csv} has no qpos rows")

    stride = max(1, round(args.source_fps / args.fps))
    max_frames = max(1, round(args.max_seconds * args.fps))
    sampled = qpos[::stride][:max_frames]

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    camera: int | str = args.camera if camera_id >= 0 else -1

    frames = []
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        for row in sampled:
            data.qpos[:] = row
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(args.output, frames, fps=args.fps)
    print(f"rendered {len(frames)} frames -> {args.output}")


if __name__ == "__main__":
    main()
