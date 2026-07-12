"""Convert retargeted Swenoid qpos CSV files to 50 Hz MjLab motion NPZ files."""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from swenoid.model_constants import SWENOID_XML


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--input-list", type=Path, action="append", default=[])
    parser.add_argument("--input-glob", action="append", default=[])
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-fps", type=float, default=120.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--compressed", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ground-align", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def _load_qpos_csv(path: Path, nq: int) -> np.ndarray:
    columns = [f"qpos_{index}" for index in range(nq)]
    rows: list[list[float]] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [
            column for column in columns if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"{path} is missing {missing[0]}")
        rows.extend([float(row[column]) for column in columns] for row in reader)
    if not rows:
        raise ValueError(f"{path} has no qpos rows")
    return np.asarray(rows, dtype=np.float64)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    output = q.copy()
    output[..., 1:] *= -1.0
    return output


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    output = np.zeros_like(values, dtype=np.float32)
    output[:-1] = ((values[1:] - values[:-1]) / dt).astype(np.float32)
    output[-1] = output[-2]
    return output


def _quat_ang_vel(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    if len(quat_wxyz) <= 1:
        return np.zeros((len(quat_wxyz), 3), dtype=np.float32)
    quat = quat_wxyz / np.maximum(
        np.linalg.norm(quat_wxyz, axis=-1, keepdims=True), 1e-12
    )
    delta = _quat_mul(quat[1:], _quat_conj(quat[:-1]))
    delta *= np.where(delta[:, :1] < 0.0, -1.0, 1.0)
    xyz = delta[:, 1:]
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, np.clip(delta[:, 0], -1.0, 1.0))
    axis = xyz / np.maximum(sin_half[:, None], 1e-12)
    output = np.zeros((len(quat), 3), dtype=np.float32)
    output[:-1] = (axis * (angle[:, None] / dt)).astype(np.float32)
    output[-1] = output[-2]
    return output


def _resample_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    input_fps: float,
    output_fps: float,
) -> np.ndarray:
    """Interpolate positions and SLERP free-joint quaternions."""
    if input_fps <= 0.0 or output_fps <= 0.0:
        raise ValueError("Input and output frame rates must be positive")
    if len(qpos) <= 1 or np.isclose(input_fps, output_fps):
        return qpos.copy()

    input_times = np.arange(len(qpos), dtype=np.float64) / input_fps
    duration = input_times[-1]
    output_times = np.arange(
        0.0,
        duration + np.finfo(np.float64).eps * max(1.0, duration),
        1.0 / output_fps,
        dtype=np.float64,
    )
    if not len(output_times):
        output_times = np.zeros(1, dtype=np.float64)

    output = np.empty((len(output_times), model.nq), dtype=np.float64)
    for qpos_index in range(model.nq):
        output[:, qpos_index] = np.interp(
            output_times, input_times, qpos[:, qpos_index]
        )

    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        quaternion_xyzw = qpos[:, qadr + 3 : qadr + 7][:, [1, 2, 3, 0]]
        interpolated = Slerp(input_times, Rotation.from_quat(quaternion_xyzw))(
            output_times
        ).as_quat()
        output[:, qadr + 3 : qadr + 7] = interpolated[:, [3, 0, 1, 2]]
    return output


def _geom_corners(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> np.ndarray:
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    return (
        data.geom_xpos[geom_id]
        + (signs * model.geom_size[geom_id]) @ data.geom_xmat[geom_id].reshape(3, 3).T
    )


def _ground_align(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if "foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
        and "collision"
        in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
    ]
    if not geom_ids:
        return qpos
    data = mujoco.MjData(model)
    offsets = []
    for row in (qpos[0], qpos[-1]):
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        lowest = min(
            float(np.min(_geom_corners(model, data, geom_id)[:, 2]))
            for geom_id in geom_ids
        )
        offsets.append(-lowest)
    output = qpos.copy()
    output[:, 2] += np.linspace(offsets[0], offsets[1], len(output))
    return output


def convert_motion(
    input_path: Path,
    output_path: Path,
    *,
    input_fps: float,
    output_fps: float,
    compressed: bool,
    ground_align: bool,
) -> None:
    """Convert one Swenoid qpos CSV to MjLab's single-motion format."""
    model = mujoco.MjModel.from_xml_path(str(SWENOID_XML))
    qpos = _load_qpos_csv(input_path, model.nq)
    source_frame_count = len(qpos)
    if ground_align:
        qpos = _ground_align(model, qpos)
    qpos = _resample_qpos(model, qpos, input_fps, output_fps)
    dt = 1.0 / output_fps

    free_qadr: int | None = None
    joint_qadr: list[int] = []
    for joint_id in range(model.njnt):
        qadr = int(model.jnt_qposadr[joint_id])
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            free_qadr = qadr
        else:
            joint_qadr.append(qadr)
    if free_qadr is None:
        raise ValueError("Swenoid model has no free joint")

    # MjLab Entity body indices exclude MuJoCo's world body. Store every robot
    # body in model order so released MjLab can apply its selected body indices.
    body_ids = list(range(1, model.nbody))
    body_names = [model.body(body_id).name for body_id in body_ids]
    data = mujoco.MjData(model)
    body_pos = np.zeros((len(qpos), len(body_ids), 3), dtype=np.float32)
    body_quat = np.zeros((len(qpos), len(body_ids), 4), dtype=np.float32)
    for frame, row in enumerate(qpos):
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[body_ids]
        body_quat[frame] = data.xquat[body_ids]

    root_pos = qpos[:, free_qadr : free_qadr + 3].astype(np.float32)
    root_quat = qpos[:, free_qadr + 3 : free_qadr + 7].astype(np.float32)
    joint_pos = qpos[:, np.asarray(joint_qadr)].astype(np.float32)
    payload = {
        "fps": np.asarray([output_fps], dtype=np.float32),
        "source_fps": np.asarray([input_fps], dtype=np.float32),
        "source_frame_count": np.asarray([source_frame_count], dtype=np.int64),
        "joint_pos": joint_pos,
        "joint_vel": _finite_difference(joint_pos, dt),
        "root_pos_w": root_pos,
        "root_quat_w": root_quat,
        "root_lin_vel_w": _finite_difference(root_pos, dt),
        "root_ang_vel_w": _quat_ang_vel(root_quat, dt),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": _finite_difference(body_pos, dt),
        "body_ang_vel_w": np.stack(
            [_quat_ang_vel(body_quat[:, index], dt) for index in range(len(body_ids))],
            axis=1,
        ),
        "body_names": np.asarray(body_names),
        "source_csv": np.asarray([str(input_path)]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(  # pyright: ignore[reportArgumentType]
            output_path, **payload
        )
    else:
        np.savez(  # pyright: ignore[reportArgumentType]
            output_path, **payload
        )


def _selected_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.input)
    for list_path in args.input_list:
        paths.extend(
            Path(line.strip())
            for line in list_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    for pattern in args.input_glob:
        paths.extend(Path(path) for path in glob.glob(pattern, recursive=True))
    return sorted(dict.fromkeys(paths))


def main() -> None:
    args = parse_args()
    paths = _selected_paths(args)
    if not paths:
        raise SystemExit("No input qpos files selected")
    converted = skipped = 0
    for index, input_path in enumerate(paths, 1):
        relative = (
            input_path.relative_to(args.input_root)
            if input_path.is_relative_to(args.input_root)
            else Path(input_path.name)
        )
        output_path = args.output_root / relative.with_suffix("").with_suffix(".npz")
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        convert_motion(
            input_path,
            output_path,
            input_fps=args.input_fps,
            output_fps=args.output_fps,
            compressed=args.compressed,
            ground_align=args.ground_align,
        )
        converted += 1
        print(f"[{index}/{len(paths)}] {input_path} -> {output_path}")
    print(f"done converted={converted} skipped={skipped}")


if __name__ == "__main__":
    main()
