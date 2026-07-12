#!/usr/bin/env python3
"""Directly retarget BONES-SEED Unitree G1 CSV trajectories to Swenoid."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import tarfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT_TRANSLATE_COLUMNS = ("root_translateX", "root_translateY", "root_translateZ")
ROOT_ROTATE_COLUMNS = ("root_rotateX", "root_rotateY", "root_rotateZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert BONES-SEED g1/csv motions into target MuJoCo qpos CSV files. "
            "The converter can read extracted CSVs or stream members from g1.tar.gz."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root of a licensed BONES-SEED download.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("bones_seed_swenoid_map.json"),
        help="Joint mapping JSON (defaults to the packaged Swenoid map).",
    )
    parser.add_argument(
        "--motion",
        action="append",
        default=[],
        help="CSV path or archive member, repeatable.",
    )
    parser.add_argument(
        "--motion-list",
        type=Path,
        action="append",
        default=[],
        help="Text file with one motion path per line.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        help="BONES-SEED metadata CSV for batch conversion.",
    )
    parser.add_argument(
        "--package", help="Optional metadata package filter, e.g. Locomotion."
    )
    parser.add_argument("--category", help="Optional metadata category filter.")
    parser.add_argument(
        "--limit", type=int, help="Maximum number of metadata motions to convert."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        config = json.load(f)
    if "target_xml" not in config:
        raise ValueError(f"{path} is missing required key: target_xml")
    if "joint_map" not in config:
        raise ValueError(f"{path} is missing required key: joint_map")
    return config


def resolve_target_xml(config_path: Path, target_xml: str) -> Path:
    path = Path(target_xml)
    if path.is_absolute():
        return path
    repo_root = config_path.resolve().parents[2]
    repo_relative = repo_root / path
    if repo_relative.exists():
        return repo_relative
    return (config_path.parent / path).resolve()


def joint_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            names.append(name)
    return names


def neutral_qpos(model: mujoco.MjModel, keyframe_name: str | None) -> np.ndarray:
    if keyframe_name:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
        if key_id < 0:
            raise ValueError(f"Target model has no keyframe named {keyframe_name!r}")
        return np.array(model.key_qpos[key_id], dtype=np.float64)
    if model.nkey:
        return np.array(model.key_qpos[0], dtype=np.float64)
    return np.array(model.qpos0, dtype=np.float64)


def source_column(name: str, columns: set[str]) -> str:
    if name in columns:
        return name
    dof_name = f"{name}_dof"
    if dof_name in columns:
        return dof_name
    raise KeyError(f"Source CSV has no column {name!r} or {dof_name!r}")


def source_to_radians(value: str, unit: str) -> float:
    number = float(value)
    if unit == "degree":
        return math.radians(number)
    if unit == "radian":
        return number
    raise ValueError(f"Unsupported source_angle_unit: {unit}")


def source_root_to_meters(values: Iterable[str], unit: str) -> np.ndarray:
    xyz = np.array([float(value) for value in values], dtype=np.float64)
    if unit == "centimeter":
        return xyz * 0.01
    if unit == "meter":
        return xyz
    raise ValueError(f"Unsupported source_root_translation_unit: {unit}")


def root_rotation(row: dict[str, str], unit: str, order: str) -> np.ndarray:
    angles = [float(row[column]) for column in ROOT_ROTATE_COLUMNS]
    degrees = unit == "degree"
    quat_xyzw = Rotation.from_euler(order, angles, degrees=degrees).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )


@contextmanager
def open_motion(dataset_root: Path, motion: str) -> Iterator[TextIO]:
    motion_path = Path(motion)
    candidates = []
    if motion_path.is_absolute():
        candidates.append(motion_path)
    else:
        candidates.append(dataset_root / motion_path)
        candidates.append(Path.cwd() / motion_path)
    for candidate in candidates:
        if candidate.exists():
            with candidate.open(newline="") as stream:
                yield stream
            return

    archive_path = dataset_root / "g1.tar.gz"
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Could not find {motion!r} as a file, and {archive_path} does not exist"
        )
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.extractfile(motion)
        if member is None and not motion.startswith("g1/"):
            member = archive.extractfile(f"g1/{motion}")
        if member is None:
            raise FileNotFoundError(f"Could not find {motion!r} in {archive_path}")
        with io.TextIOWrapper(member, newline="") as text:
            yield text


def iter_metadata_motions(args: argparse.Namespace) -> Iterable[str]:
    if not args.metadata_csv:
        return []
    count = 0
    with args.metadata_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if args.package and row.get("package") != args.package:
                continue
            if args.category and row.get("category") != args.category:
                continue
            motion = row.get("move_g1_path")
            if not motion:
                continue
            yield motion
            count += 1
            if args.limit is not None and count >= args.limit:
                break


def output_path(output_dir: Path, motion: str) -> Path:
    path = Path(motion)
    parts = path.parts
    if "g1" in parts:
        parts = parts[parts.index("g1") + 1 :]
    return output_dir.joinpath(*parts).with_suffix(".qpos.csv")


def retarget_motion(
    *,
    dataset_root: Path,
    config: dict[str, Any],
    model: mujoco.MjModel,
    neutral: np.ndarray,
    motion: str,
    output: Path,
    clip: bool,
) -> dict[str, Any]:
    angle_unit = config.get("source_angle_unit", "degree")
    root_unit = config.get("source_root_translation_unit", "centimeter")
    euler_order = config.get("source_root_euler_order", "xyz")
    root_scale_config = config.get("root_scale", 1.0)

    target_joints = set(joint_names(model))
    mapped_joints = config["joint_map"]
    for target_joint in mapped_joints:
        if target_joint not in target_joints:
            raise KeyError(f"Target model has no joint named {target_joint!r}")

    with open_motion(dataset_root, motion) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{motion} has no CSV header")
        columns = set(reader.fieldnames)
        for column in (*ROOT_TRANSLATE_COLUMNS, *ROOT_ROTATE_COLUMNS):
            if column not in columns:
                raise KeyError(f"{motion} is missing required root column {column!r}")

        source_columns = {
            target_joint: source_column(source_joint, columns)
            for target_joint, source_joint in mapped_joints.items()
        }
        rows = list(reader)

    if not rows:
        raise ValueError(f"{motion} has no trajectory rows")

    first_root_m = source_root_to_meters(
        (rows[0][column] for column in ROOT_TRANSLATE_COLUMNS), root_unit
    )
    last_root_m = source_root_to_meters(
        (rows[-1][column] for column in ROOT_TRANSLATE_COLUMNS), root_unit
    )
    if root_scale_config in ("auto", "auto_first"):
        root_scale = (
            float(neutral[2] / first_root_m[2]) if abs(first_root_m[2]) > 1e-9 else 1.0
        )
    elif root_scale_config == "auto_last":
        root_scale = (
            float(neutral[2] / last_root_m[2]) if abs(last_root_m[2]) > 1e-9 else 1.0
        )
    else:
        root_scale = float(root_scale_config)

    output.parent.mkdir(parents=True, exist_ok=True)
    qpos_columns = [f"qpos_{i}" for i in range(model.nq)]
    joint_columns = list(mapped_joints)
    clipped_samples = 0

    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", *qpos_columns, *joint_columns])

        for frame_index, row in enumerate(rows):
            qpos = neutral.copy()
            qpos[:3] = (
                source_root_to_meters(
                    (row[column] for column in ROOT_TRANSLATE_COLUMNS), root_unit
                )
                * root_scale
            )
            qpos[3:7] = root_rotation(row, angle_unit, euler_order)

            joint_values = []
            for target_joint, column in source_columns.items():
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, target_joint
                )
                qadr = model.jnt_qposadr[joint_id]
                value = source_to_radians(row[column], angle_unit)
                if clip and bool(model.jnt_limited[joint_id]):
                    lo, hi = model.jnt_range[joint_id]
                    clipped = float(np.clip(value, lo, hi))
                    clipped_samples += int(clipped != value)
                    value = clipped
                qpos[qadr] = value
                joint_values.append(value)

            writer.writerow(
                [row.get("Frame", frame_index), *qpos.tolist(), *joint_values]
            )

    return {
        "motion": motion,
        "output": str(output),
        "frames": len(rows),
        "root_scale": root_scale,
        "clipped_samples": clipped_samples,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    target_xml = resolve_target_xml(args.config, config["target_xml"])
    model = mujoco.MjModel.from_xml_path(str(target_xml))
    neutral = neutral_qpos(model, config.get("neutral_keyframe"))

    list_motions: list[str] = []
    for list_path in args.motion_list:
        with list_path.open() as f:
            list_motions.extend(
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            )

    motions = [*args.motion, *list_motions, *iter_metadata_motions(args)]
    if not motions:
        raise SystemExit("No motions selected. Pass --motion or --metadata-csv.")

    converted = 0
    for motion in motions:
        out = output_path(args.output_dir, motion)
        if out.exists() and not args.overwrite:
            print(f"skip existing: {out}")
            continue
        stats = retarget_motion(
            dataset_root=args.dataset_root,
            config=config,
            model=model,
            neutral=neutral,
            motion=motion,
            output=out,
            clip=args.clip,
        )
        converted += 1
        print(
            "converted {motion} -> {output} "
            "({frames} frames, root_scale={root_scale:.4f}, clipped={clipped_samples})".format(
                **stats
            )
        )

    print(f"done: {converted} motion(s) converted")


if __name__ == "__main__":
    main()
