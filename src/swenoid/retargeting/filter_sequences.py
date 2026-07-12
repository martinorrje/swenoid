#!/usr/bin/env python3
"""Filter retargeted MuJoCo qpos trajectories using kinematic feasibility metrics.

This script does not modify trajectories. It scores each qpos CSV and writes:
- a metrics CSV
- accepted/rejected motion lists
- optionally a filtered tree of symlinks or copies for easy visualization
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
import shutil
from pathlib import Path

import mujoco
import numpy as np

from swenoid.model_constants import SWENOID_XML


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=SWENOID_XML)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--input-glob", action="append", default=[])
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument(
        "--input-list",
        type=Path,
        action="append",
        default=[],
        help="Text file with one qpos CSV path per line.",
    )
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--accepted-list", type=Path, required=True)
    parser.add_argument("--rejected-list", type=Path, required=True)
    parser.add_argument(
        "--accepted-root",
        type=Path,
        help="Optional tree of accepted qpos symlinks/copies.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of symlink into --accepted-root.",
    )
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--ground", type=float, default=0.0)
    parser.add_argument("--contact-height", type=float, default=0.035)
    parser.add_argument("--foot-penetration-tol", type=float, default=0.035)
    parser.add_argument("--locomotion-foot-penetration-tol", type=float, default=0.06)
    parser.add_argument("--min-contact-fraction", type=float, default=0.08)
    parser.add_argument("--locomotion-min-contact-fraction", type=float, default=0.03)
    parser.add_argument("--min-support-fraction", type=float, default=0.04)
    parser.add_argument("--locomotion-min-support-fraction", type=float, default=0.01)
    parser.add_argument("--max-com-support-distance", type=float, default=0.16)
    parser.add_argument(
        "--locomotion-max-com-support-distance", type=float, default=0.30
    )
    parser.add_argument("--max-com-bad-fraction", type=float, default=0.35)
    parser.add_argument("--locomotion-max-com-bad-fraction", type=float, default=0.85)
    parser.add_argument("--max-root-jerk", type=float, default=180.0)
    parser.add_argument("--max-root-jitter", type=float, default=2.5)
    parser.add_argument("--max-joint-jerk", type=float, default=200000.0)
    parser.add_argument("--max-skating-velocity", type=float, default=0.55)
    parser.add_argument("--locomotion-max-skating-velocity", type=float, default=3.0)
    parser.add_argument("--max-joint-violation-fraction", type=float, default=0.02)
    parser.add_argument("--joint-range-margin", type=float, default=1.0)
    parser.add_argument("--locomotion-root-travel", type=float, default=0.35)
    parser.add_argument("--locomotion-root-speed", type=float, default=0.20)
    parser.add_argument(
        "--exclude-support-props",
        action="store_true",
        help="Reject motions whose names imply required stairs, boxes, ladders, chairs, or obstacle props.",
    )
    parser.add_argument(
        "--exclude-name-regex",
        action="append",
        default=[],
        help="Reject motions whose relative path/name matches this regex. Repeatable.",
    )
    return parser.parse_args()


def qpos_columns(nq: int) -> list[str]:
    return [f"qpos_{i}" for i in range(nq)]


def load_qpos(path: Path, nq: int) -> np.ndarray:
    cols = qpos_columns(nq)
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in cols if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} missing {missing[0]}")
        for row in reader:
            rows.append([float(row[c]) for c in cols])
    if not rows:
        raise ValueError(f"{path} is empty")
    return np.asarray(rows, dtype=np.float64)


def support_geom_ids(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    collision, all_foot = [], []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        body = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
            or ""
        )
        if "foot" in name or "foot" in body:
            all_foot.append(gid)
            if "collision" in name or model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                collision.append(gid)
    return collision or all_foot, all_foot


def joint_info(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qadr, ranges, limited = [], [], []
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        qadr.append(model.jnt_qposadr[jid])
        ranges.append(model.jnt_range[jid].copy())
        limited.append(bool(model.jnt_limited[jid]))
    return np.asarray(qadr), np.asarray(ranges), np.asarray(limited)


def geom_corners(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> np.ndarray:
    center = data.geom_xpos[gid]
    mat = data.geom_xmat[gid].reshape(3, 3)
    size = model.geom_size[gid]
    signs = np.array(
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
    return center + (signs * size) @ mat.T


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(set(map(tuple, points.tolist())))
    if len(pts) <= 1:
        return np.asarray(pts, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def point_in_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    if len(poly) < 3:
        return False
    x, y = point
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def point_polygon_distance(point: np.ndarray, poly: np.ndarray) -> float:
    if len(poly) == 0:
        return math.inf
    if len(poly) == 1:
        return float(np.linalg.norm(point - poly[0]))
    if point_in_polygon(point, poly):
        return 0.0
    return min(
        point_segment_distance(point, poly[i], poly[(i + 1) % len(poly)])
        for i in range(len(poly))
    )


def finite_metric(values: np.ndarray, fps: float, order: int) -> np.ndarray:
    if len(values) <= order:
        return np.zeros(0)
    out = values
    for _ in range(order):
        out = np.diff(out, axis=0) * fps
    return np.linalg.norm(out, axis=-1)


def looks_like_locomotion(
    path: Path, qpos: np.ndarray, args: argparse.Namespace
) -> bool:
    name = path.stem.lower()
    locomotion_tokens = (
        "walk",
        "jog",
        "run",
        "locomotion",
        "turn_walk",
        "turn_jog",
        "turn_run",
        "crawl",
        "dodge",
        "avoid",
        "step",
        "high_knees",
        "smoke_walk",
    )
    if any(token in name for token in locomotion_tokens):
        return True
    if len(qpos) < 2:
        return False
    root_xy = qpos[:, :2]
    travel = float(np.linalg.norm(root_xy[-1] - root_xy[0]))
    duration = max((len(qpos) - 1) / args.fps, 1e-6)
    speed = travel / duration
    return travel >= args.locomotion_root_travel or speed >= args.locomotion_root_speed


SUPPORT_PROP_RE = re.compile(
    r"("
    r"stair|ladder|obstacle|"
    r"(?:^|_)box(?:_|$)|"
    r"come_(?:up|down)_50cm|"
    r"jump_(?:on|off)(?:_front)?_50cm|"
    r"sit_on_chair|chair_overturning|"
    r"stepdown"
    r")",
    re.IGNORECASE,
)


def excluded_by_name(path: Path, args: argparse.Namespace) -> bool:
    name = path.as_posix()
    if args.exclude_support_props and SUPPORT_PROP_RE.search(name):
        return True
    return any(re.search(pattern, name) for pattern in args.exclude_name_regex)


def score_motion(
    model: mujoco.MjModel, qpos: np.ndarray, args: argparse.Namespace, path: Path
) -> dict[str, float | bool | str]:
    data = mujoco.MjData(model)
    support_ids, _foot_ids = support_geom_ids(model)
    joint_qadr, joint_ranges, joint_limited = joint_info(model)
    is_locomotion = looks_like_locomotion(path, qpos, args)

    com_distances = []
    supported = []
    contact_any = []
    min_penetration = 0.0
    foot_xy = []
    foot_contact = []
    com_positions = []

    for row in qpos:
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        com = data.subtree_com[0].copy()
        com_positions.append(com)
        support_points = []
        all_lowest = []
        for gid in support_ids:
            corners = geom_corners(model, data, gid)
            all_lowest.append(float(corners[:, 2].min()))
            near = corners[corners[:, 2] <= args.ground + args.contact_height]
            if len(near):
                support_points.extend(near[:, :2])
        min_penetration = min(min_penetration, min(all_lowest) - args.ground)
        contact_any.append(len(support_points) > 0)
        if len(support_points) >= 3:
            poly = convex_hull(np.asarray(support_points, dtype=np.float64))
            dist = point_polygon_distance(com[:2], poly)
            supported.append(True)
            com_distances.append(dist)
        else:
            supported.append(False)
            com_distances.append(math.nan)

        # Track per-foot lowest collision center-ish point for skating heuristic.
        frame_xy, frame_contact = [], []
        for gid in support_ids:
            corners = geom_corners(model, data, gid)
            lowest = corners[np.argmin(corners[:, 2])]
            frame_xy.append(lowest[:2])
            frame_contact.append(lowest[2] <= args.ground + args.contact_height)
        foot_xy.append(frame_xy)
        foot_contact.append(frame_contact)

    com_positions = np.asarray(com_positions)
    root = qpos[:, :3]
    joints = qpos[:, joint_qadr]
    foot_xy = np.asarray(foot_xy)
    foot_contact = np.asarray(foot_contact, dtype=bool)

    root_jerk_values = finite_metric(root, args.fps, 3)
    joint_jerk_values = finite_metric(joints, args.fps, 3)
    com_dist_arr = np.asarray(com_distances, dtype=np.float64)
    support_mask = np.asarray(supported, dtype=bool)
    contact_mask = np.asarray(contact_any, dtype=bool)

    if len(foot_xy) > 1:
        foot_vel = np.linalg.norm(np.diff(foot_xy, axis=0), axis=-1) * args.fps
        skate_mask = foot_contact[:-1] & foot_contact[1:]
        skating_velocity = (
            float(foot_vel[skate_mask].mean()) if skate_mask.any() else 0.0
        )
    else:
        skating_velocity = 0.0

    lo = joint_ranges[:, 0] * args.joint_range_margin
    hi = joint_ranges[:, 1] * args.joint_range_margin
    bad = np.zeros(joints.shape, dtype=bool)
    bad[:, joint_limited] = (joints[:, joint_limited] < lo[joint_limited]) | (
        joints[:, joint_limited] > hi[joint_limited]
    )
    joint_violation_fraction = float(np.any(bad, axis=1).mean())

    valid_com = com_dist_arr[np.isfinite(com_dist_arr)]
    max_com_distance = float(valid_com.max()) if len(valid_com) else math.inf
    mean_com_distance = float(valid_com.mean()) if len(valid_com) else math.inf
    com_bad_fraction = (
        float((valid_com > args.max_com_support_distance).mean())
        if len(valid_com)
        else 1.0
    )
    support_fraction = float(support_mask.mean())
    contact_fraction = float(contact_mask.mean())
    root_travel = (
        float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2])) if len(qpos) > 1 else 0.0
    )
    duration = max((len(qpos) - 1) / args.fps, 1e-6)
    root_speed = root_travel / duration
    root_jerk = float(root_jerk_values.mean()) if len(root_jerk_values) else 0.0
    root_jitter = (
        float(np.percentile(root_jerk_values, 95)) if len(root_jerk_values) else 0.0
    )
    joint_jerk = (
        float(np.percentile(joint_jerk_values, 95)) if len(joint_jerk_values) else 0.0
    )

    foot_penetration_tol = (
        args.locomotion_foot_penetration_tol
        if is_locomotion
        else args.foot_penetration_tol
    )
    min_contact_fraction = (
        args.locomotion_min_contact_fraction
        if is_locomotion
        else args.min_contact_fraction
    )
    min_support_fraction = (
        args.locomotion_min_support_fraction
        if is_locomotion
        else args.min_support_fraction
    )
    max_com_support_distance = (
        args.locomotion_max_com_support_distance
        if is_locomotion
        else args.max_com_support_distance
    )
    max_com_bad_fraction = (
        args.locomotion_max_com_bad_fraction
        if is_locomotion
        else args.max_com_bad_fraction
    )
    max_skating_velocity = (
        args.locomotion_max_skating_velocity
        if is_locomotion
        else args.max_skating_velocity
    )

    reasons = []
    if excluded_by_name(path, args):
        reasons.append("excluded_by_name")
    if min_penetration < -foot_penetration_tol:
        reasons.append("foot_penetration")
    if contact_fraction < min_contact_fraction:
        reasons.append("low_contact")
    if support_fraction < min_support_fraction:
        reasons.append("low_support")
    if (
        com_bad_fraction > max_com_bad_fraction
        and max_com_distance > max_com_support_distance
    ):
        reasons.append("com_far_from_support")
    if (
        root_jerk > args.max_root_jerk
        or root_jitter > args.max_root_jitter * args.max_root_jerk
    ):
        reasons.append("root_jitter")
    if joint_jerk > args.max_joint_jerk:
        reasons.append("joint_jitter")
    if skating_velocity > max_skating_velocity:
        reasons.append("foot_skating")
    if joint_violation_fraction > args.max_joint_violation_fraction:
        reasons.append("joint_limits")

    return {
        "frames": len(qpos),
        "pass": not reasons,
        "reasons": ";".join(reasons),
        "name_excluded": excluded_by_name(path, args),
        "locomotion": is_locomotion,
        "root_travel": root_travel,
        "root_speed": root_speed,
        "contact_fraction": contact_fraction,
        "support_fraction": support_fraction,
        "max_com_support_distance": max_com_distance,
        "mean_com_support_distance": mean_com_distance,
        "com_bad_fraction": com_bad_fraction,
        "min_foot_clearance": min_penetration,
        "root_jerk_mean": root_jerk,
        "root_jerk_p95": root_jitter,
        "joint_jerk_p95": joint_jerk,
        "skating_velocity": skating_velocity,
        "joint_violation_fraction": joint_violation_fraction,
    }


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def main() -> None:
    args = parse_args()
    paths = [*args.input]
    for list_path in args.input_list:
        with list_path.open() as f:
            paths.extend(
                Path(line.strip())
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            )
    for pattern in args.input_glob:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern, recursive=True)))
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("No input trajectories selected")

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    rows = []
    accepted, rejected = [], []
    for idx, path in enumerate(paths, 1):
        qpos = load_qpos(path, model.nq)
        metrics = score_motion(model, qpos, args, path)
        rel = (
            path.relative_to(args.input_root)
            if path.is_relative_to(args.input_root)
            else Path(path.name)
        )
        row = {"motion": str(rel), "path": str(path), **metrics}
        rows.append(row)
        if metrics["pass"]:
            accepted.append(path)
            if args.accepted_root is not None:
                link_or_copy(path, args.accepted_root / rel, args.copy)
        else:
            rejected.append(path)
        print(
            f"[{idx:05d}/{len(paths):05d}] {'PASS' if metrics['pass'] else 'DROP'} {rel} {metrics['reasons']}"
        )

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    args.accepted_list.parent.mkdir(parents=True, exist_ok=True)
    args.accepted_list.write_text(
        "\n".join(str(p) for p in accepted) + ("\n" if accepted else "")
    )
    args.rejected_list.write_text(
        "\n".join(str(p) for p in rejected) + ("\n" if rejected else "")
    )
    print(f"accepted {len(accepted)} / {len(paths)}")
    print(f"metrics: {args.metrics_csv}")


if __name__ == "__main__":
    main()
