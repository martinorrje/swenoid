from __future__ import annotations

import copy
import csv
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


def _quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _quat_ang_vel(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    if len(quat_wxyz) <= 1:
        return np.zeros((len(quat_wxyz), 3), dtype=np.float32)
    q = quat_wxyz / np.maximum(np.linalg.norm(quat_wxyz, axis=-1, keepdims=True), 1e-12)
    dq = _quat_mul_np(q[1:], _quat_conj(q[:-1]))
    sign = np.where(dq[:, :1] < 0.0, -1.0, 1.0)
    dq *= sign
    xyz = dq[:, 1:]
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, np.clip(dq[:, 0], -1.0, 1.0))
    axis = xyz / np.maximum(sin_half[:, None], 1e-12)
    vel = axis * (angle[:, None] / dt)
    out = np.zeros((len(q), 3), dtype=np.float32)
    out[:-1] = vel.astype(np.float32)
    out[-1] = out[-2]
    return out


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    out[:-1] = ((values[1:] - values[:-1]) / dt).astype(np.float32)
    out[-1] = out[-2]
    return out


def _qpos_columns(nq: int) -> list[str]:
    return [f"qpos_{i}" for i in range(nq)]


def _lerp(a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return a + (b - a) * alpha


def _quat_slerp_batch(
    q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(max=1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    slerp = (
        torch.sin((1.0 - alpha) * theta) / torch.clamp(sin_theta, min=1.0e-6) * q0
        + torch.sin(alpha * theta) / torch.clamp(sin_theta, min=1.0e-6) * q1
    )
    nlerp = _lerp(q0, q1, alpha)
    out = torch.where(sin_theta > 1.0e-6, slerp, nlerp)
    return torch.nn.functional.normalize(out, dim=-1)


@dataclass
class DatasetMotion:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    source_dt: float

    @property
    def time_step_total(self) -> int:
        return int(self.joint_pos.shape[0])


class DatasetMotionLoader:
    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        paths: list[Path],
        body_ids: list[int],
        joint_qadr: np.ndarray,
        dt: float,
        device: str,
        max_cache_size: int,
    ) -> None:
        self.model = model
        self.paths = paths
        self.body_ids = body_ids
        self.joint_qadr = joint_qadr
        self.dt = dt
        self.device = device
        self.max_cache_size = max_cache_size
        self._cache: OrderedDict[int, DatasetMotion] = OrderedDict()
        self.free_joint_qadr = self._free_joint_qadr()

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, motion_id: int) -> DatasetMotion:
        if motion_id in self._cache:
            motion = self._cache.pop(motion_id)
            self._cache[motion_id] = motion
            return motion

        motion = self._load_motion(self.paths[motion_id])
        self._cache[motion_id] = motion
        while len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)
        return motion

    def _load_qpos(self, path: Path) -> np.ndarray:
        columns = _qpos_columns(self.model.nq)
        rows = []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in columns if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path} is missing {missing[0]}")
            for row in reader:
                rows.append([float(row[c]) for c in columns])
        if not rows:
            raise ValueError(f"{path} has no qpos rows")
        return np.asarray(rows, dtype=np.float64)

    def _free_joint_qadr(self) -> np.ndarray:
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                qadr = self.model.jnt_qposadr[joint_id]
                return np.arange(qadr, qadr + 7, dtype=np.int64)
        raise ValueError("DatasetMotionLoader requires a model with a free joint")

    def _load_motion(self, path: Path) -> DatasetMotion:
        if path.suffix == ".npz":
            data = np.load(path)
            if "fps" not in data:
                raise ValueError(f"{path} has no fps metadata")
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            if not np.isfinite(fps) or fps <= 0.0:
                raise ValueError(f"{path} has invalid fps metadata: {fps}")
            body_pos_w = torch.tensor(
                data["body_pos_w"], dtype=torch.float32, device=self.device
            )
            body_quat_w = torch.tensor(
                data["body_quat_w"], dtype=torch.float32, device=self.device
            )
            body_lin_vel_w = torch.tensor(
                data["body_lin_vel_w"], dtype=torch.float32, device=self.device
            )
            body_ang_vel_w = torch.tensor(
                data["body_ang_vel_w"], dtype=torch.float32, device=self.device
            )
            if "root_pos_w" in data:
                root_pos_w = torch.tensor(
                    data["root_pos_w"], dtype=torch.float32, device=self.device
                )
                root_quat_w = torch.tensor(
                    data["root_quat_w"], dtype=torch.float32, device=self.device
                )
                root_lin_vel_w = torch.tensor(
                    data["root_lin_vel_w"], dtype=torch.float32, device=self.device
                )
                root_ang_vel_w = torch.tensor(
                    data["root_ang_vel_w"], dtype=torch.float32, device=self.device
                )
            else:
                root_pos_w = body_pos_w[:, 0]
                root_quat_w = body_quat_w[:, 0]
                root_lin_vel_w = body_lin_vel_w[:, 0]
                root_ang_vel_w = body_ang_vel_w[:, 0]
            return DatasetMotion(
                joint_pos=torch.tensor(
                    data["joint_pos"], dtype=torch.float32, device=self.device
                ),
                joint_vel=torch.tensor(
                    data["joint_vel"], dtype=torch.float32, device=self.device
                ),
                root_pos_w=root_pos_w,
                root_quat_w=root_quat_w,
                root_lin_vel_w=root_lin_vel_w,
                root_ang_vel_w=root_ang_vel_w,
                body_pos_w=body_pos_w,
                body_quat_w=body_quat_w,
                body_lin_vel_w=body_lin_vel_w,
                body_ang_vel_w=body_ang_vel_w,
                source_dt=1.0 / fps,
            )

        qpos = self._load_qpos(path)
        data = mujoco.MjData(self.model)
        body_pos = np.zeros((len(qpos), len(self.body_ids), 3), dtype=np.float32)
        body_quat = np.zeros((len(qpos), len(self.body_ids), 4), dtype=np.float32)

        for i, row in enumerate(qpos):
            data.qpos[:] = row
            mujoco.mj_forward(self.model, data)
            body_pos[i] = data.xpos[self.body_ids]
            body_quat[i] = data.xquat[self.body_ids]

        joint_pos = qpos[:, self.joint_qadr].astype(np.float32)
        joint_vel = _finite_difference(joint_pos, self.dt)
        root_pos = qpos[:, self.free_joint_qadr[:3]].astype(np.float32)
        root_quat = qpos[:, self.free_joint_qadr[3:7]].astype(np.float32)
        root_lin_vel = _finite_difference(root_pos, self.dt)
        root_ang_vel = _quat_ang_vel(root_quat, self.dt)
        body_lin_vel = _finite_difference(body_pos, self.dt)
        body_ang_vel = np.stack(
            [
                _quat_ang_vel(body_quat[:, i], self.dt)
                for i in range(body_quat.shape[1])
            ],
            axis=1,
        )

        return DatasetMotion(
            joint_pos=torch.tensor(joint_pos, dtype=torch.float32, device=self.device),
            joint_vel=torch.tensor(joint_vel, dtype=torch.float32, device=self.device),
            root_pos_w=torch.tensor(root_pos, dtype=torch.float32, device=self.device),
            root_quat_w=torch.tensor(
                root_quat, dtype=torch.float32, device=self.device
            ),
            root_lin_vel_w=torch.tensor(
                root_lin_vel, dtype=torch.float32, device=self.device
            ),
            root_ang_vel_w=torch.tensor(
                root_ang_vel, dtype=torch.float32, device=self.device
            ),
            body_pos_w=torch.tensor(body_pos, dtype=torch.float32, device=self.device),
            body_quat_w=torch.tensor(
                body_quat, dtype=torch.float32, device=self.device
            ),
            body_lin_vel_w=torch.tensor(
                body_lin_vel, dtype=torch.float32, device=self.device
            ),
            body_ang_vel_w=torch.tensor(
                body_ang_vel, dtype=torch.float32, device=self.device
            ),
            source_dt=self.dt,
        )


class DatasetMotionCommand(CommandTerm):
    cfg: DatasetMotionCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: DatasetMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        paths = self._resolve_motion_paths()
        self.motion = DatasetMotionLoader(
            model=env.sim.mj_model,
            paths=paths,
            body_ids=self.body_indexes.cpu().tolist(),
            joint_qadr=self.robot.indexing.joint_q_adr.cpu().numpy(),
            dt=cfg.source_dt,
            device=self.device,
            max_cache_size=cfg.max_cache_size,
        )
        self.active_motion_ids = self._sample_active_motion_ids()

        self.motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_times_s = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.motion_lengths = torch.ones(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_source_dt = torch.full(
            (self.num_envs,), cfg.source_dt, dtype=torch.float32, device=self.device
        )
        self._compute_dt = 0.0
        self.joint_pos_ref = torch.zeros(
            self.num_envs,
            self.robot.data.default_joint_pos.shape[1],
            device=self.device,
        )
        self.joint_vel_ref = torch.zeros_like(self.joint_pos_ref)
        self.root_pos_ref_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.root_quat_ref_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.root_quat_ref_w[:, 0] = 1.0
        self.root_lin_vel_ref_w = torch.zeros_like(self.root_pos_ref_w)
        self.root_ang_vel_ref_w = torch.zeros_like(self.root_pos_ref_w)
        self.body_pos_ref_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_ref_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_ref_w[:, :, 0] = 1.0
        self.body_lin_vel_ref_w = torch.zeros_like(self.body_pos_ref_w)
        self.body_ang_vel_ref_w = torch.zeros_like(self.body_pos_ref_w)
        self.body_pos_relative_w = torch.zeros_like(self.body_pos_ref_w)
        self.body_quat_relative_w = torch.zeros_like(self.body_quat_ref_w)
        self.body_quat_relative_w[:, :, 0] = 1.0
        self.reference_pos_offset = torch.zeros(self.num_envs, 3, device=self.device)
        self.reference_yaw_offset = torch.zeros(self.num_envs, 4, device=self.device)
        self.reference_yaw_offset[:, 0] = 1.0

        self.metrics["motion_id"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_frame"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_length"] = torch.zeros(self.num_envs, device=self.device)
        for key in (
            "error_anchor_pos",
            "error_anchor_rot",
            "error_anchor_lin_vel",
            "error_anchor_ang_vel",
            "error_body_pos",
            "error_body_rot",
            "error_body_lin_vel",
            "error_body_ang_vel",
            "error_joint_pos",
            "error_joint_vel",
            "sampling_entropy",
            "sampling_top1_prob",
            "sampling_top1_bin",
        ):
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

        self._ghost_model = None
        self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

    def _resolve_motion_paths(self) -> list[Path]:
        if self.cfg.motion_list is not None:
            list_path = Path(self.cfg.motion_list)
            if not list_path.is_file():
                raise FileNotFoundError(
                    f"General-motion list does not exist: {list_path}"
                )
            paths = [
                Path(line.strip())
                for line in list_path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        else:
            if not self.cfg.motion_root:
                raise ValueError(
                    "General motion tracking requires --env.commands.motion.motion-root "
                    "or --env.commands.motion.motion-list"
                )
            root = Path(self.cfg.motion_root)
            paths = sorted(root.glob(self.cfg.motion_glob))
        if self.cfg.max_num_motions is not None:
            paths = paths[: self.cfg.max_num_motions]
        if not paths:
            raise ValueError("DatasetMotionCommand selected no motion files")
        return paths

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.joint_pos_ref

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.joint_vel_ref

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self.root_pos_ref_w + self._env.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.root_quat_ref_w

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self.root_lin_vel_ref_w

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self.root_ang_vel_ref_w

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.body_pos_ref_w + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.body_quat_ref_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_ref_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_ref_w

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_ref_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_ref_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_ref_w[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_reference_buffers(self, env_ids: torch.Tensor) -> None:
        unique_motion_ids = torch.unique(self.motion_ids[env_ids])
        for motion_id_tensor in unique_motion_ids:
            selected_env_ids = env_ids[self.motion_ids[env_ids] == motion_id_tensor]
            motion_id = int(motion_id_tensor.item())
            motion = self.motion.get(motion_id)
            frame_pos = torch.clamp(
                self.motion_times_s[selected_env_ids] / max(motion.source_dt, 1.0e-9),
                min=0.0,
                max=float(motion.time_step_total - 1),
            )
            frame0 = torch.floor(frame_pos).long()
            frame1 = torch.clamp(frame0 + 1, max=motion.time_step_total - 1)
            alpha = (frame_pos - frame0.float()).unsqueeze(-1)
            body_alpha = alpha.unsqueeze(-1)

            self.joint_pos_ref[selected_env_ids] = _lerp(
                motion.joint_pos[frame0], motion.joint_pos[frame1], alpha
            )
            self.joint_vel_ref[selected_env_ids] = _lerp(
                motion.joint_vel[frame0], motion.joint_vel[frame1], alpha
            )
            root_pos = _lerp(
                motion.root_pos_w[frame0], motion.root_pos_w[frame1], alpha
            )
            root_quat = _quat_slerp_batch(
                motion.root_quat_w[frame0], motion.root_quat_w[frame1], alpha
            )
            root_lin_vel = _lerp(
                motion.root_lin_vel_w[frame0], motion.root_lin_vel_w[frame1], alpha
            )
            root_ang_vel = _lerp(
                motion.root_ang_vel_w[frame0], motion.root_ang_vel_w[frame1], alpha
            )
            body_pos = _lerp(
                motion.body_pos_w[frame0], motion.body_pos_w[frame1], body_alpha
            )
            body_quat = _quat_slerp_batch(
                motion.body_quat_w[frame0], motion.body_quat_w[frame1], body_alpha
            )
            body_lin_vel = _lerp(
                motion.body_lin_vel_w[frame0], motion.body_lin_vel_w[frame1], body_alpha
            )
            body_ang_vel = _lerp(
                motion.body_ang_vel_w[frame0], motion.body_ang_vel_w[frame1], body_alpha
            )

            yaw_offset = self.reference_yaw_offset[selected_env_ids]
            pos_offset = self.reference_pos_offset[selected_env_ids]
            self.root_pos_ref_w[selected_env_ids] = (
                quat_apply(yaw_offset, root_pos) + pos_offset
            )
            self.root_quat_ref_w[selected_env_ids] = quat_mul(yaw_offset, root_quat)
            self.root_lin_vel_ref_w[selected_env_ids] = quat_apply(
                yaw_offset, root_lin_vel
            )
            self.root_ang_vel_ref_w[selected_env_ids] = quat_apply(
                yaw_offset, root_ang_vel
            )
            body_yaw_offset = yaw_offset[:, None, :].expand_as(body_quat)
            self.body_pos_ref_w[selected_env_ids] = (
                quat_apply(body_yaw_offset, body_pos) + pos_offset[:, None, :]
            )
            self.body_quat_ref_w[selected_env_ids] = quat_mul(
                body_yaw_offset, body_quat
            )
            self.body_lin_vel_ref_w[selected_env_ids] = quat_apply(
                body_yaw_offset, body_lin_vel
            )
            self.body_ang_vel_ref_w[selected_env_ids] = quat_apply(
                body_yaw_offset, body_ang_vel
            )

    def _sync_time_steps_from_time(self, env_ids: torch.Tensor) -> None:
        frames = torch.floor(
            self.motion_times_s[env_ids]
            / torch.clamp(self.motion_source_dt[env_ids], min=1.0e-9)
        ).long()
        frames = torch.maximum(frames, torch.zeros_like(frames))
        frames = torch.minimum(frames, self.motion_lengths[env_ids] - 1)
        self.time_steps[env_ids] = frames

    def _sample_active_motion_ids(self) -> torch.Tensor:
        count = min(self.cfg.active_motion_count, len(self.motion))
        ids = random.sample(range(len(self.motion)), count)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _maybe_refresh_active_pool(self) -> None:
        if len(self.motion) <= self.cfg.active_motion_count:
            return
        if random.random() >= self.cfg.active_motion_refresh_prob:
            return
        replace_count = max(
            1,
            int(self.cfg.active_motion_count * self.cfg.active_motion_refresh_fraction),
        )
        replace_count = min(replace_count, len(self.active_motion_ids))
        slots = torch.randperm(len(self.active_motion_ids), device=self.device)[
            :replace_count
        ]
        new_ids = torch.tensor(
            random.sample(range(len(self.motion)), replace_count),
            dtype=torch.long,
            device=self.device,
        )
        self.active_motion_ids[slots] = new_ids

    def _sample_motions(self, env_ids: torch.Tensor) -> None:
        self._maybe_refresh_active_pool()
        pool_indexes = torch.randint(
            len(self.active_motion_ids),
            (len(env_ids),),
            dtype=torch.long,
            device=self.device,
        )
        motion_ids = self.active_motion_ids[pool_indexes]
        self.motion_ids[env_ids] = motion_ids
        motions = [self.motion.get(int(i.item())) for i in motion_ids]
        lengths = [motion.time_step_total for motion in motions]
        source_dts = [motion.source_dt for motion in motions]
        self.motion_lengths[env_ids] = torch.tensor(
            lengths, dtype=torch.long, device=self.device
        )
        self.motion_source_dt[env_ids] = torch.tensor(
            source_dts, dtype=torch.float32, device=self.device
        )

    def _sample_frames(self, env_ids: torch.Tensor) -> None:
        if self.cfg.sampling_mode == "start":
            self.time_steps[env_ids] = 0
            self.motion_times_s[env_ids] = 0.0
            return
        random_values = torch.rand(len(env_ids), device=self.device)
        max_steps = torch.clamp(self.motion_lengths[env_ids] - 1, min=1)
        self.time_steps[env_ids] = (random_values * max_steps).long()
        self.motion_times_s[env_ids] = (
            self.time_steps[env_ids].float() * self.motion_source_dt[env_ids]
        )

    def _write_reference_state_to_sim(
        self,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_ori: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

    def _resample_command(self, env_ids: torch.Tensor):
        self.reference_pos_offset[env_ids] = 0.0
        self.reference_yaw_offset[env_ids] = 0.0
        self.reference_yaw_offset[env_ids, 0] = 1.0
        self._sample_motions(env_ids)
        self._sample_frames(env_ids)
        self._update_reference_buffers(env_ids)
        if not self.cfg.initialize_robot_to_reference:
            self._align_reference_to_robot(env_ids)
            self._update_reference_buffers(env_ids)
            self.update_relative_body_poses()
            return

        root_pos = self.root_pos_w[env_ids].clone()
        root_ori = self.root_quat_w[env_ids].clone()
        root_lin_vel = self.root_lin_vel_w[env_ids].clone()
        root_ang_vel = self.root_ang_vel_w[env_ids].clone()

        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_pos += rand_samples[:, 0:3]
        root_ori = quat_mul(
            quat_from_euler_xyz(
                rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
            ),
            root_ori,
        )

        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_pos += sample_uniform(
            lower=self.cfg.joint_position_range[0],
            upper=self.cfg.joint_position_range[1],
            size=joint_pos.shape,
            device=self.device,
        )

        self._write_reference_state_to_sim(
            env_ids,
            root_pos,
            root_ori,
            root_lin_vel,
            root_ang_vel,
            joint_pos,
            self.joint_vel[env_ids],
        )

    def update_relative_body_poses(self) -> None:
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(
            quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
        )

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
        )

    def compute(self, dt: float) -> None:
        self._compute_dt = dt
        super().compute(dt)

    def _update_command(self):
        self.motion_times_s += self._compute_dt
        motion_end_times = (self.motion_lengths.float() - 1.0) * self.motion_source_dt
        env_ids = torch.where(self.motion_times_s > motion_end_times)[0]
        if env_ids.numel() > 0:
            self._resample_command(env_ids)
        self._sync_time_steps_from_time(torch.arange(self.num_envs, device=self.device))
        self._update_reference_buffers(torch.arange(self.num_envs, device=self.device))
        self.update_relative_body_poses()
        self._update_metrics()

    def _update_metrics(self):
        from mjlab.utils.lab_api.math import quat_error_magnitude

        self.metrics["motion_id"] = self.motion_ids.float()
        self.metrics["motion_frame"] = self.motion_times_s / torch.clamp(
            self.motion_source_dt, min=1.0e-9
        )
        self.metrics["motion_length"] = self.motion_lengths.float()
        self.metrics["sampling_entropy"][:] = 1.0
        self.metrics["sampling_top1_prob"][:] = 1.0 / max(len(self.motion), 1)
        self.metrics["sampling_top1_bin"][:] = 0.5
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
        )
        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)
        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - self.robot_joint_vel, dim=-1
        )

    def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
        self.reference_pos_offset[env_ids] = 0.0
        self.reference_yaw_offset[env_ids] = 0.0
        self.reference_yaw_offset[env_ids, 0] = 1.0
        frames = torch.full_like(env_ids, frame)
        frames = torch.maximum(frames, torch.zeros_like(frames))
        frames = torch.minimum(frames, self.motion_lengths[env_ids] - 1)
        self.time_steps[env_ids] = frames
        self.motion_times_s[env_ids] = (
            self.time_steps[env_ids].float() * self.motion_source_dt[env_ids]
        )
        self._update_reference_buffers(env_ids)
        if not self.cfg.initialize_robot_to_reference:
            self._align_reference_to_robot(env_ids)
            self._update_reference_buffers(env_ids)
            return
        self._write_reference_state_to_sim(
            env_ids,
            self.root_pos_w[env_ids],
            self.root_quat_w[env_ids],
            self.root_lin_vel_w[env_ids],
            self.root_ang_vel_w[env_ids],
            self.joint_pos[env_ids],
            self.joint_vel[env_ids],
        )

    def _align_reference_to_robot(self, env_ids: torch.Tensor) -> None:
        yaw_offset = yaw_quat(
            quat_mul(
                self.robot_anchor_quat_w[env_ids],
                quat_inv(self.anchor_quat_w[env_ids]),
            )
        )
        anchor_pos_local = (
            self.anchor_pos_w[env_ids] - self._env.scene.env_origins[env_ids]
        )
        robot_anchor_pos_local = (
            self.robot_anchor_pos_w[env_ids] - self._env.scene.env_origins[env_ids]
        )
        pos_offset = robot_anchor_pos_local - quat_apply(yaw_offset, anchor_pos_local)
        pos_offset[:, 2] = 0.0
        self.reference_yaw_offset[env_ids] = yaw_offset
        self.reference_pos_offset[env_ids] = pos_offset

    def select_motion(
        self, env_ids: torch.Tensor, motion_id: int, frame: int = 0
    ) -> None:
        motion_id = int(np.clip(motion_id, 0, len(self.motion) - 1))
        motion = self.motion.get(motion_id)
        self.motion_ids[env_ids] = motion_id
        self.motion_lengths[env_ids] = motion.time_step_total
        self.motion_source_dt[env_ids] = motion.source_dt
        self.reset_to_frame(env_ids, frame)
        self.update_relative_body_poses()

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        if self.cfg.viz.mode != "ghost":
            return

        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        if self._ghost_model is None:
            # Reuse the tracking task's pattern: a visual-only MuJoCo model, updated
            # from the already-loaded reference state instead of reading motion files.
            self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
            for gi in range(self._ghost_model.ngeom):
                if (
                    self._ghost_model.geom_contype[gi] != 0
                    or self._ghost_model.geom_conaffinity[gi] != 0
                ):
                    self._ghost_model.geom_rgba[gi, 3] = 0
                else:
                    self._ghost_model.geom_rgba[gi] = self._ghost_color

        entity: Entity = self._env.scene[self.cfg.entity_name]
        indexing = entity.indexing
        free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
        joint_q_adr = indexing.joint_q_adr.cpu().numpy()

        for batch in env_indices:
            qpos = np.zeros(self._env.sim.mj_model.nq)
            qpos[free_joint_q_adr[0:3]] = self.root_pos_w[batch].cpu().numpy()
            qpos[free_joint_q_adr[3:7]] = self.root_quat_w[batch].cpu().numpy()
            qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

            visualizer.add_ghost_mesh(
                qpos,
                model=self._ghost_model,
                label=f"general_motion_ghost_{batch}",
            )


@dataclass(kw_only=True)
class DatasetMotionCommandCfg(CommandTermCfg):
    anchor_body_name: str
    body_names: tuple[str, ...]
    entity_name: str
    motion_root: str = ""
    motion_glob: str = "**/*.npz"
    motion_list: str | None = None
    max_num_motions: int | None = None
    max_cache_size: int = 128
    active_motion_count: int = 128
    active_motion_refresh_prob: float = 0.02
    active_motion_refresh_fraction: float = 0.1
    source_dt: float = 1.0 / 120.0
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (-0.1, 0.1)
    sampling_mode: Literal["uniform", "start"] = "uniform"
    initialize_robot_to_reference: bool = True

    @dataclass
    class VizCfg:
        mode: Literal["ghost"] = "ghost"
        ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.45)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
        return DatasetMotionCommand(self, env)  # type: ignore[return-value]
