"""Reset and posture terms used by general-motion training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_SE3_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def _sample_se3_range(
    range_dict: dict[str, tuple[float, float]] | None,
    shape: tuple[int, ...],
    device: str,
) -> torch.Tensor:
    ranges = torch.tensor(
        [(range_dict or {}).get(key, (0.0, 0.0)) for key in _SE3_KEYS],
        device=device,
    )
    return sample_uniform(ranges[:, 0], ranges[:, 1], shape, device=device)


def reset_root_state_random_lying(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    xy_range: tuple[float, float] = (-0.2, 0.2),
    root_height_range: tuple[float, float] = (0.11, 0.18),
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    roll_noise_range: tuple[float, float] = (-0.2, 0.2),
    pitch_noise_range: tuple[float, float] = (-0.2, 0.2),
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset the floating base into randomized side, prone, or supine poses."""
    env_ids = resolve_env_ids(env, env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    if asset.is_fixed_base:
        raise ValueError("General-motion lying reset requires a floating base")

    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()
    num_envs = len(env_ids)
    positions = root_states[:, :3] + env.scene.env_origins[env_ids]
    positions[:, 0] += sample_uniform(*xy_range, num_envs, env.device)
    positions[:, 1] += sample_uniform(*xy_range, num_envs, env.device)
    positions[:, 2] = env.scene.env_origins[env_ids, 2] + sample_uniform(
        *root_height_range, num_envs, env.device
    )

    roll = torch.zeros(num_envs, device=env.device)
    pitch = torch.zeros(num_envs, device=env.device)
    lying_mode = torch.randint(0, 4, (num_envs,), device=env.device)
    roll[lying_mode == 0] = math.pi / 2.0
    roll[lying_mode == 1] = -math.pi / 2.0
    pitch[lying_mode == 2] = math.pi / 2.0
    pitch[lying_mode == 3] = -math.pi / 2.0
    roll += sample_uniform(*roll_noise_range, num_envs, env.device)
    pitch += sample_uniform(*pitch_noise_range, num_envs, env.device)
    yaw = sample_uniform(*yaw_range, num_envs, env.device)
    orientations = quat_mul(root_states[:, 3:7], quat_from_euler_xyz(roll, pitch, yaw))
    velocities = root_states[:, 7:13] + _sample_se3_range(
        velocity_range, (num_envs, 6), env.device
    )

    asset.write_root_link_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def _body_projected_gravity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    if isinstance(asset_cfg.body_ids, slice):
        body_quat_w = asset.data.root_link_quat_w[:, None, :]
    else:
        body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    batch_size, num_bodies, _ = body_quat_w.shape
    gravity_w = asset.data.gravity_vec_w[:, None, :].expand(batch_size, num_bodies, 3)
    return quat_apply_inverse(
        body_quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3)
    ).reshape(batch_size, num_bodies, 3)


def signed_upright_exp(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward upright orientation while rejecting upside-down poses."""
    gravity = _body_projected_gravity(env, asset_cfg)
    target = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    error = gravity - target
    return torch.exp(-torch.sum(torch.square(error), dim=-1).mean(dim=-1) / std**2)


def body_height_exp(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward selected body heights near a standing target."""
    asset: Entity = env.scene[asset_cfg.name]
    if isinstance(asset_cfg.body_ids, slice):
        heights = asset.data.root_link_pos_w[:, None, 2]
    else:
        heights = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    heights = heights - env.scene.env_origins[:, None, 2]
    error = heights - target_height
    return torch.exp(-torch.mean(torch.square(error), dim=-1) / std**2)
