"""General-motion reward helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.sensor import ContactSensor

from .commands import DatasetMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _contact_count(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float | None = None,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if force_threshold is None:
        assert data.found is not None
        return (data.found > 0).float().sum(dim=-1)
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)
        return (force_mag > force_threshold).float().sum(dim=(1, 2))
    if data.force is not None:
        force_mag = torch.norm(data.force, dim=-1)
        return (force_mag > force_threshold).float().sum(dim=-1)
    assert data.found is not None
    return (data.found > 0).float().sum(dim=-1)


def reference_anchor_height_gate(
    env: ManagerBasedRlEnv,
    command_name: str,
    low_height: float,
    high_height: float,
) -> torch.Tensor:
    """Return 0 while the reference is low and 1 once it reaches standing height."""
    command = cast(DatasetMotionCommand, env.command_manager.get_term(command_name))
    height = command.anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.clamp((height - low_height) / (high_height - low_height), 0.0, 1.0)


def contact_cost_by_reference_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    low_height: float,
    high_height: float,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
    force_threshold: float | None = None,
) -> torch.Tensor:
    """Penalize selected contacts mostly when the reference has risen up."""
    gate = reference_anchor_height_gate(env, command_name, low_height, high_height)
    scale = min_scale + (max_scale - min_scale) * gate
    return _contact_count(env, sensor_name, force_threshold) * scale


def no_contact_by_reference_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    low_height: float,
    high_height: float,
) -> torch.Tensor:
    """Reward no selected contacts once the reference is in its upright phase."""
    gate = reference_anchor_height_gate(env, command_name, low_height, high_height)
    no_contact = _contact_count(env, sensor_name) == 0
    return no_contact.float() * gate


def foot_support_by_reference_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    low_height: float,
    high_height: float,
    min_contacts: float = 2.0,
) -> torch.Tensor:
    """Reward foot support only once the reference is high enough to need it."""
    gate = reference_anchor_height_gate(env, command_name, low_height, high_height)
    contact_fraction = torch.clamp(
        _contact_count(env, sensor_name) / min_contacts, max=1.0
    )
    return contact_fraction * gate
