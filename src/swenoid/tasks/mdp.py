"""Swenoid-specific MDP terms layered on released MjLab tasks."""

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class SwenoidVelocityCommand(UniformVelocityCommand):
    """Velocity command that turns ambiguous low-speed samples into standing."""

    def _apply_command_deadband(self, env_ids: torch.Tensor | None = None) -> None:
        threshold = cast("SwenoidVelocityCommandCfg", self.cfg).command_deadband
        if threshold <= 0.0:
            return
        if env_ids is None:
            command = self.vel_command_b
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            command = self.vel_command_b[env_ids]
            ids = env_ids
        magnitude = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        low_command_ids = ids[magnitude <= threshold]
        self.vel_command_b[low_command_ids] = 0.0
        self.vel_command_w[low_command_ids] = 0.0

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        self._apply_command_deadband(env_ids)

    def _update_command(self) -> None:
        super()._update_command()
        self._apply_command_deadband()

    def compute(self, dt: float) -> None:
        super().compute(dt)
        self._apply_command_deadband()


@dataclass(kw_only=True)
class SwenoidVelocityCommandCfg(UniformVelocityCommandCfg):
    """Uniform velocity command with a configurable zero-command deadband."""

    command_deadband: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> SwenoidVelocityCommand:
        return SwenoidVelocityCommand(self, env)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.command_deadband < 0.0:
            raise ValueError("command_deadband must be non-negative")


def feet_still_at_low_command(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.15,
    air_cost_scale: float = 1.0,
    velocity_cost_scale: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize lifted or moving feet only when the command is near zero."""
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    assert contact_sensor.data.found is not None

    magnitude = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    low_command = (magnitude <= command_threshold).float()
    air_cost = torch.mean((contact_sensor.data.found == 0).float(), dim=1)
    foot_vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
    velocity_cost = torch.mean(torch.sum(torch.square(foot_vel), dim=-1), dim=1)

    count = torch.clamp(torch.sum(low_command), min=1.0)
    env.extras["log"]["Metrics/low_command_foot_air_fraction"] = (
        torch.sum(air_cost * low_command) / count
    )
    env.extras["log"]["Metrics/low_command_foot_speed_mean"] = (
        torch.sum(torch.mean(torch.norm(foot_vel, dim=-1), dim=1) * low_command) / count
    )
    return (
        air_cost_scale * air_cost + velocity_cost_scale * velocity_cost
    ) * low_command


class SwenoidMotionCommand(MotionCommand):
    """Single-motion command that enforces the 50 Hz reference contract."""

    def __init__(self, cfg: "SwenoidMotionCommandCfg", env: ManagerBasedRlEnv):
        with np.load(cfg.motion_file) as data:
            if "fps" not in data:
                raise ValueError(f"Motion file has no fps metadata: {cfg.motion_file}")
            motion_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        control_fps = 1.0 / env.step_dt
        if not math.isclose(motion_fps, control_fps, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError(
                f"Motion is {motion_fps:g} Hz, but the task runs at {control_fps:g} Hz. "
                "Run swenoid-convert-motion with --output-fps 50."
            )
        super().__init__(cfg, env)


@dataclass(kw_only=True)
class SwenoidMotionCommandCfg(MotionCommandCfg):
    """Motion command configuration with frame-rate validation."""

    def build(self, env: ManagerBasedRlEnv) -> SwenoidMotionCommand:
        return SwenoidMotionCommand(self, env)
