"""Tests for Swenoid-specific command and reward terms."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from swenoid.tasks.mdp import (
    SwenoidMotionCommand,
    SwenoidVelocityCommand,
    feet_still_at_low_command,
)


def test_velocity_deadband_snaps_only_tiny_commands() -> None:
    term = SwenoidVelocityCommand.__new__(SwenoidVelocityCommand)
    term.cfg = SimpleNamespace(command_deadband=0.2)
    term._env = SimpleNamespace(device="cpu", num_envs=3)
    term.vel_command_b = torch.tensor(
        [[0.1, 0.0, 0.0], [0.15, 0.15, 0.0], [0.25, 0.0, 0.0]]
    )
    term.vel_command_w = term.vel_command_b.clone()
    term._apply_command_deadband()
    expected = torch.tensor([[0.0, 0.0, 0.0], [0.15, 0.15, 0.0], [0.25, 0.0, 0.0]])
    torch.testing.assert_close(term.vel_command_b, expected)
    torch.testing.assert_close(term.vel_command_w, expected)


def test_feet_still_penalty_only_applies_at_low_command() -> None:
    command = torch.tensor([[0.01, 0.0, 0.0], [0.4, 0.0, 0.0]])
    asset = SimpleNamespace(
        data=SimpleNamespace(
            site_lin_vel_w=torch.tensor(
                [
                    [[0.0, 0.0, 0.1], [0.2, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                ]
            )
        )
    )
    sensor = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1, 0], [0, 0]])))
    env = SimpleNamespace(
        scene={"robot": asset, "feet": sensor},
        command_manager=SimpleNamespace(get_command=lambda _: command),
        extras={"log": {}},
    )
    cost = feet_still_at_low_command(
        env,
        sensor_name="feet",
        command_name="twist",
        command_threshold=0.15,
        asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]),
    )
    torch.testing.assert_close(cost, torch.tensor([0.5125, 0.0]))


def test_tracking_rejects_wrong_motion_rate_before_environment_build(tmp_path) -> None:
    motion = tmp_path / "motion.npz"
    np.savez(motion, fps=np.asarray([120.0], dtype=np.float32))
    cfg = SimpleNamespace(motion_file=str(motion))
    env = SimpleNamespace(step_dt=0.02)
    with pytest.raises(ValueError, match="120 Hz"):
        SwenoidMotionCommand(cfg, env)
