"""Tests for Swenoid's thin MjLab task extension."""

import torch
from mjlab.tasks.registry import list_tasks, load_env_cfg

from swenoid.tasks.general_motion.commands import (
    DatasetMotionCommand,
    DatasetMotionCommandCfg,
)
from swenoid.tasks.mdp import SwenoidMotionCommandCfg, SwenoidVelocityCommandCfg


def test_swenoid_tasks_are_registered() -> None:
    assert {
        "Mjlab-Velocity-Flat-Swenoid",
        "Mjlab-Velocity-Rough-Swenoid",
        "Mjlab-Tracking-Flat-Swenoid",
        "Mjlab-General-Motion-Flat-Swenoid",
    } <= set(list_tasks())


def test_velocity_actor_uses_deployable_observations() -> None:
    expected = [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
    ]
    for task_id in (
        "Mjlab-Velocity-Flat-Swenoid",
        "Mjlab-Velocity-Rough-Swenoid",
    ):
        cfg = load_env_cfg(task_id)
        assert list(cfg.observations["actor"].terms) == expected
        command = cfg.commands["twist"]
        assert isinstance(command, SwenoidVelocityCommandCfg)
        assert command.command_deadband == 0.05
        assert cfg.decimation == 4
        assert cfg.sim.mujoco.timestep == 0.005


def test_tracking_matches_velocity_proprioception() -> None:
    tracking = load_env_cfg("Mjlab-Tracking-Flat-Swenoid")
    velocity = load_env_cfg("Mjlab-Velocity-Flat-Swenoid")
    assert list(tracking.observations["actor"].terms) == list(
        velocity.observations["actor"].terms
    )
    command = tracking.commands["motion"]
    assert isinstance(command, SwenoidMotionCommandCfg)
    assert command.motion_file == ""
    assert command.anchor_body_name == "hip_mid_section"
    assert tracking.decimation == 4
    assert tracking.sim.mujoco.timestep == 0.005


def test_general_motion_matches_deployable_proprioception() -> None:
    general = load_env_cfg("Mjlab-General-Motion-Flat-Swenoid")
    tracking = load_env_cfg("Mjlab-Tracking-Flat-Swenoid")
    velocity = load_env_cfg("Mjlab-Velocity-Flat-Swenoid")
    expected = [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
    ]
    assert list(general.observations["actor"].terms) == expected
    assert list(general.observations["actor"].terms) == list(
        tracking.observations["actor"].terms
    )
    assert list(general.observations["actor"].terms) == list(
        velocity.observations["actor"].terms
    )
    command = general.commands["motion"]
    assert isinstance(command, DatasetMotionCommandCfg)
    assert command.motion_root == ""
    assert command.motion_list is None


def test_general_motion_reference_is_joint_position_and_velocity_only() -> None:
    command = DatasetMotionCommand.__new__(DatasetMotionCommand)
    command.joint_pos_ref = torch.arange(48, dtype=torch.float32).reshape(2, 24)
    command.joint_vel_ref = -command.joint_pos_ref
    reference = command.command
    assert reference.shape == (2, 48)
    torch.testing.assert_close(reference[:, :24], command.joint_pos_ref)
    torch.testing.assert_close(reference[:, 24:], command.joint_vel_ref)


def test_velocity_low_command_rewards_are_consistent() -> None:
    cfg = load_env_cfg("Mjlab-Velocity-Flat-Swenoid")
    threshold = cfg.commands["twist"].command_deadband
    assert threshold == 0.05
    assert cfg.rewards["feet_still_at_low_command"].weight < 0.0
    for name in (
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
    ):
        assert cfg.rewards[name].params["command_threshold"] == threshold
