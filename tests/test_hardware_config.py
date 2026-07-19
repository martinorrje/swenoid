"""Tests for auditable, per-robot deployment calibration."""

from dataclasses import replace

import pytest

from swenoid.deployment.calibrate import inspect_hardware
from swenoid.deployment.hardware_config import HardwareConfig


class _CalibrationHandler:
    def __init__(self, positions: list[int] | None = None):
        self.positions = positions or [2048] * 24
        self.disabled_ids: list[int] = []

    def disable_torque(self, ids):
        self.disabled_ids = list(ids)

    def read_servo_positions(self, _ids):
        return self.positions

    def read_lower_limits(self, ids):
        return [0] * len(ids)

    def read_upper_limits(self, ids):
        return [4095] * len(ids)

    def read_servo_voltages(self, ids):
        return [120] * len(ids)


def test_packaged_hardware_config_round_trip(tmp_path) -> None:
    config = HardwareConfig.load()
    assert config.motor_ids == tuple(range(1, 25))
    assert len(config.dynamixel_joint_names) == 24
    assert config.zero_positions[4] == 3072

    output = tmp_path / "robot.json"
    config.write(output)
    assert HardwareConfig.load(output) == config
    with pytest.raises(FileExistsError, match="overwrite"):
        config.write(output)


def test_hardware_config_rejects_invalid_axis_sign() -> None:
    config = HardwareConfig.load()
    with pytest.raises(ValueError, match="axis_signs"):
        replace(config, axis_signs=(0, *config.axis_signs[1:]))


def test_torque_off_inspection_captures_neutral_positions(capsys) -> None:
    config = HardwareConfig.load()
    positions = list(range(2000, 2024))
    handler = _CalibrationHandler(positions)
    captured = inspect_hardware(handler, config)
    assert captured == tuple(positions)
    assert handler.disabled_ids == list(range(1, 25))
    assert "neck_pitch_joint" in capsys.readouterr().out


def test_calibrated_config_uses_captured_positions() -> None:
    config = HardwareConfig.load()
    positions = tuple(range(2000, 2024))
    calibrated = config.with_zero_positions(positions, name="robot-one")
    assert calibrated.name == "robot-one"
    assert calibrated.zero_positions == positions
