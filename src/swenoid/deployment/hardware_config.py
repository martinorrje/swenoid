"""Validated, serializable configuration for the physical Swenoid robot."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from swenoid.model_constants import SWENOID_HARDWARE_CONFIG

SIMULATION_JOINT_NAMES = (
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_joint",
    "torso_pitch_joint",
    "torso_roll_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "neck_yaw_joint",
    "neck_pitch_joint",
    "neck_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)


def _int_tuple(values: Sequence[Any], field: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain integers") from exc


@dataclass(frozen=True)
class HardwareConfig:
    """Motor ordering, signs, and neutral encoder positions for one robot."""

    motor_ids: tuple[int, ...]
    simulation_joint_names: tuple[str, ...]
    dynamixel_joint_names: tuple[str, ...]
    axis_signs: tuple[int, ...]
    zero_positions: tuple[int, ...]
    name: str = "swenoid-default"
    schema_version: int = 1

    def __post_init__(self) -> None:
        count = len(SIMULATION_JOINT_NAMES)
        fields = {
            "motor_ids": self.motor_ids,
            "simulation_joint_names": self.simulation_joint_names,
            "dynamixel_joint_names": self.dynamixel_joint_names,
            "axis_signs": self.axis_signs,
            "zero_positions": self.zero_positions,
        }
        for field, values in fields.items():
            if len(values) != count:
                raise ValueError(f"{field} must contain {count} values")
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported hardware config schema {self.schema_version}"
            )
        if self.simulation_joint_names != SIMULATION_JOINT_NAMES:
            raise ValueError("simulation_joint_names must use the Swenoid model order")
        if len(set(self.motor_ids)) != count or any(
            motor_id < 1 or motor_id > 252 for motor_id in self.motor_ids
        ):
            raise ValueError("motor_ids must be 24 unique Dynamixel IDs in [1, 252]")
        if set(self.dynamixel_joint_names) != set(SIMULATION_JOINT_NAMES):
            raise ValueError(
                "dynamixel_joint_names must contain every Swenoid joint once"
            )
        if any(sign not in (-1, 1) for sign in self.axis_signs):
            raise ValueError("axis_signs values must be -1 or 1")
        if any(position < 0 or position > 4095 for position in self.zero_positions):
            raise ValueError("zero_positions values must be in [0, 4095]")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HardwareConfig:
        """Validate and construct a configuration from decoded JSON."""
        required = (
            "motor_ids",
            "simulation_joint_names",
            "dynamixel_joint_names",
            "axis_signs",
            "zero_positions",
        )
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError(f"Hardware config is missing: {', '.join(missing)}")
        return cls(
            motor_ids=_int_tuple(payload["motor_ids"], "motor_ids"),
            simulation_joint_names=tuple(map(str, payload["simulation_joint_names"])),
            dynamixel_joint_names=tuple(map(str, payload["dynamixel_joint_names"])),
            axis_signs=_int_tuple(payload["axis_signs"], "axis_signs"),
            zero_positions=_int_tuple(payload["zero_positions"], "zero_positions"),
            name=str(payload.get("name", "swenoid-custom")),
            schema_version=int(payload.get("schema_version", 1)),
        )

    @classmethod
    def load(cls, path: Path | str | None = None) -> HardwareConfig:
        """Load the packaged profile or a per-robot JSON profile."""
        config_path = SWENOID_HARDWARE_CONFIG if path is None else Path(path)
        try:
            with config_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid hardware config JSON: {config_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Hardware config must contain a JSON object")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible data."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "motor_ids": list(self.motor_ids),
            "simulation_joint_names": list(self.simulation_joint_names),
            "dynamixel_joint_names": list(self.dynamixel_joint_names),
            "axis_signs": list(self.axis_signs),
            "zero_positions": list(self.zero_positions),
        }

    def with_zero_positions(
        self, positions: Sequence[int], *, name: str | None = None
    ) -> HardwareConfig:
        """Return a calibrated copy using neutral encoder readings."""
        return replace(
            self,
            zero_positions=_int_tuple(positions, "zero_positions"),
            name=self.name if name is None else name,
        )

    def write(self, path: Path, *, overwrite: bool = False) -> None:
        """Write a profile without silently replacing an existing calibration."""
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
