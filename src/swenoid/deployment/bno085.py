# SPDX-FileCopyrightText: 2020 Bryan Siepert, written for Adafruit Industries
# SPDX-License-Identifier: Unlicense
"""BNO085 pelvis-IMU frame conversion and Raspberry Pi I2C driver."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

STANDARD_GRAVITY = 9.80665

# Rotation from the Swenoid body frame to the physical BNO085 sensor frame.
FRAME_TRANSFORM = Rotation.from_matrix(
    np.asarray(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
)


def rot_gravity(value: Sequence[float]) -> tuple[float, float, float]:
    """Transform the BNO gravity vector into policy coordinates."""
    x, y, z = map(float, value)
    return z, -x, y


def rot_ang_vel(value: Sequence[float]) -> tuple[float, float, float]:
    """Transform BNO angular velocity into the Swenoid pelvis frame."""
    x, y, z = map(float, value)
    return -z, x, -y


def rot_orientation(value: Sequence[float]) -> np.ndarray:
    """Transform a scalar-first sensor orientation into the body frame."""
    sensor_orientation = Rotation.from_quat(value, scalar_first=True)
    body_orientation = sensor_orientation * FRAME_TRANSFORM.inv()
    return body_orientation.as_quat(scalar_first=True)


def rotate_orientation_vector(value: Sequence[float]) -> np.ndarray:
    """Backward-compatible name for :func:`rot_orientation`."""
    return rot_orientation(value)


def scale_gravity(value: Sequence[float]) -> tuple[float, float, float]:
    """Convert a gravity vector from m/s² to units of standard gravity."""
    x, y, z = map(float, value)
    return x / STANDARD_GRAVITY, y / STANDARD_GRAVITY, z / STANDARD_GRAVITY


class BNO085:
    """BNO08x reader configured for Swenoid's pelvis-mounted sensor."""

    def __init__(
        self,
        *,
        i2c_frequency: int = 800_000,
        sensor: Any | None = None,
    ) -> None:
        if i2c_frequency <= 0:
            raise ValueError("i2c_frequency must be positive")
        self.i2c = None
        if sensor is None:
            try:
                board = import_module("board")
                busio = import_module("busio")
                bno_module = import_module("adafruit_bno08x")
                bno_i2c_module = import_module("adafruit_bno08x.i2c")
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "BNO085 hardware access requires Adafruit Blinka and "
                    "adafruit-circuitpython-bno08x. Install the Swenoid "
                    "deployment extra."
                ) from exc
            self.i2c = busio.I2C(
                board.SCL,
                board.SDA,
                frequency=i2c_frequency,
            )
            resolved_sensor: Any = bno_i2c_module.BNO08X_I2C(self.i2c)
            resolved_sensor.enable_feature(bno_module.BNO_REPORT_GYROSCOPE)
            resolved_sensor.enable_feature(bno_module.BNO_REPORT_GRAVITY)
            resolved_sensor.enable_feature(bno_module.BNO_REPORT_ROTATION_VECTOR)
        else:
            resolved_sensor = sensor
        self.bno: Any = resolved_sensor

    def get_ang_vel(self) -> tuple[float, float, float]:
        return rot_ang_vel(self.bno.gyro)

    def get_gravity_old(self) -> tuple[float, float, float]:
        """Return projected gravity in policy coordinates and units of g."""
        return scale_gravity(rot_gravity(self.bno.gravity))

    def get_gravity(self) -> tuple[float, float, float]:
        """Preferred alias for :meth:`get_gravity_old`."""
        return self.get_gravity_old()

    def get_orientation(self) -> np.ndarray:
        quat_i, quat_j, quat_k, quat_real = self.bno.quaternion
        return rot_orientation((quat_real, quat_i, quat_j, quat_k))

    def get_orientation_xyz(self) -> tuple[float, float, float]:
        euler = Rotation.from_quat(self.get_orientation(), scalar_first=True).as_euler(
            "xyz", degrees=True
        )
        return float(euler[0]), float(euler[1]), float(euler[2])


def main() -> None:
    """Print transformed orientation until interrupted."""
    import time

    sensor = BNO085()
    try:
        while True:
            x, y, z = sensor.get_orientation_xyz()
            print(f"x={x:8.3f} y={y:8.3f} z={z:8.3f}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
