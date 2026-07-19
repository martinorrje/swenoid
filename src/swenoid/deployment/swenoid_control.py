"""Dynamixel joint ordering, unit conversion, and safety checks for Swenoid.

The hardware-specific ``motor_controller`` module is imported only when the
controller is constructed, so simulation and preprocessing remain portable.
"""

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from swenoid.deployment.hardware_config import HardwareConfig


class SwenoidControl:
    """Compatibility name for the original Swenoid hardware controller."""

    def __init__(
        self,
        config: HardwareConfig | Path | str | None = None,
        device: Any = None,
        dynamixel_handler: Any = None,
        *,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 4_000_000,
        configure_latency: bool = True,
        max_retries: int = 3,
    ) -> None:
        del device
        hardware = (
            config
            if isinstance(config, HardwareConfig)
            else HardwareConfig.load(config)
        )
        if dynamixel_handler is None:
            try:
                motor_controller = import_module("swenoid.deployment.motor_controller")
                handler_type = motor_controller.DynamixelHandler
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Real-robot deployment requires the Raspberry Pi "
                    "motor_controller module. See docs/deployment.md."
                ) from exc
            dynamixel_handler = handler_type(
                port=port,
                baudrate=baudrate,
                configure_latency=configure_latency,
                max_retries=max_retries,
            )
        self.dynamixel_handler = dynamixel_handler

        self.all_ids = list(hardware.motor_ids)

        self.delay = 0.0
        self.durations_ms = [int(self.delay * 1000) for _ in range(24)]
        self._handler_configured = False
        self._closed = False

        try:
            self._setup_dxl_handler()
        except BaseException:
            self.dynamixel_handler.close()
            self._closed = True
            raise

        self.isaac_dof_names = list(hardware.simulation_joint_names)
        self.dynamixel_dof_names = list(hardware.dynamixel_joint_names)

        self.dynamixel_ids_to_isaac_ids = [
            self.dynamixel_dof_names.index(isaac_dof_name)
            for isaac_dof_name in self.isaac_dof_names
        ]

        self.isaac_ids_to_dynamixel_ids = [
            self.isaac_dof_names.index(dxl_dof_name)
            for dxl_dof_name in self.dynamixel_dof_names
        ]

        self.dynamixel_axis_mask = np.asarray(hardware.axis_signs, dtype=np.float32)

        lower_limits_raw = self.dynamixel_handler.read_lower_limits(self.all_ids)
        lower_limits = [
            int(value[0] if isinstance(value, tuple) else value)
            for value in lower_limits_raw
        ]
        upper_limits_raw = self.dynamixel_handler.read_upper_limits(self.all_ids)
        upper_limits = [
            int(value[0] if isinstance(value, tuple) else value)
            for value in upper_limits_raw
        ]
        self.lower_limits = np.asarray(lower_limits, dtype=np.int64)
        self.upper_limits = np.asarray(upper_limits, dtype=np.int64)

        self.dynamixel_base_pos = (
            np.asarray(hardware.zero_positions, dtype=np.float32) - 2048
        )

    def _setup_dxl_handler(self) -> None:
        if self._handler_configured:
            return
        torque_enabled = self.dynamixel_handler.read_torque_enabled(self.all_ids)
        configure_persistent = not any(torque_enabled)
        voltages = self.dynamixel_handler.read_servo_voltages(self.all_ids)
        print("Voltages: ", voltages)
        for voltage in voltages:
            if voltage < 100:
                raise Exception(
                    f"Voltage level is too low ({voltage}). Recharge the batteries!"
                )
        # Persistent mode and indirect-address writes require torque to already
        # be off. Deployment never changes torque state to perform this setup.
        if configure_persistent:
            self.dynamixel_handler.set_zero_return_delay_time(self.all_ids)
            self.dynamixel_handler.write_indirect_addresses(self.all_ids)
            self.dynamixel_handler.set_position_mode(self.all_ids)
        self.dynamixel_handler.add_pos_vel_group_sync_read(self.all_ids)
        self.dynamixel_handler.set_kp(self.all_ids, 800)
        self.dynamixel_handler.set_duration_accel(
            self.all_ids,
            self.durations_ms,
            [self.durations_ms[i] for i in range(24)],
            configure_drive_mode=configure_persistent,
        )
        self._handler_configured = True

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def pos_dynamixel_to_isaac(self, dof_pos: list[int]) -> np.ndarray:
        """
        Converts joint angles on dynamixel format (ints 0-4095, 2048 is 0-position) to isaac format (-pi to pi, 0 is 0-position)
        :param dof_pos: list with 24 elements, one joint angle for each motor on dynamixel format
        :return: NumPy array of shape (24,), one joint angle per motor in Isaac order
        """
        position = np.asarray(dof_pos, dtype=np.float32).copy()
        position -= self.dynamixel_base_pos
        position = position[self.dynamixel_ids_to_isaac_ids]
        position = (position - 2048) / 2048 * np.pi
        return position * self.dynamixel_axis_mask

    def pos_isaac_to_dynamixel(self, dof_pos: Any) -> list[int]:
        """
        Converts joint angles on isaac format (-pi to pi, 0 is 0-position) to dynamixel format (ints 0-4095, 2048 is 0-position)
        :param dof_pos: array-like with 24 joint angles in Isaac order
        :return: List with 24 elements, one joint angle for each motor on dynamixel format
        """
        dof_pos = self._as_numpy(dof_pos).astype(np.float32, copy=False).reshape(-1)
        if dof_pos.size != 24:
            raise ValueError(f"Expected 24 joint positions, got {dof_pos.size}")
        if not np.isfinite(dof_pos).all():
            raise ValueError("Joint targets must all be finite")
        position = self.dynamixel_axis_mask * dof_pos
        position = position[self.isaac_ids_to_dynamixel_ids]
        position = (position / np.pi) * 2048 + 2048
        position += self.dynamixel_base_pos
        for index in range(len(position)):
            lower = self.lower_limits[index]
            upper = self.upper_limits[index]
            if position[index] < lower and lower - position[index] >= 100:
                raise ValueError(
                    f"Motor {self.all_ids[index]} target {position[index]:.1f} "
                    f"is below its lower limit {lower}"
                )
            if position[index] > upper and position[index] - upper >= 100:
                raise ValueError(
                    f"Motor {self.all_ids[index]} target {position[index]:.1f} "
                    f"is above its upper limit {upper}"
                )
        position = np.clip(position, self.lower_limits, self.upper_limits)
        return [int(position[i]) for i in range(len(position))]

    def vel_dynamixel_to_isaac(self, dof_vel: list[int]) -> np.ndarray:
        """
        Converts joint velocities from dynamixel format to isaac format
        :param dof_vel: list with 24 elements, one for each motor on dynamixel format
        :return: NumPy array of shape (24,), corresponding velocities in Isaac order
        """
        velocity = np.asarray(dof_vel, dtype=np.float32)
        velocity = velocity * 0.229 * (2 * np.pi / 60)
        velocity = velocity[self.dynamixel_ids_to_isaac_ids]
        return velocity * self.dynamixel_axis_mask

    def read_pos_vel(self) -> tuple[list[int], list[int]]:
        """
        Reads position and velocities from motors
        :return: tuple of lists, first element being a list of positions, and second element a list of velocities
        """
        pos, vel = self.dynamixel_handler.read_positions_and_velocities(self.all_ids)
        return pos, vel

    def read_current(self) -> list[int]:
        currents = self.dynamixel_handler.read_servo_currents(self.all_ids)
        return currents

    def enable_torques(self) -> None:
        self.dynamixel_handler.enable_torque(self.all_ids)

    def enable_torques_at_current_position(self) -> None:
        current_position = self.dynamixel_handler.read_servo_positions(self.all_ids)
        self.dynamixel_handler.move_servos(self.all_ids, current_position)
        self.dynamixel_handler.enable_torque(self.all_ids)

    def disable_torques(self) -> None:
        self.dynamixel_handler.disable_torque(self.all_ids)

    def move_servos(self, dof_pos: list[int], last_pos: list[int]) -> None:
        if len(dof_pos) != 24 or len(last_pos) != 24:
            raise ValueError("Expected 24 current and target Dynamixel positions")
        for curr_pos, new_pos in zip(last_pos, dof_pos, strict=True):
            if abs(curr_pos - new_pos) > 800:
                raise RuntimeError(
                    f"Servo tried to move too fast, from {curr_pos} to {new_pos}"
                )
        self.dynamixel_handler.move_servos(self.all_ids, dof_pos)

    def move_servos_smooth(self, dof_pos: list[int]) -> None:
        """
        Move servos to new positions with a set velocity
        :param dof_pos: list with 24 elements, one joint angle for each motor on dynamixel format
        """
        assert len(dof_pos) == 24
        current_pos = self.dynamixel_handler.read_servo_positions(self.all_ids)
        durations = [0] * 24
        for i, (curr_pos, new_pos) in enumerate(zip(current_pos, dof_pos, strict=True)):
            if abs(curr_pos - new_pos) > 1000:
                raise Exception(
                    f"Servo tried to move too fast, from {curr_pos} to {new_pos}"
                )
            durations[i] = abs(curr_pos - new_pos) // 2
        accel_durations = [dur // 3 for dur in durations]
        self.dynamixel_handler.set_duration_accel(
            self.all_ids, durations, accel_durations
        )
        self.dynamixel_handler.move_servos(self.all_ids, dof_pos)
        self.dynamixel_handler.set_duration_accel(self.all_ids, self.durations_ms)

    def move_servos_duration(self, dof_pos: list[int], duration: list[int]) -> None:
        if len(dof_pos) != 24 or len(duration) != 24:
            raise ValueError("Expected 24 target positions and durations")
        if any(value <= 0 for value in duration):
            raise ValueError("Servo move durations must be positive")
        current_pos = self.dynamixel_handler.read_servo_positions(self.all_ids)
        for i, (curr_pos, new_pos) in enumerate(zip(current_pos, dof_pos, strict=True)):
            if abs(curr_pos - new_pos) / duration[i] > 5:
                raise RuntimeError(
                    f"Servo tried to move too fast, from {curr_pos} to {new_pos}"
                )
        self.dynamixel_handler.set_duration_accel(self.all_ids, duration)
        self.dynamixel_handler.move_servos(self.all_ids, dof_pos)
        self.dynamixel_handler.set_duration_accel(self.all_ids, self.durations_ms)

    def close(self) -> None:
        if not self._closed:
            self.dynamixel_handler.close()
            self._closed = True
