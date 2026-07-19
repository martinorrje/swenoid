"""Contract tests for hardware experiment telemetry and command provenance."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from swenoid.deployment.hardware_config import HardwareConfig
from swenoid.deployment.motor_controller import (
    ADDR_PRESENT_CURRENT,
    ADDR_PRESENT_TEMPERATURE,
    ADDR_PRESENT_VOLTAGE,
    DynamixelHandler,
)
from swenoid.deployment.swenoid_control import SwenoidControl


class _PortHandler:
    def __init__(self, port: str) -> None:
        self.port = port

    def openPort(self) -> bool:
        return True

    def setBaudRate(self, baudrate: int) -> bool:
        self.baudrate = baudrate
        return True

    def closePort(self) -> None:
        pass


class _PacketHandler:
    def __init__(self, protocol: float) -> None:
        self.protocol = protocol

    @staticmethod
    def getTxRxResult(result: int) -> str:
        return f"comm-{result}"


class _GroupSyncRead:
    instances: ClassVar[list[_GroupSyncRead]] = []

    def __init__(
        self,
        _port: _PortHandler,
        _packet: _PacketHandler,
        address: int,
        byte_len: int,
    ) -> None:
        self.address = address
        self.byte_len = byte_len
        self.ids: set[int] = set()
        self.tx_count = 0
        self.get_data_calls: list[tuple[int, int, int]] = []
        self.instances.append(self)

    def addParam(self, motor_id: int) -> bool:
        self.ids.add(motor_id)
        return True

    def clearParam(self) -> None:
        self.ids.clear()

    def txRxPacket(self) -> int:
        self.tx_count += 1
        return 0

    def isAvailable(self, motor_id: int, address: int, byte_len: int) -> bool:
        return (
            motor_id in self.ids
            and address == self.address
            and byte_len == self.byte_len
        )

    def getData(self, motor_id: int, address: int, byte_len: int) -> int:
        self.get_data_calls.append((motor_id, address, byte_len))
        if address == ADDR_PRESENT_CURRENT:
            return 0xFFFE if motor_id == 1 else 25
        if address == ADDR_PRESENT_VOLTAGE:
            return 120 + motor_id
        if address == ADDR_PRESENT_TEMPERATURE:
            return 30 + motor_id
        raise AssertionError(f"Unexpected telemetry address {address}")


class _GroupSyncWrite:
    def __init__(self, *_args: object) -> None:
        pass


class _Sdk:
    COMM_SUCCESS = 0
    COMM_NOT_AVAILABLE = -1
    PortHandler = _PortHandler
    PacketHandler = _PacketHandler
    GroupSyncRead = _GroupSyncRead
    GroupSyncWrite = _GroupSyncWrite


class _ControlHandler:
    def __init__(self) -> None:
        self.position_gain: int | None = None
        self.startup_voltage = [121 + index for index in range(24)]

    @staticmethod
    def read_torque_enabled(ids: list[int]) -> list[bool]:
        return [True] * len(ids)

    def read_servo_voltages(self, ids: list[int]) -> list[int]:
        assert len(ids) == 24
        return self.startup_voltage.copy()

    @staticmethod
    def add_pos_vel_group_sync_read(_ids: list[int]) -> None:
        pass

    def set_kp(self, ids: list[int], gain: int) -> None:
        assert len(ids) == 24
        self.position_gain = gain

    @staticmethod
    def set_duration_accel(
        _ids: list[int],
        _durations: list[int],
        _acceleration: list[int],
        *,
        configure_drive_mode: bool,
    ) -> None:
        assert configure_drive_mode is False

    @staticmethod
    def read_lower_limits(ids: list[int]) -> list[int]:
        return [0] * len(ids)

    @staticmethod
    def read_upper_limits(ids: list[int]) -> list[int]:
        return [4095] * len(ids)

    @staticmethod
    def close() -> None:
        pass


def test_servo_telemetry_uses_one_contiguous_sync_read() -> None:
    _GroupSyncRead.instances.clear()
    handler = DynamixelHandler(
        port="/dev/fake",
        baudrate=4_000_000,
        configure_latency=False,
        sdk=_Sdk,
    )

    currents, voltages, temperatures = handler.read_servo_telemetry([1, 2])

    telemetry_readers = [
        reader
        for reader in _GroupSyncRead.instances
        if reader.address == ADDR_PRESENT_CURRENT
    ]
    assert len(telemetry_readers) == 1
    reader = telemetry_readers[0]
    assert reader.byte_len == ADDR_PRESENT_TEMPERATURE - ADDR_PRESENT_CURRENT + 1
    assert reader.tx_count == 1
    assert reader.get_data_calls == [
        (1, ADDR_PRESENT_CURRENT, 2),
        (1, ADDR_PRESENT_VOLTAGE, 2),
        (1, ADDR_PRESENT_TEMPERATURE, 1),
        (2, ADDR_PRESENT_CURRENT, 2),
        (2, ADDR_PRESENT_VOLTAGE, 2),
        (2, ADDR_PRESENT_TEMPERATURE, 1),
    ]
    assert currents == [-2, 25]
    assert voltages == [121, 122]
    assert temperatures == [31, 32]


def test_position_command_and_recording_metadata_expose_applied_hardware_state() -> (
    None
):
    hardware = HardwareConfig.load()
    handler = _ControlHandler()
    control = SwenoidControl(config=hardware, dynamixel_handler=handler)
    requested = np.zeros(24, dtype=np.float32)
    requested[0] = 10.0

    command = control.encode_position_command(requested)

    motor_index = control.dynamixel_dof_names.index(control.isaac_dof_names[0])
    assert command.clipped.dtype == np.bool_
    assert command.clipped.shape == (24,)
    assert command.clipped[motor_index]
    assert command.dynamixel_goal[motor_index] in (0, 4095)
    np.testing.assert_allclose(
        command.sent_position_rad,
        control.pos_dynamixel_to_isaac(command.dynamixel_goal),
        rtol=0.0,
        atol=0.0,
    )
    assert not np.isclose(command.sent_position_rad[0], requested[0])

    metadata = control.recording_metadata()
    assert metadata["hardware_config"] == hardware.to_dict()
    assert metadata["simulation_joint_names"] == list(hardware.simulation_joint_names)
    assert metadata["dynamixel_joint_names"] == list(hardware.dynamixel_joint_names)
    assert metadata["motor_ids"] == list(hardware.motor_ids)
    assert metadata["position_lower_limit_raw"] == [0] * 24
    assert metadata["position_upper_limit_raw"] == [4095] * 24
    assert metadata["startup_voltage_raw"] == handler.startup_voltage
    assert metadata["position_p_gain"] == 800
    assert handler.position_gain == 800
