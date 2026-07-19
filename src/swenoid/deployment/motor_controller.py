"""Low-level Protocol 2.0 Dynamixel transport for the 24 Swenoid joints."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

ADDR_BAUDRATE = 8
ADDR_RETURN_DELAY_TIME = 9
ADDR_DRIVE_MODE = 10
ADDR_OPERATING_MODE = 11
ADDR_CURRENT_LIMIT = 38
ADDR_UPPER_LIMIT = 48
ADDR_LOWER_LIMIT = 52
ADDR_TORQUE_ENABLE = 64
ADDR_POSITION_GAIN = 84
ADDR_GOAL_CURRENT = 102
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146
ADDR_INDIRECT_ADDRESS_1 = 168
ADDR_INDIRECT_DATA_1 = 224

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 4_000_000
PROTOCOL_VERSION = 2.0
SERIAL_READINESS_ERROR = "device reports readiness to read but returned no data"


def _is_serial_readiness_error(error: Exception) -> bool:
    """Return whether *error* is PySerial's transient empty-read failure."""
    try:
        serial_exception = import_module("serial").SerialException
    except (AttributeError, ModuleNotFoundError):
        return False
    return isinstance(error, serial_exception) and SERIAL_READINESS_ERROR in str(error)


def unsigned_to_signed(value: int, size: int) -> int:
    """Interpret an unsigned integer as a two's-complement signed value."""
    if size <= 0:
        raise ValueError("size must be positive")
    bit_size = 8 * size
    value = int(value) & ((1 << bit_size) - 1)
    if value & (1 << (bit_size - 1)):
        value -= 1 << bit_size
    return value


class DynamixelHandler:
    """Dynamixel SDK wrapper used by :class:`SwenoidControl`.

    The controller uses one indirect Fast Sync Read for 24 positions and
    velocities and one Group Sync Write for 24 goal positions. Hardware access
    begins only when this class is constructed, so importing the Swenoid package
    remains possible on development machines without the Dynamixel SDK.
    """

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUDRATE,
        *,
        configure_latency: bool = True,
        max_retries: int = 3,
        sdk: Any | None = None,
    ) -> None:
        if baudrate <= 0:
            raise ValueError("baudrate must be positive")
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")
        self.port = port
        self.baudrate = baudrate
        self.max_retries = max_retries
        self.sdk = sdk or self._load_sdk()
        if configure_latency:
            self._ensure_latency_timer_set()

        self.portHandler = self.sdk.PortHandler(port)
        self.packetHandler = self.sdk.PacketHandler(PROTOCOL_VERSION)
        if not self.portHandler.openPort():
            raise RuntimeError(f"Failed to open Dynamixel port {port}")
        if not self.portHandler.setBaudRate(baudrate):
            self.portHandler.closePort()
            raise RuntimeError(f"Failed to set {port} to {baudrate} baud")

        self.pos_vel_groupSyncRead = self.sdk.GroupSyncRead(
            self.portHandler,
            self.packetHandler,
            ADDR_INDIRECT_DATA_1,
            8,
        )
        self.goal_pos_groupSyncWrite = self.sdk.GroupSyncWrite(
            self.portHandler,
            self.packetHandler,
            ADDR_GOAL_POSITION,
            4,
        )
        self.group_sync_reads: dict[tuple[int, int], Any] = {}
        self._pos_vel_ids: set[int] = set()
        self._time_profile_ids: set[int] = set()
        self._closed = False

    @staticmethod
    def _load_sdk() -> Any:
        try:
            return import_module("dynamixel_sdk")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Physical deployment requires dynamixel-sdk. Install the "
                "Swenoid deployment extra."
            ) from exc

    @property
    def latency_timer_path(self) -> Path:
        # Resolve stable /dev/serial/by-id symlinks to the kernel tty name.
        device = Path(self.port).resolve().name
        return Path("/sys/bus/usb-serial/devices") / device / "latency_timer"

    def _ensure_latency_timer_set(self) -> None:
        path = self.latency_timer_path
        if not path.exists():
            warnings.warn(
                f"No USB serial latency timer found at {path}; continuing without "
                "low-latency configuration",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        if path.read_text().strip() == "1":
            return
        try:
            path.write_text("1\n")
        except PermissionError as exc:
            raise PermissionError(
                f"Set {path} to 1 before deployment, for example: "
                f"echo 1 | sudo tee {path}"
            ) from exc

    def _check_result(self, result: int, error: int, operation: str) -> None:
        if result != self.sdk.COMM_SUCCESS:
            raise RuntimeError(
                f"{operation} failed: {self.packetHandler.getTxRxResult(result)}"
            )
        if error:
            raise RuntimeError(
                f"{operation} returned a servo error: "
                f"{self.packetHandler.getRxPacketError(error)}"
            )

    def _write1(self, motor_id: int, address: int, value: int, operation: str) -> None:
        self._write(
            self.packetHandler.write1ByteTxRx,
            self.packetHandler.read1ByteTxRx,
            motor_id,
            address,
            int(value),
            operation,
        )

    def _write2(self, motor_id: int, address: int, value: int, operation: str) -> None:
        self._write(
            self.packetHandler.write2ByteTxRx,
            self.packetHandler.read2ByteTxRx,
            motor_id,
            address,
            int(value) & 0xFFFF,
            operation,
        )

    def _write4(self, motor_id: int, address: int, value: int, operation: str) -> None:
        self._write(
            self.packetHandler.write4ByteTxRx,
            self.packetHandler.read4ByteTxRx,
            motor_id,
            address,
            int(value) & 0xFFFFFFFF,
            operation,
        )

    def _write(
        self,
        writer: Any,
        reader: Any,
        motor_id: int,
        address: int,
        value: int,
        operation: str,
    ) -> None:
        context = f"{operation} for motor {motor_id} at address {address}"
        write_result = self.sdk.COMM_NOT_AVAILABLE
        for _ in range(self.max_retries):
            write_result, error = writer(self.portHandler, motor_id, address, value)
            if write_result == self.sdk.COMM_SUCCESS:
                self._check_result(write_result, error, context)
                return

        read_result = self.sdk.COMM_NOT_AVAILABLE
        for _ in range(self.max_retries):
            actual, read_result, error = reader(self.portHandler, motor_id, address)
            if read_result == self.sdk.COMM_SUCCESS:
                self._check_result(read_result, error, f"verify {context}")
                if int(actual) == value:
                    return
                raise RuntimeError(
                    f"Could not verify {context}: expected {value}, read {actual}"
                )
        raise RuntimeError(
            f"{context} failed after {self.max_retries} attempts "
            f"({self.packetHandler.getTxRxResult(write_result)}), and readback "
            f"failed after {self.max_retries} attempts "
            f"({self.packetHandler.getTxRxResult(read_result)})"
        )

    def _read2(self, motor_id: int, address: int, operation: str) -> int:
        value, result, error = self.packetHandler.read2ByteTxRx(
            self.portHandler, motor_id, address
        )
        self._check_result(result, error, operation)
        return int(value)

    def _read1(self, motor_id: int, address: int, operation: str) -> int:
        value, result, error = self.packetHandler.read1ByteTxRx(
            self.portHandler, motor_id, address
        )
        self._check_result(result, error, operation)
        return int(value)

    def _read4(self, motor_id: int, address: int, operation: str) -> int:
        result = self.sdk.COMM_NOT_AVAILABLE
        for _ in range(self.max_retries):
            value, result, error = self.packetHandler.read4ByteTxRx(
                self.portHandler, motor_id, address
            )
            if result == self.sdk.COMM_SUCCESS:
                self._check_result(result, error, operation)
                return int(value)
        raise RuntimeError(
            f"{operation} for motor {motor_id} at address {address} failed after "
            f"{self.max_retries} attempts: "
            f"{self.packetHandler.getTxRxResult(result)}"
        )

    def add_pos_vel_group_sync_read(self, ids: Sequence[int]) -> None:
        for motor_id in ids:
            if motor_id in self._pos_vel_ids:
                continue
            if not self.pos_vel_groupSyncRead.addParam(int(motor_id)):
                raise RuntimeError(
                    f"Could not add motor {motor_id} to position/velocity sync read"
                )
            self._pos_vel_ids.add(int(motor_id))

    def read_lower_limits(self, ids: Sequence[int]) -> list[int]:
        return [
            self._read4(motor_id, ADDR_LOWER_LIMIT, "read lower position limit")
            for motor_id in ids
        ]

    def read_upper_limits(self, ids: Sequence[int]) -> list[int]:
        return [
            self._read4(motor_id, ADDR_UPPER_LIMIT, "read upper position limit")
            for motor_id in ids
        ]

    def set_high_baudrate(self, ids: Sequence[int]) -> None:
        for motor_id in ids:
            self._write1(motor_id, ADDR_BAUDRATE, 6, "set baudrate")

    def set_zero_return_delay_time(self, ids: Sequence[int]) -> None:
        for motor_id in ids:
            self._write1(motor_id, ADDR_RETURN_DELAY_TIME, 0, "set return delay")

    def write_indirect_addresses(self, ids: Sequence[int]) -> None:
        for motor_id in ids:
            sources = [
                *(ADDR_PRESENT_POSITION + offset for offset in range(4)),
                *(ADDR_PRESENT_VELOCITY + offset for offset in range(4)),
            ]
            for offset, source in enumerate(sources):
                self._write2(
                    motor_id,
                    ADDR_INDIRECT_ADDRESS_1 + 2 * offset,
                    source,
                    "configure indirect feedback address",
                )

    def set_duration_accel(
        self,
        ids: Sequence[int],
        durations: Sequence[int],
        accel: Sequence[int] | None = None,
        *,
        configure_drive_mode: bool = True,
    ) -> None:
        if len(ids) != len(durations):
            raise ValueError("ids and durations must have the same length")
        acceleration = [0] * len(ids) if accel is None else list(accel)
        if len(acceleration) != len(ids):
            raise ValueError("ids and acceleration must have the same length")
        for index, motor_id in enumerate(ids):
            motor_id = int(motor_id)
            if not configure_drive_mode:
                self._time_profile_ids.add(motor_id)
            if motor_id not in self._time_profile_ids:
                # Drive Mode is EEPROM on X-series servos and may only be changed
                # while torque is disabled. SwenoidControl configures it once at
                # startup; later trajectory-duration changes only touch RAM.
                self._write1(motor_id, ADDR_DRIVE_MODE, 4, "set time-based profile")
                self._time_profile_ids.add(motor_id)
            self._write4(
                motor_id,
                ADDR_PROFILE_VELOCITY,
                durations[index],
                "set profile duration",
            )
            self._write4(
                motor_id,
                ADDR_PROFILE_ACCELERATION,
                acceleration[index],
                "set profile acceleration",
            )

    def set_position_mode(self, ids: Sequence[int]) -> None:
        for motor_id in ids:
            self._write1(motor_id, ADDR_OPERATING_MODE, 3, "set position mode")

    def set_current_mode(self, ids: Sequence[int]) -> None:
        for motor_id in ids:
            self._write1(motor_id, ADDR_OPERATING_MODE, 0, "set current mode")

    def set_kp(self, ids: Sequence[int], gain: int) -> None:
        for motor_id in ids:
            self._write2(motor_id, ADDR_POSITION_GAIN, gain, "set position gain")

    def read_kp(self, ids: Sequence[int]) -> list[int]:
        return [
            self._read2(motor_id, ADDR_POSITION_GAIN, "read position gain")
            for motor_id in ids
        ]

    def command_current(self, ids: Sequence[int], currents: Sequence[int]) -> None:
        if len(ids) != len(currents):
            raise ValueError("ids and currents must have the same length")
        for motor_id, current in zip(ids, currents, strict=True):
            self._write2(motor_id, ADDR_GOAL_CURRENT, current, "command current")

    def read_current(self, ids: Sequence[int]) -> list[int]:
        return [
            unsigned_to_signed(value, 2)
            for value in self.read_servos(ids, ADDR_PRESENT_CURRENT, 2)
        ]

    def read_goal_current(self, ids: Sequence[int]) -> list[int]:
        return [
            unsigned_to_signed(value, 2)
            for value in self.read_servos(ids, ADDR_GOAL_CURRENT, 2)
        ]

    def read_current_limit(self, ids: Sequence[int]) -> list[int]:
        return self.read_servos(ids, ADDR_CURRENT_LIMIT, 2)

    def disable_torque(self, ids: Sequence[int]) -> None:
        for motor_id in ids:
            self._write1(motor_id, ADDR_TORQUE_ENABLE, 0, "disable torque")

    def read_torque_enabled(self, ids: Sequence[int]) -> list[bool]:
        return [
            bool(self._read1(motor_id, ADDR_TORQUE_ENABLE, "read torque state"))
            for motor_id in ids
        ]

    def enable_torque(self, ids: Sequence[int]) -> None:
        for motor_id in ids:
            self._write1(motor_id, ADDR_TORQUE_ENABLE, 1, "enable torque")

    def move_servos(self, ids: Sequence[int], positions: Sequence[int]) -> None:
        if len(ids) != len(positions):
            raise ValueError("ids and positions must have the same length")
        try:
            for index, motor_id in enumerate(ids):
                value = int(positions[index]) & 0xFFFFFFFF
                payload = [
                    self.sdk.DXL_LOBYTE(self.sdk.DXL_LOWORD(value)),
                    self.sdk.DXL_HIBYTE(self.sdk.DXL_LOWORD(value)),
                    self.sdk.DXL_LOBYTE(self.sdk.DXL_HIWORD(value)),
                    self.sdk.DXL_HIBYTE(self.sdk.DXL_HIWORD(value)),
                ]
                if not self.goal_pos_groupSyncWrite.addParam(int(motor_id), payload):
                    raise RuntimeError(
                        f"Could not add motor {motor_id} to goal-position sync write"
                    )
            result = self.goal_pos_groupSyncWrite.txPacket()
            if result != self.sdk.COMM_SUCCESS:
                raise RuntimeError(
                    "Goal-position sync write failed: "
                    f"{self.packetHandler.getTxRxResult(result)}"
                )
        finally:
            self.goal_pos_groupSyncWrite.clearParam()

    def read_servos(self, ids: Sequence[int], address: int, byte_len: int) -> list[int]:
        key = (address, byte_len)
        if key not in self.group_sync_reads:
            self.group_sync_reads[key] = self.sdk.GroupSyncRead(
                self.portHandler, self.packetHandler, address, byte_len
            )
        reader = self.group_sync_reads[key]
        reader.clearParam()
        for motor_id in ids:
            if not reader.addParam(int(motor_id)):
                raise RuntimeError(f"Could not add motor {motor_id} to sync read")
        try:
            result = self.sdk.COMM_NOT_AVAILABLE
            for _ in range(self.max_retries):
                result = reader.txRxPacket()
                if result == self.sdk.COMM_SUCCESS:
                    break
            if result != self.sdk.COMM_SUCCESS:
                raise RuntimeError(
                    f"Sync read failed: {self.packetHandler.getTxRxResult(result)}"
                )
            values = []
            for motor_id in ids:
                if not reader.isAvailable(int(motor_id), address, byte_len):
                    raise RuntimeError(f"No sync-read data for motor {motor_id}")
                values.append(int(reader.getData(int(motor_id), address, byte_len)))
            return values
        finally:
            reader.clearParam()

    def read_positions_and_velocities(
        self, ids: Sequence[int]
    ) -> tuple[list[int], list[int]]:
        missing = set(map(int, ids)) - self._pos_vel_ids
        if missing:
            raise RuntimeError(
                f"Position/velocity sync read is not configured for IDs {sorted(missing)}"
            )
        result = self.sdk.COMM_NOT_AVAILABLE
        for attempt in range(self.max_retries):
            try:
                result = self.pos_vel_groupSyncRead.fastSyncRead()
            except Exception as exc:
                if (
                    not _is_serial_readiness_error(exc)
                    or attempt + 1 >= self.max_retries
                ):
                    raise
                continue
            if result == self.sdk.COMM_SUCCESS:
                break
        if result != self.sdk.COMM_SUCCESS:
            raise RuntimeError(
                "Fast position/velocity sync read failed: "
                f"{self.packetHandler.getTxRxResult(result)}"
            )

        positions, velocities = [], []
        for motor_id in ids:
            if not self.pos_vel_groupSyncRead.isAvailable(
                int(motor_id), ADDR_INDIRECT_DATA_1, 8
            ):
                raise RuntimeError(f"No indirect feedback data for motor {motor_id}")
            position = self.pos_vel_groupSyncRead.getData(
                int(motor_id), ADDR_INDIRECT_DATA_1, 4
            )
            velocity = self.pos_vel_groupSyncRead.getData(
                int(motor_id), ADDR_INDIRECT_DATA_1 + 4, 4
            )
            positions.append(int(position))
            velocities.append(unsigned_to_signed(int(velocity), 4))
        return positions, velocities

    def read_servo_positions(self, ids: Sequence[int]) -> list[int]:
        return self.read_servos(ids, ADDR_PRESENT_POSITION, 4)

    def read_servo_currents(self, ids: Sequence[int]) -> list[int]:
        return self.read_current(ids)

    def read_servo_velocities(self, ids: Sequence[int]) -> list[int]:
        return [
            unsigned_to_signed(value, 4)
            for value in self.read_servos(ids, ADDR_PRESENT_VELOCITY, 4)
        ]

    def read_servo_voltages(self, ids: Sequence[int]) -> list[int]:
        return self.read_servos(ids, ADDR_PRESENT_VOLTAGE, 2)

    def read_servo_telemetry(
        self, ids: Sequence[int]
    ) -> tuple[list[int], list[int], list[int]]:
        """Read current, voltage, and temperature in one Group Sync Read.

        The X-series present-state registers occupy one contiguous block from
        present current (126) through present temperature (146). Keeping this
        separate from the 8-byte indirect position/velocity read avoids a
        torque-off migration of existing indirect-address mappings.
        """
        start = ADDR_PRESENT_CURRENT
        byte_len = ADDR_PRESENT_TEMPERATURE - start + 1
        key = (start, byte_len)
        if key not in self.group_sync_reads:
            self.group_sync_reads[key] = self.sdk.GroupSyncRead(
                self.portHandler, self.packetHandler, start, byte_len
            )
        reader = self.group_sync_reads[key]
        reader.clearParam()
        for motor_id in ids:
            if not reader.addParam(int(motor_id)):
                raise RuntimeError(
                    f"Could not add motor {motor_id} to telemetry sync read"
                )
        try:
            result = self.sdk.COMM_NOT_AVAILABLE
            for _ in range(self.max_retries):
                result = reader.txRxPacket()
                if result == self.sdk.COMM_SUCCESS:
                    break
            if result != self.sdk.COMM_SUCCESS:
                raise RuntimeError(
                    "Servo telemetry sync read failed: "
                    f"{self.packetHandler.getTxRxResult(result)}"
                )
            currents, voltages, temperatures = [], [], []
            for motor_id in ids:
                if not reader.isAvailable(int(motor_id), start, byte_len):
                    raise RuntimeError(
                        f"No telemetry sync-read data for motor {motor_id}"
                    )
                current = reader.getData(int(motor_id), ADDR_PRESENT_CURRENT, 2)
                voltage = reader.getData(int(motor_id), ADDR_PRESENT_VOLTAGE, 2)
                temperature = reader.getData(int(motor_id), ADDR_PRESENT_TEMPERATURE, 1)
                currents.append(unsigned_to_signed(int(current), 2))
                voltages.append(int(voltage))
                temperatures.append(int(temperature))
            return currents, voltages, temperatures
        finally:
            reader.clearParam()

    def disable_torques(self, ids: Sequence[int]) -> None:
        self.disable_torque(ids)

    def reboot_servos(self, ids: Sequence[int]) -> None:
        for motor_id in ids:
            result, error = self.packetHandler.reboot(self.portHandler, motor_id)
            self._check_result(result, error, "reboot servo")
            self._time_profile_ids.discard(int(motor_id))

    def close(self) -> None:
        if not self._closed:
            self.portHandler.closePort()
            self._closed = True
