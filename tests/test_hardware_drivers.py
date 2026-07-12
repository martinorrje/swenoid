"""Hardware-driver tests using protocol fakes rather than connected devices."""

import numpy as np
import pytest

from swenoid.deployment.bno085 import BNO085, rot_ang_vel, rot_gravity
from swenoid.deployment.motor_controller import (
    ADDR_DRIVE_MODE,
    ADDR_INDIRECT_DATA_1,
    DynamixelHandler,
    unsigned_to_signed,
)


class _FakePortHandler:
    def __init__(self, port):
        self.port = port
        self.closed = False

    def openPort(self):
        return True

    def setBaudRate(self, baudrate):
        self.baudrate = baudrate
        return True

    def closePort(self):
        self.closed = True


class _FakePacketHandler:
    def __init__(self, protocol):
        self.protocol = protocol
        self.write1_calls = []

    def write1ByteTxRx(self, _port, motor_id, address, value):
        self.write1_calls.append((motor_id, address, value))
        return 0, 0

    def write2ByteTxRx(self, *_):
        return 0, 0

    def write4ByteTxRx(self, *_):
        return 0, 0

    def read2ByteTxRx(self, _port, motor_id, address):
        return motor_id * 100 + address, 0, 0

    def read4ByteTxRx(self, _port, motor_id, address):
        return motor_id * 100 + address, 0, 0

    def reboot(self, *_):
        return 0, 0

    def getTxRxResult(self, result):
        return f"comm-{result}"

    def getRxPacketError(self, error):
        return f"servo-{error}"


class _FakeGroupSyncRead:
    def __init__(self, _port, _packet, address, byte_len):
        self.address = address
        self.byte_len = byte_len
        self.ids = set()

    def addParam(self, motor_id):
        self.ids.add(motor_id)
        return True

    def clearParam(self):
        self.ids.clear()

    def txRxPacket(self):
        return 0

    def fastSyncRead(self):
        return 0

    def isAvailable(self, motor_id, _address, _byte_len):
        return motor_id in self.ids

    def getData(self, motor_id, address, _byte_len):
        if self.address == ADDR_INDIRECT_DATA_1:
            if address == ADDR_INDIRECT_DATA_1:
                return 2000 + motor_id
            return 0xFFFFFFFF
        return motor_id * 1000 + address


class _FakeGroupSyncWrite:
    def __init__(self, *_):
        self.payloads = {}

    def addParam(self, motor_id, payload):
        self.payloads[motor_id] = payload
        return True

    def txPacket(self):
        return 0

    def clearParam(self):
        self.payloads.clear()


class _FakeSdk:
    COMM_SUCCESS = 0
    COMM_NOT_AVAILABLE = -1
    PortHandler = _FakePortHandler
    PacketHandler = _FakePacketHandler
    GroupSyncRead = _FakeGroupSyncRead
    GroupSyncWrite = _FakeGroupSyncWrite

    @staticmethod
    def DXL_LOBYTE(value):
        return value & 0xFF

    @staticmethod
    def DXL_HIBYTE(value):
        return (value >> 8) & 0xFF

    @staticmethod
    def DXL_LOWORD(value):
        return value & 0xFFFF

    @staticmethod
    def DXL_HIWORD(value):
        return (value >> 16) & 0xFFFF


class _FakeBnoSensor:
    gyro = (1.0, 2.0, 3.0)
    gravity = (9.80665, 0.0, 0.0)
    quaternion = (0.0, 0.0, 0.0, 1.0)


def test_signed_dynamixel_conversion() -> None:
    assert unsigned_to_signed(0xFFFF, 2) == -1
    assert unsigned_to_signed(0xFFFFFFFF, 4) == -1
    assert unsigned_to_signed(42, 2) == 42


def test_dynamixel_group_read_and_write_without_hardware() -> None:
    handler = DynamixelHandler(
        port="/dev/fake",
        baudrate=4_000_000,
        configure_latency=False,
        sdk=_FakeSdk,
    )
    handler.add_pos_vel_group_sync_read([1, 2])
    positions, velocities = handler.read_positions_and_velocities([1, 2])
    assert positions == [2001, 2002]
    assert velocities == [-1, -1]
    assert handler.read_lower_limits([1, 2]) == [152, 252]
    handler.move_servos([1, 2], [2048, 2049])
    handler.set_duration_accel([1, 2], [100, 100])
    handler.set_duration_accel([1, 2], [200, 200])
    drive_mode_writes = [
        call
        for call in handler.packetHandler.write1_calls
        if call[1] == ADDR_DRIVE_MODE
    ]
    assert drive_mode_writes == [(1, ADDR_DRIVE_MODE, 4), (2, ADDR_DRIVE_MODE, 4)]
    handler.close()
    assert handler.portHandler.closed


def test_bno085_frame_conversion_without_i2c() -> None:
    assert rot_ang_vel((1.0, 2.0, 3.0)) == (-3.0, 1.0, -2.0)
    assert rot_gravity((1.0, 2.0, 3.0)) == (3.0, -1.0, 2.0)
    sensor = BNO085(sensor=_FakeBnoSensor())
    assert sensor.get_ang_vel() == (-3.0, 1.0, -2.0)
    assert sensor.get_gravity() == pytest.approx((0.0, -1.0, 0.0))
    assert np.linalg.norm(sensor.get_orientation()) == pytest.approx(1.0)
