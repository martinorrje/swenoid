"""Hardware-free integration tests for real-experiment deployment records."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

import swenoid.deployment.real as real
import swenoid.deployment.recording as recording
from swenoid.deployment.recording import ExperimentRecorder
from swenoid.deployment.swenoid_control import PositionCommand


def _record_args(path: Path, *extra: str) -> real.argparse.Namespace:
    return real.parse_args(
        [
            "--onnx",
            "policy.onnx",
            "--record",
            str(path),
            "--trial-id",
            "velocity-nominal-001",
            "--condition",
            "nominal",
            "--robot-id",
            "swenoid-01",
            "--trial-steps",
            "3",
            *extra,
        ]
    )


def _metadata() -> dict[str, object]:
    return {
        "trial": {
            "trial_id": "velocity-nominal-001",
            "condition": "nominal",
            "robot_id": "swenoid-01",
            "experiment_context": {
                "external_tracking": {"estimated_clock_offset_ms": 12.5}
            },
        },
        "policy": {"policy_kind": "velocity"},
    }


def test_recording_args_require_identifiers_and_forbid_loop(tmp_path: Path) -> None:
    missing = real.parse_args(
        [
            "--onnx",
            "policy.onnx",
            "--record",
            str(tmp_path / "missing.npz"),
            "--trial-steps",
            "3",
        ]
    )
    with pytest.raises(
        SystemExit,
        match=r"--trial-id, --condition, --robot-id",
    ):
        real._validate_args(missing)

    looped = _record_args(tmp_path / "looped.npz", "--loop")
    with pytest.raises(SystemExit, match="cannot use --loop"):
        real._validate_args(looped)

    valid = _record_args(tmp_path / "valid.npz")
    real._validate_args(valid)


def test_recording_metadata_options_require_record() -> None:
    args = real.parse_args(["--onnx", "policy.onnx", "--trial-id", "unrecorded-trial"])

    with pytest.raises(SystemExit, match="require --record"):
        real._validate_args(args)


def test_recorded_velocity_main_requires_trial_horizon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy = SimpleNamespace(
        policy_kind="velocity",
        is_general_motion=False,
        is_tracking=False,
    )
    monkeypatch.setattr(real, "resolve_policy_path", lambda **_: Path("policy.onnx"))
    monkeypatch.setattr(real, "OnnxPolicy", lambda _: policy)

    with pytest.raises(SystemExit, match="require --trial-steps"):
        real.main(
            [
                "--onnx",
                "policy.onnx",
                "--record",
                str(tmp_path / "trial.npz"),
                "--trial-id",
                "velocity-001",
                "--condition",
                "nominal",
                "--robot-id",
                "swenoid-01",
            ]
        )


class _VelocityPolicy:
    is_tracking = False
    motion_length = None
    observation_names = (
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
    )
    observation_size = 81
    action_scale = np.full(24, 0.5, dtype=np.float32)
    default_joint_pos = np.linspace(-0.2, 0.2, 24, dtype=np.float32)

    @staticmethod
    def reference_at(_time_step: int) -> None:
        return None

    @staticmethod
    def infer(_observations: np.ndarray, time_step: int) -> list[np.ndarray]:
        action = np.full((1, 24), 0.1 * (time_step + 1), dtype=np.float32)
        return [action]


class _FakeImu:
    def __init__(self) -> None:
        self.sequence = 0
        self.stop = SimpleNamespace(set=lambda: None)
        self.thread = SimpleNamespace(join=lambda timeout: None)

    def snapshot(self) -> real.ImuSnapshot:
        index = self.sequence
        self.sequence += 1
        raw_angular_velocity = np.asarray(
            [index + 1.0, index + 2.0, index + 3.0], dtype=np.float32
        )
        gravity = np.asarray([index * 0.01, 0.0, -1.0], dtype=np.float32)
        return real.ImuSnapshot(
            angular_velocity_raw=raw_angular_velocity,
            angular_velocity=raw_angular_velocity * 0.5,
            projected_gravity_raw=gravity,
            projected_gravity=gravity,
            orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            monotonic_ns=time.perf_counter_ns() - 500_000,
            sequence=index,
        )


class _FakeControl:
    def __init__(self) -> None:
        self.read_count = 0
        self.last_read_index = -1
        self.moves: list[tuple[np.ndarray, np.ndarray]] = []
        self.last_move_return_ns: int | None = None
        self.last_move_return_unix_ns: int | None = None
        self.disabled = False
        self.closed = False

    def read_pos_vel(self) -> tuple[list[int], list[int]]:
        index = self.read_count
        self.read_count += 1
        self.last_read_index = index
        position = np.arange(24, dtype=np.int32) + 2000 + index
        velocity = np.arange(24, dtype=np.int32) - 12 + index
        return position.tolist(), velocity.tolist()

    @staticmethod
    def pos_dynamixel_to_isaac(position: list[int]) -> np.ndarray:
        return (np.asarray(position, dtype=np.float32) - 2000.0) * 0.01

    @staticmethod
    def vel_dynamixel_to_isaac(velocity: list[int]) -> np.ndarray:
        return np.asarray(velocity, dtype=np.float32) * 0.02

    @staticmethod
    def encode_position_command(target: np.ndarray) -> PositionCommand:
        requested = np.asarray(target, dtype=np.float32)
        goal = np.rint(requested * 1000.0).astype(np.int32) + 2048
        return PositionCommand(
            dynamixel_goal=goal.tolist(),
            sent_position_rad=requested.copy(),
            clipped=np.zeros(24, dtype=np.bool_),
        )

    def move_servos(self, goal: list[int], position: list[int]) -> None:
        self.moves.append(
            (np.asarray(goal, dtype=np.int32), np.asarray(position, dtype=np.int32))
        )
        self.last_move_return_ns = time.perf_counter_ns()
        self.last_move_return_unix_ns = time.time_ns()

    def disable_torques(self) -> None:
        self.disabled = True

    def close(self) -> None:
        self.closed = True

    def read_servo_telemetry(self) -> tuple[list[int], list[int], list[int]]:
        index = self.last_read_index
        current = np.arange(24, dtype=np.int16) + index
        voltage = np.full(24, 120 + index, dtype=np.uint16)
        temperature = np.full(24, 30 + index, dtype=np.uint8)
        return current.tolist(), voltage.tolist(), temperature.tolist()


def _fake_runner(
    *,
    args: real.argparse.Namespace,
    recorder: ExperimentRecorder,
) -> tuple[real.RealRobotRunner, _FakeControl]:
    runner = object.__new__(real.RealRobotRunner)
    control = _FakeControl()
    runner.policy = _VelocityPolicy()  # pyright: ignore[reportAttributeAccessIssue]
    runner.args = args
    runner.recorder = recorder
    runner.requested_command = np.asarray(args.command, dtype=np.float32)
    runner.last_actions = np.zeros((1, 24), dtype=np.float32)
    runner.time_step = 0
    runner.completed_steps = 0
    runner._last_cycle_start_ns = None
    runner._last_recorded_command = None
    runner._last_goal_write_completion_ns = None
    runner._last_goal_write_completion_unix_ns = None
    runner.imu = _FakeImu()  # pyright: ignore[reportAttributeAccessIssue]
    runner.control = control  # pyright: ignore[reportAttributeAccessIssue]
    runner.external_pose = None
    return runner, control


def test_short_velocity_trial_records_aligned_paper_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "velocity-trial.npz"
    args = _record_args(
        path,
        "--command",
        "0.4",
        "-0.1",
        "0.2",
        "--command-start-step",
        "1",
        "--velocity-steps",
        "1",
        "--telemetry-every",
        "2",
    )
    real._validate_args(args)
    recorder = ExperimentRecorder(
        path,
        metadata=_metadata(),
        observation_size=_VelocityPolicy.observation_size,
        command_size=3,
        checkpoint_steps=50,
    )
    runner, control = _fake_runner(args=args, recorder=recorder)
    monkeypatch.setattr(real.RealRobotRunner, "move_to_start", lambda _self: None)
    monkeypatch.setattr(real.time, "sleep", lambda _duration: None)

    termination_reason = runner.run()
    recorder.finalize(termination_reason=termination_reason)

    assert termination_reason == "trial_steps_reached"
    assert runner.completed_steps == 3
    assert len(control.moves) == 3
    with np.load(path, allow_pickle=False) as archive:
        assert set(recording._STEP_SPECS) <= set(archive.files)
        assert set(recording._TELEMETRY_SPECS) <= set(archive.files)
        np.testing.assert_array_equal(archive["sample_index"], [0, 1, 2])
        np.testing.assert_array_equal(archive["policy_frame"], [0, 1, 2])

        expected_command = np.asarray(
            [[0.0, 0.0, 0.0], [0.4, -0.1, 0.2], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(archive["command"], expected_command)
        np.testing.assert_allclose(
            archive["policy_observation"][:, -3:], expected_command
        )

        expected_action = np.stack(
            [np.full(24, value, dtype=np.float32) for value in (0.1, 0.2, 0.3)]
        )
        np.testing.assert_allclose(archive["policy_action"], expected_action)
        np.testing.assert_allclose(
            archive["policy_observation"][0, 54:78], np.zeros(24)
        )
        np.testing.assert_allclose(
            archive["policy_observation"][1:, 54:78], expected_action[:-1]
        )
        assert np.isnan(archive["reference_joint_position_rad"]).all()
        assert np.isnan(archive["reference_joint_velocity_rad_s"]).all()

        expected_position_raw = np.stack(
            [np.arange(24, dtype=np.int32) + 2000 + index for index in range(3)]
        )
        expected_velocity_raw = np.stack(
            [np.arange(24, dtype=np.int32) - 12 + index for index in range(3)]
        )
        np.testing.assert_array_equal(
            archive["dynamixel_position_raw"], expected_position_raw
        )
        np.testing.assert_array_equal(
            archive["dynamixel_velocity_raw"], expected_velocity_raw
        )
        np.testing.assert_allclose(
            archive["joint_position_rad"],
            (expected_position_raw.astype(np.float32) - 2000.0) * 0.01,
        )
        np.testing.assert_allclose(
            archive["joint_velocity_rad_s"],
            expected_velocity_raw.astype(np.float32) * 0.02,
        )

        expected_target = (
            expected_action * _VelocityPolicy.action_scale
            + _VelocityPolicy.default_joint_pos
        )
        expected_goal = np.rint(expected_target * 1000.0).astype(np.int32) + 2048
        np.testing.assert_allclose(
            archive["joint_target_requested_rad"], expected_target
        )
        np.testing.assert_allclose(archive["joint_target_sent_rad"], expected_target)
        np.testing.assert_array_equal(
            archive["dynamixel_goal_position_raw"], expected_goal
        )
        assert not archive["dynamixel_goal_clipped"].any()

        np.testing.assert_array_equal(archive["servo_telemetry_sample_index"], [0, 2])
        expected_current = np.stack(
            [np.arange(24, dtype=np.int16) + index for index in (0, 2)]
        )
        np.testing.assert_array_equal(archive["servo_current_raw"], expected_current)
        np.testing.assert_allclose(
            archive["servo_current_a"],
            expected_current.astype(np.float32) * recording.CURRENT_AMPERE_PER_RAW_UNIT,
        )
        np.testing.assert_allclose(
            archive["servo_voltage_v"], [[12.0] * 24, [12.2] * 24]
        )
        np.testing.assert_allclose(
            archive["servo_temperature_c"], [[30.0] * 24, [32.0] * 24]
        )

        assert archive["time_s"].shape == (3,)
        assert np.all(np.diff(archive["time_s"]) > 0.0)
        assert np.isnan(archive["control_dt_s"][0])
        assert np.isfinite(archive["control_dt_s"][1:]).all()
        for field in (
            "read_duration_s",
            "inference_duration_s",
            "write_duration_s",
            "cycle_duration_s",
            "imu_sample_time_s",
            "imu_age_s",
            "servo_telemetry_time_s",
        ):
            assert np.isfinite(archive[field]).all(), field
        assert (archive["imu_age_s"] >= 0.0).all()
        np.testing.assert_array_equal(archive["imu_sequence"], [0, 1, 2])

    events = [
        json.loads(line)
        for line in recorder.paths.events.read_text(encoding="utf-8").splitlines()
    ]
    changes = [event for event in events if event["name"] == "velocity_command_changed"]
    assert [event["sample_index"] for event in changes] == [0, 1, 2]
    assert [event["detail"]["active"] for event in changes] == [False, True, False]
    assert [event["detail"]["command"] for event in changes] == [
        [0.0, 0.0, 0.0],
        pytest.approx([0.4, -0.1, 0.2]),
        [0.0, 0.0, 0.0],
    ]


class _JoinedThread:
    def join(self, timeout: float) -> None:
        assert timeout == 1.0


class _StoppedImu:
    def __init__(self) -> None:
        self.stop = SimpleNamespace(set=lambda: None)
        self.thread = _JoinedThread()


class _ClosingControl:
    def __init__(self) -> None:
        self.disabled = False
        self.closed = False

    def disable_torques(self) -> None:
        self.disabled = True

    def close(self) -> None:
        self.closed = True


def test_controller_error_finalizes_empty_record_without_disabling_torque(
    tmp_path: Path,
) -> None:
    path = tmp_path / "failed.npz"
    recorder = ExperimentRecorder(
        path,
        metadata=_metadata(),
        observation_size=81,
        command_size=3,
    )
    recorder.start()
    runner = object.__new__(real.RealRobotRunner)
    control = _ClosingControl()
    runner.args = SimpleNamespace(disable_torque_on_exit=True)  # pyright: ignore[reportAttributeAccessIssue]
    runner.recorder = recorder
    runner.external_pose = None
    runner.imu = _StoppedImu()  # pyright: ignore[reportAttributeAccessIssue]
    runner.control = control  # pyright: ignore[reportAttributeAccessIssue]
    runner.run = lambda: (_ for _ in ()).throw(RuntimeError("serial failure"))

    with pytest.raises(RuntimeError, match="serial failure"):
        runner.execute()

    assert not control.disabled
    assert control.closed
    manifest = json.loads(recorder.paths.manifest.read_text(encoding="utf-8"))
    assert manifest["recording"]["status"] == "finalized"
    assert manifest["recording"]["termination_reason"] == "controller_error"
    assert manifest["recording"]["sample_count"] == 0
    assert manifest["recording"]["error"] == {
        "type": "RuntimeError",
        "message": "serial failure",
    }


class _FakeExternalPoseReceiver:
    instances: ClassVar[list[_FakeExternalPoseReceiver]] = []
    baseline_ready = True
    endpoint_ready = True
    fatal_error: BaseException | None = None

    def __init__(
        self,
        *,
        bind: str,
        port: int,
        on_packet: object,
        max_clock_offset_s: float,
    ) -> None:
        self.bind = bind
        self.port = port
        self.on_packet = on_packet
        self.max_clock_offset_s = max_clock_offset_s
        self.started = False
        self.stopped = False
        self.baseline_waits: list[tuple[int, float]] = []
        self.endpoint_waits: list[tuple[int, float]] = []
        self.endpoint_source_waits: list[tuple[int, float, int | None]] = []
        self.failure_checks = 0
        type(self).instances.append(self)

    @classmethod
    def configure(
        cls,
        *,
        baseline_ready: bool = True,
        endpoint_ready: bool = True,
        fatal_error: BaseException | None = None,
    ) -> None:
        cls.instances = []
        cls.baseline_ready = baseline_ready
        cls.endpoint_ready = endpoint_ready
        cls.fatal_error = fatal_error

    def start(self) -> None:
        self.started = True

    def wait_for_valid_samples(self, minimum: int, timeout_s: float) -> bool:
        self.baseline_waits.append((minimum, timeout_s))
        return type(self).baseline_ready

    def wait_for_valid_after(self, monotonic_ns: int, timeout_s: float) -> bool:
        self.endpoint_waits.append((monotonic_ns, timeout_s))
        return True

    def wait_for_valid_source_after(
        self,
        source_time_ns: int,
        timeout_s: float,
        *,
        receive_monotonic_ns: int | None = None,
    ) -> bool:
        self.endpoint_source_waits.append(
            (source_time_ns, timeout_s, receive_monotonic_ns)
        )
        return type(self).endpoint_ready

    def raise_if_failed(self) -> None:
        self.failure_checks += 1
        fatal_error = type(self).fatal_error
        if fatal_error is not None:
            raise fatal_error

    def stop(self) -> None:
        self.stopped = True

    def metadata(self) -> dict[str, object]:
        return {
            "transport": "udp_json",
            "bind": self.bind,
            "port": self.port,
            "max_clock_offset_s": self.max_clock_offset_s,
        }


def test_external_pose_options_are_validated_and_recorded(
    tmp_path: Path,
) -> None:
    args = _record_args(
        tmp_path / "tracked.npz",
        "--external-pose-port",
        "15555",
        "--external-pose-bind",
        "127.0.0.1",
        "--external-pose-baseline-samples",
        "4",
        "--external-pose-timeout-s",
        "0.75",
        "--external-pose-max-clock-offset-ms",
        "25",
    )
    real._validate_args(args)
    policy_path = tmp_path / "policy.onnx"
    policy_path.write_bytes(b"test policy")
    policy = SimpleNamespace(
        metadata={},
        policy_kind="velocity",
        motion_length=None,
        observation_size=81,
        observation_names=("command",),
        action_scale=np.ones(24, dtype=np.float32),
        default_joint_pos=np.zeros(24, dtype=np.float32),
    )

    metadata = real.build_manifest_metadata(policy, policy_path, args)  # pyright: ignore[reportArgumentType]

    assert metadata["control"]["external_pose"] == {
        "transport": "udp_json",
        "bind": "127.0.0.1",
        "port": 15555,
        "baseline_samples": 4,
        "wait_timeout_s": 0.75,
        "max_clock_offset_ms": 25.0,
    }


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        (
            "--external-pose-baseline-samples",
            "0",
            "baseline-samples must be positive",
        ),
        ("--external-pose-timeout-s", "nan", "timeout-s must be finite"),
        (
            "--external-pose-max-clock-offset-ms",
            "0",
            "clock-offset-ms must be finite and positive",
        ),
    ],
)
def test_external_pose_options_reject_invalid_values(
    tmp_path: Path,
    option: str,
    value: str,
    message: str,
) -> None:
    args = _record_args(tmp_path / "invalid.npz", option, value)

    with pytest.raises(SystemExit, match=message):
        real._validate_args(args)


def test_external_pose_baseline_precedes_control_and_endpoint_covers_last_goal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _FakeExternalPoseReceiver.configure()
    monkeypatch.setattr(real, "ExternalPoseReceiver", _FakeExternalPoseReceiver)
    monkeypatch.setattr(real.RealRobotRunner, "move_to_start", lambda _self: None)
    monkeypatch.setattr(real.time, "sleep", lambda _duration: None)
    path = tmp_path / "tracked-success.npz"
    args = _record_args(
        path,
        "--external-pose-port",
        "15555",
        "--external-pose-baseline-samples",
        "3",
        "--external-pose-timeout-s",
        "0.25",
        "--external-pose-max-clock-offset-ms",
        "40",
    )
    recorder = ExperimentRecorder(
        path,
        metadata=_metadata(),
        observation_size=81,
        command_size=3,
    )
    runner, control = _fake_runner(args=args, recorder=recorder)
    original_read = control.read_pos_vel

    def guarded_read() -> tuple[list[int], list[int]]:
        assert _FakeExternalPoseReceiver.instances[0].baseline_waits
        return original_read()

    monkeypatch.setattr(control, "read_pos_vel", guarded_read)

    runner.execute()

    receiver = _FakeExternalPoseReceiver.instances[0]
    assert receiver.started and receiver.stopped
    assert receiver.baseline_waits == [(3, 0.25)]
    assert not receiver.endpoint_waits
    assert len(receiver.endpoint_source_waits) == 1
    source_deadline_ns, endpoint_timeout_s, receive_deadline_ns = (
        receiver.endpoint_source_waits[0]
    )
    assert endpoint_timeout_s == 0.25
    assert receive_deadline_ns is not None
    assert runner._last_goal_write_completion_ns is not None
    assert runner._last_goal_write_completion_unix_ns is not None
    assert receive_deadline_ns == runner._last_goal_write_completion_ns
    assert source_deadline_ns == (
        runner._last_goal_write_completion_unix_ns + 12_500_000
    )
    assert control.last_move_return_ns is not None
    assert receive_deadline_ns >= control.last_move_return_ns
    assert control.last_move_return_unix_ns is not None
    assert source_deadline_ns >= control.last_move_return_unix_ns + 12_500_000
    assert receiver.max_clock_offset_s == pytest.approx(0.04)
    assert getattr(receiver.on_packet, "__self__", None) is recorder
    assert getattr(receiver.on_packet, "__name__", "") == (
        "append_external_pose_packet"
    )
    assert receiver.failure_checks == 6
    assert not control.disabled
    assert control.closed

    manifest = json.loads(recorder.paths.manifest.read_text(encoding="utf-8"))
    assert manifest["recording"]["termination_reason"] == "trial_steps_reached"
    assert manifest["external_pose_recording"]["port"] == 15555
    events = [
        json.loads(line)
        for line in recorder.paths.events.read_text(encoding="utf-8").splitlines()
    ]
    assert "external_pose_baseline_ready" in {event["name"] for event in events}
    assert "external_pose_endpoint_ready" in {event["name"] for event in events}


def test_external_pose_baseline_timeout_finalizes_as_controller_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _FakeExternalPoseReceiver.configure(baseline_ready=False)
    monkeypatch.setattr(real, "ExternalPoseReceiver", _FakeExternalPoseReceiver)
    monkeypatch.setattr(real.RealRobotRunner, "move_to_start", lambda _self: None)
    path = tmp_path / "baseline-timeout.npz"
    args = _record_args(
        path,
        "--external-pose-port",
        "15555",
        "--external-pose-timeout-s",
        "0.01",
    )
    recorder = ExperimentRecorder(
        path,
        metadata=_metadata(),
        observation_size=81,
        command_size=3,
    )
    runner, control = _fake_runner(args=args, recorder=recorder)

    with pytest.raises(TimeoutError, match="baseline did not receive"):
        runner.execute()

    receiver = _FakeExternalPoseReceiver.instances[0]
    assert receiver.stopped
    assert control.read_count == 0
    assert not control.moves
    assert not control.disabled
    assert control.closed
    manifest = json.loads(recorder.paths.manifest.read_text(encoding="utf-8"))
    assert manifest["recording"]["termination_reason"] == "controller_error"
    assert manifest["recording"]["sample_count"] == 0
    assert manifest["recording"]["error"]["type"] == "TimeoutError"


def test_delayed_prewrite_pose_cannot_satisfy_external_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _FakeExternalPoseReceiver.configure(endpoint_ready=False)
    monkeypatch.setattr(real, "ExternalPoseReceiver", _FakeExternalPoseReceiver)
    monkeypatch.setattr(real.RealRobotRunner, "move_to_start", lambda _self: None)
    monkeypatch.setattr(real.time, "sleep", lambda _duration: None)
    path = tmp_path / "endpoint-timeout.npz"
    args = _record_args(
        path,
        "--external-pose-port",
        "15555",
        "--external-pose-timeout-s",
        "0.01",
    )
    recorder = ExperimentRecorder(
        path,
        metadata=_metadata(),
        observation_size=81,
        command_size=3,
    )
    runner, control = _fake_runner(args=args, recorder=recorder)

    with pytest.raises(TimeoutError, match="endpoint was not received"):
        runner.execute()

    receiver = _FakeExternalPoseReceiver.instances[0]
    # The legacy receive-only fake would accept a delayed pre-write packet. The
    # source-time gate must reject it and therefore time out.
    assert not receiver.endpoint_waits
    assert len(receiver.endpoint_source_waits) == 1
    source_deadline_ns, _, receive_deadline_ns = receiver.endpoint_source_waits[0]
    assert receive_deadline_ns is not None
    assert runner._last_goal_write_completion_ns is not None
    assert runner._last_goal_write_completion_unix_ns is not None
    assert source_deadline_ns == (
        runner._last_goal_write_completion_unix_ns + 12_500_000
    )
    assert receive_deadline_ns == runner._last_goal_write_completion_ns
    assert len(control.moves) == 3
    assert not control.disabled
    assert control.closed
    manifest = json.loads(recorder.paths.manifest.read_text(encoding="utf-8"))
    assert manifest["recording"]["termination_reason"] == "controller_error"
    assert manifest["recording"]["sample_count"] == 3
    assert manifest["recording"]["error"]["type"] == "TimeoutError"


def test_external_pose_fatal_error_is_surfaced_and_finalized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _FakeExternalPoseReceiver.configure(
        baseline_ready=False,
        fatal_error=RuntimeError("tracker receive thread failed"),
    )
    monkeypatch.setattr(real, "ExternalPoseReceiver", _FakeExternalPoseReceiver)
    monkeypatch.setattr(real.RealRobotRunner, "move_to_start", lambda _self: None)
    path = tmp_path / "tracker-fatal.npz"
    args = _record_args(path, "--external-pose-port", "15555")
    recorder = ExperimentRecorder(
        path,
        metadata=_metadata(),
        observation_size=81,
        command_size=3,
    )
    runner, control = _fake_runner(args=args, recorder=recorder)

    with pytest.raises(RuntimeError, match="tracker receive thread failed"):
        runner.execute()

    assert not control.disabled
    assert control.closed
    manifest = json.loads(recorder.paths.manifest.read_text(encoding="utf-8"))
    assert manifest["recording"]["termination_reason"] == "controller_error"
    assert manifest["recording"]["error"] == {
        "type": "RuntimeError",
        "message": "tracker receive thread failed",
    }
