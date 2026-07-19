"""Run and record a paper-grade Swenoid hardware experiment at 50 Hz."""

from __future__ import annotations

import argparse
import contextlib
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from swenoid.deployment.policy import (
    OnnxPolicy,
    benchmark,
    concatenate_observations,
    resolve_policy_path,
)
from swenoid.deployment.recording import (
    CURRENT_AMPERE_PER_RAW_UNIT,
    VOLTAGE_VOLT_PER_RAW_UNIT,
    ExperimentRecorder,
    ExternalPoseReceiver,
    load_json_object,
    sha256_file,
    sha256_json,
)
from swenoid.deployment.swenoid_control import SwenoidControl
from swenoid.model_constants import SWENOID_HARDWARE_CONFIG

CONTROL_PERIOD_S = 0.02
JOINT_COUNT = 24


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse deployment and experiment-recording arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--onnx", type=Path)
    source.add_argument("--wandb-run", help="Run ID or ENTITY/PROJECT/RUN_ID.")
    parser.add_argument("--entity", default="morrje-kth-royal-institute-of-technology")
    parser.add_argument("--project", default="mjlab")
    parser.add_argument("--onnx-name")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument(
        "--hardware-config",
        type=Path,
        help="Per-robot motor mapping and neutral calibration JSON.",
    )
    parser.add_argument("--baudrate", type=int, default=4_000_000)
    parser.add_argument(
        "--serial-retries",
        type=int,
        default=100,
        help="Maximum attempts for each Dynamixel transaction.",
    )
    parser.add_argument("--i2c-frequency", type=int, default=800_000)
    parser.add_argument(
        "--configure-latency",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--command", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--velocity-steps",
        type=int,
        default=200,
        help="Number of policy steps for which the velocity command is active.",
    )
    parser.add_argument(
        "--command-start-step",
        type=int,
        default=0,
        help="First policy step at which the velocity command is active.",
    )
    parser.add_argument(
        "--trial-steps",
        type=int,
        help="Stop naturally after this many completed control steps.",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--trial-id")
    parser.add_argument("--condition")
    parser.add_argument("--robot-id")
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--replicate", type=int)
    parser.add_argument(
        "--experiment-metadata",
        type=Path,
        help="JSON object containing floor, setup, operator, and protocol context.",
    )
    parser.add_argument("--overwrite-record", action="store_true")
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=250,
        help="Write a recoverable partial NPZ every N completed steps.",
    )
    parser.add_argument(
        "--telemetry-every",
        type=int,
        default=1,
        help="Read motor current, voltage, and temperature every N recorded steps.",
    )
    parser.add_argument(
        "--external-pose-port",
        type=int,
        help="UDP port receiving timestamped external 6-DoF ground truth.",
    )
    parser.add_argument(
        "--external-pose-bind",
        default="0.0.0.0",
        help="Local address for the external-pose UDP receiver.",
    )
    parser.add_argument(
        "--external-pose-baseline-samples",
        type=int,
        default=2,
        help="Valid external-pose samples required before policy execution.",
    )
    parser.add_argument(
        "--external-pose-timeout-s",
        type=float,
        default=2.0,
        help="Maximum wait for external-pose baseline and endpoint coverage.",
    )
    parser.add_argument(
        "--external-pose-max-clock-offset-ms",
        type=float,
        default=1000.0,
        help="Maximum absolute tracker-source/receiver clock offset.",
    )
    parser.add_argument(
        "--disable-torque-on-exit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable motor torque after a graceful exit; never used after errors.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.velocity_steps < 0:
        raise SystemExit("--velocity-steps must be non-negative")
    if args.command_start_step < 0:
        raise SystemExit("--command-start-step must be non-negative")
    if args.trial_steps is not None and args.trial_steps <= 0:
        raise SystemExit("--trial-steps must be positive")
    if args.telemetry_every <= 0 or args.checkpoint_steps <= 0:
        raise SystemExit("Telemetry and checkpoint intervals must be positive")
    if args.baudrate <= 0 or args.serial_retries <= 0 or args.i2c_frequency <= 0:
        raise SystemExit("Hardware rates and retry counts must be positive")
    if args.external_pose_port is not None and not (
        1 <= args.external_pose_port <= 65535
    ):
        raise SystemExit("--external-pose-port must be in [1, 65535]")
    if args.external_pose_baseline_samples <= 0:
        raise SystemExit("--external-pose-baseline-samples must be positive")
    if (
        not np.isfinite(args.external_pose_timeout_s)
        or args.external_pose_timeout_s <= 0
    ):
        raise SystemExit("--external-pose-timeout-s must be finite and positive")
    if (
        not np.isfinite(args.external_pose_max_clock_offset_ms)
        or args.external_pose_max_clock_offset_ms <= 0
    ):
        raise SystemExit(
            "--external-pose-max-clock-offset-ms must be finite and positive"
        )
    if args.replicate is not None and args.replicate < 0:
        raise SystemExit("--replicate must be non-negative")
    if args.training_seed is not None and args.training_seed < 0:
        raise SystemExit("--training-seed must be non-negative")

    recording_only_options = (
        args.experiment_metadata is not None,
        args.overwrite_record,
        args.external_pose_port is not None,
        args.trial_id is not None,
        args.condition is not None,
        args.robot_id is not None,
        args.training_seed is not None,
        args.replicate is not None,
    )
    if args.record is None and any(recording_only_options):
        raise SystemExit("Experiment metadata and recording options require --record")
    if args.record is not None:
        if args.loop:
            raise SystemExit("Recorded trials cannot use --loop")
        missing = [
            option
            for option, value in (
                ("--trial-id", args.trial_id),
                ("--condition", args.condition),
                ("--robot-id", args.robot_id),
            )
            if not value
        ]
        if missing:
            raise SystemExit("Recorded trials require " + ", ".join(missing))


@dataclass(frozen=True)
class ImuSnapshot:
    """One atomically captured BNO085 sample and its filtered policy values."""

    angular_velocity_raw: np.ndarray
    angular_velocity: np.ndarray
    projected_gravity_raw: np.ndarray
    projected_gravity: np.ndarray
    orientation_wxyz: np.ndarray
    monotonic_ns: int
    sequence: int


class ImuReader:
    """Continuously sample the pelvis BNO085 and publish atomic snapshots."""

    def __init__(self, i2c_frequency: int = 800_000):
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.i2c_frequency = i2c_frequency
        self._lock = threading.Lock()
        self._snapshot = ImuSnapshot(
            angular_velocity_raw=np.zeros(3, dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            projected_gravity_raw=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            projected_gravity=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            monotonic_ns=0,
            sequence=-1,
        )
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            bno085 = import_module("swenoid.deployment.bno085")
            sensor_type = bno085.BNO085
            sensor = sensor_type(i2c_frequency=self.i2c_frequency)
            gravity_fn = getattr(sensor, "get_gravity_old", None)
            if gravity_fn is None:
                gravity_fn = sensor.get_gravity
            for _ in range(1000):
                gravity_fn()
                sensor.get_ang_vel()

            angular_velocity = self._validate_vector(
                sensor.get_ang_vel(), "angular velocity"
            )
            gravity = self._validate_gravity(gravity_fn()).reshape(3)
            orientation = self._validate_orientation(sensor.get_orientation())
            self._publish(
                angular_velocity=angular_velocity,
                gravity=gravity,
                orientation=orientation,
                sequence=0,
                initialize=True,
            )
            self.ready.set()

            sequence = 1
            while not self.stop.is_set():
                angular_velocity = self._validate_vector(
                    sensor.get_ang_vel(), "angular velocity"
                )
                gravity = self._validate_gravity(gravity_fn()).reshape(3)
                orientation = self._validate_orientation(sensor.get_orientation())
                self._publish(
                    angular_velocity=angular_velocity,
                    gravity=gravity,
                    orientation=orientation,
                    sequence=sequence,
                    initialize=False,
                )
                sequence += 1
        except BaseException as exc:
            self.error = exc
            self.ready.set()

    def _publish(
        self,
        *,
        angular_velocity: np.ndarray,
        gravity: np.ndarray,
        orientation: np.ndarray,
        sequence: int,
        initialize: bool,
    ) -> None:
        with self._lock:
            previous = self._snapshot
            filtered_angular_velocity = (
                angular_velocity
                if initialize
                else 0.45 * angular_velocity + 0.55 * previous.angular_velocity
            )
            filtered_gravity = (
                gravity
                if initialize
                else 0.55 * gravity + 0.45 * previous.projected_gravity
            )
            self._snapshot = ImuSnapshot(
                angular_velocity_raw=angular_velocity.astype(np.float32, copy=True),
                angular_velocity=np.asarray(
                    filtered_angular_velocity, dtype=np.float32
                ).copy(),
                projected_gravity_raw=gravity.astype(np.float32, copy=True),
                projected_gravity=np.asarray(filtered_gravity, dtype=np.float32).copy(),
                orientation_wxyz=orientation.astype(np.float32, copy=True),
                monotonic_ns=time.perf_counter_ns(),
                sequence=sequence,
            )

    @staticmethod
    def _validate_vector(value: Any, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(3)
        if not np.isfinite(vector).all():
            raise ValueError(f"Invalid {name} from BNO085: {vector}")
        return vector

    @staticmethod
    def _validate_gravity(value: Any) -> np.ndarray:
        gravity = np.asarray(value, dtype=np.float32).reshape(1, 3)
        norm = float(np.linalg.norm(gravity))
        if not np.isfinite(norm) or norm < 1e-6:
            raise ValueError(f"Invalid gravity vector from BNO085: {gravity}")
        return gravity

    @staticmethod
    def _validate_orientation(value: Any) -> np.ndarray:
        orientation = np.asarray(value, dtype=np.float32).reshape(4)
        norm = float(np.linalg.norm(orientation))
        if not np.isfinite(norm) or norm < 1e-6:
            raise ValueError(
                f"Invalid orientation quaternion from BNO085: {orientation}"
            )
        return orientation / norm

    def wait(self) -> None:
        self.ready.wait()
        if self.error is not None:
            raise RuntimeError("Failed to initialize BNO085") from self.error

    def snapshot(self) -> ImuSnapshot:
        """Return a copy of all fields from one published IMU sample."""
        if self.error is not None:
            raise RuntimeError("BNO085 reader failed") from self.error
        with self._lock:
            sample = self._snapshot
            return ImuSnapshot(
                angular_velocity_raw=sample.angular_velocity_raw.copy(),
                angular_velocity=sample.angular_velocity.copy(),
                projected_gravity_raw=sample.projected_gravity_raw.copy(),
                projected_gravity=sample.projected_gravity.copy(),
                orientation_wxyz=sample.orientation_wxyz.copy(),
                monotonic_ns=sample.monotonic_ns,
                sequence=sample.sequence,
            )


def git_provenance(source_path: Path) -> dict[str, Any]:
    """Return best-effort repository provenance without failing deployment."""

    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(source_path), *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    root = run("rev-parse", "--show-toplevel")
    commit = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=normal")
    return {
        "available": root is not None and commit is not None,
        "repository_root": root,
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
    }


def software_provenance() -> dict[str, Any]:
    """Describe the software runtime used for a physical trial."""
    try:
        package_version = version("swenoid")
    except PackageNotFoundError:
        package_version = "uninstalled"
    return {
        "swenoid_version": package_version,
        "git": git_provenance(Path(__file__).resolve().parent),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "numpy_version": np.__version__,
    }


def _policy_source(args: argparse.Namespace) -> dict[str, Any]:
    if args.onnx is not None:
        return {"kind": "local_onnx", "requested_path": str(args.onnx)}
    return {
        "kind": "wandb_run",
        "run": args.wandb_run,
        "entity": args.entity,
        "project": args.project,
        "onnx_name": args.onnx_name,
    }


def build_manifest_metadata(
    policy: OnnxPolicy,
    policy_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build manifest sections available before hardware initialization."""
    experiment_context: dict[str, Any] = {}
    experiment_source: dict[str, Any] | None = None
    if args.experiment_metadata is not None:
        experiment_context = load_json_object(args.experiment_metadata)
        experiment_source = {
            "path": str(args.experiment_metadata.resolve()),
            "sha256": sha256_file(args.experiment_metadata),
        }

    onnx_metadata = dict(policy.metadata)
    return {
        "policy": {
            "source": _policy_source(args),
            "resolved_path": str(policy_path.resolve()),
            "sha256": sha256_file(policy_path),
            "policy_kind": policy.policy_kind,
            "motion_length": policy.motion_length,
            "observation_size": policy.observation_size,
            "observation_names": list(policy.observation_names),
            "action_scale": policy.action_scale.tolist(),
            "default_joint_position_rad": policy.default_joint_pos.tolist(),
            "onnx_metadata": onnx_metadata,
            "onnx_metadata_sha256": sha256_json(onnx_metadata),
        },
        "software": software_provenance(),
        "trial": {
            "trial_id": args.trial_id,
            "condition": args.condition,
            "robot_id": args.robot_id,
            "training_seed": args.training_seed,
            "replicate": args.replicate,
            "experiment_context": experiment_context,
            "experiment_context_sha256": sha256_json(experiment_context),
            "experiment_context_source": experiment_source,
        },
        "control": {
            "control_period_s": CONTROL_PERIOD_S,
            "policy_rate_hz": 1.0 / CONTROL_PERIOD_S,
            "command": list(map(float, args.command)),
            "command_start_step": args.command_start_step,
            "velocity_steps": args.velocity_steps,
            "trial_steps": args.trial_steps,
            "loop": args.loop,
            "telemetry_every": args.telemetry_every,
            "checkpoint_steps": args.checkpoint_steps,
            "port": args.port,
            "baudrate": args.baudrate,
            "serial_retries": args.serial_retries,
            "i2c_frequency": args.i2c_frequency,
            "configure_latency": args.configure_latency,
            "disable_torque_on_exit": args.disable_torque_on_exit,
            "external_pose": (
                None
                if args.external_pose_port is None
                else {
                    "transport": "udp_json",
                    "bind": args.external_pose_bind,
                    "port": args.external_pose_port,
                    "baseline_samples": args.external_pose_baseline_samples,
                    "wait_timeout_s": args.external_pose_timeout_s,
                    "max_clock_offset_ms": args.external_pose_max_clock_offset_ms,
                }
            ),
        },
    }


class RealRobotRunner:
    """Hardware policy loop with complete, aligned experimental recording."""

    def __init__(
        self,
        policy: OnnxPolicy,
        args: argparse.Namespace,
        recorder: ExperimentRecorder | None = None,
    ) -> None:
        self.policy = policy
        self.args = args
        self.recorder = recorder
        self.requested_command = np.asarray(args.command, dtype=np.float32).reshape(3)
        self.last_actions = np.zeros((1, JOINT_COUNT), dtype=np.float32)
        self.time_step = 0
        self.completed_steps = 0
        self._last_cycle_start_ns: int | None = None
        self._last_recorded_command: np.ndarray | None = None
        self._last_goal_write_completion_ns: int | None = None
        self._last_goal_write_completion_unix_ns: int | None = None
        self.imu: ImuReader | None = None
        self.control: SwenoidControl | None = None
        self.external_pose: ExternalPoseReceiver | None = None

        try:
            self.imu = ImuReader(i2c_frequency=args.i2c_frequency)
            self.imu.wait()
            self.control = SwenoidControl(
                config=args.hardware_config,
                port=args.port,
                baudrate=args.baudrate,
                configure_latency=args.configure_latency,
                max_retries=args.serial_retries,
            )
            if self.recorder is not None:
                self.recorder.set_metadata("hardware", self._hardware_metadata())
        except BaseException as exc:
            self._record_initialization_failure(exc)
            self._close_unstarted_resources()
            raise

    def _hardware_metadata(self) -> dict[str, Any]:
        assert self.control is not None
        metadata = self.control.recording_metadata()
        hardware_config = metadata["hardware_config"]
        config_path = (
            SWENOID_HARDWARE_CONFIG
            if self.args.hardware_config is None
            else self.args.hardware_config
        )
        return {
            **metadata,
            "robot_id": self.args.robot_id,
            "hardware_config_path": str(config_path.resolve()),
            "hardware_config_file_sha256": sha256_file(config_path),
            "hardware_config_json_sha256": sha256_json(hardware_config),
        }

    def _record_initialization_failure(self, error: BaseException) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.set_metadata(
                "hardware",
                {
                    "robot_id": self.args.robot_id,
                    "initialization_status": "failed",
                    "requested_hardware_config": (
                        None
                        if self.args.hardware_config is None
                        else str(self.args.hardware_config)
                    ),
                },
            )
            self.recorder.finalize(
                termination_reason="controller_error",
                error=error,
            )
        except BaseException as recording_error:
            print(
                f"[WARN] Could not finalize the failed trial record: {recording_error}"
            )

    def _close_unstarted_resources(self) -> None:
        if self.imu is not None:
            self.imu.stop.set()
        if self.control is not None:
            with contextlib.suppress(BaseException):
                self.control.close()
        if self.imu is not None:
            self.imu.thread.join(timeout=1.0)

    def move_to_start(self) -> None:
        assert self.control is not None
        target = self.control.pos_isaac_to_dynamixel(self.policy.default_joint_pos)
        self.control.enable_torques_at_current_position()
        self.control.move_servos_duration(target, [3000] * JOINT_COUNT)
        input("Robot is at the default pose. Press Enter to start the policy...")

    def _start_recording(self) -> None:
        if self.recorder is None:
            return
        self.recorder.start()
        if self.args.external_pose_port is not None:
            self.external_pose = ExternalPoseReceiver(
                bind=self.args.external_pose_bind,
                port=self.args.external_pose_port,
                on_packet=self.recorder.append_external_pose_packet,
                max_clock_offset_s=(
                    self.args.external_pose_max_clock_offset_ms / 1000.0
                ),
            )
            self.external_pose.start()
            self.recorder.event(
                "external_pose_receiver_started",
                sample_index=-1,
                detail={
                    "bind": self.args.external_pose_bind,
                    "port": self.args.external_pose_port,
                    "baseline_samples": self.args.external_pose_baseline_samples,
                    "wait_timeout_s": self.args.external_pose_timeout_s,
                    "max_clock_offset_ms": (
                        self.args.external_pose_max_clock_offset_ms
                    ),
                },
            )
            baseline_ready = self.external_pose.wait_for_valid_samples(
                self.args.external_pose_baseline_samples,
                self.args.external_pose_timeout_s,
            )
            self.external_pose.raise_if_failed()
            if not baseline_ready:
                raise TimeoutError(
                    "External pose baseline did not receive "
                    f"{self.args.external_pose_baseline_samples} valid sample(s) "
                    f"within {self.args.external_pose_timeout_s:g} s"
                )
            self.recorder.event(
                "external_pose_baseline_ready",
                sample_index=-1,
                detail={
                    "valid_samples": self.args.external_pose_baseline_samples,
                },
            )

    def _external_pose_clock_offset(self) -> tuple[float, int]:
        """Return tracker-minus-controller clock offset from trial metadata."""
        offset_ms = 0.0
        if self.recorder is not None:
            trial = self.recorder.manifest.get("trial", {})
            if not isinstance(trial, Mapping):
                raise ValueError("Trial metadata must be an object")
            context = trial.get("experiment_context", {})
            if not isinstance(context, Mapping):
                raise ValueError("trial.experiment_context must be an object")
            external_tracking = context.get("external_tracking")
            if external_tracking is not None:
                if not isinstance(external_tracking, Mapping):
                    raise ValueError(
                        "trial.experiment_context.external_tracking must be an object"
                    )
                raw_offset = external_tracking.get("estimated_clock_offset_ms")
                if raw_offset is not None:
                    if isinstance(raw_offset, bool) or not isinstance(
                        raw_offset, (int, float)
                    ):
                        raise ValueError(
                            "external_tracking.estimated_clock_offset_ms must be numeric"
                        )
                    offset_ms = float(raw_offset)
                    if not np.isfinite(offset_ms):
                        raise ValueError(
                            "external_tracking.estimated_clock_offset_ms must be finite"
                        )
        return offset_ms, round(offset_ms * 1_000_000.0)

    def _require_external_pose_endpoint(self) -> None:
        """Require post-command ground truth in both controller and source time."""
        if self.external_pose is None:
            return
        if (
            self._last_goal_write_completion_ns is None
            or self._last_goal_write_completion_unix_ns is None
        ):
            raise RuntimeError(
                "Cannot establish external-pose endpoint without a motor command"
            )
        offset_ms, offset_ns = self._external_pose_clock_offset()
        source_time_threshold_ns = self._last_goal_write_completion_unix_ns + offset_ns
        endpoint_ready = self.external_pose.wait_for_valid_source_after(
            source_time_threshold_ns,
            self.args.external_pose_timeout_s,
            receive_monotonic_ns=self._last_goal_write_completion_ns,
        )
        self.external_pose.raise_if_failed()
        if not endpoint_ready:
            raise TimeoutError(
                "External pose endpoint was not received after the final motor "
                f"command within {self.args.external_pose_timeout_s:g} s"
            )
        if self.recorder is not None:
            self.recorder.event(
                "external_pose_endpoint_ready",
                sample_index=self.recorder.sample_count - 1,
                detail={
                    "goal_write_completion_monotonic_ns": (
                        self._last_goal_write_completion_ns
                    ),
                    "goal_write_completion_unix_ns": (
                        self._last_goal_write_completion_unix_ns
                    ),
                    "estimated_clock_offset_ms": offset_ms,
                    "source_time_threshold_ns": source_time_threshold_ns,
                },
            )

    def _velocity_command(self, policy_frame: int) -> tuple[np.ndarray, bool]:
        first = self.args.command_start_step
        last = first + self.args.velocity_steps
        active = first <= policy_frame < last
        return (
            self.requested_command.copy() if active else np.zeros(3, dtype=np.float32),
            active,
        )

    def _record_command_change(
        self,
        command: np.ndarray,
        *,
        active: bool,
        sample_index: int,
        policy_frame: int,
    ) -> None:
        if self.recorder is None or self.policy.is_tracking:
            return
        if self._last_recorded_command is not None and np.array_equal(
            command, self._last_recorded_command
        ):
            return
        self.recorder.event(
            "velocity_command_changed",
            sample_index=sample_index,
            detail={
                "policy_frame": policy_frame,
                "active": active,
                "command": command.tolist(),
            },
        )
        self._last_recorded_command = command.copy()

    def _next_task_input(
        self, policy_frame: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        reference = self.policy.reference_at(policy_frame)
        if reference is None:
            command, active = self._velocity_command(policy_frame)
            missing_reference = np.full(JOINT_COUNT, np.nan, dtype=np.float32)
            return command, missing_reference, missing_reference.copy(), active
        reference_position = np.asarray(reference[0], dtype=np.float32).reshape(
            JOINT_COUNT
        )
        reference_velocity = np.asarray(reference[1], dtype=np.float32).reshape(
            JOINT_COUNT
        )
        return (
            np.concatenate((reference_position, reference_velocity)),
            reference_position,
            reference_velocity,
            False,
        )

    def _termination_cause(self) -> str | None:
        if (
            self.args.trial_steps is not None
            and self.completed_steps >= self.args.trial_steps
        ):
            return "trial_steps_reached"
        if (
            self.policy.motion_length is not None
            and self.time_step >= self.policy.motion_length
        ):
            if self.args.loop:
                self.time_step = 0
                return None
            return "motion_complete"
        return None

    def run(self) -> str:
        """Run until a configured natural terminal condition is reached."""
        self.move_to_start()
        self._start_recording()
        missed_deadlines = 0

        while True:
            termination_cause = self._termination_cause()
            if termination_cause is not None:
                self._require_external_pose_endpoint()
                return termination_cause
            if self.external_pose is not None:
                self.external_pose.raise_if_failed()

            assert self.control is not None
            assert self.imu is not None
            cycle_start_ns = time.perf_counter_ns()
            cycle_start_unix_ns = time.time_ns()
            policy_frame = self.time_step
            sample_index = self.completed_steps
            if self.recorder is not None:
                sample_index = self.recorder.sample_count

            command, reference_position, reference_velocity, command_active = (
                self._next_task_input(policy_frame)
            )
            self._record_command_change(
                command,
                active=command_active,
                sample_index=sample_index,
                policy_frame=policy_frame,
            )

            read_start_ns = time.perf_counter_ns()
            raw_position, raw_velocity = self.control.read_pos_vel()
            imu = self.imu.snapshot()
            imu_read_ns = time.perf_counter_ns()
            telemetry: tuple[list[int], list[int], list[int]] | None = None
            telemetry_time_ns: int | None = None
            if (
                self.recorder is not None
                and sample_index % self.args.telemetry_every == 0
            ):
                telemetry = self.control.read_servo_telemetry()
                telemetry_time_ns = time.perf_counter_ns()
            read_duration_s = (time.perf_counter_ns() - read_start_ns) * 1e-9

            joint_position = self.control.pos_dynamixel_to_isaac(raw_position)
            joint_velocity = self.control.vel_dynamixel_to_isaac(raw_velocity)
            relative_joint_position = (
                joint_position - self.policy.default_joint_pos
            ).reshape(1, JOINT_COUNT)
            observation_parts = {
                "command": command.reshape(1, -1),
                "base_ang_vel": imu.angular_velocity.reshape(1, 3),
                "body_ang_vel": imu.angular_velocity.reshape(1, 3),
                "projected_gravity": imu.projected_gravity.reshape(1, 3),
                "body_projected_gravity": imu.projected_gravity.reshape(1, 3),
                "joint_pos": relative_joint_position,
                "joint_vel": joint_velocity.reshape(1, JOINT_COUNT),
                "actions": self.last_actions,
            }
            observations = concatenate_observations(self.policy, observation_parts)

            inference_start_ns = time.perf_counter_ns()
            outputs = self.policy.infer(observations, policy_frame)
            inference_duration_s = (time.perf_counter_ns() - inference_start_ns) * 1e-9
            action = outputs[0].astype(np.float32, copy=False)
            requested_target = (
                action * self.policy.action_scale + self.policy.default_joint_pos
            )[0]
            position_command = self.control.encode_position_command(requested_target)

            write_start_ns = time.perf_counter_ns()
            self.control.move_servos(position_command.dynamixel_goal, raw_position)
            write_completion_ns = time.perf_counter_ns()
            write_completion_unix_ns = time.time_ns()
            self._last_goal_write_completion_ns = write_completion_ns
            self._last_goal_write_completion_unix_ns = write_completion_unix_ns
            write_duration_s = (write_completion_ns - write_start_ns) * 1e-9
            self.last_actions = action.copy()

            work_duration_s = (time.perf_counter_ns() - cycle_start_ns) * 1e-9
            deadline_missed = work_duration_s > CONTROL_PERIOD_S
            cycle_duration_s = work_duration_s

            if deadline_missed:
                missed_deadlines += 1
                if missed_deadlines % 50 == 1:
                    print(f"[WARN] Missed 50 Hz deadline {missed_deadlines} time(s)")

            if self.recorder is not None:
                control_dt_s = (
                    np.nan
                    if self._last_cycle_start_ns is None
                    else (cycle_start_ns - self._last_cycle_start_ns) * 1e-9
                )
                self.recorder.append_step(
                    {
                        "sample_index": sample_index,
                        "policy_frame": policy_frame,
                        "time_s": self.recorder.elapsed_s(cycle_start_ns),
                        "time_unix_ns": cycle_start_unix_ns,
                        "control_dt_s": control_dt_s,
                        "read_duration_s": read_duration_s,
                        "inference_duration_s": inference_duration_s,
                        "write_duration_s": write_duration_s,
                        "cycle_duration_s": cycle_duration_s,
                        "deadline_missed": deadline_missed,
                        "policy_observation": observations[0],
                        "policy_action": action[0],
                        "command": command,
                        "reference_joint_position_rad": reference_position,
                        "reference_joint_velocity_rad_s": reference_velocity,
                        "joint_position_rad": joint_position,
                        "joint_velocity_rad_s": joint_velocity,
                        "joint_target_requested_rad": requested_target,
                        "joint_target_sent_rad": position_command.sent_position_rad,
                        "dynamixel_position_raw": np.asarray(
                            raw_position, dtype=np.int32
                        ),
                        "dynamixel_velocity_raw": np.asarray(
                            raw_velocity, dtype=np.int32
                        ),
                        "dynamixel_goal_position_raw": np.asarray(
                            position_command.dynamixel_goal, dtype=np.int32
                        ),
                        "dynamixel_goal_clipped": position_command.clipped,
                        "base_angular_velocity_raw_rad_s": (imu.angular_velocity_raw),
                        "base_angular_velocity_rad_s": imu.angular_velocity,
                        "projected_gravity_raw": imu.projected_gravity_raw,
                        "projected_gravity": imu.projected_gravity,
                        "base_orientation_wxyz": imu.orientation_wxyz,
                        "imu_sample_time_s": self.recorder.elapsed_s(imu.monotonic_ns),
                        "imu_age_s": max(0.0, (imu_read_ns - imu.monotonic_ns) * 1e-9),
                        "imu_sequence": imu.sequence,
                    }
                )
                if telemetry is not None:
                    assert telemetry_time_ns is not None
                    current_raw, voltage_raw, temperature_c = telemetry
                    current = np.asarray(current_raw, dtype=np.int16)
                    voltage = np.asarray(voltage_raw, dtype=np.uint16)
                    self.recorder.append_servo_telemetry(
                        {
                            "servo_telemetry_sample_index": sample_index,
                            "servo_telemetry_time_s": self.recorder.elapsed_s(
                                telemetry_time_ns
                            ),
                            "servo_current_raw": current,
                            "servo_current_a": current.astype(np.float32)
                            * CURRENT_AMPERE_PER_RAW_UNIT,
                            "servo_voltage_raw": voltage,
                            "servo_voltage_v": voltage.astype(np.float32)
                            * VOLTAGE_VOLT_PER_RAW_UNIT,
                            "servo_temperature_c": np.asarray(
                                temperature_c, dtype=np.float32
                            ),
                        }
                    )

            self._last_cycle_start_ns = cycle_start_ns
            self.time_step += 1
            self.completed_steps += 1
            remaining_s = CONTROL_PERIOD_S - (
                (time.perf_counter_ns() - cycle_start_ns) * 1e-9
            )
            if remaining_s > 0.0:
                time.sleep(remaining_s)

    def execute(self) -> None:
        """Run and finalize the trial without disabling torque after errors."""
        self._termination_reason = "controller_error"
        self._execution_error: BaseException | None = None
        disable_torque = False
        recorder = getattr(self, "recorder", None)
        try:
            natural_cause = self.run()
            self._termination_reason = natural_cause
            disable_torque = self.args.disable_torque_on_exit
            if recorder is not None:
                recorder.event(
                    "trial_completed",
                    sample_index=recorder.sample_count - 1,
                    detail={"termination_reason": natural_cause},
                )
        except KeyboardInterrupt:
            self._termination_reason = "operator_interrupt"
            disable_torque = self.args.disable_torque_on_exit
            if recorder is not None and recorder.start_monotonic_ns is not None:
                recorder.event(
                    "operator_interrupt",
                    sample_index=recorder.sample_count - 1,
                )
        except BaseException as exc:
            self._execution_error = exc
            if recorder is not None and recorder.start_monotonic_ns is not None:
                recorder.event(
                    "controller_error",
                    sample_index=recorder.sample_count - 1,
                    detail={"type": type(exc).__name__, "message": str(exc)},
                )
            raise
        finally:
            self.close(disable_torque=disable_torque)

    def close(self, *, disable_torque: bool) -> None:
        """Close hardware and atomically finalize all experiment artifacts."""
        termination_reason = getattr(self, "_termination_reason", "controller_error")
        error = getattr(self, "_execution_error", None)
        close_error: BaseException | None = None
        external_metadata: Mapping[str, Any] | None = None

        if self.external_pose is not None:
            try:
                self.external_pose.stop()
                if (
                    self.recorder is not None
                    and self.recorder.start_monotonic_ns is not None
                ):
                    external_metadata = self.external_pose.metadata()
                self.external_pose.raise_if_failed()
            except BaseException as exc:
                close_error = exc
                termination_reason = "controller_error"
                disable_torque = False

        if self.imu is not None:
            self.imu.stop.set()
        if self.control is not None:
            try:
                if disable_torque:
                    self.control.disable_torques()
                else:
                    print(
                        "[SAFETY] Motor torque remains enabled. Support the robot "
                        "before powering it down."
                    )
            except BaseException as exc:
                close_error = close_error or exc
            try:
                self.control.close()
            except BaseException as exc:
                close_error = close_error or exc
        if self.imu is not None:
            self.imu.thread.join(timeout=1.0)

        if self.recorder is not None:
            final_error = error or close_error
            try:
                self.recorder.finalize(
                    termination_reason=termination_reason,
                    external_metadata=external_metadata,
                    error=final_error,
                )
            except BaseException as exc:
                close_error = close_error or exc

        if error is None and close_error is not None:
            raise close_error


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    path = resolve_policy_path(
        onnx_path=args.onnx,
        wandb_run=args.wandb_run,
        entity=args.entity,
        project=args.project,
        onnx_name=args.onnx_name,
    )
    policy = OnnxPolicy(path)
    print(f"Loaded {policy.policy_kind.replace('_', ' ')} policy: {path}")
    if policy.is_general_motion:
        raise SystemExit(
            "General-motion deployment requires a live 48-value reference "
            "provider, which is deferred to the real-time imitation release."
        )
    if args.record is not None and not policy.is_tracking and args.trial_steps is None:
        raise SystemExit("Recorded velocity trials require --trial-steps")
    if args.benchmark:
        benchmark(policy)

    recorder = None
    if args.record is not None:
        command_size = 48 if policy.is_tracking else 3
        recorder = ExperimentRecorder(
            args.record,
            metadata=build_manifest_metadata(policy, path, args),
            observation_size=policy.observation_size,
            command_size=command_size,
            overwrite=args.overwrite_record,
            checkpoint_steps=args.checkpoint_steps,
        )
    runner = RealRobotRunner(policy, args, recorder)
    runner.execute()


if __name__ == "__main__":
    main()
