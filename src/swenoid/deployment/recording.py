"""Versioned, crash-recoverable records for physical Swenoid experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np

SCHEMA_NAME = "swenoid.real_experiment"
SCHEMA_VERSION = 1

CURRENT_AMPERE_PER_RAW_UNIT = 0.00269
VOLTAGE_VOLT_PER_RAW_UNIT = 0.1

OUTCOMES = (
    "success",
    "fall",
    "safety_abort",
    "infrastructure_failure",
    "incomplete",
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Mapping[str, Any]) -> str:
    """Hash a JSON object using a stable encoding."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object and reject other top-level values."""
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_npz_atomic(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(  # pyright: ignore[reportArgumentType]
                stream,
                **payload,  # pyright: ignore[reportArgumentType]
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class RecordingPaths:
    """Files belonging to one experiment record."""

    samples: Path
    manifest: Path
    events: Path
    partial_samples: Path
    directory_layout: bool

    @classmethod
    def resolve(cls, requested: Path) -> RecordingPaths:
        requested = requested.expanduser()
        if requested.suffix == ".npz":
            return cls(
                samples=requested,
                manifest=requested.with_suffix(".json"),
                events=requested.with_suffix(".events.jsonl"),
                partial_samples=requested.with_suffix(".partial.npz"),
                directory_layout=False,
            )
        return cls(
            samples=requested / "samples.npz",
            manifest=requested / "manifest.json",
            events=requested / "events.jsonl",
            partial_samples=requested / "samples.partial.npz",
            directory_layout=True,
        )

    def existing_files(self) -> list[Path]:
        return [
            path
            for path in (
                self.samples,
                self.manifest,
                self.events,
                self.partial_samples,
            )
            if path.exists()
        ]


_STEP_SPECS: dict[str, tuple[tuple[int, ...] | None, np.dtype[Any], bool]] = {
    "sample_index": ((), np.dtype(np.int64), True),
    "policy_frame": ((), np.dtype(np.int64), True),
    "time_s": ((), np.dtype(np.float64), True),
    "time_unix_ns": ((), np.dtype(np.int64), True),
    "control_dt_s": ((), np.dtype(np.float32), False),
    "read_duration_s": ((), np.dtype(np.float32), True),
    "inference_duration_s": ((), np.dtype(np.float32), True),
    "write_duration_s": ((), np.dtype(np.float32), True),
    "cycle_duration_s": ((), np.dtype(np.float32), True),
    "deadline_missed": ((), np.dtype(np.bool_), True),
    "policy_observation": (None, np.dtype(np.float32), True),
    "policy_action": ((24,), np.dtype(np.float32), True),
    "command": (None, np.dtype(np.float32), True),
    "reference_joint_position_rad": ((24,), np.dtype(np.float32), False),
    "reference_joint_velocity_rad_s": ((24,), np.dtype(np.float32), False),
    "joint_position_rad": ((24,), np.dtype(np.float32), True),
    "joint_velocity_rad_s": ((24,), np.dtype(np.float32), True),
    "joint_target_requested_rad": ((24,), np.dtype(np.float32), True),
    "joint_target_sent_rad": ((24,), np.dtype(np.float32), True),
    "dynamixel_position_raw": ((24,), np.dtype(np.int32), True),
    "dynamixel_velocity_raw": ((24,), np.dtype(np.int32), True),
    "dynamixel_goal_position_raw": ((24,), np.dtype(np.int32), True),
    "dynamixel_goal_clipped": ((24,), np.dtype(np.bool_), True),
    "base_angular_velocity_raw_rad_s": ((3,), np.dtype(np.float32), True),
    "base_angular_velocity_rad_s": ((3,), np.dtype(np.float32), True),
    "projected_gravity_raw": ((3,), np.dtype(np.float32), True),
    "projected_gravity": ((3,), np.dtype(np.float32), True),
    "base_orientation_wxyz": ((4,), np.dtype(np.float32), True),
    "imu_sample_time_s": ((), np.dtype(np.float64), True),
    "imu_age_s": ((), np.dtype(np.float32), True),
    "imu_sequence": ((), np.dtype(np.int64), True),
}

_TELEMETRY_SPECS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "servo_telemetry_sample_index": ((), np.dtype(np.int64)),
    "servo_telemetry_time_s": ((), np.dtype(np.float64)),
    "servo_current_raw": ((24,), np.dtype(np.int16)),
    "servo_current_a": ((24,), np.dtype(np.float32)),
    "servo_voltage_raw": ((24,), np.dtype(np.uint16)),
    "servo_voltage_v": ((24,), np.dtype(np.float32)),
    "servo_temperature_c": ((24,), np.dtype(np.float32)),
}

_EXTERNAL_EMPTY: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "external_pose_receive_time_s": ((), np.dtype(np.float64)),
    "external_pose_receive_unix_ns": ((), np.dtype(np.int64)),
    "external_pose_source_time_ns": ((), np.dtype(np.int64)),
    "external_pose_sequence": ((), np.dtype(np.int64)),
    "external_base_position_world_m": ((3,), np.dtype(np.float32)),
    "external_base_orientation_world_wxyz": ((4,), np.dtype(np.float32)),
    "external_base_linear_velocity_world_m_s": ((3,), np.dtype(np.float32)),
    "external_base_angular_velocity_world_rad_s": ((3,), np.dtype(np.float32)),
    "external_pose_valid": ((), np.dtype(np.bool_)),
    "external_pose_quality": ((), np.dtype(np.float32)),
    "external_pose_frame_id": ((), np.dtype("<U128")),
}

EXTERNAL_POSE_DEFAULT_MAX_CLOCK_OFFSET_S = 1.0


class ExperimentRecorder:
    """Collect aligned samples and maintain a recoverable trial manifest."""

    def __init__(
        self,
        requested_path: Path,
        *,
        metadata: Mapping[str, Any],
        observation_size: int,
        command_size: int,
        overwrite: bool = False,
        checkpoint_steps: int = 250,
    ) -> None:
        if observation_size <= 0 or command_size <= 0:
            raise ValueError("observation_size and command_size must be positive")
        if checkpoint_steps <= 0:
            raise ValueError("checkpoint_steps must be positive")
        self.paths = RecordingPaths.resolve(requested_path)
        existing = self.paths.existing_files()
        if existing and not overwrite:
            names = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Refusing to overwrite an existing experiment record: {names}"
            )
        self.paths.samples.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for path in existing:
                path.unlink()

        self.observation_size = observation_size
        self.command_size = command_size
        self.checkpoint_steps = checkpoint_steps
        self._step_samples = {name: [] for name in _STEP_SPECS}
        self._telemetry_samples = {name: [] for name in _TELEMETRY_SPECS}
        self._external_samples = {name: [] for name in _EXTERNAL_EMPTY}
        self._lock = threading.Lock()
        self._checkpoint_thread: threading.Thread | None = None
        self._checkpoint_error: BaseException | None = None
        self._started = False
        self._finalized = False
        self.start_monotonic_ns: int | None = None
        self.start_unix_ns: int | None = None

        self.manifest: dict[str, Any] = {
            "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
            **dict(metadata),
            "recording": {
                "status": "initializing",
                "samples_file": self.paths.samples.name,
                "events_file": self.paths.events.name,
                "checkpoint_file": self.paths.partial_samples.name,
                "created_at_utc": utc_now(),
                "sample_count": 0,
                "servo_telemetry_count": 0,
                "external_pose_count": 0,
            },
        }
        self.manifest.setdefault(
            "result",
            {
                "annotation_status": "unreviewed",
                "outcome": None,
                "success": None,
                "fall": None,
                "external_intervention": None,
                "valid_for_analysis": None,
                "exclusion_reason": None,
            },
        )
        self.paths.events.write_text("", encoding="utf-8")
        _write_json_atomic(self.paths.manifest, self.manifest)

    @property
    def sample_count(self) -> int:
        return len(self._step_samples["sample_index"])

    @property
    def external_pose_count(self) -> int:
        return len(self._external_samples["external_pose_sequence"])

    def set_metadata(self, section: str, value: Mapping[str, Any]) -> None:
        """Set one manifest section before the policy starts."""
        if self._started:
            raise RuntimeError("Experiment metadata cannot change after start")
        self.manifest[section] = dict(value)
        _write_json_atomic(self.paths.manifest, self.manifest)

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Experiment recorder already started")
        self.start_monotonic_ns = time.perf_counter_ns()
        self.start_unix_ns = time.time_ns()
        self._started = True
        recording = self.manifest["recording"]
        recording["status"] = "recording"
        recording["started_at_utc"] = utc_now()
        recording["start_unix_ns"] = self.start_unix_ns
        _write_json_atomic(self.paths.manifest, self.manifest)
        self.event("trial_started", sample_index=-1)

    def elapsed_s(self, monotonic_ns: int | None = None) -> float:
        if self.start_monotonic_ns is None:
            raise RuntimeError("Experiment recorder has not started")
        now = time.perf_counter_ns() if monotonic_ns is None else monotonic_ns
        return (now - self.start_monotonic_ns) * 1e-9

    def event(
        self,
        name: str,
        *,
        sample_index: int,
        detail: Mapping[str, Any] | None = None,
        monotonic_ns: int | None = None,
    ) -> None:
        """Append one immediately durable event."""
        record = {
            "time_s": (
                self.elapsed_s(monotonic_ns)
                if self.start_monotonic_ns is not None
                else None
            ),
            "time_unix_ns": time.time_ns(),
            "sample_index": int(sample_index),
            "name": str(name),
            "detail": {} if detail is None else dict(detail),
        }
        with self.paths.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def append_step(self, values: Mapping[str, Any]) -> None:
        """Validate and append one completely executed control step."""
        missing = set(_STEP_SPECS) - set(values)
        extra = set(values) - set(_STEP_SPECS)
        if missing or extra:
            raise ValueError(
                f"Invalid step fields; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        converted: dict[str, np.ndarray] = {}
        for name, (shape, dtype, finite) in _STEP_SPECS.items():
            array = np.asarray(values[name], dtype=dtype)
            expected_shape = shape
            if name == "policy_observation":
                expected_shape = (self.observation_size,)
            elif name == "command":
                expected_shape = (self.command_size,)
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {array.shape}; expected {expected_shape}"
                )
            if finite and not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
            converted[name] = array.copy()

        expected_index = self.sample_count
        if int(converted["sample_index"]) != expected_index:
            raise ValueError(
                f"sample_index must be {expected_index}, got "
                f"{int(converted['sample_index'])}"
            )
        if expected_index and float(converted["time_s"]) <= float(
            self._step_samples["time_s"][-1]
        ):
            raise ValueError("time_s must increase strictly")

        with self._lock:
            for name, value in converted.items():
                self._step_samples[name].append(value)
        if self.sample_count % self.checkpoint_steps == 0:
            self._start_checkpoint()

    def append_servo_telemetry(self, values: Mapping[str, Any]) -> None:
        """Append a timestamped, motor-order electrical telemetry sample."""
        missing = set(_TELEMETRY_SPECS) - set(values)
        extra = set(values) - set(_TELEMETRY_SPECS)
        if missing or extra:
            raise ValueError(
                "Invalid telemetry fields; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        converted: dict[str, np.ndarray] = {}
        for name, (shape, dtype) in _TELEMETRY_SPECS.items():
            array = np.asarray(values[name], dtype=dtype)
            if array.shape != shape:
                raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
            converted[name] = array.copy()
        with self._lock:
            for name, value in converted.items():
                self._telemetry_samples[name].append(value)

    def append_external_pose_packet(
        self,
        receive_monotonic_ns: int,
        receive_unix_ns: int,
        packet: ExternalPosePacket,
    ) -> None:
        """Append one accepted external-pose packet to the durable schema."""
        if not self._started or self.start_monotonic_ns is None:
            raise RuntimeError("Experiment recorder has not started")
        if self._finalized:
            raise RuntimeError("Experiment recorder has already finalized")
        receive_monotonic_ns = int(receive_monotonic_ns)
        receive_unix_ns = int(receive_unix_ns)
        if receive_monotonic_ns < self.start_monotonic_ns:
            raise ValueError("External pose was received before the trial started")
        if receive_unix_ns <= 0:
            raise ValueError("External pose receive Unix time must be positive")
        if len(packet.frame_id) > 128:
            raise ValueError("External pose frame_id exceeds 128 characters")

        values: dict[str, Any] = {
            "external_pose_receive_time_s": self.elapsed_s(receive_monotonic_ns),
            "external_pose_receive_unix_ns": receive_unix_ns,
            "external_pose_source_time_ns": packet.source_time_ns,
            "external_pose_sequence": packet.sequence,
            "external_base_position_world_m": packet.position_m,
            "external_base_orientation_world_wxyz": packet.orientation_wxyz,
            "external_base_linear_velocity_world_m_s": packet.linear_velocity_m_s,
            "external_base_angular_velocity_world_rad_s": packet.angular_velocity_rad_s,
            "external_pose_valid": packet.valid,
            "external_pose_quality": packet.quality,
            "external_pose_frame_id": packet.frame_id,
        }
        converted: dict[str, np.ndarray] = {}
        for name, (shape, dtype) in _EXTERNAL_EMPTY.items():
            array = np.asarray(values[name], dtype=dtype)
            if array.shape != shape:
                raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
            converted[name] = array.copy()

        if not math.isfinite(float(converted["external_pose_receive_time_s"])):
            raise ValueError("External pose receive time must be finite")
        if not math.isfinite(float(converted["external_pose_quality"])) or not (
            0.0 <= float(converted["external_pose_quality"]) <= 1.0
        ):
            raise ValueError("External pose quality must be in [0, 1]")
        valid = bool(converted["external_pose_valid"])
        position = converted["external_base_position_world_m"]
        orientation = converted["external_base_orientation_world_wxyz"]
        if valid:
            if not np.isfinite(position).all() or not np.isfinite(orientation).all():
                raise ValueError("A valid external pose must have finite pose values")
            if abs(float(np.linalg.norm(orientation)) - 1.0) > 0.01:
                raise ValueError("External pose quaternion must be normalized")
        for name in (
            "external_base_linear_velocity_world_m_s",
            "external_base_angular_velocity_world_rad_s",
        ):
            array = converted[name]
            if not (np.isfinite(array).all() or np.isnan(array).all()):
                raise ValueError(f"{name} must be finite or entirely NaN")

        with self._lock:
            if self._external_samples["external_pose_sequence"]:
                previous_sequence = int(
                    self._external_samples["external_pose_sequence"][-1]
                )
                previous_source_time = int(
                    self._external_samples["external_pose_source_time_ns"][-1]
                )
                previous_frame = str(
                    self._external_samples["external_pose_frame_id"][-1]
                )
                if int(converted["external_pose_sequence"]) <= previous_sequence:
                    raise ValueError("External pose sequence must increase strictly")
                if (
                    int(converted["external_pose_source_time_ns"])
                    <= previous_source_time
                ):
                    raise ValueError("External pose source time must increase strictly")
                if str(converted["external_pose_frame_id"]) != previous_frame:
                    raise ValueError("External pose frame_id changed within the trial")
            for name, value in converted.items():
                self._external_samples[name].append(value)

    def _empty_array(self, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
        return np.empty((0, *shape), dtype=dtype)

    def _payload(self) -> dict[str, np.ndarray]:
        with self._lock:
            step_copy = {
                name: list(values) for name, values in self._step_samples.items()
            }
            telemetry_copy = {
                name: list(values) for name, values in self._telemetry_samples.items()
            }
            external_copy = {
                name: list(values) for name, values in self._external_samples.items()
            }

        payload: dict[str, np.ndarray] = {
            "schema_name": np.asarray(SCHEMA_NAME),
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        }
        for name, values in step_copy.items():
            shape, dtype, _ = _STEP_SPECS[name]
            if name == "policy_observation":
                shape = (self.observation_size,)
            elif name == "command":
                shape = (self.command_size,)
            assert shape is not None
            payload[name] = (
                np.stack(values).astype(dtype, copy=False)
                if values
                else self._empty_array(shape, dtype)
            )
        for name, values in telemetry_copy.items():
            shape, dtype = _TELEMETRY_SPECS[name]
            payload[name] = (
                np.stack(values).astype(dtype, copy=False)
                if values
                else self._empty_array(shape, dtype)
            )
        for name, values in external_copy.items():
            shape, dtype = _EXTERNAL_EMPTY[name]
            payload[name] = (
                np.stack(values).astype(dtype, copy=False)
                if values
                else self._empty_array(shape, dtype)
            )

        # Read-only compatibility aliases. Canonical analysis must use the
        # unit-bearing names above; there is intentionally no ambiguous
        # ``commanded_joint_pos`` alias.
        payload["joint_pos"] = payload["joint_position_rad"]
        payload["joint_vel"] = payload["joint_velocity_rad_s"]
        payload["ang_vel"] = payload["base_angular_velocity_rad_s"]
        payload["gravity"] = payload["projected_gravity"]
        return payload

    def _checkpoint_worker(self) -> None:
        try:
            _write_npz_atomic(self.paths.partial_samples, self._payload())
        except BaseException as exc:
            self._checkpoint_error = exc

    def _start_checkpoint(self) -> None:
        if self._checkpoint_thread is not None and self._checkpoint_thread.is_alive():
            return
        self._checkpoint_thread = threading.Thread(
            target=self._checkpoint_worker,
            daemon=True,
        )
        self._checkpoint_thread.start()

    def finalize(
        self,
        *,
        termination_reason: str,
        external_arrays: Mapping[str, np.ndarray] | None = None,
        external_metadata: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Atomically write final samples and terminal metadata."""
        if self._finalized:
            return
        if self._checkpoint_thread is not None:
            self._checkpoint_thread.join()
        if self._started:
            self.event(
                "trial_stopped",
                sample_index=self.sample_count - 1,
                detail={"termination_reason": termination_reason},
            )

        payload = self._payload()
        external_count = self.external_pose_count
        if external_arrays is not None:
            if external_count:
                raise ValueError(
                    "Cannot combine callback-recorded and legacy external-pose arrays"
                )
            provided_external = dict(external_arrays)
            missing = set(_EXTERNAL_EMPTY) - set(provided_external)
            unknown = set(provided_external) - set(_EXTERNAL_EMPTY)
            if missing or unknown:
                raise ValueError(
                    "Invalid external-pose arrays; "
                    f"missing={sorted(missing)}, extra={sorted(unknown)}"
                )
            lengths: set[int] = set()
            for name, (shape, dtype) in _EXTERNAL_EMPTY.items():
                array = np.asarray(provided_external[name], dtype=dtype)
                if array.ndim != len(shape) + 1 or array.shape[1:] != shape:
                    raise ValueError(
                        f"{name} has shape {array.shape}; expected (M, {shape})"
                    )
                lengths.add(array.shape[0])
                payload[name] = array
            if len(lengths) != 1:
                raise ValueError("External-pose arrays must have the same sample count")
            external_count = next(iter(lengths))

        _write_npz_atomic(self.paths.samples, payload)
        self.paths.partial_samples.unlink(missing_ok=True)

        recording = self.manifest["recording"]
        recording.update(
            {
                "status": "finalized",
                "finalized_at_utc": utc_now(),
                "termination_reason": termination_reason,
                "sample_count": self.sample_count,
                "servo_telemetry_count": len(
                    self._telemetry_samples["servo_telemetry_sample_index"]
                ),
                "external_pose_count": external_count,
            }
        )
        if self._checkpoint_error is not None:
            recording["checkpoint_error"] = {
                "type": type(self._checkpoint_error).__name__,
                "message": str(self._checkpoint_error),
            }
        if external_metadata is not None:
            self.manifest["external_pose_recording"] = dict(external_metadata)
        if error is not None:
            recording["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        _write_json_atomic(self.paths.manifest, self.manifest)
        self._finalized = True


@dataclass(frozen=True)
class ExternalPosePacket:
    """One validated pose datagram from an external tracking system."""

    source_time_ns: int
    sequence: int
    position_m: np.ndarray
    orientation_wxyz: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray
    valid: bool
    quality: float
    frame_id: str

    @classmethod
    def parse(cls, data: bytes) -> ExternalPosePacket:
        try:
            value = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("External pose packet is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("External pose packet must be a JSON object")
        try:
            raw_source_time = value["source_time_ns"]
            raw_sequence = value["sequence"]
            raw_valid = value["valid"]
            raw_frame_id = value["frame_id"]
            if (
                not isinstance(raw_source_time, int)
                or isinstance(raw_source_time, bool)
                or not isinstance(raw_sequence, int)
                or isinstance(raw_sequence, bool)
            ):
                raise TypeError("timestamps and sequence must be JSON integers")
            if not isinstance(raw_valid, bool):
                raise TypeError("valid must be a JSON boolean")
            if not isinstance(raw_frame_id, str):
                raise TypeError("frame_id must be a JSON string")
            source_time_ns = raw_source_time
            sequence = raw_sequence
            valid = raw_valid
            quality = float(value.get("quality", 1.0 if valid else 0.0))
            frame_id = raw_frame_id
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "External pose requires integer source_time_ns and sequence, "
                "boolean valid, and string frame_id"
            ) from exc
        if source_time_ns <= 0 or sequence < 0:
            raise ValueError(
                "External pose timestamps and sequence must be non-negative"
            )
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("External pose quality must be in [0, 1]")
        if not frame_id:
            raise ValueError("External pose frame_id cannot be empty")

        def vector(name: str, width: int, *, optional: bool = False) -> np.ndarray:
            raw = value.get(name)
            if raw is None and optional:
                return np.full(width, np.nan, dtype=np.float32)
            try:
                array = np.asarray(raw, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"External pose {name} must be numeric") from exc
            if array.shape != (width,):
                raise ValueError(f"External pose {name} must have {width} values")
            return array

        position = vector("position_m", 3, optional=not valid)
        orientation = vector("orientation_wxyz", 4, optional=not valid)
        linear_velocity = vector("linear_velocity_m_s", 3, optional=True)
        angular_velocity = vector("angular_velocity_rad_s", 3, optional=True)
        if valid:
            if not np.isfinite(position).all() or not np.isfinite(orientation).all():
                raise ValueError(
                    "A valid external pose must contain finite pose values"
                )
            norm = float(np.linalg.norm(orientation))
            if not math.isfinite(norm) or abs(norm - 1.0) > 0.01:
                raise ValueError("External pose quaternion must be normalized")
            orientation = orientation / norm
        for name, array in (
            ("linear_velocity_m_s", linear_velocity),
            ("angular_velocity_rad_s", angular_velocity),
        ):
            if not (np.isfinite(array).all() or np.isnan(array).all()):
                raise ValueError(f"External pose {name} must be finite or omitted")
        return cls(
            source_time_ns=source_time_ns,
            sequence=sequence,
            position_m=position,
            orientation_wxyz=orientation,
            linear_velocity_m_s=linear_velocity,
            angular_velocity_rad_s=angular_velocity,
            valid=valid,
            quality=quality,
            frame_id=frame_id,
        )


class ExternalPoseReceiver:
    """Receive validated, timestamped 6-DoF ground truth asynchronously."""

    def __init__(
        self,
        *,
        bind: str,
        port: int,
        on_packet: Callable[[int, int, ExternalPosePacket], None] | None = None,
        max_clock_offset_s: float = EXTERNAL_POSE_DEFAULT_MAX_CLOCK_OFFSET_S,
    ) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("External pose UDP port must be in [1, 65535]")
        if not math.isfinite(max_clock_offset_s) or max_clock_offset_s <= 0.0:
            raise ValueError("External pose clock-offset limit must be positive")
        self.bind = bind
        self.port = port
        self.on_packet = on_packet
        self.max_clock_offset_s = float(max_clock_offset_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._rows: list[tuple[int, int, ExternalPosePacket]] = []
        self._errors: list[str] = []
        self._dropped_packets = 0
        self._last_source_time_ns = -1
        self._last_sequence = -1
        self._frame_id: str | None = None
        self._fatal_error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("External pose receiver already started")
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(0.2)
        self._socket.bind((self.bind, self.port))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _record_rejection(self, error: ValueError) -> None:
        with self._condition:
            self._dropped_packets += 1
            if len(self._errors) < 20:
                self._errors.append(str(error))
            self._condition.notify_all()

    def _set_fatal(self, error: BaseException) -> None:
        with self._condition:
            if self._fatal_error is None:
                self._fatal_error = error
            self._condition.notify_all()
        self._stop.set()

    def _run(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                data, _ = self._socket.recvfrom(65_535)
            except TimeoutError:
                continue
            except OSError as exc:
                if self._stop.is_set():
                    break
                self._set_fatal(exc)
                return
            receive_monotonic_ns = time.perf_counter_ns()
            receive_unix_ns = time.time_ns()
            try:
                packet = ExternalPosePacket.parse(data)
                if packet.source_time_ns < 946_684_800_000_000_000:
                    raise ValueError(
                        "External pose source_time_ns is not Unix epoch time"
                    )
                clock_offset_s = abs(packet.source_time_ns - receive_unix_ns) * 1e-9
                if clock_offset_s > self.max_clock_offset_s:
                    raise ValueError(
                        "External pose source/receive clock offset "
                        f"{clock_offset_s:.6f}s exceeds {self.max_clock_offset_s:.6f}s"
                    )
                if (
                    packet.source_time_ns <= self._last_source_time_ns
                    or packet.sequence <= self._last_sequence
                ):
                    raise ValueError("Out-of-order external pose packet")
                if self._frame_id is not None and packet.frame_id != self._frame_id:
                    raise ValueError(
                        f"External pose frame_id changed from {self._frame_id!r} "
                        f"to {packet.frame_id!r}"
                    )
            except ValueError as exc:
                self._record_rejection(exc)
                continue

            if self.on_packet is not None:
                try:
                    self.on_packet(receive_monotonic_ns, receive_unix_ns, packet)
                except BaseException as exc:
                    self._set_fatal(exc)
                    return

            with self._condition:
                self._last_source_time_ns = packet.source_time_ns
                self._last_sequence = packet.sequence
                if self._frame_id is None:
                    self._frame_id = packet.frame_id
                self._rows.append((receive_monotonic_ns, receive_unix_ns, packet))
                self._condition.notify_all()

    def stop(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @staticmethod
    def _usable(packet: ExternalPosePacket) -> bool:
        return packet.valid and packet.quality > 0.0

    def wait_for_valid_samples(self, minimum: int, timeout_s: float) -> bool:
        """Wait for ``minimum`` usable packets, returning false on timeout."""
        if minimum <= 0:
            raise ValueError("minimum must be positive")
        if not math.isfinite(timeout_s) or timeout_s < 0.0:
            raise ValueError("timeout_s must be finite and non-negative")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._fatal_error is not None:
                    error = self._fatal_error
                    raise RuntimeError("External pose receiver failed") from error
                count = sum(self._usable(packet) for _, _, packet in self._rows)
                if count >= minimum:
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)

    def wait_for_valid_after(self, monotonic_ns: int, timeout_s: float) -> bool:
        """Wait for a usable packet received at or after a monotonic deadline."""
        if not math.isfinite(timeout_s) or timeout_s < 0.0:
            raise ValueError("timeout_s must be finite and non-negative")
        threshold = int(monotonic_ns)
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._fatal_error is not None:
                    error = self._fatal_error
                    raise RuntimeError("External pose receiver failed") from error
                if any(
                    receive_ns >= threshold and self._usable(packet)
                    for receive_ns, _, packet in self._rows
                ):
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)

    def wait_for_valid_source_after(
        self,
        source_time_ns: int,
        timeout_s: float,
        *,
        receive_monotonic_ns: int | None = None,
    ) -> bool:
        """Wait for one usable row acquired after the supplied source deadline."""
        if int(source_time_ns) <= 0:
            raise ValueError("source_time_ns must be positive")
        if not math.isfinite(timeout_s) or timeout_s < 0.0:
            raise ValueError("timeout_s must be finite and non-negative")
        source_threshold = int(source_time_ns)
        receive_threshold = (
            None if receive_monotonic_ns is None else int(receive_monotonic_ns)
        )
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._fatal_error is not None:
                    error = self._fatal_error
                    raise RuntimeError("External pose receiver failed") from error
                if any(
                    packet.source_time_ns >= source_threshold
                    and (receive_threshold is None or receive_ns >= receive_threshold)
                    and self._usable(packet)
                    for receive_ns, _, packet in self._rows
                ):
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)

    def raise_if_failed(self) -> None:
        """Raise a controller-thread-visible error after a receiver failure."""
        with self._lock:
            error = self._fatal_error
        if error is not None:
            raise RuntimeError("External pose receiver failed") from error

    def arrays(self, *, trial_start_monotonic_ns: int) -> dict[str, np.ndarray]:
        """Return received rows for compatibility with schema-v1 callers."""
        with self._lock:
            rows = list(self._rows)
        if not rows:
            return {
                name: np.empty((0, *shape), dtype=dtype)
                for name, (shape, dtype) in _EXTERNAL_EMPTY.items()
            }
        return {
            "external_pose_receive_time_s": np.asarray(
                [
                    (receive_ns - trial_start_monotonic_ns) * 1e-9
                    for receive_ns, _, _ in rows
                ],
                dtype=np.float64,
            ),
            "external_pose_receive_unix_ns": np.asarray(
                [unix_ns for _, unix_ns, _ in rows], dtype=np.int64
            ),
            "external_pose_source_time_ns": np.asarray(
                [packet.source_time_ns for _, _, packet in rows], dtype=np.int64
            ),
            "external_pose_sequence": np.asarray(
                [packet.sequence for _, _, packet in rows], dtype=np.int64
            ),
            "external_base_position_world_m": np.stack(
                [packet.position_m for _, _, packet in rows]
            ).astype(np.float32),
            "external_base_orientation_world_wxyz": np.stack(
                [packet.orientation_wxyz for _, _, packet in rows]
            ).astype(np.float32),
            "external_base_linear_velocity_world_m_s": np.stack(
                [packet.linear_velocity_m_s for _, _, packet in rows]
            ).astype(np.float32),
            "external_base_angular_velocity_world_rad_s": np.stack(
                [packet.angular_velocity_rad_s for _, _, packet in rows]
            ).astype(np.float32),
            "external_pose_valid": np.asarray(
                [packet.valid for _, _, packet in rows], dtype=np.bool_
            ),
            "external_pose_quality": np.asarray(
                [packet.quality for _, _, packet in rows], dtype=np.float32
            ),
            "external_pose_frame_id": np.asarray(
                [packet.frame_id for _, _, packet in rows], dtype="<U128"
            ),
        }

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            fatal = self._fatal_error
            return {
                "transport": "udp_json",
                "bind": self.bind,
                "port": self.port,
                "received_packets": len(self._rows),
                "dropped_packets": self._dropped_packets,
                "errors": list(self._errors),
                "frame_id": self._frame_id,
                "max_source_receive_clock_offset_s": self.max_clock_offset_s,
                "fatal_error": (
                    None
                    if fatal is None
                    else {"type": type(fatal).__name__, "message": str(fatal)}
                ),
            }


def _manifest_path(path: Path) -> tuple[RecordingPaths, Path]:
    paths = RecordingPaths.resolve(path)
    if path.is_dir():
        return paths, paths.manifest
    if path.name in {"manifest.json", "samples.npz"}:
        paths = RecordingPaths.resolve(path.parent)
        return paths, paths.manifest
    if path.suffix == ".json" and path.name.endswith(".events.json"):
        raise ValueError(f"Cannot infer a manifest from {path}")
    return paths, paths.manifest


def annotate_experiment(
    path: Path,
    *,
    outcome: str,
    external_intervention: bool,
    exclusion_reason: str | None,
    notes: str | None,
) -> Path:
    """Attach the human-observed outcome required for paper statistics."""
    if outcome not in OUTCOMES:
        raise ValueError(f"Unknown outcome {outcome!r}")
    paths, manifest_path = _manifest_path(path)
    manifest = load_json_object(manifest_path)
    if manifest.get("schema") != {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}:
        raise ValueError(f"Unsupported experiment manifest: {manifest_path}")
    if outcome == "infrastructure_failure" and not exclusion_reason:
        raise ValueError("Infrastructure failures require --exclusion-reason")
    if outcome == "success" and external_intervention:
        raise ValueError("A successful trial cannot include external intervention")
    result = manifest.setdefault("result", {})
    history = result.setdefault("annotation_history", [])
    if result.get("annotation_status") == "reviewed":
        history.append(
            {key: value for key, value in result.items() if key != "annotation_history"}
        )
    valid_for_analysis = outcome not in {"infrastructure_failure", "incomplete"}
    result.update(
        {
            "annotation_status": "reviewed",
            "annotated_at_utc": utc_now(),
            "outcome": outcome,
            "success": outcome == "success",
            "fall": outcome == "fall",
            "external_intervention": external_intervention,
            "valid_for_analysis": valid_for_analysis,
            "exclusion_reason": exclusion_reason,
            "notes": notes,
        }
    )
    _write_json_atomic(manifest_path, manifest)
    event = {
        "time_s": None,
        "time_unix_ns": time.time_ns(),
        "sample_index": -1,
        "name": "result_annotated",
        "detail": {
            "outcome": outcome,
            "external_intervention": external_intervention,
            "exclusion_reason": exclusion_reason,
        },
    }
    with paths.events.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return manifest_path


def _nested_value(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _require_context_string(
    context: Mapping[str, Any], errors: list[str], *keys: str
) -> str | None:
    value = _nested_value(context, *keys)
    label = ".".join(keys)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"experiment context is missing {label}")
        return None
    return value.strip()


def _paper_protocol_metadata(
    manifest: Mapping[str, Any], task: Any, errors: list[str]
) -> tuple[float, float, int, str | None]:
    trial = manifest.get("trial")
    context_value = (
        trial.get("experiment_context") if isinstance(trial, Mapping) else None
    )
    if not isinstance(context_value, Mapping):
        errors.append("trial metadata is missing experiment_context")
        return (0.0, 0.0, 0, None)
    context = context_value

    model = _require_context_string(
        context, errors, "condition_details", "simulator_actuator_model"
    )
    _require_context_string(context, errors, "condition_details", "training_run")
    for path in (
        ("protocol", "success_criterion"),
        ("protocol", "evaluation_window"),
        ("protocol", "infrastructure_exclusion_rule"),
        ("environment", "surface"),
        ("robot_configuration", "battery_pack_id"),
        ("trial_design", "block_id"),
        ("trial_design", "command_sequence_id"),
        ("trial_design", "order_assignment"),
    ):
        _require_context_string(context, errors, *path)
    for name in (
        "tethered",
        "spotter_present",
        "external_support_allowed_after_start",
    ):
        if not isinstance(_nested_value(context, "safety", name), bool):
            errors.append(f"experiment context safety.{name} must be boolean")

    parameter_files = _nested_value(
        context, "condition_details", "actuator_parameter_files"
    )
    if not isinstance(parameter_files, list):
        errors.append(
            "experiment context condition_details.actuator_parameter_files "
            "must be a list"
        )
        parameter_files = []
    condition = _nested_value(manifest, "trial", "condition")
    identity = f"{condition or ''} {model or ''}".lower()
    identified = any(
        token in identity for token in ("identified", "bam", "sysid", "system-id")
    )
    if identified and not parameter_files:
        errors.append("identified conditions require actuator parameter files")
    for index, entry in enumerate(parameter_files):
        label = f"actuator_parameter_files[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{label} must be an object with an identifier and SHA-256")
            continue
        identifier = entry.get("path") or entry.get("identifier")
        if not isinstance(identifier, str) or not identifier.strip():
            errors.append(f"{label} is missing path or identifier")
        digest = entry.get("sha256")
        if not (
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdefABCDEF" for character in digest)
        ):
            errors.append(f"{label} has an invalid SHA-256")

    if task != "velocity":
        return (0.0, 0.0, 0, None)

    world_frame = _require_context_string(
        context, errors, "external_tracking", "world_frame"
    )
    for name in (
        "system",
        "rigid_body",
        "calibration_id",
        "clock_sync_method",
        "quality_definition",
    ):
        _require_context_string(context, errors, "external_tracking", name)
    minimum_quality = _nested_value(context, "external_tracking", "minimum_quality")
    if (
        isinstance(minimum_quality, bool)
        or not isinstance(minimum_quality, (int, float))
        or not math.isfinite(float(minimum_quality))
        or not 0.0 <= float(minimum_quality) <= 1.0
    ):
        errors.append(
            "experiment context external_tracking.minimum_quality must be in [0, 1]"
        )
        minimum_quality_value = 0.0
    else:
        minimum_quality_value = float(minimum_quality)
    clock_offset = _nested_value(
        context, "external_tracking", "estimated_clock_offset_ms"
    )
    if (
        isinstance(clock_offset, bool)
        or not isinstance(clock_offset, (int, float))
        or not math.isfinite(float(clock_offset))
    ):
        errors.append(
            "experiment context external_tracking.estimated_clock_offset_ms "
            "must be finite"
        )
        clock_offset_ns = 0
    else:
        clock_offset_ns = round(float(clock_offset) * 1e6)
    maximum_gap = _nested_value(context, "external_tracking", "maximum_gap_ms")
    if (
        isinstance(maximum_gap, bool)
        or not isinstance(maximum_gap, (int, float))
        or not math.isfinite(float(maximum_gap))
        or float(maximum_gap) <= 0.0
    ):
        errors.append(
            "experiment context external_tracking.maximum_gap_ms must be positive"
        )
        maximum_gap_s = 0.0
    else:
        maximum_gap_s = float(maximum_gap) * 1e-3
    return minimum_quality_value, maximum_gap_s, clock_offset_ns, world_frame


def _manifest_count(
    recording: Mapping[str, Any], name: str, errors: list[str]
) -> int | None:
    value = recording.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"manifest recording.{name} must be a non-negative integer")
        return None
    return value


def _dtype_matches(array: np.ndarray, expected: np.dtype[Any]) -> bool:
    if expected.kind == "U":
        return array.dtype.kind == "U"
    return array.dtype == expected


def _valid_hex(value: Any, lengths: set[int]) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def validate_experiment(path: Path, *, paper_ready: bool) -> list[str]:
    """Return validation failures for a finalized experiment record."""
    paths, manifest_path = _manifest_path(path)
    errors: list[str] = []
    try:
        manifest = load_json_object(manifest_path)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    if manifest.get("schema") != {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}:
        errors.append("manifest schema is not swenoid.real_experiment version 1")

    recording_value = manifest.get("recording")
    if not isinstance(recording_value, Mapping):
        errors.append("manifest recording section is missing or invalid")
        recording: Mapping[str, Any] = {}
    else:
        recording = recording_value
    if recording.get("status") != "finalized":
        errors.append("recording is not finalized")
    manifest_step_count = _manifest_count(recording, "sample_count", errors)
    manifest_telemetry_count = _manifest_count(
        recording, "servo_telemetry_count", errors
    )
    manifest_external_count = _manifest_count(recording, "external_pose_count", errors)

    policy_value = manifest.get("policy")
    policy = policy_value if isinstance(policy_value, Mapping) else {}
    task = policy.get("policy_kind")
    if task not in {"velocity", "single_motion", "general_motion"}:
        errors.append("manifest policy_kind is unsupported")
    observation_size = policy.get("observation_size")
    if (
        isinstance(observation_size, bool)
        or not isinstance(observation_size, int)
        or observation_size <= 0
    ):
        errors.append("manifest policy observation_size must be positive")
        observation_size = None
    command_size = 3 if task == "velocity" else 48

    minimum_quality, maximum_gap_s, clock_offset_ns, declared_world_frame = (
        _paper_protocol_metadata(manifest, task, errors)
        if paper_ready
        else (
            0.0,
            math.inf,
            0,
            _nested_value(
                manifest,
                "trial",
                "experiment_context",
                "external_tracking",
                "world_frame",
            ),
        )
    )

    if not paths.samples.is_file():
        errors.append(f"samples file is missing: {paths.samples}")
        return errors

    try:
        archive = np.load(paths.samples, allow_pickle=False)
    except (OSError, ValueError) as exc:
        errors.append(f"cannot load samples without pickle: {exc}")
        return errors

    with archive:
        required = (
            {"schema_name", "schema_version"}
            | set(_STEP_SPECS)
            | set(_TELEMETRY_SPECS)
            | set(_EXTERNAL_EMPTY)
        )
        missing = required - set(archive.files)
        if missing:
            errors.append(f"samples are missing fields: {', '.join(sorted(missing))}")

        arrays: dict[str, np.ndarray] = {}
        for name in required & set(archive.files):
            try:
                arrays[name] = np.asarray(archive[name])
            except (OSError, TypeError, ValueError) as exc:
                errors.append(f"cannot read samples field {name}: {exc}")

        schema_name = arrays.get("schema_name")
        if schema_name is not None:
            if schema_name.shape != () or schema_name.dtype.kind != "U":
                errors.append("sample archive schema_name must be a Unicode scalar")
            elif schema_name.item() != SCHEMA_NAME:
                errors.append("sample archive has the wrong schema name")
        schema_version = arrays.get("schema_version")
        if schema_version is not None:
            if schema_version.shape != () or schema_version.dtype != np.dtype(np.int64):
                errors.append("sample archive schema_version must be an int64 scalar")
            elif int(schema_version.item()) != SCHEMA_VERSION:
                errors.append("sample archive has the wrong schema version")

        sample_index = arrays.get("sample_index")
        step_count: int | None = None
        if sample_index is not None:
            if sample_index.ndim != 1:
                errors.append("sample_index must have shape (N,)")
            else:
                step_count = sample_index.shape[0]
        if (
            step_count is not None
            and manifest_step_count is not None
            and step_count != manifest_step_count
        ):
            errors.append(
                "manifest sample_count does not match the sample archive "
                f"({manifest_step_count} != {step_count})"
            )

        for name, (tail_shape, dtype, finite) in _STEP_SPECS.items():
            array = arrays.get(name)
            if array is None:
                continue
            expected_tail = tail_shape
            if name == "policy_observation" and observation_size is not None:
                expected_tail = (observation_size,)
            elif name == "command":
                expected_tail = (command_size,)
            if expected_tail is not None and (
                array.ndim != len(expected_tail) + 1 or array.shape[1:] != expected_tail
            ):
                errors.append(
                    f"{name} has shape {array.shape}; expected (N, {expected_tail})"
                )
            if step_count is not None and array.ndim >= 1 and len(array) != step_count:
                errors.append(f"{name} has a different control-sample count")
            if not _dtype_matches(array, dtype):
                errors.append(f"{name} has dtype {array.dtype}; expected {dtype}")
            if finite and not np.isfinite(array).all():
                errors.append(f"{name} contains non-finite values")

        if step_count is not None and sample_index is not None:
            if step_count and not np.array_equal(
                sample_index, np.arange(step_count, dtype=np.int64)
            ):
                errors.append("sample_index is not contiguous from zero")
            time_s = arrays.get("time_s")
            time_unix_ns = arrays.get("time_unix_ns")
            if time_s is not None and time_s.shape == (step_count,):  # noqa: SIM102
                if step_count > 1 and np.any(time_s[1:] <= time_s[:-1]):
                    errors.append("time_s is not strictly increasing")
            if time_unix_ns is not None and time_unix_ns.shape == (step_count,):
                if np.any(time_unix_ns <= 0):
                    errors.append("time_unix_ns must be positive")
                if step_count > 1 and np.any(time_unix_ns[1:] <= time_unix_ns[:-1]):
                    errors.append("time_unix_ns is not strictly increasing")
            control_dt = arrays.get("control_dt_s")
            if control_dt is not None and control_dt.shape == (step_count,):
                if step_count and not np.isnan(control_dt[0]):
                    errors.append("the first control_dt_s value must be NaN")
                if step_count > 1 and (
                    not np.isfinite(control_dt[1:]).all()
                    or np.any(control_dt[1:] <= 0.0)
                ):
                    errors.append(
                        "later control_dt_s values must be finite and positive"
                    )
            imu_time = arrays.get("imu_sample_time_s")
            imu_sequence = arrays.get("imu_sequence")
            if imu_time is not None and imu_time.shape == (step_count,):  # noqa: SIM102
                if step_count > 1 and np.any(imu_time[1:] < imu_time[:-1]):
                    errors.append("imu_sample_time_s is not monotonic")
            if imu_sequence is not None and imu_sequence.shape == (step_count,):  # noqa: SIM102
                if np.any(imu_sequence < 0) or (
                    step_count > 1 and np.any(imu_sequence[1:] < imu_sequence[:-1])
                ):
                    errors.append("imu_sequence must be non-negative and monotonic")
            for name in (
                "read_duration_s",
                "inference_duration_s",
                "write_duration_s",
                "cycle_duration_s",
                "imu_age_s",
            ):
                value = arrays.get(name)
                if (
                    value is not None
                    and value.shape == (step_count,)
                    and np.any(value < 0.0)
                ):
                    errors.append(f"{name} contains negative values")

        for name in (
            "reference_joint_position_rad",
            "reference_joint_velocity_rad_s",
        ):
            reference = arrays.get(name)
            if reference is None:
                continue
            if (
                task in {"single_motion", "general_motion"}
                and not np.isfinite(reference).all()
            ):
                errors.append(f"tracking {name} contains missing values")
            if task == "velocity" and not np.isnan(reference).all():
                errors.append(f"velocity-policy {name} must contain only NaN")

        telemetry_index = arrays.get("servo_telemetry_sample_index")
        telemetry_count: int | None = None
        if telemetry_index is not None:
            if telemetry_index.ndim != 1:
                errors.append("servo_telemetry_sample_index must have shape (S,)")
            else:
                telemetry_count = telemetry_index.shape[0]
        if (
            telemetry_count is not None
            and manifest_telemetry_count is not None
            and telemetry_count != manifest_telemetry_count
        ):
            errors.append(
                "manifest servo_telemetry_count does not match the sample archive "
                f"({manifest_telemetry_count} != {telemetry_count})"
            )
        for name, (tail_shape, dtype) in _TELEMETRY_SPECS.items():
            array = arrays.get(name)
            if array is None:
                continue
            if array.ndim != len(tail_shape) + 1 or array.shape[1:] != tail_shape:
                errors.append(
                    f"{name} has shape {array.shape}; expected (S, {tail_shape})"
                )
            if (
                telemetry_count is not None
                and array.ndim >= 1
                and len(array) != telemetry_count
            ):
                errors.append(f"{name} has a different servo-telemetry count")
            if not _dtype_matches(array, dtype):
                errors.append(f"{name} has dtype {array.dtype}; expected {dtype}")
            if not np.isfinite(array).all():
                errors.append(f"{name} contains non-finite values")
        telemetry_time = arrays.get("servo_telemetry_time_s")
        if telemetry_count is not None and telemetry_index is not None:
            if telemetry_count > 1 and np.any(
                telemetry_index[1:] <= telemetry_index[:-1]
            ):
                errors.append(
                    "servo telemetry sample indices are not strictly increasing"
                )
            if (
                step_count is not None
                and telemetry_count
                and (
                    np.any(telemetry_index < 0) or np.any(telemetry_index >= step_count)
                )
            ):
                errors.append(
                    "servo telemetry sample indices are outside control range"
                )
            if telemetry_time is not None and telemetry_time.shape == (
                telemetry_count,
            ):
                if np.any(telemetry_time < 0.0) or (
                    telemetry_count > 1
                    and np.any(telemetry_time[1:] <= telemetry_time[:-1])
                ):
                    errors.append(
                        "servo telemetry timestamps must be non-negative and increasing"
                    )
                time_s = arrays.get("time_s")
                if (
                    time_s is not None
                    and time_s.ndim == 1
                    and step_count is not None
                    and telemetry_count
                    and not np.any(telemetry_index < 0)
                    and not np.any(telemetry_index >= step_count)
                    and np.any(telemetry_time < time_s[telemetry_index])
                ):
                    errors.append(
                        "servo telemetry predates its associated control sample"
                    )

        external_sequence = arrays.get("external_pose_sequence")
        external_count: int | None = None
        if external_sequence is not None:
            if external_sequence.ndim != 1:
                errors.append("external_pose_sequence must have shape (M,)")
            else:
                external_count = external_sequence.shape[0]
        if (
            external_count is not None
            and manifest_external_count is not None
            and external_count != manifest_external_count
        ):
            errors.append(
                "manifest external_pose_count does not match the sample archive "
                f"({manifest_external_count} != {external_count})"
            )
        for name, (tail_shape, dtype) in _EXTERNAL_EMPTY.items():
            array = arrays.get(name)
            if array is None:
                continue
            if array.ndim != len(tail_shape) + 1 or array.shape[1:] != tail_shape:
                errors.append(
                    f"{name} has shape {array.shape}; expected (M, {tail_shape})"
                )
            if (
                external_count is not None
                and array.ndim >= 1
                and len(array) != external_count
            ):
                errors.append(f"{name} has a different external-pose count")
            if not _dtype_matches(array, dtype):
                errors.append(f"{name} has dtype {array.dtype}; expected {dtype}")

        usable_external = np.zeros(0, dtype=np.bool_)
        if external_count is not None:
            receive_time = arrays.get("external_pose_receive_time_s")
            receive_unix = arrays.get("external_pose_receive_unix_ns")
            source_time = arrays.get("external_pose_source_time_ns")
            valid = arrays.get("external_pose_valid")
            quality = arrays.get("external_pose_quality")
            frame_id = arrays.get("external_pose_frame_id")
            position = arrays.get("external_base_position_world_m")
            orientation = arrays.get("external_base_orientation_world_wxyz")
            linear_velocity = arrays.get("external_base_linear_velocity_world_m_s")
            angular_velocity = arrays.get("external_base_angular_velocity_world_rad_s")

            if receive_time is not None and receive_time.shape == (external_count,):
                if not np.isfinite(receive_time).all() or np.any(receive_time < 0.0):
                    errors.append(
                        "external pose receive times must be finite and non-negative"
                    )
                if external_count > 1 and np.any(receive_time[1:] <= receive_time[:-1]):
                    errors.append(
                        "external pose receive times are not strictly increasing"
                    )
            for name, value in (
                ("receive Unix time", receive_unix),
                ("source time", source_time),
                ("sequence", external_sequence),
            ):
                if value is not None and value.shape == (external_count,):  # noqa: SIM102
                    if np.any(value < 0) or (
                        external_count > 1 and np.any(value[1:] <= value[:-1])
                    ):
                        errors.append(
                            f"external pose {name} is not strictly increasing"
                        )
            if source_time is not None and source_time.shape == (external_count,):  # noqa: SIM102
                if np.any(source_time < 946_684_800_000_000_000):
                    errors.append(
                        "external pose source timestamps are not Unix epoch time"
                    )
            if quality is not None and quality.shape == (external_count,):  # noqa: SIM102
                if (
                    not np.isfinite(quality).all()
                    or np.any(quality < 0.0)
                    or np.any(quality > 1.0)
                ):
                    errors.append("external pose quality must be finite and in [0, 1]")
            if frame_id is not None and frame_id.shape == (external_count,):
                normalized_frames = [str(value) for value in frame_id]
                if any(not value for value in normalized_frames):
                    errors.append("external pose frame_id cannot be empty")
                if len(set(normalized_frames)) > 1:
                    errors.append("external pose frame_id changes within the trial")
                if (
                    normalized_frames
                    and isinstance(declared_world_frame, str)
                    and declared_world_frame.strip()
                    and any(
                        value != declared_world_frame.strip()
                        for value in normalized_frames
                    )
                ):
                    errors.append(
                        "external pose frame_id does not match the declared world_frame"
                    )

            if valid is not None and valid.shape == (external_count,):
                valid_mask = valid.astype(np.bool_, copy=False)
                if quality is not None and quality.shape == (external_count,):
                    usable_external = (
                        valid_mask & (quality > 0.0) & (quality >= minimum_quality)
                    )
                else:
                    usable_external = valid_mask
                if position is not None and position.shape == (external_count, 3):  # noqa: SIM102
                    if not np.isfinite(position[valid_mask]).all():
                        errors.append(
                            "valid external pose positions contain non-finite values"
                        )
                if orientation is not None and orientation.shape == (external_count, 4):
                    valid_orientation = orientation[valid_mask]
                    if not np.isfinite(valid_orientation).all():
                        errors.append(
                            "valid external pose quaternions contain non-finite values"
                        )
                    elif valid_orientation.size and np.any(
                        np.abs(np.linalg.norm(valid_orientation, axis=1) - 1.0) > 0.01
                    ):
                        errors.append(
                            "valid external pose quaternions are not normalized"
                        )
            for name, value in (
                ("linear velocity", linear_velocity),
                ("angular velocity", angular_velocity),
            ):
                if value is not None and value.shape == (external_count, 3):
                    row_valid = np.isfinite(value).all(axis=1) | np.isnan(value).all(
                        axis=1
                    )
                    if not row_valid.all():
                        errors.append(
                            f"external pose {name} rows must be finite or entirely NaN"
                        )

            receiver_metadata = manifest.get("external_pose_recording")
            max_offset: float | None = None
            if isinstance(receiver_metadata, Mapping):
                raw_max_offset = receiver_metadata.get(
                    "max_source_receive_clock_offset_s"
                )
                if (
                    not isinstance(raw_max_offset, bool)
                    and isinstance(raw_max_offset, (int, float))
                    and math.isfinite(float(raw_max_offset))
                    and float(raw_max_offset) > 0.0
                ):
                    max_offset = float(raw_max_offset)
                elif external_count:
                    errors.append(
                        "external receiver clock-offset limit is missing or invalid"
                    )
                if receiver_metadata.get("fatal_error") is not None:
                    errors.append("external pose receiver reported a fatal error")
                received_packets = receiver_metadata.get("received_packets")
                if (
                    isinstance(received_packets, bool)
                    or not isinstance(received_packets, int)
                    or received_packets < 0
                ):
                    errors.append("external receiver received_packets is invalid")
                elif received_packets != external_count:
                    errors.append(
                        "external receiver received_packets does not match the archive"
                    )
            elif external_count:
                errors.append("external pose receiver metadata is missing")
            if (
                max_offset is not None
                and receive_unix is not None
                and source_time is not None
                and receive_unix.shape == source_time.shape == (external_count,)
                and np.any(np.abs(source_time - receive_unix) * 1e-9 > max_offset)
            ):
                errors.append("external pose source/receive clock offset exceeds limit")

            if paper_ready and task == "velocity":
                usable_count = int(np.count_nonzero(usable_external))
                if usable_count < 2:
                    errors.append(
                        "velocity paper metrics require at least two usable external poses"
                    )
                control_unix = arrays.get("time_unix_ns")
                if (
                    usable_count
                    and source_time is not None
                    and control_unix is not None
                    and control_unix.ndim == 1
                    and control_unix.size
                ):
                    usable_source_time = source_time[usable_external] - clock_offset_ns
                    control_start = int(control_unix[0])
                    control_end = int(control_unix[-1])
                    cycle_duration = arrays.get("cycle_duration_s")
                    if (
                        cycle_duration is not None
                        and cycle_duration.shape == control_unix.shape
                        and np.isfinite(cycle_duration[-1])
                        and cycle_duration[-1] >= 0.0
                    ):
                        control_end += round(float(cycle_duration[-1]) * 1e9)
                    if int(np.min(usable_source_time)) > control_start:
                        errors.append(
                            "external pose does not provide a baseline before control starts"
                        )
                    if int(np.max(usable_source_time)) < control_end:
                        errors.append(
                            "external pose does not cover the final goal write"
                        )
                    if maximum_gap_s > 0.0:
                        baseline_indices = np.flatnonzero(
                            usable_source_time <= control_start
                        )
                        endpoint_indices = np.flatnonzero(
                            usable_source_time >= control_end
                        )
                        if baseline_indices.size and endpoint_indices.size:
                            first = int(baseline_indices[-1])
                            last = int(endpoint_indices[0])
                            spanning_times = usable_source_time[first : last + 1]
                            if spanning_times.size > 1 and np.any(
                                np.diff(spanning_times) * 1e-9 > maximum_gap_s
                            ):
                                errors.append(
                                    "usable external pose gap exceeds the predeclared maximum"
                                )

    if paper_ready:
        if manifest_step_count is None or manifest_step_count <= 0:
            errors.append("paper-ready trials require at least one control sample")
        policy_digest = policy.get("sha256")
        if not _valid_hex(policy_digest, {64}):
            errors.append("paper-ready policy SHA-256 is missing or invalid")
        hardware = manifest.get("hardware")
        hardware_digest = (
            hardware.get("hardware_config_json_sha256")
            if isinstance(hardware, Mapping)
            else None
        )
        if not _valid_hex(hardware_digest, {64}):
            errors.append("paper-ready hardware-config SHA-256 is missing or invalid")
        elif isinstance(hardware, Mapping):
            hardware_config = hardware.get("hardware_config")
            if (
                isinstance(hardware_config, Mapping)
                and sha256_json(hardware_config) != hardware_digest.lower()
            ):
                errors.append(
                    "hardware-config SHA-256 does not match the embedded configuration"
                )
        commit = _nested_value(manifest, "software", "git", "commit")
        if not _valid_hex(commit, {40, 64}):
            errors.append("paper-ready software Git commit is missing or invalid")
        git = _nested_value(manifest, "software", "git")
        if not isinstance(git, Mapping) or git.get("dirty") is not False:
            errors.append(
                "paper-ready software provenance requires a clean Git worktree"
            )
        trial = manifest.get("trial")
        if not isinstance(trial, Mapping):
            errors.append("trial metadata section is missing")
            trial = {}
        for name in ("trial_id", "condition", "robot_id"):
            if not trial.get(name):
                errors.append(f"trial metadata is missing {name}")
        experiment_context = trial.get("experiment_context")
        context_digest = trial.get("experiment_context_sha256")
        if not _valid_hex(context_digest, {64}):
            errors.append("trial experiment_context_sha256 is missing or invalid")
        elif (
            isinstance(experiment_context, Mapping)
            and sha256_json(experiment_context) != context_digest.lower()
        ):
            errors.append("trial experiment_context_sha256 does not match its context")
        context_source = trial.get("experiment_context_source")
        source_digest = (
            context_source.get("sha256")
            if isinstance(context_source, Mapping)
            else None
        )
        if not _valid_hex(source_digest, {64}):
            errors.append(
                "trial experiment_context_source SHA-256 is missing or invalid"
            )
        result = manifest.get("result")
        if not isinstance(result, Mapping):
            errors.append("trial result section is missing")
            result = {}
        if result.get("annotation_status") != "reviewed":
            errors.append("trial outcome has not been reviewed and annotated")
        elif result.get("valid_for_analysis") is not True:
            errors.append("trial is marked invalid for analysis")
        if recording.get("error") is not None:
            errors.append("recording contains a controller or finalization error")
        termination_reason = recording.get("termination_reason")
        if termination_reason == "controller_error":
            errors.append("controller-error trials are not paper-ready")
        if result.get("success") is True and termination_reason not in {
            "trial_steps_reached",
            "motion_complete",
        }:
            errors.append("a successful trial must have a natural termination reason")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or annotate a recorded Swenoid experiment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate an experiment record.")
    validate.add_argument("path", type=Path)
    validate.add_argument("--paper-ready", action="store_true")
    annotate = subparsers.add_parser("annotate", help="Record the observed outcome.")
    annotate.add_argument("path", type=Path)
    annotate.add_argument("--outcome", required=True, choices=OUTCOMES)
    annotate.add_argument("--external-intervention", action="store_true")
    annotate.add_argument("--exclusion-reason")
    annotate.add_argument("--notes")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "annotate":
        path = annotate_experiment(
            args.path,
            outcome=args.outcome,
            external_intervention=args.external_intervention,
            exclusion_reason=args.exclusion_reason,
            notes=args.notes,
        )
        print(f"Updated {path}")
        return
    errors = validate_experiment(args.path, paper_ready=args.paper_ready)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Experiment record is valid.")


if __name__ == "__main__":
    main()
