import json
import socket
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from swenoid.deployment.recording import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    ExperimentRecorder,
    ExternalPosePacket,
    ExternalPoseReceiver,
    annotate_experiment,
    sha256_json,
    validate_experiment,
)

_BASE_UNIX_NS = 1_700_000_000_000_000_000


def _experiment_context(*, identified: bool = False) -> dict[str, object]:
    parameter_files: list[dict[str, str]] = []
    if identified:
        parameter_files = [
            {
                "path": "params/xm430.json",
                "sha256": "d" * 64,
            }
        ]
    return {
        "condition_details": {
            "simulator_actuator_model": "identified-bam" if identified else "nominal",
            "actuator_parameter_files": parameter_files,
            "training_run": "wandb/entity/project/run-001",
        },
        "protocol": {
            "success_criterion": "Complete the configured horizon without a fall.",
            "evaluation_window": "All recorded control samples.",
            "infrastructure_exclusion_rule": "Exclude only acquisition failures.",
        },
        "robot_configuration": {"battery_pack_id": "battery-01"},
        "environment": {"surface": "level rubber mat"},
        "trial_design": {
            "block_id": "block-01",
            "command_sequence_id": "sequence-01",
            "order_assignment": "interleaved-a",
        },
        "external_tracking": {
            "system": "test-mocap",
            "rigid_body": "pelvis",
            "world_frame": "mocap_world",
            "calibration_id": "calibration-001",
            "clock_sync_method": "PTP",
            "estimated_clock_offset_ms": 0.1,
            "minimum_quality": 0.5,
            "maximum_gap_ms": 100.0,
            "quality_definition": "Normalized marker-fit confidence.",
        },
        "safety": {
            "tethered": False,
            "spotter_present": True,
            "external_support_allowed_after_start": False,
        },
    }


def _metadata(
    *,
    policy_kind: str = "single_motion",
    condition: str = "nominal",
) -> dict[str, object]:
    context = _experiment_context(identified=condition == "identified")
    hardware_config = {"name": "test-hardware", "schema_version": 1}
    return {
        "trial": {
            "trial_id": "trial-001",
            "condition": condition,
            "robot_id": "swenoid-01",
            "experiment_context": context,
            "experiment_context_sha256": sha256_json(context),
            "experiment_context_source": {
                "path": "metadata.json",
                "sha256": "e" * 64,
            },
        },
        "policy": {
            "policy_kind": policy_kind,
            "observation_size": 6,
            "sha256": "a" * 64,
        },
        "hardware": {
            "hardware_config": hardware_config,
            "hardware_config_json_sha256": sha256_json(hardware_config),
        },
        "software": {"git": {"commit": "c" * 40, "dirty": False}},
    }


def _step(index: int, *, policy_kind: str = "single_motion") -> dict[str, object]:
    tracking = policy_kind != "velocity"
    command_size = 48 if tracking else 3
    reference_position = np.linspace(-0.2, 0.2, 24)
    reference_velocity = np.zeros(24)
    if not tracking:
        reference_position[:] = np.nan
        reference_velocity[:] = np.nan
    return {
        "sample_index": index,
        "policy_frame": index,
        "time_s": 0.02 * (index + 1),
        "time_unix_ns": _BASE_UNIX_NS + 20_000_000 * (index + 1),
        "control_dt_s": np.nan if index == 0 else 0.02,
        "read_duration_s": 0.001,
        "inference_duration_s": 0.002,
        "write_duration_s": 0.001,
        "cycle_duration_s": 0.004,
        "deadline_missed": False,
        "policy_observation": np.arange(6, dtype=np.float32),
        "policy_action": np.full(24, index, dtype=np.float32),
        "command": np.arange(command_size, dtype=np.float32),
        "reference_joint_position_rad": reference_position,
        "reference_joint_velocity_rad_s": reference_velocity,
        "joint_position_rad": np.linspace(-0.1, 0.1, 24),
        "joint_velocity_rad_s": np.zeros(24),
        "joint_target_requested_rad": np.linspace(-0.15, 0.15, 24),
        "joint_target_sent_rad": np.linspace(-0.15, 0.15, 24),
        "dynamixel_position_raw": np.arange(24, dtype=np.int32) + 2_000,
        "dynamixel_velocity_raw": np.arange(24, dtype=np.int32),
        "dynamixel_goal_position_raw": np.arange(24, dtype=np.int32) + 2_001,
        "dynamixel_goal_clipped": np.zeros(24, dtype=np.bool_),
        "base_angular_velocity_raw_rad_s": np.array([0.1, 0.2, 0.3]),
        "base_angular_velocity_rad_s": np.array([0.1, 0.2, 0.3]),
        "projected_gravity_raw": np.array([0.0, 0.0, -1.0]),
        "projected_gravity": np.array([0.0, 0.0, -1.0]),
        "base_orientation_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        "imu_sample_time_s": 0.02 * (index + 1),
        "imu_age_s": 0.001,
        "imu_sequence": index,
    }


def _telemetry(index: int = 0) -> dict[str, object]:
    current_raw = np.arange(24, dtype=np.int16)
    voltage_raw = np.full(24, 120, dtype=np.uint16)
    return {
        "servo_telemetry_sample_index": index,
        "servo_telemetry_time_s": 0.02 * (index + 1) + 0.001,
        "servo_current_raw": current_raw,
        "servo_current_a": current_raw.astype(np.float32) * 0.00269,
        "servo_voltage_raw": voltage_raw,
        "servo_voltage_v": voltage_raw.astype(np.float32) * 0.1,
        "servo_temperature_c": np.full(24, 32.0, dtype=np.float32),
    }


def _packet(
    source_time_ns: int,
    sequence: int,
    *,
    frame_id: str = "mocap_world",
    quality: float = 0.95,
) -> ExternalPosePacket:
    return ExternalPosePacket(
        source_time_ns=source_time_ns,
        sequence=sequence,
        position_m=np.asarray([sequence * 0.1, 0.0, 0.7], dtype=np.float32),
        orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        linear_velocity_m_s=np.asarray([0.2, 0.0, 0.0], dtype=np.float32),
        angular_velocity_rad_s=np.asarray([0.0, 0.0, 0.1], dtype=np.float32),
        valid=True,
        quality=quality,
        frame_id=frame_id,
    )


def _append_external(
    recorder: ExperimentRecorder,
    *,
    source_time_ns: int,
    sequence: int,
    receive_offset_s: float,
    quality: float = 0.95,
) -> None:
    assert recorder.start_monotonic_ns is not None
    recorder.append_external_pose_packet(
        recorder.start_monotonic_ns + int(receive_offset_s * 1e9),
        source_time_ns + 1_000_000,
        _packet(source_time_ns, sequence, quality=quality),
    )


def _receiver_metadata() -> dict[str, object]:
    return {
        "transport": "udp_json",
        "received_packets": 2,
        "dropped_packets": 0,
        "errors": [],
        "frame_id": "mocap_world",
        "max_source_receive_clock_offset_s": 0.1,
        "fatal_error": None,
    }


def _finalized_record(
    path: Path,
    *,
    policy_kind: str = "single_motion",
    external_pose: bool = False,
    condition: str = "nominal",
    termination_reason: str | None = None,
    error: BaseException | None = None,
) -> ExperimentRecorder:
    command_size = 3 if policy_kind == "velocity" else 48
    recorder = ExperimentRecorder(
        path,
        metadata=_metadata(policy_kind=policy_kind, condition=condition),
        observation_size=6,
        command_size=command_size,
    )
    recorder.start()
    if external_pose:
        _append_external(
            recorder,
            source_time_ns=_BASE_UNIX_NS,
            sequence=0,
            receive_offset_s=0.005,
        )
    recorder.append_step(_step(0, policy_kind=policy_kind))
    if external_pose:
        _append_external(
            recorder,
            source_time_ns=_BASE_UNIX_NS + 40_000_000,
            sequence=1,
            receive_offset_s=0.04,
        )
    natural_reason = (
        "trial_steps_reached" if policy_kind == "velocity" else "motion_complete"
    )
    recorder.finalize(
        termination_reason=termination_reason or natural_reason,
        external_metadata=_receiver_metadata() if external_pose else None,
        error=error,
    )
    return recorder


def _rewrite_npz(path: Path, updates: Mapping[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload.update(updates)
    np.savez_compressed(path, **payload)


def _annotate_success(path: Path) -> None:
    annotate_experiment(
        path,
        outcome="success",
        external_intervention=False,
        exclusion_reason=None,
        notes="Completed the configured trial.",
    )


def test_npz_round_trip_without_pickle_preserves_schema_and_shapes(tmp_path) -> None:
    path = tmp_path / "trial.npz"
    recorder = ExperimentRecorder(
        path,
        metadata=_metadata(),
        observation_size=6,
        command_size=48,
    )
    recorder.start()
    recorder.append_step(_step(0))
    recorder.append_step(_step(1))
    recorder.append_servo_telemetry(_telemetry())
    recorder.finalize(termination_reason="motion_complete")

    with np.load(path, allow_pickle=False) as archive:
        assert archive["schema_name"].item() == SCHEMA_NAME
        assert archive["schema_version"].item() == SCHEMA_VERSION
        assert archive["sample_index"].shape == (2,)
        assert archive["policy_observation"].shape == (2, 6)
        assert archive["policy_action"].shape == (2, 24)
        assert archive["command"].shape == (2, 48)
        assert archive["joint_position_rad"].shape == (2, 24)
        assert archive["servo_current_a"].shape == (1, 24)
        assert archive["external_base_position_world_m"].shape == (0, 3)
        assert archive["external_pose_frame_id"].shape == (0,)
        np.testing.assert_array_equal(archive["sample_index"], [0, 1])
        np.testing.assert_array_equal(
            archive["joint_pos"], archive["joint_position_rad"]
        )
        assert all(archive[name].dtype.kind != "O" for name in archive.files)


def test_checkpoint_contains_external_pose_and_finalize_counts_it(tmp_path) -> None:
    path = tmp_path / "trial.npz"
    recorder = ExperimentRecorder(
        path,
        metadata=_metadata(),
        observation_size=6,
        command_size=48,
        checkpoint_steps=1,
    )
    recorder.start()
    _append_external(
        recorder,
        source_time_ns=_BASE_UNIX_NS,
        sequence=0,
        receive_offset_s=0.005,
    )
    recorder.append_step(_step(0))

    assert recorder._checkpoint_thread is not None
    recorder._checkpoint_thread.join()
    with np.load(recorder.paths.partial_samples, allow_pickle=False) as checkpoint:
        np.testing.assert_array_equal(checkpoint["sample_index"], [0])
        np.testing.assert_array_equal(checkpoint["external_pose_sequence"], [0])
        assert all(checkpoint[name].dtype.kind != "O" for name in checkpoint.files)

    recorder.finalize(
        termination_reason="motion_complete",
        external_metadata=_receiver_metadata(),
    )
    manifest = json.loads(recorder.paths.manifest.read_text(encoding="utf-8"))
    assert manifest["recording"]["external_pose_count"] == 1
    assert not recorder.paths.partial_samples.exists()


def test_checkpoint_overwrite_refusal_and_finalize_idempotence(tmp_path) -> None:
    path = tmp_path / "trial.npz"
    recorder = _finalized_record(path)
    recorder.finalize(termination_reason="ignored-second-finalize")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ExperimentRecorder(
            path,
            metadata=_metadata(),
            observation_size=6,
            command_size=48,
        )
    manifest = json.loads(recorder.paths.manifest.read_text(encoding="utf-8"))
    assert manifest["recording"]["termination_reason"] == "motion_complete"

    replacement = ExperimentRecorder(
        path,
        metadata=_metadata(),
        observation_size=6,
        command_size=48,
        overwrite=True,
    )
    assert not replacement.paths.samples.exists()
    assert replacement.paths.manifest.is_file()


def test_outcome_annotation_makes_tracking_trial_paper_ready(tmp_path) -> None:
    path = tmp_path / "tracking.npz"
    recorder = _finalized_record(path)

    assert validate_experiment(path, paper_ready=False) == []
    assert "trial outcome has not been reviewed and annotated" in validate_experiment(
        path, paper_ready=True
    )
    manifest_path = annotate_experiment(
        path,
        outcome="success",
        external_intervention=False,
        exclusion_reason=None,
        notes="Completed the sequence without a fall.",
    )

    assert manifest_path == recorder.paths.manifest
    assert validate_experiment(path, paper_ready=True) == []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["result"]["success"] is True
    assert manifest["result"]["fall"] is False
    assert manifest["result"]["annotation_history"] == []


def test_external_pose_packet_parses_optional_velocities_and_strict_types() -> None:
    packet = ExternalPosePacket.parse(
        json.dumps(
            {
                "source_time_ns": _BASE_UNIX_NS,
                "sequence": 4,
                "position_m": [1.0, -2.0, 0.7],
                "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "valid": True,
                "quality": 0.95,
                "frame_id": "mocap_world",
            }
        ).encode()
    )

    assert packet.sequence == 4
    assert np.isnan(packet.linear_velocity_m_s).all()
    assert np.isnan(packet.angular_velocity_rad_s).all()
    with pytest.raises(ValueError, match="integer source_time_ns"):
        ExternalPosePacket.parse(
            json.dumps(
                {
                    "source_time_ns": str(_BASE_UNIX_NS),
                    "sequence": 5,
                    "position_m": [0.0, 0.0, 0.7],
                    "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "valid": True,
                    "frame_id": "world",
                }
            ).encode()
        )


def test_external_pose_packet_rejects_malformed_quaternion() -> None:
    data = json.dumps(
        {
            "source_time_ns": _BASE_UNIX_NS,
            "sequence": 0,
            "position_m": [0.0, 0.0, 0.7],
            "orientation_wxyz": [2.0, 0.0, 0.0, 0.0],
            "valid": True,
            "frame_id": "world",
        }
    ).encode()
    with pytest.raises(ValueError, match="quaternion must be normalized"):
        ExternalPosePacket.parse(data)


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _send_pose(port: int, *, source_time_ns: int, sequence: int, frame: str) -> None:
    payload = {
        "source_time_ns": source_time_ns,
        "sequence": sequence,
        "position_m": [0.0, 0.0, 0.7],
        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "valid": True,
        "quality": 0.9,
        "frame_id": frame,
    }
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.sendto(json.dumps(payload).encode(), ("127.0.0.1", port))


def test_receiver_callback_readiness_and_rejects_frame_and_clock_changes() -> None:
    port = _free_udp_port()
    received: list[tuple[int, int, ExternalPosePacket]] = []
    receiver = ExternalPoseReceiver(
        bind="127.0.0.1",
        port=port,
        on_packet=lambda monotonic, unix, packet: received.append(
            (monotonic, unix, packet)
        ),
        max_clock_offset_s=0.2,
    )
    receiver.start()
    before = time.perf_counter_ns()
    now = time.time_ns()
    _send_pose(port, source_time_ns=now, sequence=0, frame="world")
    assert receiver.wait_for_valid_samples(1, 1.0)
    assert receiver.wait_for_valid_after(before, 0.0)
    assert receiver.wait_for_valid_source_after(now, 0.0, receive_monotonic_ns=before)

    _send_pose(port, source_time_ns=time.time_ns(), sequence=1, frame="other")
    _send_pose(port, source_time_ns=1, sequence=2, frame="world")
    assert not receiver.wait_for_valid_samples(2, 0.2)
    receiver.stop()
    receiver.raise_if_failed()

    assert len(received) == 1
    metadata = receiver.metadata()
    assert metadata["received_packets"] == 1
    assert metadata["dropped_packets"] == 2
    assert metadata["frame_id"] == "world"


def test_receiver_surfaces_callback_failure() -> None:
    port = _free_udp_port()

    def fail_callback(_monotonic: int, _unix: int, _packet: ExternalPosePacket) -> None:
        raise RuntimeError("archive failed")

    receiver = ExternalPoseReceiver(
        bind="127.0.0.1",
        port=port,
        on_packet=fail_callback,
    )
    receiver.start()
    _send_pose(port, source_time_ns=time.time_ns(), sequence=0, frame="world")
    with pytest.raises(RuntimeError, match="receiver failed"):
        receiver.wait_for_valid_samples(1, 1.0)
    receiver.stop()
    with pytest.raises(RuntimeError, match="receiver failed"):
        receiver.raise_if_failed()
    assert receiver.metadata()["fatal_error"]["message"] == "archive failed"


def test_velocity_trial_requires_usable_external_pose_with_endpoint_coverage(
    tmp_path,
) -> None:
    missing_path = tmp_path / "missing.npz"
    _finalized_record(missing_path, policy_kind="velocity")
    _annotate_success(missing_path)
    assert (
        "velocity paper metrics require at least two usable external poses"
        in validate_experiment(missing_path, paper_ready=True)
    )

    complete_path = tmp_path / "complete.npz"
    _finalized_record(complete_path, policy_kind="velocity", external_pose=True)
    _annotate_success(complete_path)
    assert validate_experiment(complete_path, paper_ready=True) == []


def test_validator_rejects_manifest_and_multirate_count_mismatches(tmp_path) -> None:
    path = tmp_path / "counts.npz"
    recorder = _finalized_record(path)
    manifest = json.loads(recorder.paths.manifest.read_text(encoding="utf-8"))
    manifest["recording"]["sample_count"] = 2
    recorder.paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    _rewrite_npz(path, {"servo_current_raw": np.empty((1, 24), dtype=np.int16)})

    errors = validate_experiment(path, paper_ready=False)
    assert any("manifest sample_count does not match" in error for error in errors)
    assert "servo_current_raw has a different servo-telemetry count" in errors


def test_validator_rejects_external_frame_clock_and_coverage_corruption(
    tmp_path,
) -> None:
    path = tmp_path / "external.npz"
    _finalized_record(path, policy_kind="velocity", external_pose=True)
    _annotate_success(path)
    _rewrite_npz(
        path,
        {
            "external_pose_frame_id": np.asarray(["world", "other"], dtype="<U128"),
            "external_pose_source_time_ns": np.asarray(
                [_BASE_UNIX_NS + 21_000_000, _BASE_UNIX_NS + 22_000_000],
                dtype=np.int64,
            ),
            "external_pose_receive_unix_ns": np.asarray(
                [_BASE_UNIX_NS + 500_000_000, _BASE_UNIX_NS + 600_000_000],
                dtype=np.int64,
            ),
        },
    )

    errors = validate_experiment(path, paper_ready=True)
    assert "external pose frame_id changes within the trial" in errors
    assert "external pose source/receive clock offset exceeds limit" in errors
    assert "external pose does not provide a baseline before control starts" in errors


def test_validator_applies_predeclared_external_quality_threshold(tmp_path) -> None:
    path = tmp_path / "quality.npz"
    _finalized_record(path, policy_kind="velocity", external_pose=True)
    _annotate_success(path)
    _rewrite_npz(
        path,
        {"external_pose_quality": np.asarray([0.4, 0.4], dtype=np.float32)},
    )

    assert (
        "velocity paper metrics require at least two usable external poses"
        in validate_experiment(path, paper_ready=True)
    )


def test_validator_rejects_trial_long_external_pose_gap(tmp_path) -> None:
    path = tmp_path / "gap.npz"
    recorder = _finalized_record(path, policy_kind="velocity", external_pose=True)
    _annotate_success(path)
    manifest = json.loads(recorder.paths.manifest.read_text(encoding="utf-8"))
    manifest["trial"]["experiment_context"]["external_tracking"]["maximum_gap_ms"] = (
        10.0
    )
    recorder.paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    assert (
        "usable external pose gap exceeds the predeclared maximum"
        in validate_experiment(path, paper_ready=True)
    )


def test_validator_applies_signed_source_clock_offset(tmp_path) -> None:
    path = tmp_path / "clock-offset.npz"
    recorder = _finalized_record(path, policy_kind="velocity", external_pose=True)
    _annotate_success(path)
    manifest = json.loads(recorder.paths.manifest.read_text(encoding="utf-8"))
    context = manifest["trial"]["experiment_context"]
    context["external_tracking"]["estimated_clock_offset_ms"] = 5.0
    manifest["trial"]["experiment_context_sha256"] = sha256_json(context)
    recorder.paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    source_times = np.asarray(
        [_BASE_UNIX_NS + 24_000_000, _BASE_UNIX_NS + 30_000_000],
        dtype=np.int64,
    )
    _rewrite_npz(
        path,
        {
            "external_pose_source_time_ns": source_times,
            "external_pose_receive_unix_ns": source_times + 1_000_000,
        },
    )

    assert validate_experiment(path, paper_ready=True) == []


def test_paper_ready_requires_protocol_metadata_and_identified_hashes(tmp_path) -> None:
    path = tmp_path / "identified.npz"
    recorder = _finalized_record(path, condition="identified")
    _annotate_success(path)
    manifest = json.loads(recorder.paths.manifest.read_text(encoding="utf-8"))
    context = manifest["trial"]["experiment_context"]
    context["protocol"]["evaluation_window"] = ""
    context["condition_details"]["actuator_parameter_files"] = [
        {"path": "params/xm430.json", "sha256": "bad"}
    ]
    context["safety"]["tethered"] = "false"
    manifest["policy"]["sha256"] = "not-a-digest"
    manifest["hardware"]["hardware_config"]["name"] = "mutated"
    manifest["trial"]["experiment_context_source"]["sha256"] = "bad"
    recorder.paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    errors = validate_experiment(path, paper_ready=True)
    assert "experiment context is missing protocol.evaluation_window" in errors
    assert "actuator_parameter_files[0] has an invalid SHA-256" in errors
    assert "experiment context safety.tethered must be boolean" in errors
    assert "paper-ready policy SHA-256 is missing or invalid" in errors
    assert "hardware-config SHA-256 does not match the embedded configuration" in errors
    assert "trial experiment_context_sha256 does not match its context" in errors
    assert "trial experiment_context_source SHA-256 is missing or invalid" in errors


def test_paper_ready_rejects_recording_errors_and_unnatural_success(tmp_path) -> None:
    error_path = tmp_path / "error.npz"
    _finalized_record(
        error_path,
        termination_reason="controller_error",
        error=RuntimeError("serial failure"),
    )
    annotate_experiment(
        error_path,
        outcome="fall",
        external_intervention=False,
        exclusion_reason=None,
        notes="The controller failed during the fall.",
    )
    error_messages = validate_experiment(error_path, paper_ready=True)
    assert "recording contains a controller or finalization error" in error_messages
    assert "controller-error trials are not paper-ready" in error_messages

    interrupt_path = tmp_path / "interrupt.npz"
    _finalized_record(interrupt_path, termination_reason="operator_interrupt")
    _annotate_success(interrupt_path)
    assert (
        "a successful trial must have a natural termination reason"
        in validate_experiment(interrupt_path, paper_ready=True)
    )


def test_paper_ready_rejects_empty_or_excluded_trial(tmp_path) -> None:
    path = tmp_path / "empty.npz"
    recorder = ExperimentRecorder(
        path,
        metadata=_metadata(),
        observation_size=6,
        command_size=48,
    )
    recorder.finalize(
        termination_reason="controller_error",
        error=RuntimeError("initialization failed"),
    )
    annotate_experiment(
        path,
        outcome="incomplete",
        external_intervention=False,
        exclusion_reason=None,
        notes="Initialization failed before the first command.",
    )

    errors = validate_experiment(path, paper_ready=True)
    assert "paper-ready trials require at least one control sample" in errors
    assert "trial is marked invalid for analysis" in errors
