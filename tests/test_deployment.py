"""Tests for ONNX policy inspection and hardware joint conversion."""

import os
import sys
import sysconfig
from pathlib import Path

import numpy as np
import onnx
import pytest
import wandb
from onnx import TensorProto, helper, numpy_helper

import swenoid.deployment.sim2sim as sim2sim
from swenoid.deployment.policy import OnnxPolicy, download_wandb_onnx, motion_length
from swenoid.deployment.sim2sim import build_standalone_model
from swenoid.deployment.swenoid_control import SwenoidControl


def _metadata(
    model: onnx.ModelProto,
    observation_size: int,
    observation_names: str = "command",
) -> None:
    values = {
        "observation_names": observation_names,
        "action_scale": ",".join(["0.1"] * 24),
        "default_joint_pos": ",".join(["0.0"] * 24),
    }
    for key, value in values.items():
        item = model.metadata_props.add()
        item.key = key
        item.value = value
    assert observation_size in (3, 48, 126)


def _make_velocity_onnx(path: Path) -> None:
    actions = numpy_helper.from_array(
        np.zeros((1, 24), dtype=np.float32), name="actions_value"
    )
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["actions"], value=actions)],
        "velocity",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 24])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    _metadata(model, 3)
    onnx.save(model, path)


def _make_general_motion_onnx(path: Path) -> None:
    actions = numpy_helper.from_array(
        np.zeros((1, 24), dtype=np.float32), name="actions_value"
    )
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["actions"], value=actions)],
        "general_motion",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 126])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 24])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    _metadata(
        model,
        126,
        "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command",
    )
    onnx.save(model, path)


def _make_tracking_onnx(path: Path, frames: int = 7) -> None:
    joint_pos = np.arange(frames * 24, dtype=np.float32).reshape(frames, 24)
    joint_vel = -joint_pos
    actions = numpy_helper.from_array(
        np.zeros((1, 24), dtype=np.float32), name="actions_value"
    )
    initializers = [
        numpy_helper.from_array(joint_pos, name="motion_joint_pos"),
        numpy_helper.from_array(joint_vel, name="motion_joint_vel"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="squeeze_axes"),
    ]
    nodes = [
        helper.make_node("Constant", [], ["actions"], value=actions),
        helper.make_node("Squeeze", ["time_step", "squeeze_axes"], ["time_flat"]),
        helper.make_node("Cast", ["time_flat"], ["time_index"], to=TensorProto.INT64),
        helper.make_node(
            "Gather", ["motion_joint_pos", "time_index"], ["joint_pos"], axis=0
        ),
        helper.make_node(
            "Gather", ["motion_joint_vel", "time_index"], ["joint_vel"], axis=0
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "tracking",
        [
            helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 48]),
            helper.make_tensor_value_info("time_step", TensorProto.FLOAT, [1, 1]),
        ],
        [
            helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 24]),
            helper.make_tensor_value_info("joint_pos", TensorProto.FLOAT, [1, 24]),
            helper.make_tensor_value_info("joint_vel", TensorProto.FLOAT, [1, 24]),
        ],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    _metadata(model, 48)
    onnx.checker.check_model(model)
    onnx.save(model, path)


class _FakeWandbFile:
    def __init__(self, name: str, updated_at: str):
        self.name = name
        self.updated_at = updated_at

    def download(self, root: str, replace: bool = False) -> None:
        del replace
        output = Path(root) / self.name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()


class _FakeWandbRun:
    id = "run-id"

    def __init__(self, files: list[_FakeWandbFile]):
        self._files = files

    def files(self) -> list[_FakeWandbFile]:
        return self._files

    def file(self, name: str) -> _FakeWandbFile:
        return next(item for item in self._files if item.name == name)


def _use_fake_wandb_run(monkeypatch, files: list[_FakeWandbFile]) -> None:
    run = _FakeWandbRun(files)
    monkeypatch.setattr(
        wandb, "Api", lambda: type("Api", (), {"run": lambda _, __: run})()
    )


def test_wandb_download_selects_newest_onnx(monkeypatch, tmp_path) -> None:
    _use_fake_wandb_run(
        monkeypatch,
        [
            _FakeWandbFile("older.onnx", "2026-07-18T01:00:00Z"),
            _FakeWandbFile("newer.onnx", "2026-07-19T01:00:00Z"),
            _FakeWandbFile("metadata.json", "2026-07-20T01:00:00Z"),
        ],
    )

    path = download_wandb_onnx("entity/project/run", cache_root=tmp_path)

    assert path == tmp_path / "run-id" / "newer.onnx"


def test_wandb_download_breaks_timestamp_ties_by_filename(
    monkeypatch, tmp_path
) -> None:
    _use_fake_wandb_run(
        monkeypatch,
        [
            _FakeWandbFile("a-first.onnx", "2026-07-19T01:00:00Z"),
            _FakeWandbFile("z-last.onnx", "2026-07-19T01:00:00Z"),
        ],
    )

    path = download_wandb_onnx("entity/project/run", cache_root=tmp_path)

    assert path == tmp_path / "run-id" / "z-last.onnx"


def test_wandb_download_honors_exact_filename(monkeypatch, tmp_path) -> None:
    _use_fake_wandb_run(
        monkeypatch,
        [
            _FakeWandbFile("requested.onnx", "2026-07-18T01:00:00Z"),
            _FakeWandbFile("newer.onnx", "2026-07-19T01:00:00Z"),
        ],
    )

    path = download_wandb_onnx(
        "entity/project/run",
        cache_root=tmp_path,
        filename="requested.onnx",
    )

    assert path == tmp_path / "run-id" / "requested.onnx"


def test_wandb_download_rejects_missing_exact_filename(monkeypatch, tmp_path) -> None:
    _use_fake_wandb_run(
        monkeypatch,
        [_FakeWandbFile("available.onnx", "2026-07-19T01:00:00Z")],
    )

    with pytest.raises(ValueError, match=r"missing\.onnx"):
        download_wandb_onnx(
            "entity/project/run",
            cache_root=tmp_path,
            filename="missing.onnx",
        )


def test_wandb_download_rejects_run_without_onnx(monkeypatch, tmp_path) -> None:
    _use_fake_wandb_run(
        monkeypatch,
        [_FakeWandbFile("metadata.json", "2026-07-19T01:00:00Z")],
    )

    with pytest.raises(ValueError, match="No ONNX files"):
        download_wandb_onnx("entity/project/run", cache_root=tmp_path)


def test_mjpython_relaunches_sim2sim_on_macos(monkeypatch, tmp_path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    mjpython = bin_dir / "mjpython"
    mjpython.touch()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr(
        sysconfig,
        "get_config_var",
        lambda name: "/uv/lib" if name == "LIBDIR" else None,
    )
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "/existing/lib")
    monkeypatch.setattr(
        sys,
        "argv",
        ["swenoid-sim2sim", "--onnx", "policy.onnx", "--loop"],
    )
    monkeypatch.setattr(sim2sim.mujoco.viewer, "_MJPYTHON", None)
    call = None

    class ExecCalled(Exception):
        pass

    def fake_execv(path, argv):
        nonlocal call
        call = (path, argv)
        raise ExecCalled

    monkeypatch.setattr(os, "execv", fake_execv)

    with pytest.raises(ExecCalled):
        sim2sim._ensure_mjpython()

    assert call == (
        str(mjpython),
        [
            str(mjpython),
            "-m",
            "swenoid.deployment.sim2sim",
            "--onnx",
            "policy.onnx",
            "--loop",
        ],
    )
    assert os.environ["DYLD_FALLBACK_LIBRARY_PATH"] == "/uv/lib:/existing/lib"


def test_mjpython_preflight_is_noop_outside_macos(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        os,
        "execv",
        lambda *_: pytest.fail("unexpected mjpython relaunch"),
    )

    sim2sim._ensure_mjpython()


def test_mjpython_preflight_is_noop_when_dispatcher_is_active(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sim2sim.mujoco.viewer, "_MJPYTHON", object())
    monkeypatch.setattr(
        os,
        "execv",
        lambda *_: pytest.fail("unexpected mjpython relaunch"),
    )

    sim2sim._ensure_mjpython()


def test_mjpython_preflight_rejects_missing_launcher(monkeypatch, tmp_path) -> None:
    python = tmp_path / "bin" / "python"
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr(sim2sim.mujoco.viewer, "_MJPYTHON", None)

    with pytest.raises(RuntimeError, match=str(python.with_name("mjpython"))):
        sim2sim._ensure_mjpython()


def test_velocity_policy_is_detected_from_inputs(tmp_path) -> None:
    path = tmp_path / "velocity.onnx"
    _make_velocity_onnx(path)
    policy = OnnxPolicy(path)
    assert not policy.is_tracking
    assert policy.policy_kind == "velocity"
    assert policy.motion_length is None
    assert motion_length(path) is None
    assert policy.infer(np.zeros((1, 3), dtype=np.float32), 0)[0].shape == (1, 24)


def test_general_motion_policy_is_not_misclassified_as_velocity(tmp_path) -> None:
    path = tmp_path / "general_motion.onnx"
    _make_general_motion_onnx(path)
    policy = OnnxPolicy(path)
    assert not policy.is_tracking
    assert policy.is_general_motion
    assert policy.policy_kind == "general_motion"
    assert policy.observation_size == 126


def test_tracking_policy_reads_embedded_reference(tmp_path) -> None:
    path = tmp_path / "tracking.onnx"
    _make_tracking_onnx(path)
    policy = OnnxPolicy(path)
    assert policy.is_tracking
    assert policy.policy_kind == "single_motion"
    assert policy.motion_length == 7
    reference = policy.reference_at(3)
    assert reference is not None
    np.testing.assert_array_equal(reference[0][0], np.arange(72, 96))
    np.testing.assert_array_equal(reference[1], -reference[0])


class _FakeDynamixelHandler:
    def disable_torque(self, *_):
        pass

    def set_zero_return_delay_time(self, *_):
        pass

    def read_servo_voltages(self, ids):
        return [120] * len(ids)

    def write_indirect_addresses(self, *_):
        pass

    def add_pos_vel_group_sync_read(self, *_):
        pass

    def set_position_mode(self, *_):
        pass

    def set_kp(self, *_):
        pass

    def set_duration_accel(self, *_):
        pass

    def read_lower_limits(self, ids):
        return [(0,)] * len(ids)

    def read_upper_limits(self, ids):
        return [(4095,)] * len(ids)

    def close(self):
        pass


def test_hardware_joint_conversion_round_trip() -> None:
    control = SwenoidControl(dynamixel_handler=_FakeDynamixelHandler())
    source = np.linspace(-0.5, 0.5, 24, dtype=np.float32)
    encoded = control.pos_isaac_to_dynamixel(source)
    decoded = control.pos_dynamixel_to_isaac(encoded)
    np.testing.assert_allclose(decoded, source, atol=2 * np.pi / 4096, rtol=0.0)


def test_hardware_step_safety_rejects_large_jump() -> None:
    control = SwenoidControl(dynamixel_handler=_FakeDynamixelHandler())
    with pytest.raises(Exception, match="move too fast"):
        control.move_servos([3000] * 24, [2000] * 24)


def test_hardware_conversion_rejects_target_far_outside_motor_limit() -> None:
    control = SwenoidControl(dynamixel_handler=_FakeDynamixelHandler())
    target = np.zeros(24, dtype=np.float32)
    target[0] = 4.0
    with pytest.raises(ValueError, match="limit"):
        control.pos_isaac_to_dynamixel(target)


def test_standalone_bam_model_covers_all_actuators() -> None:
    model, data, controllers = build_standalone_model("bam")
    assert model.nu == 24
    assert sorted(len(controller.actuator) for controller in controllers) == [9, 15]
    for controller in controllers:
        controller.reset(data.qpos)
        controller.update()
    assert np.isfinite(data.ctrl).all()
