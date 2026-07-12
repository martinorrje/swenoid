"""Tests for ONNX policy inspection and hardware joint conversion."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from swenoid.deployment.policy import OnnxPolicy, motion_length
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
