"""Run an exported Swenoid ONNX policy in standalone MuJoCo."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, cast

import mujoco
import mujoco.viewer
import numpy as np
from bam.model import load_model
from bam.mujoco import MujocoController
from scipy.spatial.transform import Rotation

from swenoid.bam_xm_actuators import register_xm_actuators
from swenoid.deployment.policy import (
    OnnxPolicy,
    benchmark,
    concatenate_observations,
    resolve_policy_path,
)
from swenoid.model_constants import (
    SWENOID_XM430_BAM_PARAMS,
    SWENOID_XM540_BAM_PARAMS,
    SWENOID_XML,
    XM430_EFFORT_LIMIT,
    XM430_TARGET_NAMES_EXPR,
    XM540_EFFORT_LIMIT,
    XM540_TARGET_NAMES_EXPR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--onnx", type=Path)
    source.add_argument("--wandb-run", help="Run ID or ENTITY/PROJECT/RUN_ID.")
    parser.add_argument("--entity", default="morrje-kth-royal-institute-of-technology")
    parser.add_argument("--project", default="mjlab")
    parser.add_argument("--onnx-name")
    parser.add_argument("--command", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--actuator-model", choices=("bam", "dc"), default="bam")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--record", type=Path)
    return parser.parse_args()


def build_standalone_model(
    actuator_model: str,
) -> tuple[mujoco.MjModel, mujoco.MjData, list[MujocoController]]:
    """Build standalone MuJoCo with fitted BAM or the XML's direct DC servos."""
    spec = mujoco.MjSpec.from_file(str(SWENOID_XML))
    if actuator_model == "bam":
        for actuator in spec.actuators:
            actuator.set_to_motor()
    elif actuator_model != "dc":
        raise ValueError(f"Unknown actuator model: {actuator_model!r}")

    model = spec.compile()
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    if actuator_model == "dc":
        return model, data, []

    register_xm_actuators()
    controllers = []
    groups = (
        (SWENOID_XM540_BAM_PARAMS, XM540_TARGET_NAMES_EXPR[0], XM540_EFFORT_LIMIT),
        (SWENOID_XM430_BAM_PARAMS, XM430_TARGET_NAMES_EXPR[0], XM430_EFFORT_LIMIT),
    )
    for parameter_file, joint_pattern, effort_limit in groups:
        actuator_names = []
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            joint_name = model.joint(joint_id).name
            if joint_name is not None and re.fullmatch(joint_pattern, joint_name):
                actuator_name = model.actuator(actuator_id).name
                if actuator_name is None:
                    raise ValueError(f"Actuator {actuator_id} has no name")
                actuator_names.append(actuator_name)
        bam_model = cast(Any, load_model(str(parameter_file)))
        bam_model.actuator.kp = 800.0
        bam_model.actuator.vin = 12.0
        controllers.append(
            MujocoController(
                bam_model,
                cast(Any, actuator_names),
                model,
                data,
                max_current=effort_limit / float(bam_model.kt.value),
            )
        )
    if sorted(len(controller.actuator) for controller in controllers) != [9, 15]:
        raise ValueError("BAM actuator grouping did not cover Swenoid's 24 joints")
    return model, data, controllers


class Sim2SimRunner:
    """Standalone MuJoCo runner with the real-robot observation contract."""

    def __init__(self, policy: OnnxPolicy, args: argparse.Namespace):
        self.policy = policy
        self.args = args
        self.command = np.asarray(args.command, dtype=np.float32).reshape(1, 3)
        self.last_actions = np.zeros((1, 24), dtype=np.float32)
        self.time_step = 0

        self.model, self.data, self.bam_controllers = build_standalone_model(
            args.actuator_model
        )
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self._key_callback
        )
        self.viewer.cam.azimuth = 90.0
        self._reset_state()
        self.samples: dict[str, list[np.ndarray]] = {
            "joint_pos": [],
            "joint_vel": [],
            "commanded_joint_pos": [],
            "ang_vel": [],
            "gravity": [],
        }

    def _reset_state(self) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[:3] = (0.0, 0.0, 0.301207)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[7:] = self.policy.default_joint_pos
        self.data.qvel[:] = 0.0
        self.time_step = 0
        self.last_actions[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for controller in self.bam_controllers:
            controller.reset(self.data.qpos)
            controller.last_ts = self.data.time

    def _key_callback(self, key: int) -> None:
        if self.policy.is_tracking:
            return
        changes = {
            265: (0, 0.1),
            264: (0, -0.1),
            263: (1, 0.1),
            262: (1, -0.1),
            81: (2, 0.1),
            82: (2, -0.1),
        }
        if key in changes:
            axis, delta = changes[key]
            self.command[0, axis] += delta

    def _robot_observations(self) -> dict[str, np.ndarray]:
        joint_pos_raw = self.data.qpos[7:].copy().reshape(1, -1)
        joint_pos = joint_pos_raw - self.policy.default_joint_pos
        joint_vel = self.data.qvel[6:].copy().reshape(1, -1)
        hip_quat = self.data.xquat[self.model.body("hip_mid_section").id]
        inverse_hip = Rotation.from_quat(hip_quat, scalar_first=True).inv()
        gravity = inverse_hip.apply((0.0, 0.0, -1.0)).reshape(1, 3)
        ang_vel = self.data.sensor("hip_gyro").data.copy().reshape(1, 3)

        reference = self.policy.reference_at(self.time_step)
        command = (
            np.concatenate(reference, axis=1) if reference is not None else self.command
        )
        self.samples["joint_pos"].append(joint_pos_raw.copy())
        self.samples["joint_vel"].append(joint_vel.copy())
        self.samples["ang_vel"].append(ang_vel.copy())
        self.samples["gravity"].append(gravity.copy())
        return {
            "command": command,
            "base_ang_vel": ang_vel,
            "body_ang_vel": ang_vel,
            "projected_gravity": gravity,
            "body_projected_gravity": gravity,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "actions": self.last_actions,
        }

    def run(self) -> None:
        while self.viewer.is_running():
            start = time.perf_counter()
            if (
                self.policy.motion_length is not None
                and self.time_step >= self.policy.motion_length
            ):
                if not self.args.loop:
                    break
                self._reset_state()

            observations = concatenate_observations(
                self.policy, self._robot_observations()
            )
            outputs = self.policy.infer(observations, self.time_step)
            self.last_actions = outputs[0].astype(np.float32)
            target = (
                self.last_actions * self.policy.action_scale
                + self.policy.default_joint_pos
            )[0]
            self.samples["commanded_joint_pos"].append(target.copy())
            if self.bam_controllers:
                for controller in self.bam_controllers:
                    controller.q_target[:] = target[controller.act_indexes]
                for _ in range(4):
                    for controller in self.bam_controllers:
                        controller.update()
                    mujoco.mj_step(self.model, self.data)
            else:
                self.data.ctrl[:] = target
                for _ in range(4):
                    mujoco.mj_step(self.model, self.data)
            self.viewer.cam.lookat = self.data.xpos[self.model.body("chest").id]
            self.viewer.cam.distance = 1.2
            self.viewer.cam.elevation = -20.0
            self.viewer.sync()
            self.time_step += 1
            if not self.args.no_realtime:
                time.sleep(max(0.0, 0.02 - (time.perf_counter() - start)))

    def save(self) -> None:
        if self.args.record is None or not self.samples["joint_pos"]:
            return
        self.args.record.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: np.concatenate(value) for key, value in self.samples.items() if value
        }
        np.savez(self.args.record, **payload)  # pyright: ignore[reportArgumentType]


def main() -> None:
    args = parse_args()
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
            "General-motion sim2sim requires an external 48-value reference "
            "provider, which is deferred to the real-time imitation release."
        )
    if args.benchmark:
        benchmark(policy)
    runner = Sim2SimRunner(policy, args)
    try:
        runner.run()
    finally:
        runner.save()
        runner.viewer.close()


if __name__ == "__main__":
    main()
