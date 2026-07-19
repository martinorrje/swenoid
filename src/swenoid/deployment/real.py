"""Run a Swenoid ONNX policy on the physical robot at 50 Hz."""

from __future__ import annotations

import argparse
import threading
import time
from importlib import import_module
from pathlib import Path

import numpy as np

from swenoid.deployment.policy import (
    OnnxPolicy,
    benchmark,
    concatenate_observations,
    resolve_policy_path,
)
from swenoid.deployment.swenoid_control import SwenoidControl


def parse_args() -> argparse.Namespace:
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
        help="Set a velocity command to zero after this many 50 Hz steps.",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument(
        "--disable-torque-on-exit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable motor torque after a graceful exit; never used after errors.",
    )
    return parser.parse_args()


class ImuReader:
    """Continuously sample and lightly filter the pelvis BNO085."""

    def __init__(self, i2c_frequency: int = 800_000):
        self.angular_velocity = np.zeros((1, 3), dtype=np.float32)
        self.projected_gravity = np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32)
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.i2c_frequency = i2c_frequency
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
            self.angular_velocity[:] = np.asarray(sensor.get_ang_vel()).reshape(1, 3)
            self.projected_gravity[:] = self._validate_gravity(gravity_fn())
            self.ready.set()
            while not self.stop.is_set():
                angular_velocity = np.asarray(sensor.get_ang_vel()).reshape(1, 3)
                gravity = self._validate_gravity(gravity_fn())
                self.angular_velocity = (
                    0.45 * angular_velocity + 0.55 * self.angular_velocity
                ).astype(np.float32)
                self.projected_gravity = (
                    0.55 * gravity + 0.45 * self.projected_gravity
                ).astype(np.float32)
        except BaseException as exc:
            self.error = exc
            self.ready.set()

    @staticmethod
    def _validate_gravity(value) -> np.ndarray:
        gravity = np.asarray(value, dtype=np.float32).reshape(1, 3)
        norm = float(np.linalg.norm(gravity))
        if not np.isfinite(norm) or norm < 1e-6:
            raise ValueError(f"Invalid gravity vector from BNO085: {gravity}")
        return gravity

    def wait(self) -> None:
        self.ready.wait()
        if self.error is not None:
            raise RuntimeError("Failed to initialize BNO085") from self.error


class RealRobotRunner:
    """Hardware policy loop with position-step and deadline safety checks."""

    def __init__(self, policy: OnnxPolicy, args: argparse.Namespace):
        self.policy = policy
        self.args = args
        self.command = np.asarray(args.command, dtype=np.float32).reshape(1, 3)
        self.last_actions = np.zeros((1, 24), dtype=np.float32)
        self.time_step = 0
        self.imu = ImuReader(i2c_frequency=args.i2c_frequency)
        self.imu.wait()
        self.control = SwenoidControl(
            config=args.hardware_config,
            port=args.port,
            baudrate=args.baudrate,
            configure_latency=args.configure_latency,
            max_retries=args.serial_retries,
        )
        self.samples: dict[str, list[np.ndarray]] = {
            "joint_pos": [],
            "joint_vel": [],
            "commanded_joint_pos": [],
            "ang_vel": [],
            "gravity": [],
        }

    def move_to_start(self) -> None:
        target = self.control.pos_isaac_to_dynamixel(self.policy.default_joint_pos)
        self.control.enable_torques_at_current_position()
        self.control.move_servos_duration(target, [3000] * 24)
        input("Robot is at the default pose. Press Enter to start the policy...")

    def _observations(self) -> tuple[dict[str, np.ndarray], list[int]]:
        raw_position, raw_velocity = self.control.read_pos_vel()
        joint_pos_raw = self.control.pos_dynamixel_to_isaac(raw_position).reshape(1, -1)
        joint_pos = joint_pos_raw - self.policy.default_joint_pos
        joint_vel = self.control.vel_dynamixel_to_isaac(raw_velocity).reshape(1, -1)
        reference = self.policy.reference_at(self.time_step)
        command = (
            np.concatenate(reference, axis=1) if reference is not None else self.command
        )
        self.samples["joint_pos"].append(joint_pos_raw.copy())
        self.samples["joint_vel"].append(joint_vel.copy())
        self.samples["ang_vel"].append(self.imu.angular_velocity.copy())
        self.samples["gravity"].append(self.imu.projected_gravity.copy())
        return (
            {
                "command": command,
                "base_ang_vel": self.imu.angular_velocity,
                "body_ang_vel": self.imu.angular_velocity,
                "projected_gravity": self.imu.projected_gravity,
                "body_projected_gravity": self.imu.projected_gravity,
                "joint_pos": joint_pos,
                "joint_vel": joint_vel,
                "actions": self.last_actions,
            },
            raw_position,
        )

    def run(self) -> None:
        self.move_to_start()
        missed_deadlines = 0
        while True:
            start = time.perf_counter()
            if (
                self.policy.motion_length is not None
                and self.time_step >= self.policy.motion_length
            ):
                if not self.args.loop:
                    break
                self.time_step = 0

            observation_parts, raw_position = self._observations()
            observations = concatenate_observations(self.policy, observation_parts)
            outputs = self.policy.infer(observations, self.time_step)
            self.last_actions = outputs[0].astype(np.float32)
            target_rad = (
                self.last_actions * self.policy.action_scale
                + self.policy.default_joint_pos
            )[0]
            target_dynamixel = self.control.pos_isaac_to_dynamixel(target_rad)
            self.control.move_servos(target_dynamixel, raw_position)
            self.samples["commanded_joint_pos"].append(
                np.asarray(target_dynamixel).reshape(1, -1)
            )
            self.time_step += 1
            if (
                not self.policy.is_tracking
                and self.time_step >= self.args.velocity_steps
            ):
                self.command[:] = 0.0

            remaining = 0.02 - (time.perf_counter() - start)
            if remaining > 0.0:
                time.sleep(remaining)
            else:
                missed_deadlines += 1
                if missed_deadlines % 50 == 1:
                    print(f"[WARN] Missed 50 Hz deadline {missed_deadlines} time(s)")

    def execute(self) -> None:
        disable_torque = False
        try:
            self.run()
        except KeyboardInterrupt:
            disable_torque = self.args.disable_torque_on_exit
        else:
            disable_torque = self.args.disable_torque_on_exit
        finally:
            self.close(disable_torque=disable_torque)

    def close(self, *, disable_torque: bool) -> None:
        self.imu.stop.set()
        try:
            if disable_torque:
                self.control.disable_torques()
            else:
                print(
                    "[SAFETY] Motor torque remains enabled. Support the robot "
                    "before powering it down."
                )
        finally:
            try:
                self.control.close()
            finally:
                self.imu.thread.join(timeout=1.0)
                if self.args.record is not None and self.samples["joint_pos"]:
                    self.args.record.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        key: np.concatenate(value)
                        for key, value in self.samples.items()
                        if value
                    }
                    np.savez(  # pyright: ignore[reportArgumentType]
                        self.args.record,
                        **payload,  # pyright: ignore[reportArgumentType]
                    )


def main() -> None:
    args = parse_args()
    if args.velocity_steps < 0:
        raise SystemExit("--velocity-steps must be non-negative")
    if args.baudrate <= 0 or args.serial_retries <= 0 or args.i2c_frequency <= 0:
        raise SystemExit("Hardware rates and retry counts must be positive")
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
    if args.benchmark:
        benchmark(policy)
    runner = RealRobotRunner(policy, args)
    runner.execute()


if __name__ == "__main__":
    main()
