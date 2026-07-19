"""Inspect Dynamixel wiring and capture a torque-off neutral calibration."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from swenoid.deployment.hardware_config import HardwareConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Existing hardware JSON profile.")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=4_000_000)
    parser.add_argument("--serial-retries", type=int, default=3)
    parser.add_argument(
        "--configure-latency",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--capture-zero",
        action="store_true",
        help="Save current torque-off positions as the simulation zero pose.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("hardware-calibration.json")
    )
    parser.add_argument("--name", help="Name stored in a captured profile.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_values(handler: Any, method: str, ids: Sequence[int]) -> list[int]:
    values = getattr(handler, method)(ids)
    return [int(value[0] if isinstance(value, tuple) else value) for value in values]


def inspect_hardware(handler: Any, config: HardwareConfig) -> tuple[int, ...]:
    """Disable torque, read all motors, and print an auditable mapping table."""
    ids = list(config.motor_ids)
    handler.disable_torque(ids)
    positions = _read_values(handler, "read_servo_positions", ids)
    lower = _read_values(handler, "read_lower_limits", ids)
    upper = _read_values(handler, "read_upper_limits", ids)
    voltages = _read_values(handler, "read_servo_voltages", ids)
    print("id  dynamixel_joint                    position  limits       voltage")
    for motor_id, joint, position, low, high, voltage in zip(
        ids,
        config.dynamixel_joint_names,
        positions,
        lower,
        upper,
        voltages,
        strict=True,
    ):
        status = "OK" if low <= position <= high else "OUTSIDE LIMITS"
        print(
            f"{motor_id:>2}  {joint:<34} {position:>4}      "
            f"{low:>4}..{high:<4}  {voltage / 10:>5.1f} V  {status}"
        )
    outside = [
        motor_id
        for motor_id, position, low, high in zip(
            ids, positions, lower, upper, strict=True
        )
        if not low <= position <= high
    ]
    if outside:
        raise RuntimeError(f"Motors outside configured limits: {outside}")
    return tuple(positions)


def main() -> None:
    args = parse_args()
    if args.baudrate <= 0 or args.serial_retries <= 0:
        raise SystemExit("Hardware rates and retry counts must be positive")
    config = HardwareConfig.load(args.config)
    motor_controller = import_module("swenoid.deployment.motor_controller")
    handler = motor_controller.DynamixelHandler(
        port=args.port,
        baudrate=args.baudrate,
        configure_latency=args.configure_latency,
        max_retries=args.serial_retries,
    )
    try:
        positions = inspect_hardware(handler, config)
    finally:
        handler.close()
    if not args.capture_zero:
        print("Inspection complete; no calibration file was written.")
        return
    calibrated = config.with_zero_positions(
        positions, name=args.name or f"{config.name}-calibrated"
    )
    calibrated.write(args.output, overwrite=args.overwrite)
    print(f"Wrote torque-off neutral calibration to {args.output}")


if __name__ == "__main__":
    main()
