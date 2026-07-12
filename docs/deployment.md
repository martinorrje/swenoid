# ONNX deployment

Swenoid includes two runners built from the deployment programs used during
development:

- `swenoid-sim2sim` runs an exported policy in standalone MuJoCo;
- `swenoid-deploy` runs the same policy on the physical robot at 50 Hz.

Both runners inspect ONNX inputs to distinguish velocity from tracking policies.
A tracking model is identified by its `time_step` input, so detection does not
depend on fragile shape-inference heuristics. They also read
`observation_names`, `action_scale`, and `default_joint_pos` from ONNX metadata.

General-motion policies are recognized separately from velocity policies by
their 126-value deployable observation contract. They are deliberately rejected
by the current runners because they require a live 48-value joint-reference
provider. Camera input and online retargeting will be added with the real-time
imitation pipeline.

## Install

For sim2sim on a workstation:

```bash
uv sync --extra sim2sim
```

For physical deployment on the Raspberry Pi:

```bash
uv sync --extra deployment
```

The deployment extra installs ONNX Runtime, the ROBOTIS Dynamixel SDK, Adafruit
Blinka, and the Adafruit BNO08x driver. The repository itself now contains:

- `swenoid.deployment.motor_controller.DynamixelHandler`, including indirect
  position/velocity Fast Sync Read and goal-position Group Sync Write;
- `swenoid.deployment.bno085.BNO085`, including the physical-sensor to pelvis
  frame transform; and
- `swenoid.deployment.swenoid_control.SwenoidControl`, including joint ordering,
  direction, zero offsets, limits, and unit conversion.

### Raspberry Pi device setup

Enable I2C and reboot once:

```bash
sudo raspi-config nonint do_i2c 0
sudo reboot
```

Confirm that the BNO085 and U2D2 are visible before enabling motor power:

```bash
ls -l /dev/i2c-1 /dev/ttyUSB0
```

The 50 Hz loop requires the U2D2 USB latency timer to be 1 ms. The driver checks
this value but never invokes `sudo` internally. Configure it once after the U2D2
is connected:

```bash
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

If the U2D2 appears under a different name, pass `--port` and use the matching
sysfs directory. The defaults are `/dev/ttyUSB0`, 4 Mbaud, three bounded serial
attempts, and an 800 kHz I2C bus. They can be overridden with `--port`,
`--baudrate`, `--serial-retries`, and `--i2c-frequency`.

Test the transformed IMU output before connecting motor power:

```bash
uv run swenoid-bno085
```

## Sim2sim

Run a local ONNX file:

```bash
uv run swenoid-sim2sim --onnx /path/to/policy.onnx --loop
```

Or download it from W&B:

```bash
uv run swenoid-sim2sim \
  --wandb-run ENTITY/PROJECT/RUN_ID \
  --loop
```

For velocity policies, the initial command is zero. Pass
`--command 0.25 0.0 0.0`, or use the arrow keys for planar velocity and `Q`/`R`
for yaw. Tracking policies take their reference joint position and velocity from
the motion embedded in the ONNX model. `--loop` resets standalone MuJoCo at the
end of the motion. Add `--record data/sim2sim.npz` to save a rollout.

Standalone MuJoCo uses the selected XM430 M6 and XM540 M5 BAM fits by default,
matching policy training. Pass `--actuator-model dc` to use the XML's simple
position-servo baseline (`kp=20`, damping `1.0`, armature `0.01`) for comparison.

## Physical robot

Run from a supported Raspberry Pi control environment:

```bash
uv run swenoid-deploy \
  --wandb-run ENTITY/PROJECT/RUN_ID \
  --port /dev/ttyUSB0 \
  --command 0.25 0.0 0.0 \
  --record data/real_rollout.npz
```

The program warms up the IMU, checks battery voltage through the Dynamixel
handler, disables torque while configuring EEPROM-backed servo modes, moves to
the ONNX default pose over three seconds, and waits for an explicit Enter press
before starting. Keep the robot supported throughout initialization. It rejects
target jumps larger than 800 Dynamixel counts and warns about missed 20 ms
deadlines. A velocity command is zeroed after 200 steps by default; change this
with `--velocity-steps`.

The physical runner defaults to a zero velocity command. A nonzero command must
be supplied explicitly with `--command`. Targets outside the limits read from
the servos are rejected (with only sub-100-count numerical overshoots clamped),
and non-finite policy outputs are never sent.

By default, torque is disabled when the program exits. Use
`--no-disable-torque-on-exit` only when the robot is externally supported and a
separate controller will immediately take ownership.

## Safety checklist

Before an untethered test:

1. Support or suspend the robot and keep a physical emergency stop accessible.
2. Verify Dynamixel IDs, joint ordering, axis signs, zero offsets, and limits.
3. Run sim2sim with the exact ONNX file.
4. Benchmark inference and confirm substantial margin within the 20 ms budget.
5. Test zero actions and the initial pose with all joints unloaded where possible.
6. Start with conservative commands and a spotter.

The software checks do not replace current limiting, thermal protection,
mechanical stops, or an independent emergency-stop path.
