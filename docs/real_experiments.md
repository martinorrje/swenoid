# Recording physical-robot experiments

The physical runner records the policy inputs, outputs, motor feedback, IMU
state, electrical telemetry, timing, and trial provenance needed to reproduce
the hardware results. Global position and linear velocity are not observable
from the onboard sensors. Locomotion trials therefore require an external pose
source when they will be used for linear-velocity error or endpoint-deviation
results.

This document describes schema version 1 (`swenoid.real_experiment`). Treat the
schema version in the archive and manifest as authoritative when writing
analysis code.

## Before collecting data

Define the experimental protocol before the first trial. At minimum, fix:

- the policies, commands or reference motions, trial duration, and success
  criterion;
- the robot configuration, calibration, surface, support arrangement, and
  battery-handling rule;
- which failures are experimental outcomes and which are infrastructure
  failures; and
- the number of policy seeds and repetitions justified by the planned
  analysis.

Do not discard a fall, safety abort, poor tracking run, deadline miss, or clipped
command because it makes a condition look worse. Those are results. An
infrastructure failure may be excluded only under a rule fixed in advance, and
its reason must be recorded.

Copy [the metadata template](../examples/real_experiments/metadata.json) for
each trial or block and fill every field relevant to the setup. Empty strings
and `null` values are placeholders, not publication-ready metadata. Use stable
identifiers for the robot, battery, command sequence, motion, and experimental
block.

The standard hardware safety checklist still applies. Recording is diagnostic;
it is not an emergency stop and does not replace current limits, mechanical
support, a spotter, or an independent means of removing power.

## Run a recorded trial

A fixed-duration velocity trial with external tracking has the following form:

The values are illustrative; choose the trial length and repeat structure before
collection.

```bash
uv run --no-sync swenoid-deploy \
  --onnx /absolute/path/to/policy.onnx \
  --hardware-config /absolute/path/to/hardware-calibration.json \
  --command 0.25 0.0 0.0 \
  --command-start-step 50 \
  --trial-steps 500 \
  --trial-id velocity-nominal-seed03-r02 \
  --condition nominal \
  --robot-id swenoid-01 \
  --training-seed 3 \
  --replicate 2 \
  --experiment-metadata examples/real_experiments/metadata.json \
  --telemetry-every 1 \
  --external-pose-bind 0.0.0.0 \
  --external-pose-port 5501 \
  --external-pose-baseline-samples 2 \
  --external-pose-timeout-s 2.0 \
  --external-pose-max-clock-offset-ms 1000 \
  --record data/real/velocity-nominal-seed03-r02
```

Use `--wandb-run` instead of `--onnx` when appropriate. A tracking trial ends at
the embedded sequence boundary; `--trial-steps` can impose an earlier, explicit
limit across conditions. Recorded trials reject `--loop`, because an implicit
number of repetitions makes completion ambiguous. For a velocity policy,
`--command-start-step` and `--velocity-steps` define the half-open active-command
interval. They are separate from `--trial-steps`, which stops recording and ends
the trial. A step limit is mandatory for every recorded velocity trial.

Recording options are:

| Option | Meaning |
| --- | --- |
| `--record PATH` | Enable schema-v1 recording. A path ending in `.npz` selects the flat layout; any other path selects a trial directory. |
| `--trial-id ID` | Stable, unique trial identifier. Required with `--record`. |
| `--condition NAME` | Experimental condition, such as `nominal` or `identified`. Required with `--record`. |
| `--robot-id ID` | Identifier for the physical robot. Required with `--record`. |
| `--training-seed N` | Training seed for the policy under test, when known. |
| `--replicate N` | Replicate number within the matched experimental design. |
| `--experiment-metadata FILE` | JSON object merged into the trial context in the manifest; requires `--record`. |
| `--trial-steps N` | Stop after exactly `N` completed control samples; `N` must be positive. Required for recorded velocity trials. |
| `--command-start-step N` | Non-negative first policy step at which a velocity command is active; defaults to zero. |
| `--velocity-steps N` | Number of steps for which the velocity command remains active after `--command-start-step`; defaults to 200. |
| `--telemetry-every N` | Read current, input voltage, and temperature every positive `N` recorded control samples; defaults to every sample. Increase it if telemetry threatens the 20 ms budget, and use the same value across conditions. |
| `--checkpoint-steps N` | Replace the recoverable partial archive every positive `N` completed samples; defaults to 250. |
| `--overwrite-record` | Replace an existing record at the exact target. Omit this during normal collection. |
| `--external-pose-bind ADDRESS` | Local IPv4 address on which to receive pose datagrams; defaults to `0.0.0.0` and requires `--record`. |
| `--external-pose-port PORT` | Enable the UDP JSON external-pose receiver on this port; requires `--record`. |
| `--external-pose-baseline-samples N` | Require `N` usable packets (`valid` and quality above zero) before the first policy goal; defaults to 2. |
| `--external-pose-timeout-s S` | Maximum wait for each baseline or final-endpoint evidence phase; defaults to 2.0 s. |
| `--external-pose-max-clock-offset-ms MS` | Set a positive source-to-receipt offset limit; farther timestamps are rejected; defaults to 1000 ms. |

With external pose enabled, the receiver must collect the configured number of
usable baseline packets before the runner sends its first policy goal. After the
last goal, it waits for a newer usable endpoint before finalizing a natural
completion. Either wait is bounded by `--external-pose-timeout-s`; a timeout is
recorded as an error, not a clean success. The waits send no additional policy
goals and do not change the established torque-on-exit behavior. After the
robot is secure, annotate a tracker timeout or fatal receiver failure as
`infrastructure_failure` with an exclusion reason, never as `success`.

The runner refuses to overwrite any file belonging to an existing record unless
`--overwrite-record` is explicit. Use a new trial ID instead of overwriting a
completed experiment. Checkpoints are written on a background thread with an
atomic replace; the final archive and manifest are also replaced atomically.
An exception or operator interruption still finalizes the samples collected up
to the last completely recorded control step. The manifest records the automatic
termination reason and any exception. This termination reason is not a
substitute for the observed trial outcome.

## Files and recovery

Directory layout, recommended for archival:

```text
velocity-nominal-seed03-r02/
├── manifest.json
├── events.jsonl
└── samples.npz
```

While recording, `samples.partial.npz` contains the latest completed
checkpoint, including the control-rate arrays, servo telemetry, and all external
pose rows captured by that checkpoint snapshot. It is removed after
`samples.npz` has been finalized. If the host loses power, retain the partial
archive, manifest, and event log; do not rename the partial archive to a final
result or mark it successful.

For `--record data/trial.npz`, the equivalent flat layout is:

```text
data/trial.npz
data/trial.json
data/trial.events.jsonl
```

The temporary checkpoint is `data/trial.partial.npz`.

`manifest.json` (or the flat `.json`) contains the schema, trial and policy
identity, policy and configuration hashes, calibrated joint mapping, controller
settings, recorder status and counts, termination details, external-receiver
configuration and rejection statistics, and the reviewed outcome.
`events.jsonl` is an append-only stream; each line is an independently parseable
JSON object with local monotonic trial time, Unix time, sample index, event name,
and details. Runtime events are flushed and synced immediately. They include
lifecycle and velocity-command changes, controller errors, the
`external_pose_receiver_started`, `external_pose_baseline_ready`, and
`external_pose_endpoint_ready` handshake events, and later result annotations.
Per-step clipping and deadline state belongs in the NPZ arrays, not the event
log.

The NPZ contains only numeric or Unicode arrays and can always be opened with
pickle disabled:

```python
import json
from pathlib import Path

import numpy as np

trial = Path("data/real/velocity-nominal-seed03-r02")
manifest = json.loads((trial / "manifest.json").read_text(encoding="utf-8"))

with np.load(trial / "samples.npz", allow_pickle=False) as samples:
    time_s = samples["time_s"].copy()
    measured_q = samples["joint_position_rad"].copy()
    requested_q = samples["joint_target_requested_rad"].copy()
    clipped = samples["dynamixel_goal_clipped"].copy()
```

Copy arrays that must outlive the `with` block. Never load an experiment archive
with `allow_pickle=True`.

## Array reference

Let `N` be the number of completed 50 Hz control samples, `S` the number of
lower-rate servo-telemetry samples, `M` the number of asynchronous external-pose
packets, `O` the ONNX observation width, and `C` the command width. Joint-space
SI arrays use the simulation joint order recorded in the manifest. Arrays whose
names begin with `dynamixel_` and all servo-telemetry arrays use the Dynamixel
bus order recorded as `dynamixel_joint_names` and `motor_ids` in the manifest.

### Schema and control-rate state

| Field | Shape | Type | Unit and meaning |
| --- | ---: | --- | --- |
| `schema_name` | scalar | Unicode | `swenoid.real_experiment`. |
| `schema_version` | scalar | int64 | `1`. |
| `sample_index` | `(N,)` | int64 | Contiguous record index starting at zero. |
| `policy_frame` | `(N,)` | int64 | Reference frame supplied to the policy. |
| `time_s` | `(N,)` | float64 | Monotonic seconds since recorder start, sampled in the controller process. |
| `time_unix_ns` | `(N,)` | int64 | Host Unix time in nanoseconds for cross-device alignment. |
| `control_dt_s` | `(N,)` | float32 | Interval since the previous control sample; the first value is `NaN`. |
| `read_duration_s` | `(N,)` | float32 | Position/velocity feedback, IMU snapshot, and any scheduled servo-telemetry read time. |
| `inference_duration_s` | `(N,)` | float32 | ONNX inference time. |
| `write_duration_s` | `(N,)` | float32 | Per-step movement safety check and Dynamixel goal-write time. |
| `cycle_duration_s` | `(N,)` | float32 | Controller work from cycle start through the motor goal write; excludes recording and deadline sleep. |
| `deadline_missed` | `(N,)` | bool | `true` when that pre-recording controller work exceeded the 20 ms budget. |
| `policy_observation` | `(N, O)` | float32 | Exact flattened observation passed to ONNX, in the metadata-defined order. |
| `policy_action` | `(N, 24)` | float32 | Normalized ONNX output before action scaling. |
| `command` | `(N, C)` | float32 | Velocity command `(vx, vy, yaw_rate)` or tracking reference `(q_ref, dq_ref)`. |
| `reference_joint_position_rad` | `(N, 24)` | float32 | Tracking position reference; `NaN` for a velocity policy. |
| `reference_joint_velocity_rad_s` | `(N, 24)` | float32 | Tracking velocity reference; `NaN` for a velocity policy. |
| `joint_position_rad` | `(N, 24)` | float32 | Encoder position converted to radians and simulation order. |
| `joint_velocity_rad_s` | `(N, 24)` | float32 | Encoder velocity converted to rad/s and simulation order. |
| `joint_target_requested_rad` | `(N, 24)` | float32 | Policy target before hardware position-limit clipping. |
| `joint_target_sent_rad` | `(N, 24)` | float32 | Exact radian target represented by the integer goal sent to each motor. |
| `dynamixel_position_raw` | `(N, 24)` | int32 | Present-position register values in motor order. |
| `dynamixel_velocity_raw` | `(N, 24)` | int32 | Signed present-velocity register values in motor order. |
| `dynamixel_goal_position_raw` | `(N, 24)` | int32 | Integer goal-position values sent in motor order. |
| `dynamixel_goal_clipped` | `(N, 24)` | bool | Per-motor indication that a requested goal exceeded a configured limit. |
| `base_angular_velocity_raw_rad_s` | `(N, 3)` | float32 | Unfiltered pelvis-frame BNO085 gyroscope sample. |
| `base_angular_velocity_rad_s` | `(N, 3)` | float32 | Gyroscope sample after the deployment filter; this is the value seen by the policy. |
| `projected_gravity_raw` | `(N, 3)` | float32 | Unfiltered, normalized gravity direction in the pelvis frame. |
| `projected_gravity` | `(N, 3)` | float32 | Filtered projected gravity supplied to the policy. |
| `base_orientation_wxyz` | `(N, 4)` | float32 | BNO085 pelvis orientation quaternion, scalar first. |
| `imu_sample_time_s` | `(N,)` | float64 | Monotonic recorder-relative time at which the IMU thread acquired the sample. |
| `imu_age_s` | `(N,)` | float32 | Age of that IMU sample when the control loop read it. |
| `imu_sequence` | `(N,)` | int64 | IMU-thread sequence number; repeated values reveal reused samples. |

`joint_pos`, `joint_vel`, `ang_vel`, and `gravity` are read-only compatibility
aliases for the canonical SI arrays. New analysis must use the unit-bearing
names. There is deliberately no ambiguous `commanded_joint_pos` alias.

### Multi-rate servo telemetry

The electrical read is a separate Dynamixel Group Sync Read. Its cadence is
controlled by `--telemetry-every` so it does not have to run in every
20 ms control cycle.

| Field | Shape | Type | Unit and meaning |
| --- | ---: | --- | --- |
| `servo_telemetry_sample_index` | `(S,)` | int64 | Control sample at which the telemetry read was requested. |
| `servo_telemetry_time_s` | `(S,)` | float64 | Recorder-relative completion time of the telemetry read. |
| `servo_current_raw` | `(S, 24)` | int16 | Signed Present Current register. |
| `servo_current_a` | `(S, 24)` | float32 | Present current in amperes (`raw * 0.00269`). |
| `servo_voltage_raw` | `(S, 24)` | uint16 | Present Input Voltage register. |
| `servo_voltage_v` | `(S, 24)` | float32 | Input voltage in volts (`raw * 0.1`). |
| `servo_temperature_c` | `(S, 24)` | float32 | Present Temperature in degrees Celsius. |

The scaling comes from the official ROBOTIS control tables for both the
[XM430-W350](https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/) and
[XM540-W270](https://emanual.robotis.com/docs/en/dxl/x/xm540-w270/): Present
Current uses 2.69 mA per unit, Present Input Voltage uses 0.1 V per unit, and
Present Temperature uses 1 degree Celsius per unit. Retain the raw arrays so a
future firmware or hardware-specific interpretation can be audited.

### External pose

External pose is asynchronous and is not repeated to match the control rate.
Apply the signed clock conversion below, then interpolate eligible samples
onto `time_s`. State the interpolation and filtering method in the paper. A packet is eligible for
metric coverage only when `external_pose_valid` is true and
`external_pose_quality` is greater than or equal to the predeclared
`external_tracking.minimum_quality`. The sender defines that quality scale;
record its meaning in `external_tracking.quality_definition`.
`external_tracking.maximum_gap_ms` predeclares the largest permitted gap
between consecutive eligible packets over the control interval; baseline and
endpoint packets alone are not continuous trajectory evidence.

| Field | Shape | Type | Unit and meaning |
| --- | ---: | --- | --- |
| `external_pose_receive_time_s` | `(M,)` | float64 | Controller monotonic time at UDP receipt, relative to trial start. |
| `external_pose_receive_unix_ns` | `(M,)` | int64 | Controller Unix receipt time in nanoseconds. |
| `external_pose_source_time_ns` | `(M,)` | int64 | Acquisition time supplied by the tracking host. |
| `external_pose_sequence` | `(M,)` | int64 | Strictly increasing sender sequence number. |
| `external_base_position_world_m` | `(M, 3)` | float32 | Pelvis-origin position in the named world frame. |
| `external_base_orientation_world_wxyz` | `(M, 4)` | float32 | Unit quaternion rotating pelvis-frame vectors into the named world frame, scalar first. |
| `external_base_linear_velocity_world_m_s` | `(M, 3)` | float32 | Optional world-frame linear velocity; all `NaN` when omitted. |
| `external_base_angular_velocity_world_rad_s` | `(M, 3)` | float32 | Optional world-frame angular velocity; all `NaN` when omitted. |
| `external_pose_valid` | `(M,)` | bool | Tracking-system validity flag. |
| `external_pose_quality` | `(M,)` | float32 | Sender-defined quality in `[0, 1]`. |
| `external_pose_frame_id` | `(M,)` | Unicode | Sender frame name. It must remain constant within a trial. |

## External-pose UDP contract

Send one UTF-8 JSON object per UDP datagram to the configured port:

```json
{
  "source_time_ns": 1784451600123456789,
  "sequence": 42,
  "position_m": [1.240, -0.031, 0.615],
  "orientation_wxyz": [0.9998, 0.0, 0.0, 0.0200],
  "linear_velocity_m_s": [0.247, -0.004, 0.001],
  "angular_velocity_rad_s": [0.002, -0.006, 0.101],
  "valid": true,
  "quality": 0.98,
  "frame_id": "mocap_world"
}
```

`source_time_ns`, `sequence`, `valid`, and `frame_id` are required. A valid
sample also requires finite `position_m` and `orientation_wxyz`; the quaternion
norm must be within 0.01 of one. Linear and angular velocity are optional and
become `NaN` arrays when omitted. For an invalid sample, pose may be omitted.
Quality defaults to one for a valid sample and zero for an invalid sample.
The first accepted packet fixes `frame_id` for the trial. The receiver rejects
malformed packets, repeated or decreasing timestamps or sequence numbers, any
subsequent frame-ID change, timestamps that are not Unix nanoseconds, and any
packet whose absolute source-to-receipt offset exceeds
`--external-pose-max-clock-offset-ms`. Rejections are counted in the manifest
and cannot satisfy baseline, endpoint, or metric coverage. UDP delivery itself
is not guaranteed.

`source_time_ns` is the tracking measurement's Unix acquisition time, not its
send time. Synchronize the tracking host, controller, and any video host with
PTP or a measured NTP setup, and record the method and observed offset in trial
metadata. Receipt time alone includes unknown acquisition and network latency
and is not sufficient for publication-grade dynamic alignment. The default
1000 ms limit catches wrong clock domains; it is not an acceptable sync target.
Use a tighter bound justified by the tracking system. Define
`external_tracking.estimated_clock_offset_ms` as tracker/source clock minus
controller clock. Convert each acquisition timestamp to controller time by
subtracting that signed offset:

```text
source_time_on_controller_ns = source_time_ns - estimated_clock_offset_ms * 1e6
```

Use the corrected epoch time to map each acquisition onto control `time_s`
before interpolation. Record how the offset was measured.

Define the transform before collection. `position_m` is the pelvis rigid-body
origin expressed in `frame_id`; `orientation_wxyz` rotates pelvis-frame
vectors into that same right-handed world frame. Keep rigid-body marker
placement, axis directions, handedness, quaternion convention, and world origin
fixed across conditions. Archive the tracking-system rigid-body definition and
calibration.

## Outcome annotation and validation

The runner can determine that a sequence ended, a step limit was reached, an
operator interrupted it, or software raised an exception. It cannot determine
whether the robot fell, was caught, or satisfied the experiment's success
criterion. Review each trial against the event log and synchronized video, then
annotate it:

```bash
uv run --no-sync swenoid-experiment annotate \
  data/real/velocity-nominal-seed03-r02 \
  --outcome success \
  --notes "Completed the fixed command sequence without contact or support."
```

Allowed outcomes are `success`, `fall`, `safety_abort`,
`infrastructure_failure`, and `incomplete`. Add `--external-intervention` if a
person, tether, or support altered the motion after the trial began. A success
cannot include external intervention. `infrastructure_failure` requires
`--exclusion-reason`; it and `incomplete` are marked invalid for the primary
analysis. Re-annotation retains the previous reviewed result in the manifest's
annotation history.

Validate immediately after annotation and again before aggregation:

```bash
uv run --no-sync swenoid-experiment validate \
  data/real/velocity-nominal-seed03-r02 \
  --paper-ready
```

Basic validation checks schema, finalization, required arrays, aligned control
lengths, contiguous indices, monotonic timestamps, and tracking references.

For `--paper-ready`, a record must also have:

- at least one control sample; `trial_id`, `condition`, and `robot_id`; immutable
  policy and hardware-configuration evidence; and clean Git provenance;
- a reviewed outcome marked valid for analysis, no recorded controller or
  finalization error, and no `controller_error` termination. A success must end
  naturally with `trial_steps_reached` or `motion_complete`;
- nonempty `condition_details.simulator_actuator_model` and `training_run`;
  protocol `success_criterion`, `evaluation_window`, and
  `infrastructure_exclusion_rule`; `environment.surface`;
  `robot_configuration.battery_pack_id`; and trial-design `block_id`,
  `command_sequence_id`, and `order_assignment`; and
- for an identified or BAM condition, a nonempty
  `condition_details.actuator_parameter_files` list. Every entry must be an
  object with a nonempty `path` or `identifier` and a 64-character hexadecimal
  `sha256`; include the fitting revision as well. Nominal conditions may use an
  empty list, but must spell out `kp=20`, damping `1`, and armature `0.01` in
  `simulator_actuator_model`.

A paper-ready velocity trial additionally requires external-tracking `system`,
`rigid_body`, `world_frame`, `calibration_id`, `clock_sync_method`, finite
`estimated_clock_offset_ms`, `minimum_quality` in `[0, 1]`, a finite positive
`maximum_gap_ms`, and a nonempty `quality_definition`. Validation checks equal
external-array counts and one stable archive frame matching the declared
`world_frame`. It subtracts `estimated_clock_offset_ms` from source timestamps
before testing coverage: an eligible baseline must precede the first control
sample and an eligible endpoint must follow final goal-write completion. The
corrected acquisition timeline must cover the control interval without a gap
between consecutive eligible packets above `maximum_gap_ms`. A packet is
eligible only when it is valid, has positive quality, and meets the declared
quality threshold.

Passing validation says that the evidence is structurally complete and
internally consistent; it does not certify the calibration, metric definition,
or experimental design.

## From fields to paper results

Compute all per-trial quantities before aggregating across trials. Preserve the
per-trial values and analysis code used to produce every table cell.

| Paper result | Recorded source and calculation |
| --- | --- |
| Linear-velocity RMSE | Resample eligible `external_base_linear_velocity_world_m_s` to `time_s`, rotate it into the commanded body frame with the external quaternion, and compare it with `command[:, :2]` over the predeclared evaluation interval. If the sender omits velocity, differentiate and filter `external_base_position_world_m` with a fixed, reported method. External pose is mandatory. |
| Yaw-rate RMSE | Compare `command[:, 2]` with eligible externally measured yaw rate when available. `base_angular_velocity_rad_s[:, 2]` provides an onboard pelvis-IMU result and diagnostic; state explicitly which source is reported. |
| Endpoint lateral deviation | Express the final eligible `external_base_position_world_m` relative to the initial position and heading, then take the lateral component. External pose is mandatory. |
| Successful locomotion trials | Count reviewed `result.success` over trials marked `valid_for_analysis`. Falls and safety aborts remain in the denominator as failures. Report infrastructure exclusions and reasons separately. |
| Tracking joint-position MAE | Mean absolute difference between `joint_position_rad` and `reference_joint_position_rad`, in the simulation joint order, over the declared tracking interval. |
| Tracking joint-velocity MAE | Mean absolute difference between `joint_velocity_rad_s` and `reference_joint_velocity_rad_s` over the same interval. |
| Motion completion | Apply the predeclared completion rule to `policy_frame`, the manifest's reference identity and length, termination reason, and reviewed outcome. A caught or fallen robot is not a completion even if commands continued. |
| Falls per motion | Count reviewed `result.fall` for each motion and report the valid-trial denominator. Do not infer falls solely from IMU thresholds. |

The projected-gravity vector supports a combined tilt diagnostic, for example
the angle between `projected_gravity` and the upright vector. External pose
supports pelvis height and world-referenced roll and pitch. Neither onboard
joint encoders nor the IMU alone provide global planar displacement or linear
velocity.

Use timing and actuator signals to explain failures without redefining them:

- report the rate and longest run of `deadline_missed`, with the distributions
  of `control_dt_s`, `cycle_duration_s`, and `inference_duration_s`; use
  `control_dt_s` to detect cadence slips caused by recording overhead;
- report the fraction of samples and joints for which
  `dynamixel_goal_clipped` is true, and inspect requested versus sent targets;
- inspect motor-order current, voltage, and temperature alongside the mapped
  joint names; and
- inspect IMU age, repeated `imu_sequence` values, tilt, joint error, and
  external tracking validity around falls and safety aborts.

These diagnostics are especially important when comparing actuator models:
apparent model differences must not be artifacts of control overruns, stale
sensors, depleted batteries, command clipping, or missing tracking data.

## Comparing nominal and identified actuator models

Treat the nominal and identified cases as experimental conditions, not as
assumed explanations. Record the exact policy ONNX hash, training run, actuator
configuration, hardware calibration, and repository revision for every trial.
The nominal training condition should name the actual simulator settings
(`kp=20`, joint damping `1`, armature `0.01`); the identified condition should
name the exact parameter files and fitting revision. The physical Dynamixel
position gain is a controller setting and must not be described as the
simulator's `kp`.

For an identified condition, use auditable parameter entries such as:

```json
[
  {
    "path": "params/xm430_m6.json",
    "revision": "git:<fitting-revision>",
    "sha256": "<64-hex-file-digest>"
  }
]
```

`identifier` may replace `path` for an immutable artifact-store identity. The
digest must be the actual parameter-file SHA-256, not the metadata-file digest.

Use matched policy seeds, commands or motions, step limits, and evaluation
windows. Interleave or randomize condition order within blocks so warm motors,
battery discharge, surface changes, and operator learning do not coincide with
one condition. Record the order assignment and block ID. Keep the same robot,
calibration, support policy, initial pose, surface, and external-tracking
calibration unless the change itself is under study. Select the number of seeds
and repeats before seeing the final result; this repository does not prescribe
a universal sample count.

Report negative transfer directly. A policy that flails, falls, or is stopped
for safety is a failed hardware trial, not missing tracking data. The recorder
supports a fair comparison, but it does not establish why an identified model
failed to transfer.

## Circle-drawing measurements

Circle dimensions are manual measurements and do not come from the policy
recorder. Copy
[drawing_measurements.csv](../examples/real_experiments/drawing_measurements.csv)
and add one row per attempt. Do not store only the mean. `horizontal_diameter_mm`
and `vertical_diameter_mm` are the maximum extents of the same traced curve;
record the target, robot-to-board distance, board tilt from vertical, image
scale, measurement method and instrument resolution, observer, and UTC
measurement time as well.

Photograph or scan each drawing with a scale in the board plane, retain the
original file, and put its relative path and SHA-256 digest in the row:

```bash
sha256sum artifacts/drawing/circle-001.tif
```

The hash ties a manual measurement to an immutable source image; it does not
replace calibration or measurement uncertainty. Archive the pen attachment
revision, pen and board details, target trajectory, IK configuration, setup
photographs, and any corresponding control record under the same trial ID.

## Archive checklist

For each reported experiment, preserve:

- the complete finalized record, including manifest, events, and samples;
- the exact ONNX file, hardware-calibration JSON, trial metadata, and hashes;
- raw external-tracking exports, rigid-body and coordinate-frame definitions,
  clock-synchronization evidence, and synchronized video;
- drawing images and the completed manual-measurement CSV where applicable;
- the analysis scripts, generated per-trial metrics, repository commit, and
  locked Python environment used to build the tables and figures; and
- a trial index listing every attempted run, including excluded infrastructure
  failures and their reasons.

Make the archive read-only after analysis and checksum it. The recorder covers
velocity and stored single-motion hardware evaluation. The live camera
retargeting and real-time imitation implementation remains outside the current
repository and this recording protocol.
