# Swenoid

Open-source software for **Swenoid**, a 0.68 m, 7.1 kg, 24-DoF humanoid robot.
This repository contains the MuJoCo digital twin, MjLab environments for
velocity control, single-motion tracking, and dataset-wide general motion
tracking, the BONES-SEED retargeting tools, and the policy training/export
workflow used for the robot.

The code is based on [MjLab](https://github.com/mujocolab/mjlab) and trains PPO
policies with [RSL-RL](https://github.com/leggedrobotics/rsl_rl) in
GPU-parallel MuJoCo environments. The simulation uses identified Dynamixel
XM430-W350 and XM540-W270 actuator models from
[BAM](https://github.com/Rhoban/bam).

> **Release status:** simulation, retargeting, velocity/single-motion/general-
> motion training, sim2sim, physical deployment, the 24-motor Dynamixel
> transport, and the pelvis BNO085 driver are included. Live camera retargeting
> and real-time imitation will be added later. The separate Swenoid CAD
> repository is being prepared and will be linked here before the hardware
> release.

## Repository ecosystem

| Repository | Contents | Status |
| --- | --- | --- |
| **`martinorrje/swenoid`** (this repository) | MuJoCo model, MjLab tasks, BONES-SEED retargeting, training, visualization, and ONNX export | Available |
| [`martinorrje/bam`](https://github.com/martinorrje/bam) | Swenoid BAM fork, raw-data processing, XM430/XM540 fitting, parameters, and fit plots | Fork publication pending |
| [`martinorrje/swenoid_cad`](https://github.com/martinorrje/swenoid_cad) | Editable CAD, print exports, assembly information, and bill of materials | Repository publication pending |

The MuJoCo meshes needed to run this repository are included, so the unpublished
editable CAD sources do not block simulation or training.

## What is included

- The complete Swenoid MJCF model and visual/collision meshes under
  `src/swenoid/assets/swenoid/`.
- Identified BAM parameter files for XM430-W350 M6 and XM540-W270 M5.
- Flat- and rough-terrain velocity tasks.
- A **single-reference-motion tracking** task. This is distinct from a general
  multi-motion tracking policy: one policy is trained for one motion file.
- A **general motion tracking** task that samples a large retargeted motion
  dataset through a bounded active-motion pool and cache.
- Actor observations that can be measured on the real robot; neither policy
  requires base position, heading, or base linear velocity.
- Direct BONES-SEED Unitree G1-to-Swenoid retargeting, optional feasibility
  filtering, 50 Hz motion conversion, rendering, and W&B Registry upload.
- Automatic PyTorch checkpoint and ONNX export with W&B logging.
- Cleaned real-robot and standalone-MuJoCo ONNX deployment programs based on the
  programs used for the experiments.

The released retargeting pipeline does **not** use PHUMA adaptation modules or a
PHUMA post-processing stage. It directly maps the provided G1 trajectories to
Swenoid, then optionally rejects infeasible sequences without modifying them.

## Requirements

Training requires:

- Linux on x86-64;
- a recent NVIDIA GPU and driver supported by PyTorch, Warp, and MuJoCo Warp;
- Python 3.12 or 3.13; and
- [`uv`](https://docs.astral.sh/uv/).

CPU and Apple Silicon can be used for light preprocessing and MuJoCo evaluation,
but GPU-parallel training requires NVIDIA CUDA.

## Installation

```bash
git clone https://github.com/martinorrje/swenoid.git
cd swenoid
uv sync --extra training --extra retargeting --extra sim2sim
uv run swenoid-list-envs | grep Swenoid
```

Install ONNX Runtime for sim2sim:

```bash
uv sync --extra sim2sim
```

On the Raspberry Pi, install the complete hardware deployment stack:

```bash
uv sync --extra deployment
```

The environment is locked by `uv.lock`. The training, retargeting, sim2sim, and
deployment extras are separate so a Raspberry Pi does not install MjLab or the
CUDA training stack. Do not install packages into a separate Conda environment
and then run bare `python`; use `uv run ...` from this checkout so each workflow
resolves consistently.

Log in once if W&B will be used:

```bash
uv run wandb login
```

For development, run:

```bash
uv run ruff format
uv run ruff check
uv run pyright
uv run pytest tests/
```

## Simulation and control rates

Both tasks use a 5 ms MuJoCo physics step and a decimation of four:

- physics and sensor state: **200 Hz**;
- policy inference and position commands: **50 Hz**.

Retargeted motions must therefore also be converted to 50 Hz. The converter in
this repository performs real interpolation—including quaternion SLERP—and
recomputes velocities at 50 Hz. The tracking loader rejects an NPZ whose stored
frame rate does not match the environment control rate, preventing a 120 Hz
trajectory from being silently played at the wrong speed.

The simulated actuator split is:

- XM540-W270: hips, knees, torso, and shoulder pitch/roll (15 joints);
- XM430-W350: ankles, shoulder yaw, elbows, and neck (9 joints).

See the BAM fork for the identification data, fitting commands, model comparison,
and the rationale for selecting XM430 M6 and XM540 M5.

The two compact X-series BAM registry adapters needed at simulation runtime are
also included here. They mirror the BAM fork definitions, allowing a clean
installation from the released `better-actuator-models` package while keeping
the full identification workflow in its own repository.

## Policy observation and action contract

The velocity and tracking actors share the same robot-side proprioception, in
this exact order:

1. base angular velocity from the pelvis IMU (3);
2. gravity projected into the pelvis frame (3);
3. joint positions relative to the default pose (24);
4. joint velocities (24); and
5. the previous normalized policy action (24).

The task command is appended last:

- velocity: desired body-frame `x`, `y`, and yaw velocity (3), for 81 actor
  observations in total;
- single and general motion tracking: reference joint position and velocity
  (24 + 24), for 126 actor observations in total.

The actor does not observe global position, global orientation, heading, base
linear velocity, motion-anchor position, or terrain height. It outputs 24
normalized actions. Each action is scaled and added to the default joint pose to
form a 50 Hz Dynamixel position target. Exported ONNX files include
`observation_names`, `action_scale`, and `default_joint_pos` metadata so a
deployment program can reproduce this contract.

## Train a velocity policy

Flat ground:

```bash
uv run swenoid-train Mjlab-Velocity-Flat-Swenoid \
  --env.scene.num-envs 4096 \
  --agent.logger wandb \
  --agent.wandb-project swenoid \
  --agent.run-name velocity-flat
```

Rough terrain:

```bash
uv run swenoid-train Mjlab-Velocity-Rough-Swenoid \
  --env.scene.num-envs 4096 \
  --agent.logger wandb \
  --agent.wandb-project swenoid \
  --agent.run-name velocity-rough
```

The default Swenoid runner uses 30,000 PPO iterations and saves every 50
iterations. Override either value through `--agent.max-iterations` or
`--agent.save-interval`.

Visualize a W&B run in Viser without interrupting training:

```bash
uv run swenoid-play Mjlab-Velocity-Flat-Swenoid \
  --wandb-run-path ENTITY/swenoid/RUN_ID \
  --viewer viser \
  --num-envs 1
```

## Retarget BONES-SEED

### 1. Obtain the dataset

Download BONES-SEED from its
[official dataset page](https://bones.studio/datasets/seed) or
[Hugging Face repository](https://huggingface.co/datasets/bones-studio/seed)
and accept its license. The raw dataset and retargeted motion files are not
redistributed by this repository.

Set local paths:

```bash
export BONES_SEED_ROOT=/absolute/path/to/bones-seed
export SWENOID_MOTIONS=/absolute/path/to/swenoid-motion-workspace
mkdir -p "$SWENOID_MOTIONS"
```

The expected dataset tree contains `g1/csv/...` and `metadata/...`.

### 2. Directly map G1 trajectories to Swenoid

Convert one motion:

```bash
uv run swenoid-retarget \
  --dataset-root "$BONES_SEED_ROOT" \
  --motion g1/csv/221129/dancing_routine_2_004__A081_M.csv \
  --output-dir "$SWENOID_MOTIONS/qpos"
```

Convert every motion named by the metadata file:

```bash
uv run swenoid-retarget \
  --dataset-root "$BONES_SEED_ROOT" \
  --metadata-csv "$BONES_SEED_ROOT/metadata/seed_metadata_v004.csv" \
  --output-dir "$SWENOID_MOTIONS/qpos"
```

Use the metadata filename included in your dataset version. `--package`,
`--category`, and `--limit` can select a smaller batch.

The mapping copies compatible leg, torso, shoulder, and elbow joints; maps G1
ankle pitch to Swenoid's one-DoF ankle; converts degrees to radians and
centimeters to meters; scales root translation to the Swenoid neutral height;
clips target joint limits; and leaves unavailable wrist/ankle-roll and neck
signals at the neutral Swenoid pose.

### 3. Inspect and optionally filter

Render a retargeted trajectory:

```bash
uv run swenoid-render-motion \
  --qpos-csv "$SWENOID_MOTIONS/qpos/csv/221129/dancing_routine_2_004__A081_M.qpos.csv" \
  --output "$SWENOID_MOTIONS/dancing_routine_2.mp4"
```

The optional filter measures foot penetration/contact, center-of-mass support,
root and joint jitter, foot skating, and joint-limit violations. It only accepts
or rejects files; it never alters a trajectory:

```bash
uv run swenoid-filter-motions \
  --input-root "$SWENOID_MOTIONS/qpos" \
  --input-glob "$SWENOID_MOTIONS/qpos/**/*.qpos.csv" \
  --metrics-csv "$SWENOID_MOTIONS/curation/metrics.csv" \
  --accepted-list "$SWENOID_MOTIONS/curation/accepted.txt" \
  --rejected-list "$SWENOID_MOTIONS/curation/rejected.txt" \
  --accepted-root "$SWENOID_MOTIONS/accepted" \
  --exclude-support-props
```

### 4. Convert qpos to a 50 Hz tracking NPZ

For one or many accepted qpos files:

```bash
uv run swenoid-convert-motion \
  --input-root "$SWENOID_MOTIONS/qpos" \
  --input-list "$SWENOID_MOTIONS/curation/accepted.txt" \
  --output-root "$SWENOID_MOTIONS/npz" \
  --input-fps 120 \
  --output-fps 50
```

Set `--input-fps` to the true rate of the source release if it differs
from 120 Hz. The output stores joint and body pose/velocity arrays plus `fps=50`.

### 5. Optionally upload a motion to W&B Registry

```bash
uv run swenoid-upload-motion \
  "$SWENOID_MOTIONS/npz/csv/221129/dancing_routine_2_004__A081_M.npz" \
  --name dancing-routine-2
```

The command prints the exact registry identifier to pass to training.

## Train a single-motion tracking policy

Using a local motion:

```bash
uv run swenoid-train Mjlab-Tracking-Flat-Swenoid \
  --env.commands.motion.motion-file "$SWENOID_MOTIONS/npz/csv/221129/dancing_routine_2_004__A081_M.npz" \
  --env.scene.num-envs 4096 \
  --agent.logger wandb \
  --agent.wandb-project swenoid \
  --agent.run-name track-dancing-routine-2
```

Using W&B Registry instead:

```bash
uv run swenoid-train Mjlab-Tracking-Flat-Swenoid \
  --registry-name ENTITY/wandb-registry-motions/dancing-routine-2:latest \
  --env.scene.num-envs 4096 \
  --agent.logger wandb \
  --agent.wandb-project swenoid \
  --agent.run-name track-dancing-routine-2
```

Train separate runs for separate reference motions. This task intentionally does
not sample a multi-motion dataset.

Visualize a tracking run:

```bash
uv run swenoid-play Mjlab-Tracking-Flat-Swenoid \
  --wandb-run-path ENTITY/swenoid/RUN_ID \
  --motion-file "$SWENOID_MOTIONS/npz/csv/221129/dancing_routine_2_004__A081_M.npz" \
  --viewer viser \
  --num-envs 1
```

When a tracking run consumed a W&B motion artifact, `play` can recover that
artifact from the run and `--motion-file` may be omitted.

## Train a general motion tracking policy

General motion tracking samples all NPZ files below a retargeted motion root
while keeping only a bounded active pool and cache on the training device:

```bash
uv run swenoid-train Mjlab-General-Motion-Flat-Swenoid \
  --env.commands.motion.motion-root "$SWENOID_MOTIONS/npz" \
  --env.commands.motion.active-motion-count 128 \
  --env.commands.motion.max-cache-size 128 \
  --env.scene.num-envs 4096 \
  --agent.logger wandb \
  --agent.wandb-project swenoid \
  --agent.run-name general-motion
```

Alternatively, pass `--env.commands.motion.motion-list /path/to/list.txt` with
one absolute NPZ path per line. The actor receives the same measurable
proprioception as the other two policies. Its 48-value reference command is
only desired joint position and velocity; root/body reference state remains
available to the critic, rewards, resets, and diagnostics but not the actor.

The exported general-motion ONNX contains the policy and metadata, but does not
embed a particular motion. Supplying live reference joint trajectories from a
camera/retargeting process is intentionally deferred until the real-time
imitation pipeline is released. See
[docs/general_motion.md](docs/general_motion.md) for the dataset contract,
observation table, cache behavior, and export status.

## Checkpoints, ONNX, and deployment

Training writes local runs under `logs/rsl_rl/<experiment>/...` and logs to W&B
by default. Each save produces a PyTorch checkpoint and attempts an ONNX export.
With `--agent.upload-model true`, both are uploaded to the W&B run.

- A velocity ONNX policy has one input: `obs`.
- A single-motion ONNX policy has `obs` and `time_step` inputs and embeds its one
  reference motion. Its outputs are actions and the reference joint/body arrays.
- A general-motion ONNX policy has one `obs` input and expects reference joint
  position and velocity inside its 126-value observation. It does not embed a
  motion.

Real-robot inference must run every 20 ms, reproduce the observation ordering
listed above, use the exported action scale/default pose, and use the exact joint
ordering stored in the model metadata. Start with the robot supported, verify
axis signs and joint limits one actuator at a time, provide a physical emergency
stop, and do not enable all torques until zero-action behavior has been checked.

Run the exported policy in standalone MuJoCo:

```bash
uv run swenoid-sim2sim --wandb-run ENTITY/swenoid/RUN_ID --loop
```

Sim2sim uses the selected XM430 M6 and XM540 M5 BAM fits by default. Add
`--actuator-model dc` to compare against the XML's direct `kp=20`, damping
`1.0`, armature `0.01` servo model.

Run it on the robot after completing the Raspberry Pi device setup:

```bash
uv run swenoid-deploy --wandb-run ENTITY/swenoid/RUN_ID
```

See [docs/deployment.md](docs/deployment.md) for driver interfaces, local ONNX
usage, recording, controls, timing behavior, and the hardware safety checklist.

## Project layout

```text
src/swenoid/assets/swenoid/  Swenoid MJCF, meshes, and actuator parameters
src/swenoid/tasks/           Velocity, single-motion, and general-motion tasks
src/swenoid/retargeting/     BONES-SEED retarget/filter/convert/render/upload tools
src/swenoid/deployment/      Shared ONNX, sim2sim, and physical-robot runners
tests/                       Robot, task, conversion, ONNX, and controller tests
typings/                     Development-only MuJoCo type information
```

## Dataset attribution

Training data includes [Motion Data by Bones Studio](https://bones.studio/).
Use of the underlying dataset is subject to the BONES Motion Capture Dataset
License Agreement. Consult the
[current BONES-SEED license](https://huggingface.co/datasets/bones-studio/seed/blob/main/LICENSE.md)
before downloading, processing, or distributing derived results.

## Upstream projects and citation

This repository is a thin extension of released MjLab rather than a vendored
copy. Please cite MjLab, RSL-RL, MuJoCo, BAM, BONES-SEED as required by its
license, and the Swenoid paper when using this work.

```bibtex
@misc{orrje2026swenoid,
  title  = {Swenoid: An Accessible Open-Source Humanoid Robot},
  author = {Orrje, Martin and Eriksson, Arvid and Broman, David},
  year   = {2026}
}
```

## License

Code is released under the [MIT License](LICENSE), except where a file
states another upstream license. BONES-SEED data is not covered by this license
and is not included. The forthcoming CAD repository will state its own hardware
license separately.
