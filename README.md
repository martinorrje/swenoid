# Swenoid

Software, simulation models, and training tools for **Swenoid**, a 0.68 m,
7.1 kg humanoid robot with 24 actuated joints.

This repository contains the MuJoCo digital twin, MjLab environments,
BONES-SEED retargeting tools, and ONNX deployment programs used on the robot.
Policies are trained with [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
in GPU-parallel [MjLab](https://github.com/mujocolab/mjlab) environments. The
simulation uses identified Dynamixel XM430-W350 and XM540-W270 actuator models
from [BAM](https://github.com/Rhoban/bam).

The release covers velocity control, single-motion tracking, dataset-wide
general-motion tracking, sim2sim evaluation, and physical deployment. Live
camera retargeting and real-time imitation are outside this release. The
general-motion policy expects a live joint-reference provider that the
current deployment runners deliberately do not supply.

## Related repositories

| Repository | Purpose |
| --- | --- |
| **[`martinorrje/swenoid`](https://github.com/martinorrje/swenoid)** | MuJoCo model, MjLab tasks, retargeting, training, ONNX export, and deployment |
| [`martinorrje/bam`](https://github.com/martinorrje/bam) | Swenoid actuator-identification data, model fitting, parameters, and fit plots |
| [`martinorrje/swenoid_cad`](https://github.com/martinorrje/swenoid_cad) | MIT-licensed STEP assembly; print exports, assembly instructions, and bill of materials are in progress |

The meshes required by the MuJoCo model are included here. The CAD repository
contains the hardware assembly.

## Getting started

Swenoid supports Python 3.12 and 3.13. GPU-parallel training requires Linux, an
x86-64 host, and an NVIDIA GPU supported by PyTorch, Warp, and MuJoCo Warp.
Retargeting and light MuJoCo evaluation can also run on CPU and Apple Silicon.
All workflows use [`uv`](https://docs.astral.sh/uv/) and the checked-in
`uv.lock`.

Clone the repository:

```bash
git clone https://github.com/martinorrje/swenoid.git
cd swenoid
```

For a contributor checkout, install the locked development environment:

```bash
uv sync --frozen
uv run --no-sync swenoid-list-envs
```

For a smaller runtime environment, install only the extra for the job at hand:

| Workflow | Install command |
| --- | --- |
| Training | `uv sync --frozen --no-dev --extra training` |
| Retargeting | `uv sync --frozen --no-dev --extra retargeting` |
| Standalone MuJoCo deployment | `uv sync --frozen --no-dev --extra sim2sim` |
| Raspberry Pi deployment | `uv sync --frozen --no-dev --extra deployment` |

The commands below use `uv run --no-sync` after this explicit setup. That keeps
`uv` from adding the default development group to a focused environment. If you
change extras, run the corresponding `uv sync` command again.

Log in once before using W&B:

```bash
uv run --no-sync wandb login
```

Contributor setup and the exact CI commands are documented in
[`CONTRIBUTING.md`](CONTRIBUTING.md). Detailed runtime documentation is split
between the [deployment guide](docs/deployment.md), the
[general-motion contract](docs/general_motion.md), and the
[reference-model notes](docs/reference_models.md).

## Simulation model

All registered environments use a 5 ms MuJoCo step and a decimation of four:

- physics and sensor state run at 200 Hz;
- policy inference and position commands run at 50 Hz.

Retargeted motions must be converted to 50 Hz. The converter interpolates
translation and joint state, applies quaternion SLERP to orientation, and
recomputes velocity at the output rate. The tracking loader rejects NPZ files
whose stored rate differs from the environment control rate.

The actuator split is:

- XM540-W270: hips, knees, torso, and shoulder pitch/roll (15 joints);
- XM430-W350: ankles, shoulder yaw, elbows, and neck (9 joints).

The two compact X-series BAM registry adapters needed at runtime are included in
this package. The complete identification workflow and raw measurements remain
in the Swenoid BAM repository.

## Policy interface

Every policy actor receives the same robot-side proprioception in this order:

| Input | Size | Source |
| --- | ---: | --- |
| Base angular velocity | 3 | Pelvis IMU gyroscope |
| Projected gravity | 3 | Pelvis IMU orientation |
| Joint position | 24 | Offset from the default pose |
| Joint velocity | 24 | Motor state |
| Previous action | 24 | Previous normalized policy output |

The task command is appended last:

| Policy | Command | Actor size |
| --- | --- | ---: |
| Velocity | Body-frame x/y velocity and yaw rate (3) | 81 |
| Single-motion tracking | Reference joint position and velocity (24 + 24) | 126 |
| General-motion tracking | Reference joint position and velocity (24 + 24) | 126 |

The actor does not observe global position, global orientation, heading, base
linear velocity, motion-anchor position, or terrain height. Tracking tasks may
use reference root and body state in the critic, rewards, resets, and
diagnostics, but never in the actor.

All policies output 24 normalized actions. Each action is scaled and added to
the default joint pose to produce a 50 Hz Dynamixel position target. Exported
ONNX files store `observation_names`, `action_scale`, and `default_joint_pos`
metadata so deployment code can reproduce the contract.

## Train a velocity policy

Flat ground:

```bash
uv run --no-sync swenoid-train Mjlab-Velocity-Flat-Swenoid \
  --env.scene.num-envs 4096 \
  --agent.logger wandb \
  --agent.wandb-project swenoid \
  --agent.run-name velocity-flat
```

Rough terrain:

```bash
uv run --no-sync swenoid-train Mjlab-Velocity-Rough-Swenoid \
  --env.scene.num-envs 4096 \
  --agent.logger wandb \
  --agent.wandb-project swenoid \
  --agent.run-name velocity-rough
```

The velocity runner defaults to 30,000 PPO iterations and saves every 50
iterations. Override these values with `--agent.max-iterations` and
`--agent.save-interval`.

Visualize a W&B run in Viser without interrupting training:

```bash
uv run --no-sync swenoid-play Mjlab-Velocity-Flat-Swenoid \
  --wandb-run-path ENTITY/swenoid/RUN_ID \
  --viewer viser \
  --num-envs 1
```

## Retarget BONES-SEED

### 1. Obtain the dataset

Download BONES-SEED from the
[official dataset page](https://bones.studio/datasets/seed) or its
[Hugging Face repository](https://huggingface.co/datasets/bones-studio/seed)
and accept the dataset license. This repository does not redistribute the raw
dataset or retargeted motion files.

Set local paths:

```bash
export BONES_SEED_ROOT=/absolute/path/to/bones-seed
export SWENOID_MOTIONS=/absolute/path/to/swenoid-motion-workspace
mkdir -p "$SWENOID_MOTIONS"
```

The expected dataset tree contains `g1/csv/...` and `metadata/...`.

### 2. Map G1 trajectories to Swenoid

Convert one motion:

```bash
uv run --no-sync swenoid-retarget \
  --dataset-root "$BONES_SEED_ROOT" \
  --motion g1/csv/221129/dancing_routine_2_004__A081_M.csv \
  --output-dir "$SWENOID_MOTIONS/qpos"
```

Convert every motion listed by the metadata file:

```bash
uv run --no-sync swenoid-retarget \
  --dataset-root "$BONES_SEED_ROOT" \
  --metadata-csv "$BONES_SEED_ROOT/metadata/seed_metadata_v004.csv" \
  --output-dir "$SWENOID_MOTIONS/qpos"
```

Use the metadata filename supplied with your dataset version. `--package`,
`--category`, and `--limit` select a smaller batch.

The mapping copies compatible leg, torso, shoulder, and elbow joints; maps G1
ankle pitch to Swenoid's single-axis ankle; converts degrees to radians and
centimeters to meters; scales root translation to the Swenoid neutral height;
clips joint limits; and leaves joints with no source signal at the neutral pose.
It does not use PHUMA adaptation or post-processing modules.

### 3. Inspect and filter

Render a trajectory before training on it:

```bash
uv run --no-sync swenoid-render-motion \
  --qpos-csv "$SWENOID_MOTIONS/qpos/csv/221129/dancing_routine_2_004__A081_M.qpos.csv" \
  --output "$SWENOID_MOTIONS/dancing_routine_2.mp4"
```

The optional filter measures contact, foot penetration and skating, center-of-
mass support, root and joint jitter, and joint-limit violations. It classifies
files but never modifies a trajectory:

```bash
uv run --no-sync swenoid-filter-motions \
  --input-root "$SWENOID_MOTIONS/qpos" \
  --input-glob "$SWENOID_MOTIONS/qpos/**/*.qpos.csv" \
  --metrics-csv "$SWENOID_MOTIONS/curation/metrics.csv" \
  --accepted-list "$SWENOID_MOTIONS/curation/accepted.txt" \
  --rejected-list "$SWENOID_MOTIONS/curation/rejected.txt" \
  --accepted-root "$SWENOID_MOTIONS/accepted" \
  --exclude-support-props
```

### 4. Convert to the tracking format

Convert accepted qpos files to 50 Hz NPZ:

```bash
uv run --no-sync swenoid-convert-motion \
  --input-root "$SWENOID_MOTIONS/qpos" \
  --input-list "$SWENOID_MOTIONS/curation/accepted.txt" \
  --output-root "$SWENOID_MOTIONS/npz" \
  --input-fps 120 \
  --output-fps 50
```

Set `--input-fps` to the source release's actual rate. The output contains joint
and body pose and velocity arrays together with `fps=50`.

Upload a converted motion to W&B Registry if desired:

```bash
uv run --no-sync swenoid-upload-motion \
  "$SWENOID_MOTIONS/npz/csv/221129/dancing_routine_2_004__A081_M.npz" \
  --name dancing-routine-2
```

The command prints the registry identifier accepted by training.

## Train a single-motion policy

A single-motion policy embeds one reference clip. Train from a local NPZ:

```bash
uv run --no-sync swenoid-train Mjlab-Tracking-Flat-Swenoid \
  --env.commands.motion.motion-file "$SWENOID_MOTIONS/npz/csv/221129/dancing_routine_2_004__A081_M.npz" \
  --env.scene.num-envs 4096 \
  --agent.logger wandb \
  --agent.wandb-project swenoid \
  --agent.run-name track-dancing-routine-2
```

Or resolve the motion from W&B Registry:

```bash
uv run --no-sync swenoid-train Mjlab-Tracking-Flat-Swenoid \
  --registry-name ENTITY/wandb-registry-motions/dancing-routine-2:latest \
  --env.scene.num-envs 4096 \
  --agent.logger wandb \
  --agent.wandb-project swenoid \
  --agent.run-name track-dancing-routine-2
```

Train separate runs for separate clips. This task does not sample a motion
dataset.

Visualize a tracking run:

```bash
uv run --no-sync swenoid-play Mjlab-Tracking-Flat-Swenoid \
  --wandb-run-path ENTITY/swenoid/RUN_ID \
  --motion-file "$SWENOID_MOTIONS/npz/csv/221129/dancing_routine_2_004__A081_M.npz" \
  --viewer viser \
  --num-envs 1
```

If the run consumed a W&B motion artifact, `swenoid-play` can recover it and
`--motion-file` may be omitted.

## Train a general-motion policy

General-motion training samples a directory of NPZ files while keeping a
bounded active pool and cache on the training device:

```bash
uv run --no-sync swenoid-train Mjlab-General-Motion-Flat-Swenoid \
  --env.commands.motion.motion-root "$SWENOID_MOTIONS/npz" \
  --env.commands.motion.active-motion-count 128 \
  --env.commands.motion.max-cache-size 128 \
  --env.scene.num-envs 4096 \
  --agent.logger wandb \
  --agent.wandb-project swenoid \
  --agent.run-name general-motion
```

Alternatively, pass `--env.commands.motion.motion-list /path/to/list.txt` with
one absolute NPZ path per line.

The exported ONNX contains the policy but no reference motion. Its 48-value
reference command must be supplied by another process. Live camera input and
online retargeting remain outside this release. See
[`docs/general_motion.md`](docs/general_motion.md) for the NPZ contract,
observation layout, cache behavior, and deployment boundary.

## Checkpoints and deployment

Checkpoint and ONNX pairs are attached to the
[`v0.1.0` release](https://github.com/martinorrje/swenoid/releases/tag/v0.1.0).
Source runs, iterations, metrics, filenames, and SHA-256 checksums are recorded
in [`reproducibility/reference_models.json`](reproducibility/reference_models.json).
The [reference-model notes](docs/reference_models.md) describe their limitations
and data boundary.

Training writes local runs under `logs/rsl_rl/<experiment>/...`. Each save
creates a PyTorch checkpoint and attempts an ONNX export. With
`--agent.upload-model true`, both are uploaded to W&B.

| Policy export | ONNX interface |
| --- | --- |
| Velocity | One `obs` input |
| Single motion | `obs` and `time_step` inputs; embeds one reference clip |
| General motion | One 126-value `obs` input; expects an external reference |

Run a velocity or single-motion export in standalone MuJoCo:

```bash
uv run --no-sync swenoid-sim2sim \
  --wandb-run ENTITY/swenoid/RUN_ID \
  --loop
```

Sim2sim uses the identified BAM actuator models by default. Pass
`--actuator-model dc` to compare against the XML's direct position-servo model.

After configuring and calibrating the Raspberry Pi, run the same policy on the
robot:

```bash
uv run --no-sync swenoid-deploy --wandb-run ENTITY/swenoid/RUN_ID
```

Physical inference runs every 20 ms and must use the joint order, observation
order, action scale, and default pose stored in the model metadata. Start with
the robot supported, verify every axis sign and limit independently, and keep a
physical emergency stop accessible. The software is research software and is
not a substitute for mechanical support, current limiting, thermal protection,
or an independent emergency-stop path.

See [`docs/deployment.md`](docs/deployment.md) for device setup, calibration,
local ONNX usage, controls, recording, timing checks, and the complete bring-up
checklist.

## Repository layout

```text
src/swenoid/assets/swenoid/  MJCF, meshes, and actuator parameters
src/swenoid/tasks/           Velocity and motion-tracking environments
src/swenoid/retargeting/     Retarget, filter, convert, render, and upload tools
src/swenoid/deployment/      Shared ONNX, sim2sim, and robot runners
tests/                       Unit and integration tests
typings/                     Development-only MuJoCo type information
reproducibility/             Reference-model provenance and checksums
```

## Data, attribution, and citation

The released general-motion preview was trained on trajectories derived from
[Motion Data by Bones Studio](https://bones.studio/). BONES-SEED source and
retargeted trajectories are not distributed here. Download and use the dataset
under the
[current BONES-SEED license](https://huggingface.co/datasets/bones-studio/seed/blob/main/LICENSE.md).

This repository extends MjLab rather than vendoring it. Publications should
credit the upstream work they rely on, including MjLab, RSL-RL, MuJoCo, BAM,
and BONES-SEED, and cite Swenoid as:

```bibtex
@misc{orrje2026swenoid,
  title  = {Swenoid: An Accessible Open-Source Humanoid Robot},
  author = {Orrje, Martin and Eriksson, Arvid and Broman, David},
  year   = {2026}
}
```

Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request. It
documents the development environment, CI checks, experiment evidence, and the
additional review requirements for hardware-facing changes.

## License

This repository is released under the [MIT License](LICENSE), except where a
file carries an upstream license notice. BONES-SEED data is not part of this
repository and is not covered by the MIT License. The
[Swenoid CAD repository](https://github.com/martinorrje/swenoid_cad) is also
released under MIT.
