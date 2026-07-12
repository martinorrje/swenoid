# General motion tracking

`Mjlab-General-Motion-Flat-Swenoid` trains one policy across a directory or
manifest of retargeted motion NPZ files. It is distinct from
`Mjlab-Tracking-Flat-Swenoid`, which embeds one reference clip in each policy.

## Observation contract

The actor input has 126 values in this fixed order:

| Term | Size | Source |
| --- | ---: | --- |
| `base_ang_vel` | 3 | Pelvis IMU gyroscope |
| `projected_gravity` | 3 | Pelvis IMU orientation/gravity |
| `joint_pos` | 24 | Position relative to the default pose |
| `joint_vel` | 24 | Joint velocity |
| `actions` | 24 | Previous normalized action |
| `command` | 48 | Reference joint position and velocity |

The reference root pose, root velocity, body poses, and body velocities are not
actor inputs. They remain privileged critic observations and are used by the
tracking rewards, reset logic, terminations, and diagnostics.

## Dataset format

Each NPZ must contain:

- `joint_pos`, `joint_vel`;
- `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`; and
- `fps`.

Optional root arrays are accepted. If they are absent, the configured anchor
body supplies the root reference. Motions may have different lengths and frame
rates; references are interpolated at the 50 Hz control times.

Pass either a recursive root:

```bash
uv run swenoid-train Mjlab-General-Motion-Flat-Swenoid \
  --env.commands.motion.motion-root /absolute/path/to/npz \
  --agent.logger wandb \
  --agent.wandb-project swenoid \
  --agent.run-name general-motion
```

or a text manifest containing one absolute NPZ path per line:

```bash
uv run swenoid-train Mjlab-General-Motion-Flat-Swenoid \
  --env.commands.motion.motion-list /absolute/path/to/accepted_npz.txt \
  --agent.logger wandb \
  --agent.wandb-project swenoid \
  --agent.run-name general-motion
```

`active_motion_count` controls how many motions participate in the current
sampling pool. `max_cache_size` bounds decoded motion memory. The active pool is
periodically refreshed from the complete manifest, allowing large datasets to
be covered without loading every trajectory simultaneously.

## Export and deployment status

Checkpoints export a generic ONNX policy with one `obs` input of shape
`(1, 126)`. Unlike a single-motion ONNX, it does not include a `time_step` input
or embed a trajectory. The current physical and sim2sim runners reject this
policy type with an explicit message. A live camera/retargeting reference
provider will be connected in the later real-time imitation release.
