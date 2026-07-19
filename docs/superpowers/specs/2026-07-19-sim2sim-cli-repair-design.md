# Sim2Sim CLI Repair Design

## Goal

Make the documented `uv run swenoid-sim2sim` command work when a W&B run has
multiple exported ONNX checkpoints and when the command launches MuJoCo's
passive viewer on macOS.

## W&B Policy Selection

`download_wandb_onnx` will retain exact selection through `--onnx-name`. When
no name is supplied, it will select the ONNX file with the newest W&B
`updated_at` value. The filename will be the deterministic tie-breaker when
timestamps are equal.

The downloader will continue to fail when the run has no ONNX files or when an
explicit filename does not match exactly. Those errors will identify the run
and requested filename. The selected file will continue to use the existing
`wandb_checkpoints/<run-id>/` cache.

Because policy resolution is shared, the same selection rule will apply to
sim2sim and real-robot deployment. An explicit `--onnx-name` remains the way to
pin deployment to a specific checkpoint.

## macOS Viewer Launch

At the beginning of the sim2sim entry point, macOS execution will be checked
before policy resolution or model construction. If MuJoCo's `mjpython`
dispatcher is not active, the process will replace itself with the `mjpython`
executable from the same Python environment and run
`swenoid.deployment.sim2sim` as a module with the original arguments.

The relaunched process will detect that the dispatcher is active and proceed
normally. Non-macOS platforms will not relaunch. A missing `mjpython`
executable will produce a direct error that identifies the expected path.

## Compatibility

The existing `swenoid-sim2sim` command and arguments remain unchanged. Linux
and explicitly named W&B policies retain their existing behavior. The real
robot entry point will gain newest-checkpoint selection but will never invoke
`mjpython`.

## Testing

Focused unit tests will verify:

- newest W&B ONNX selection by `updated_at`;
- deterministic filename tie-breaking;
- exact `--onnx-name` selection and missing-name errors;
- no-ONNX errors;
- macOS relaunch command and argument forwarding;
- no relaunch outside macOS or when already under `mjpython`.

The deployment test module and full test suite will then be run, followed by
lint and type checks configured by the project.
