# Reference models

The `v0.1.0` GitHub release provides checkpoint and ONNX pairs for a completed
flat-ground velocity policy and a clearly labeled general-motion preview. Exact
run IDs, iterations, metrics, filenames, and SHA-256 checksums are recorded in
[`reproducibility/reference_models.json`](../reproducibility/reference_models.json).

Download the assets with GitHub CLI:

```bash
gh release download v0.1.0 --repo martinorrje/swenoid
sha256sum -c SHA256SUMS
```

Run the velocity ONNX in standalone MuJoCo:

```bash
uv run --no-sync swenoid-sim2sim --onnx swenoid-velocity-flat.onnx --loop
```

The general-motion preview validates the 126-value policy contract and can be
evaluated by a custom reference provider. The bundled deployment runners reject
it intentionally until live imitation supplies the required 48-value joint
reference. It comes from an active run and is not presented as a final result.

## Training-data boundary

Training data for the general-motion model includes Motion Data by
[Bones Studio](https://bones.studio/). Use of the underlying dataset is subject
to the BONES Motion Capture Dataset License Agreement. The released generic
policy does not embed source trajectories.

Single-motion tracking ONNX files are not redistributed because that export
embeds its complete reference trajectory. Users with authorized BONES-SEED
access can reproduce one with the documented retargeting and training commands.

## Safety and limitations

The reference models are research artifacts, not safety-certified controllers.
Validate the exact ONNX in sim2sim, use the per-robot calibration workflow, keep
the robot supported during bring-up, and use an independent physical emergency
stop. The general-motion preview is not accepted by the physical runner.

