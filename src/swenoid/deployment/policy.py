"""Shared ONNX and W&B utilities for Swenoid deployment."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def motion_length(model_path: Path) -> int | None:
    """Return an embedded tracking motion length, or ``None`` for velocity."""
    model = onnx.load(str(model_path))
    if "time_step" not in {value.name for value in model.graph.input}:
        return None
    initializer_shapes = {
        initializer.name: tuple(initializer.dims)
        for initializer in model.graph.initializer
    }
    lengths = []
    for node in model.graph.node:
        if node.op_type != "Gather" or not node.input:
            continue
        shape = initializer_shapes.get(node.input[0])
        if shape and shape[0] > 0:
            lengths.append(int(shape[0]))
    if not lengths:
        raise ValueError(
            "Tracking ONNX has a time_step input but no embedded motion tensor"
        )
    return Counter(lengths).most_common(1)[0][0]


def download_wandb_onnx(
    run_path: str,
    *,
    cache_root: Path = Path("wandb_checkpoints"),
    filename: str | None = None,
) -> Path:
    """Download an ONNX policy attached to a W&B run."""
    import wandb

    run = wandb.Api().run(run_path)
    candidates = [item for item in run.files() if item.name.endswith(".onnx")]
    if filename is not None:
        selected = next((item for item in candidates if item.name == filename), None)
        if selected is None:
            available = [item.name for item in candidates]
            raise ValueError(
                f"ONNX file {filename!r} not found in {run_path}; "
                f"available: {available}"
            )
    else:
        if not candidates:
            raise ValueError(f"No ONNX files found in {run_path}")
        selected = max(candidates, key=lambda item: (item.updated_at or "", item.name))
    name = selected.name
    output_dir = cache_root / run.id
    output_path = output_dir / name
    if not output_path.exists():
        selected.download(str(output_dir), replace=True)
    return output_path


def resolve_policy_path(
    *,
    onnx_path: Path | None,
    wandb_run: str | None,
    entity: str,
    project: str,
    onnx_name: str | None = None,
) -> Path:
    """Resolve a local ONNX path or an eight-character/full W&B run path."""
    if (onnx_path is None) == (wandb_run is None):
        raise ValueError("Provide exactly one of --onnx or --wandb-run")
    if onnx_path is not None:
        if not onnx_path.is_file():
            raise FileNotFoundError(onnx_path)
        return onnx_path
    assert wandb_run is not None
    run_path = f"{entity}/{project}/{wandb_run}" if "/" not in wandb_run else wandb_run
    return download_wandb_onnx(run_path, filename=onnx_name)


class OnnxPolicy:
    """ONNX policy plus the metadata required to reproduce MjLab inference."""

    def __init__(self, path: Path):
        self.path = path
        self.session = ort.InferenceSession(str(path))
        self.metadata = self.session.get_modelmeta().custom_metadata_map
        self.input_names = {item.name for item in self.session.get_inputs()}
        expected_inputs = (
            {"obs", "time_step"} if "time_step" in self.input_names else {"obs"}
        )
        if self.input_names != expected_inputs:
            raise ValueError(
                f"Unsupported ONNX inputs {sorted(self.input_names)}; expected "
                f"{sorted(expected_inputs)}"
            )
        self.is_tracking = "time_step" in self.input_names
        self.motion_length = motion_length(path)
        if self.is_tracking != (self.motion_length is not None):
            raise ValueError("Inconsistent ONNX tracking inputs and embedded motion")
        self.observation_names = self._metadata_values("observation_names", str)
        self.action_scale = np.asarray(
            self._metadata_values("action_scale", float), dtype=np.float32
        )
        self.default_joint_pos = np.asarray(
            self._metadata_values("default_joint_pos", float), dtype=np.float32
        )
        if self.action_scale.shape != (24,) or self.default_joint_pos.shape != (24,):
            raise ValueError(
                "Swenoid ONNX metadata must contain 24 action scales and "
                "24 default joint positions"
            )
        if not self.observation_names or any(
            not name for name in self.observation_names
        ):
            raise ValueError("ONNX observation_names metadata is empty or malformed")
        obs_input = next(
            item for item in self.session.get_inputs() if item.name == "obs"
        )
        try:
            self.observation_size = int(obs_input.shape[-1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "ONNX obs input must have a fixed feature dimension"
            ) from exc
        if self.is_tracking:
            self.policy_kind = "single_motion"
        elif self.observation_size == 126 and self.observation_names == [
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "actions",
            "command",
        ]:
            self.policy_kind = "general_motion"
        else:
            self.policy_kind = "velocity"

    @property
    def is_general_motion(self) -> bool:
        return self.policy_kind == "general_motion"

    def _metadata_values(self, key: str, convert):
        if key not in self.metadata:
            raise ValueError(f"ONNX metadata is missing {key!r}")
        return [convert(value) for value in self.metadata[key].split(",")]

    def reference_at(self, time_step: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Get reference joint position and velocity at one tracking frame."""
        if not self.is_tracking:
            return None
        outputs = self.session.run(
            ["joint_pos", "joint_vel"],
            {
                "obs": np.zeros((1, self.observation_size), dtype=np.float32),
                "time_step": np.asarray([[time_step]], dtype=np.float32),
            },
        )
        joint_pos, joint_vel = np.asarray(outputs[0]), np.asarray(outputs[1])
        if joint_pos.shape != (1, 24) or joint_vel.shape != (1, 24):
            raise ValueError(
                "Tracking ONNX reference outputs must both have shape (1, 24)"
            )
        return joint_pos, joint_vel

    def infer(self, observations: np.ndarray, time_step: int) -> list[np.ndarray]:
        feed = {"obs": observations.astype(np.float32, copy=False)}
        if self.is_tracking:
            feed["time_step"] = np.asarray([[time_step]], dtype=np.float32)
        outputs = [np.asarray(output) for output in self.session.run(None, feed)]
        if not outputs or outputs[0].shape != (1, 24):
            shape = None if not outputs else outputs[0].shape
            raise ValueError(f"ONNX action output must have shape (1, 24), got {shape}")
        return outputs


def concatenate_observations(
    policy: OnnxPolicy,
    observations: dict[str, np.ndarray],
) -> np.ndarray:
    """Concatenate observations in the order embedded by MjLab's exporter."""
    missing = [name for name in policy.observation_names if name not in observations]
    if missing:
        raise KeyError(f"No deployment observation provider for: {missing}")
    output = np.concatenate(
        [observations[name] for name in policy.observation_names], axis=1
    ).astype(np.float32)
    if output.shape != (1, policy.observation_size):
        raise ValueError(
            f"Observation shape {output.shape} does not match ONNX "
            f"(1, {policy.observation_size})"
        )
    return output


def benchmark(policy: OnnxPolicy, iterations: int = 100) -> None:
    """Print warmed-up ONNX inference latency."""
    import time

    obs = np.zeros((1, policy.observation_size), dtype=np.float32)
    for _ in range(10):
        policy.infer(obs, 0)
    start = time.perf_counter()
    for index in range(iterations):
        policy.infer(obs, index % (policy.motion_length or 1))
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iterations
    print(f"ONNX Runtime: {elapsed_ms:.3f} ms/inference ({iterations} runs)")
