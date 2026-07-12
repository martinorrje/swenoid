#!/usr/bin/env python3
"""Upload one Swenoid tracking motion to a Weights & Biases Registry."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path, help="Converted Swenoid motion NPZ.")
    parser.add_argument("--name", required=True, help="Registry collection name.")
    parser.add_argument(
        "--project",
        default="swenoid-motion-upload",
        help="Temporary W&B project used to log the artifact.",
    )
    parser.add_argument(
        "--registry",
        default="motions",
        help="W&B Registry name and artifact type.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.motion.is_file():
        raise SystemExit(f"Motion file does not exist: {args.motion}")

    run = wandb.init(project=args.project, name=f"upload-{args.name}")
    assert run is not None
    artifact = wandb.Artifact(name=args.name, type=args.registry)
    artifact.add_file(str(args.motion), name="motion.npz")
    logged_artifact = run.log_artifact(artifact)
    run.link_artifact(
        artifact=logged_artifact,
        target_path=f"wandb-registry-{args.registry}/{args.name}",
    )
    entity = run.entity
    run.finish()
    print(
        "Uploaded motion. Train with "
        f"--registry-name {entity}/wandb-registry-{args.registry}/{args.name}:latest"
    )


if __name__ == "__main__":
    main()
