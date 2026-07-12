"""Swenoid RSL-RL runner configuration and W&B-compatible ONNX upload."""

from pathlib import Path

import wandb
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


def _model_cfg() -> RslRlModelCfg:
    return RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    )


def swenoid_velocity_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    actor = _model_cfg()
    actor.distribution_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    }
    return RslRlOnPolicyRunnerCfg(
        actor=actor,
        critic=_model_cfg(),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="swenoid_velocity",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=30_000,
    )


def swenoid_tracking_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    actor = _model_cfg()
    actor.distribution_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    }
    return RslRlOnPolicyRunnerCfg(
        actor=actor,
        critic=_model_cfg(),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="swenoid_tracking",
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=30_000,
    )


def swenoid_general_motion_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO configuration for dataset-wide motion tracking."""
    cfg = swenoid_tracking_ppo_cfg()
    cfg.experiment_name = "swenoid_general_motion"
    return cfg


def _upload_exported_onnx(runner, checkpoint_path: str) -> None:
    """Handle the logger name used by RSL-RL 5.4 and MjLab 1.4."""
    if runner.logger.logger_type != "WandbLogWriter" or not runner.cfg["upload_model"]:
        return
    policy_dir, _, onnx_path = runner._get_export_paths(checkpoint_path)
    if Path(onnx_path).is_file() and wandb.run is not None:
        wandb.save(str(onnx_path), base_path=str(policy_dir))


class SwenoidVelocityRunner(VelocityOnPolicyRunner):
    """Velocity runner with ONNX upload support for the current W&B logger."""

    def save(self, path: str, infos=None):
        super().save(path, infos)
        _upload_exported_onnx(self, path)


class SwenoidGeneralMotionRunner(VelocityOnPolicyRunner):
    """General-motion runner exporting a policy without an embedded motion."""

    def save(self, path: str, infos=None):
        super().save(path, infos)
        _upload_exported_onnx(self, path)


class SwenoidTrackingRunner(MotionTrackingOnPolicyRunner):
    """Tracking runner with ONNX and consumed-motion artifact logging."""

    def save(self, path: str, infos=None):
        super().save(path, infos)
        _upload_exported_onnx(self, path)
        if (
            self.logger.logger_type == "WandbLogWriter"
            and self.cfg["upload_model"]
            and self.registry_name is not None
            and wandb.run is not None
        ):
            wandb.run.use_artifact(self.registry_name)
            self.registry_name = None
