"""Register Swenoid environments with MjLab's task registry."""

from mjlab.tasks.registry import register_mjlab_task

from swenoid.tasks.general_motion.config import swenoid_general_motion_flat_env_cfg
from swenoid.tasks.runners import (
    SwenoidGeneralMotionRunner,
    SwenoidTrackingRunner,
    SwenoidVelocityRunner,
    swenoid_general_motion_ppo_cfg,
    swenoid_tracking_ppo_cfg,
    swenoid_velocity_ppo_cfg,
)
from swenoid.tasks.tracking import swenoid_flat_tracking_env_cfg
from swenoid.tasks.velocity import swenoid_flat_env_cfg, swenoid_rough_env_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Swenoid",
    env_cfg=swenoid_flat_env_cfg(),
    play_env_cfg=swenoid_flat_env_cfg(play=True),
    rl_cfg=swenoid_velocity_ppo_cfg(),
    runner_cls=SwenoidVelocityRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Swenoid",
    env_cfg=swenoid_rough_env_cfg(),
    play_env_cfg=swenoid_rough_env_cfg(play=True),
    rl_cfg=swenoid_velocity_ppo_cfg(),
    runner_cls=SwenoidVelocityRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tracking-Flat-Swenoid",
    env_cfg=swenoid_flat_tracking_env_cfg(),
    play_env_cfg=swenoid_flat_tracking_env_cfg(play=True),
    rl_cfg=swenoid_tracking_ppo_cfg(),
    runner_cls=SwenoidTrackingRunner,
)

register_mjlab_task(
    task_id="Mjlab-General-Motion-Flat-Swenoid",
    env_cfg=swenoid_general_motion_flat_env_cfg(),
    play_env_cfg=swenoid_general_motion_flat_env_cfg(play=True),
    rl_cfg=swenoid_general_motion_ppo_cfg(),
    runner_cls=SwenoidGeneralMotionRunner,
)
