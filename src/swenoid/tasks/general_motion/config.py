"""Swenoid general motion tracking environment configuration."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE, make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from swenoid.robot import SWENOID_ACTION_SCALE, get_swenoid_robot_cfg
from swenoid.tasks.general_motion import rewards as general_motion_rewards
from swenoid.tasks.general_motion import support
from swenoid.tasks.general_motion.commands import DatasetMotionCommandCfg
from swenoid.tasks.tracking import SWENOID_TRACKING_BODIES


def swenoid_general_motion_flat_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Swenoid flat-ground general motion tracking configuration."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": get_swenoid_robot_cfg(actuator_model="bam")}
    cfg.scene.num_envs = 4096
    cfg.episode_length_s = 10.0
    cfg.sim.njmax = 300
    cfg.sim.nconmax = 70
    cfg.sim.contact_sensor_maxmatch = 128

    # Match the deployable velocity and single-motion policies exactly. The
    # general-motion command contributes only reference joint position and
    # velocity; all remaining actor inputs are robot-side proprioception.
    base_actor_terms = cfg.observations["actor"].terms
    base_actor_terms["joint_pos"].params = {}
    base_actor_terms["joint_vel"].noise = Unoise(n_min=-1.5, n_max=1.5)
    cfg.observations["actor"] = ObservationGroupCfg(
        terms={
            "base_ang_vel": base_actor_terms["base_ang_vel"],
            "projected_gravity": ObservationTermCfg(
                func=envs_mdp.projected_gravity_from_sensor,
                params={"sensor_name": "robot/imu_upvector"},
                noise=Unoise(n_min=-0.05, n_max=0.05),
            ),
            "joint_pos": base_actor_terms["joint_pos"],
            "joint_vel": base_actor_terms["joint_vel"],
            "actions": base_actor_terms["actions"],
            "command": base_actor_terms["command"],
        },
        concatenate_terms=True,
        enable_corruption=True,
    )

    foot_geom_names = ("left_foot_collision", "right_foot_collision")
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=foot_geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    non_foot_ground_cfg = ContactSensorCfg(
        name="non_foot_ground_touch",
        primary=ContactMatch(
            mode="geom",
            pattern=".*_collision",
            entity="robot",
            exclude=foot_geom_names,
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="hip_mid_section", entity="robot"),
        secondary=ContactMatch(
            mode="subtree", pattern="hip_mid_section", entity="robot"
        ),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (feet_ground_cfg, non_foot_ground_cfg, self_collision_cfg)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = SWENOID_ACTION_SCALE

    cfg.commands["motion"] = DatasetMotionCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        pose_range={
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.01, 0.01),
            "roll": (-0.08, 0.08),
            "pitch": (-0.08, 0.08),
            "yaw": (-0.15, 0.15),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.08, 0.08),
        anchor_body_name="hip_mid_section",
        body_names=SWENOID_TRACKING_BODIES,
        motion_root="",
        motion_glob="**/*.npz",
        motion_list=None,
        active_motion_count=128,
        max_cache_size=128,
    )

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = ("chest",)
    cfg.events["reset_base"] = EventTermCfg(
        func=support.reset_root_state_random_lying,
        mode="reset",
        params={
            "xy_range": (-0.2, 0.2),
            "root_height_range": (0.11, 0.18),
            "yaw_range": (-math.pi, math.pi),
            "roll_noise_range": (-0.25, 0.25),
            "pitch_noise_range": (-0.25, 0.25),
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )
    cfg.events["reset_robot_joints"] = EventTermCfg(
        func=envs_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.35, 0.35),
            "velocity_range": (-0.05, 0.05),
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    )

    cfg.terminations.pop("ee_body_pos", None)
    cfg.terminations["anchor_pos"].params["threshold"] = 0.75
    cfg.terminations["anchor_ori"].params["threshold"] = math.pi

    cfg.rewards["motion_global_root_pos"].weight = 0.15
    cfg.rewards["motion_global_root_ori"].weight = 0.15
    cfg.rewards["motion_body_pos"].weight = 1.5
    cfg.rewards["motion_body_ori"].weight = 1.5

    torso_cfg = SceneEntityCfg("robot", body_names=("chest",))
    cfg.rewards["getup_upright"] = RewardTermCfg(
        func=support.signed_upright_exp,
        weight=2.0,
        params={"std": math.sqrt(0.2), "asset_cfg": torso_cfg},
    )
    cfg.rewards["getup_chest_height"] = RewardTermCfg(
        func=support.body_height_exp,
        weight=1.5,
        params={"target_height": 0.42, "std": 0.10, "asset_cfg": torso_cfg},
    )

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=general_motion_rewards.contact_cost_by_reference_height,
        weight=-0.5,
        params={
            "command_name": "motion",
            "sensor_name": self_collision_cfg.name,
            "low_height": 0.18,
            "high_height": 0.30,
            "min_scale": 0.15,
            "max_scale": 1.0,
            "force_threshold": 10.0,
        },
    )
    cfg.rewards["non_foot_ground_collision"] = RewardTermCfg(
        func=general_motion_rewards.contact_cost_by_reference_height,
        weight=-2.0,
        params={
            "command_name": "motion",
            "sensor_name": non_foot_ground_cfg.name,
            "low_height": 0.18,
            "high_height": 0.30,
            "min_scale": 0.0,
            "max_scale": 1.0,
            "force_threshold": None,
        },
    )
    cfg.rewards["no_non_foot_ground_contact"] = RewardTermCfg(
        func=general_motion_rewards.no_contact_by_reference_height,
        weight=0.5,
        params={
            "command_name": "motion",
            "sensor_name": non_foot_ground_cfg.name,
            "low_height": 0.22,
            "high_height": 0.32,
        },
    )
    cfg.rewards["feet_support"] = RewardTermCfg(
        func=general_motion_rewards.foot_support_by_reference_height,
        weight=0.5,
        params={
            "command_name": "motion",
            "sensor_name": feet_ground_cfg.name,
            "low_height": 0.22,
            "high_height": 0.32,
            "min_contacts": 2.0,
        },
    )

    cfg.viewer.body_name = "chest"
    cfg.viewer.distance = 1.4

    if play:
        cfg.scene.num_envs = 64
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        motion_cmd = cfg.commands["motion"]
        assert isinstance(motion_cmd, DatasetMotionCommandCfg)
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.joint_position_range = (0.0, 0.0)
        motion_cmd.sampling_mode = "start"
        motion_cmd.active_motion_count = 16
        motion_cmd.max_cache_size = 16

    return cfg
