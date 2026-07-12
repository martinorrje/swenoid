"""Swenoid single-reference-motion tracking task."""

from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from swenoid.robot import SWENOID_ACTION_SCALE, get_swenoid_robot_cfg
from swenoid.tasks.mdp import SwenoidMotionCommandCfg

SWENOID_TRACKING_BODIES = (
    "hip_mid_section",
    "l_hip_y",
    "l_thigh",
    "left_shin_part",
    "left_foot",
    "r_hip_y",
    "r_thigh",
    "right_shin_part",
    "right_foot",
    "torso",
    "chest",
    "l_shoulder_x",
    "l_arm_mid_part",
    "l_elbow_part",
    "left_hand",
    "neck",
    "head_platform",
    "r_shoulder_x",
    "r_arm_mid_part",
    "r_elbow_part",
    "right_hand",
)


def _replace_motion_command(
    cfg: ManagerBasedRlEnvCfg,
) -> SwenoidMotionCommandCfg:
    original = cfg.commands["motion"]
    assert isinstance(original, MotionCommandCfg)
    kwargs = {item.name: getattr(original, item.name) for item in fields(original)}
    command = SwenoidMotionCommandCfg(**kwargs)
    cfg.commands["motion"] = command
    return command


def swenoid_flat_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the deployable single-motion tracking environment."""
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": get_swenoid_robot_cfg(actuator_model="bam")}
    cfg.scene.num_envs = 4096
    cfg.episode_length_s = 10.0
    cfg.sim.njmax = 300
    cfg.sim.nconmax = 70
    cfg.sim.contact_sensor_maxmatch = 128

    original_terms = cfg.observations["actor"].terms
    original_terms["joint_pos"].params = {}
    original_terms["joint_vel"].noise = Unoise(n_min=-1.5, n_max=1.5)
    cfg.observations["actor"] = ObservationGroupCfg(
        terms={
            "base_ang_vel": original_terms["base_ang_vel"],
            "projected_gravity": ObservationTermCfg(
                func=envs_mdp.projected_gravity_from_sensor,
                params={"sensor_name": "robot/imu_upvector"},
                noise=Unoise(n_min=-0.05, n_max=0.05),
            ),
            "joint_pos": original_terms["joint_pos"],
            "joint_vel": original_terms["joint_vel"],
            "actions": original_terms["actions"],
            "command": original_terms["command"],
        },
        concatenate_terms=True,
        enable_corruption=True,
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
    cfg.scene.sensors = (self_collision_cfg,)

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = SWENOID_ACTION_SCALE

    motion_cmd = _replace_motion_command(cfg)
    motion_cmd.motion_file = ""
    motion_cmd.anchor_body_name = "hip_mid_section"
    motion_cmd.body_names = SWENOID_TRACKING_BODIES

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = ("chest",)
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_foot",
        "right_foot",
        "left_hand",
        "right_hand",
    )
    cfg.viewer.body_name = "chest"
    cfg.viewer.distance = 1.4

    if play:
        cfg.scene.num_envs = 64
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.joint_position_range = (0.0, 0.0)
        motion_cmd.sampling_mode = "start"
    return cfg
