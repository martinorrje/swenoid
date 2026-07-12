"""Swenoid velocity task configuration."""

from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from swenoid.robot import SWENOID_ACTION_SCALE, get_swenoid_robot_cfg
from swenoid.tasks import mdp


def _replace_velocity_command(
    cfg: ManagerBasedRlEnvCfg,
) -> mdp.SwenoidVelocityCommandCfg:
    original = cfg.commands["twist"]
    assert isinstance(original, UniformVelocityCommandCfg)
    kwargs = {item.name: getattr(original, item.name) for item in fields(original)}
    command = mdp.SwenoidVelocityCommandCfg(**kwargs)
    cfg.commands["twist"] = command
    return command


def swenoid_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the rough-terrain Swenoid velocity environment."""
    cfg = make_velocity_env_cfg()
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 70
    cfg.scene.entities = {"robot": get_swenoid_robot_cfg(actuator_model="bam")}

    actor_terms = cfg.observations["actor"].terms
    actor_terms.pop("base_lin_vel", None)
    actor_terms.pop("height_scan", None)
    actor_terms["projected_gravity"].func = envs_mdp.projected_gravity_from_sensor
    actor_terms["projected_gravity"].params = {"sensor_name": "robot/imu_upvector"}

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "hip_mid_section"

    site_names = ("left_foot", "right_foot")
    geom_names = ("left_foot_collision", "right_foot_collision")
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=site, entity="robot") for site in site_names
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
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
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = SWENOID_ACTION_SCALE
    cfg.viewer.body_name = "chest"
    cfg.viewer.distance = 1.0

    twist_cmd = _replace_velocity_command(cfg)
    twist_cmd.viz.z_offset = 0.55
    twist_cmd.rel_standing_envs = 0.25
    twist_cmd.ranges.lin_vel_x = (-0.5, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = ("chest",)
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        r".*hip_pitch.*": 0.3,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.35,
        r".*ankle.*": 0.2,
        r".*torso_yaw.*": 0.2,
        r".*torso_roll.*": 0.08,
        r".*torso_pitch.*": 0.1,
        r".*shoulder_pitch.*": 0.15,
        r".*shoulder_roll.*": 0.15,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.15,
        r".*neck.*": 0.2,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.2,
        r".*hip_yaw.*": 0.2,
        r".*knee.*": 0.6,
        r".*ankle.*": 0.3,
        r".*torso_yaw.*": 0.3,
        r".*torso_roll.*": 0.08,
        r".*torso_pitch.*": 0.2,
        r".*shoulder_pitch.*": 0.5,
        r".*shoulder_roll.*": 0.2,
        r".*shoulder_yaw.*": 0.15,
        r".*elbow.*": 0.35,
        r".*neck.*": 0.3,
    }

    low_command_threshold = 0.05
    twist_cmd.command_deadband = low_command_threshold
    cfg.rewards["pose"].params["walking_threshold"] = low_command_threshold
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("chest",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("chest",)
    for reward_name in ("foot_clearance", "foot_slip"):
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["action_rate_l2"].weight = -0.06
    cfg.rewards["air_time"].weight = 0.25
    cfg.rewards["air_time"].params.update(
        threshold_min=0.12,
        threshold_max=0.7,
        command_threshold=low_command_threshold,
    )
    for reward_name in (
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
    ):
        cfg.rewards[reward_name].params["command_threshold"] = low_command_threshold
    cfg.rewards["foot_clearance"].params["target_height"] = 0.06
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.06

    cfg.rewards["feet_still_at_low_command"] = RewardTermCfg(
        func=mdp.feet_still_at_low_command,
        weight=-1.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "command_name": "twist",
            "command_threshold": low_command_threshold,
            "air_cost_scale": 1.0,
            "velocity_cost_scale": 0.5,
            "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
        },
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=velocity_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    )

    cfg.curriculum["command_vel"].params["velocity_stages"] = [
        {
            "step": 0,
            "lin_vel_x": (-0.5, 0.8),
            "lin_vel_y": (-0.3, 0.3),
            "ang_vel_z": (-0.5, 0.5),
        },
        {
            "step": 5000 * 24,
            "lin_vel_x": (-0.75, 1.0),
            "lin_vel_y": (-0.4, 0.4),
            "ang_vel_z": (-0.6, 0.6),
        },
        {
            "step": 10000 * 24,
            "lin_vel_x": (-1.0, 1.2),
            "lin_vel_y": (-0.5, 0.5),
            "ang_vel_z": (-0.7, 0.7),
        },
    ]

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )
        if cfg.scene.terrain is not None:
            generator = cfg.scene.terrain.terrain_generator
            if generator is not None:
                generator.curriculum = False
                generator.num_cols = 5
                generator.num_rows = 5
                generator.border_width = 10.0
    return cfg


def swenoid_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the flat-ground Swenoid velocity environment."""
    cfg = swenoid_rough_env_cfg(play=play)
    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = tuple(
        sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
    )
    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms.pop("height_scan", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-0.75, 1.0)
        twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)
    return cfg
