"""Swenoid MuJoCo model and MjLab entity configuration."""

import json
from pathlib import Path
from typing import Literal

import mujoco
from bam.mjlab import BamActuatorCfg
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from swenoid.bam_xm_actuators import register_xm_actuators
from swenoid.model_constants import (
    JOINT_ARMATURE,
    SWENOID_XM430_BAM_PARAMS,
    SWENOID_XM540_BAM_PARAMS,
    SWENOID_XML,
    XM430_DAMPING,
    XM430_EFFORT_LIMIT,
    XM430_STIFFNESS,
    XM430_TARGET_NAMES_EXPR,
    XM430_VELOCITY_LIMIT,
    XM540_DAMPING,
    XM540_EFFORT_LIMIT,
    XM540_STIFFNESS,
    XM540_TARGET_NAMES_EXPR,
    XM540_VELOCITY_LIMIT,
)
from swenoid.model_constants import (
    SWENOID_ACTION_SCALE as SWENOID_ACTION_SCALE,
)


def get_swenoid_spec() -> mujoco.MjSpec:
    """Load a fresh Swenoid spec suitable for insertion into an MjLab scene."""
    spec = mujoco.MjSpec.from_file(str(SWENOID_XML))

    for geom in list(spec.worldbody.geoms):
        if geom.name == "floor":
            spec.delete(geom)

    for actuator in list(spec.actuators):
        spec.delete(actuator)

    spec.add_sensor(
        name="imu_ang_vel",
        type=mujoco.mjtSensor.mjSENS_GYRO,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="hip_imu",
    )
    spec.add_sensor(
        name="imu_upvector",
        type=mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname="world",
        reftype=mujoco.mjtObj.mjOBJ_SITE,
        refname="hip_imu",
    )
    spec.add_sensor(
        name="imu_lin_vel",
        type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="hip_imu",
    )
    spec.add_sensor(
        name="root_angmom",
        type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname="hip_mid_section",
    )
    return spec


SWENOID_XM540_DC_ACTUATOR_CFG = DcMotorActuatorCfg(
    target_names_expr=XM540_TARGET_NAMES_EXPR,
    stiffness=XM540_STIFFNESS,
    damping=XM540_DAMPING,
    effort_limit=XM540_EFFORT_LIMIT,
    saturation_effort=XM540_EFFORT_LIMIT,
    velocity_limit=XM540_VELOCITY_LIMIT,
    armature=JOINT_ARMATURE,
)
SWENOID_XM430_DC_ACTUATOR_CFG = DcMotorActuatorCfg(
    target_names_expr=XM430_TARGET_NAMES_EXPR,
    stiffness=XM430_STIFFNESS,
    damping=XM430_DAMPING,
    effort_limit=XM430_EFFORT_LIMIT,
    saturation_effort=XM430_EFFORT_LIMIT,
    velocity_limit=XM430_VELOCITY_LIMIT,
    armature=JOINT_ARMATURE,
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.29),
    joint_pos={
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    },
    joint_vel={".*": 0.0},
)

_FOOT_REGEX = r"^(left|right)_foot_collision$"
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={_FOOT_REGEX: 3, ".*_collision": 1},
    priority={_FOOT_REGEX: 1},
    friction={_FOOT_REGEX: (0.6,)},
)

SWENOID_DC_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(SWENOID_XM540_DC_ACTUATOR_CFG, SWENOID_XM430_DC_ACTUATOR_CFG),
    soft_joint_pos_limit_factor=0.9,
)


def _make_bam_actuator_cfg(
    *,
    json_path: Path,
    target_names_expr: tuple[str, ...],
    torque_limit: float,
) -> BamActuatorCfg:
    register_xm_actuators()
    with json_path.open() as stream:
        kt = float(json.load(stream)["kt"])
    return BamActuatorCfg(
        json_path=str(json_path),
        target_names_expr=target_names_expr,
        vin=12.0,
        kp_fw=800.0,
        max_current=torque_limit / kt,
    )


def make_swenoid_articulation(
    actuator_model: Literal["bam", "dc"] = "bam",
) -> EntityArticulationInfoCfg:
    """Create an articulation using identified BAM or baseline DC actuators."""
    if actuator_model == "dc":
        return SWENOID_DC_ARTICULATION
    if actuator_model != "bam":
        raise ValueError(f"Unknown actuator model: {actuator_model!r}")
    return EntityArticulationInfoCfg(
        actuators=(
            _make_bam_actuator_cfg(
                json_path=SWENOID_XM540_BAM_PARAMS,
                target_names_expr=XM540_TARGET_NAMES_EXPR,
                torque_limit=XM540_EFFORT_LIMIT,
            ),
            _make_bam_actuator_cfg(
                json_path=SWENOID_XM430_BAM_PARAMS,
                target_names_expr=XM430_TARGET_NAMES_EXPR,
                torque_limit=XM430_EFFORT_LIMIT,
            ),
        ),
        soft_joint_pos_limit_factor=0.9,
    )


def get_swenoid_robot_cfg(
    actuator_model: Literal["bam", "dc"] = "bam",
) -> EntityCfg:
    """Return a fresh Swenoid entity configuration."""
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_swenoid_spec,
        articulation=make_swenoid_articulation(actuator_model),
    )
