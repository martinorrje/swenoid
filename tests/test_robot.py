"""Tests for the packaged Swenoid model and actuator configuration."""

import json

import pytest
from mjlab.entity import Entity

from swenoid.robot import (
    SWENOID_XM430_BAM_PARAMS,
    SWENOID_XM540_BAM_PARAMS,
    get_swenoid_robot_cfg,
)


def test_packaged_model_compiles() -> None:
    entity = Entity(get_swenoid_robot_cfg(actuator_model="dc"))
    model = entity.spec.compile()
    assert entity.num_joints == 24
    assert entity.num_actuators == 24
    assert "floor" not in {model.geom(index).name for index in range(model.ngeom)}
    sensors = {model.sensor(index).name for index in range(model.nsensor)}
    assert {"imu_ang_vel", "imu_upvector", "imu_lin_vel", "root_angmom"} <= sensors


def test_bam_articulation_uses_selected_fits() -> None:
    cfg = get_swenoid_robot_cfg(actuator_model="bam")
    assert cfg.articulation is not None
    xm540, xm430 = cfg.articulation.actuators
    assert type(xm540).__name__ == "BamActuatorCfg"
    assert xm540.json_path == str(SWENOID_XM540_BAM_PARAMS)
    assert xm540.max_current == pytest.approx(2.9290450291371037)
    assert type(xm430).__name__ == "BamActuatorCfg"
    assert xm430.json_path == str(SWENOID_XM430_BAM_PARAMS)
    assert xm430.max_current == pytest.approx(1.5902010671375502)


def test_fit_parameter_files_identify_xm_models() -> None:
    with SWENOID_XM430_BAM_PARAMS.open() as stream:
        xm430 = json.load(stream)
    with SWENOID_XM540_BAM_PARAMS.open() as stream:
        xm540 = json.load(stream)
    assert (xm430["actuator"], xm430["model"]) == ("xm430_w350", "m6")
    assert (xm540["actuator"], xm540["model"]) == ("xm540_w270", "m5")
