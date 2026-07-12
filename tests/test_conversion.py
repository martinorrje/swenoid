"""Tests for control-rate-aware motion conversion."""

import csv

import mujoco
import numpy as np

from swenoid.retargeting.convert_motion import _resample_qpos, convert_motion
from swenoid.robot import SWENOID_XML


def test_resample_qpos_preserves_duration_and_quaternion_norm() -> None:
    model = mujoco.MjModel.from_xml_path(str(SWENOID_XML))
    qpos = np.repeat(model.qpos0[None, :], 121, axis=0)
    qpos[:, 0] = np.linspace(0.0, 1.0, 121)
    qpos[:, 7] = np.linspace(-0.5, 0.5, 121)
    output = _resample_qpos(model, qpos, input_fps=120.0, output_fps=50.0)
    assert output.shape == (51, model.nq)
    np.testing.assert_allclose(output[:, 0], np.linspace(0.0, 1.0, 51))
    np.testing.assert_allclose(np.linalg.norm(output[:, 3:7], axis=1), 1.0)


def test_convert_motion_stores_full_body_layout(tmp_path) -> None:
    model = mujoco.MjModel.from_xml_path(str(SWENOID_XML))
    qpos = np.repeat(model.qpos0[None, :], 13, axis=0)
    source = tmp_path / "motion.qpos.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Frame", *[f"qpos_{index}" for index in range(model.nq)]])
        for index, row in enumerate(qpos):
            writer.writerow([index, *row])
    output = tmp_path / "motion.npz"
    convert_motion(
        source,
        output,
        input_fps=120.0,
        output_fps=50.0,
        compressed=True,
        ground_align=False,
    )
    with np.load(output) as data:
        assert data["fps"].item() == 50.0
        assert data["source_fps"].item() == 120.0
        assert data["joint_pos"].shape[1] == 24
        assert data["body_pos_w"].shape[1] == model.nbody - 1
        assert data["body_names"].tolist() == [
            model.body(index).name for index in range(1, model.nbody)
        ]
