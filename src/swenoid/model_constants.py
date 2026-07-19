"""Packaged model paths and simulator-independent Swenoid constants."""

from __future__ import annotations

import math
from pathlib import Path

ASSET_DIR = Path(__file__).parent / "assets" / "swenoid"
SWENOID_XML = ASSET_DIR / "xmls" / "swenoid.xml"
SWENOID_XM430_BAM_PARAMS = ASSET_DIR / "params" / "xm430_m6.json"
SWENOID_XM540_BAM_PARAMS = ASSET_DIR / "params" / "xm540_m5.json"
SWENOID_HARDWARE_CONFIG = ASSET_DIR / "params" / "hardware.json"

for required_file in (
    SWENOID_XML,
    SWENOID_XM430_BAM_PARAMS,
    SWENOID_XM540_BAM_PARAMS,
    SWENOID_HARDWARE_CONFIG,
):
    if not required_file.is_file():
        raise FileNotFoundError(f"Missing packaged Swenoid asset: {required_file}")

XM540_TARGET_NAMES_EXPR = (
    r".*(hip_(roll|yaw|pitch)|knee|torso_(pitch|roll|yaw)|shoulder_(pitch|roll))_joint",
)
XM430_TARGET_NAMES_EXPR = (r".*(ankle|shoulder_yaw|elbow|neck_(yaw|pitch|roll))_joint",)

XM540_STIFFNESS = 20.0
XM540_DAMPING = 1.0
XM540_EFFORT_LIMIT = 10.6
XM540_VELOCITY_LIMIT = 30.0 * 2.0 * math.pi / 60.0
XM430_STIFFNESS = 20.0
XM430_DAMPING = 1.0
XM430_EFFORT_LIMIT = 4.1
XM430_VELOCITY_LIMIT = 46.0 * 2.0 * math.pi / 60.0
JOINT_ARMATURE = 0.01

SWENOID_ACTION_SCALE: dict[str, float] = {
    XM540_TARGET_NAMES_EXPR[0]: 0.25 * XM540_EFFORT_LIMIT / XM540_STIFFNESS,
    XM430_TARGET_NAMES_EXPR[0]: 0.25 * XM430_EFFORT_LIMIT / XM430_STIFFNESS,
}
