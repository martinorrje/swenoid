"""Swenoid robot software with an optional MjLab training extension."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from swenoid.model_constants import (
        SWENOID_ACTION_SCALE as SWENOID_ACTION_SCALE,
    )
    from swenoid.model_constants import SWENOID_XML as SWENOID_XML
    from swenoid.robot import get_swenoid_robot_cfg as get_swenoid_robot_cfg

__version__ = "0.1.0"

__all__ = ["SWENOID_ACTION_SCALE", "SWENOID_XML", "get_swenoid_robot_cfg"]


def __getattr__(name: str) -> Any:
    """Load MjLab-facing exports only when explicitly requested."""
    if name in {"SWENOID_ACTION_SCALE", "SWENOID_XML"}:
        from swenoid import model_constants

        return getattr(model_constants, name)
    if name == "get_swenoid_robot_cfg":
        from swenoid.robot import get_swenoid_robot_cfg

        return get_swenoid_robot_cfg
    raise AttributeError(name)
