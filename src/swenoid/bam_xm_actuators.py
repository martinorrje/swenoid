"""BAM runtime definitions for Swenoid's Dynamixel X-series servos."""

import math
from typing import cast

from bam.actuator import VoltageControlledActuator
from bam.actuators import actuators
from bam.parameter import Parameter
from bam.testbench import Pendulum, Testbench

X_SERIES_ENCODER_COUNTS_PER_REV = 4096
X_SERIES_KP_DIVISOR = 128
X_SERIES_PWM_LIMIT = 885


class _XmActuator(VoltageControlledActuator):
    def __init__(self, testbench_class: Testbench):
        super().__init__(
            testbench_class,
            vin=12.0,
            kp=800,
            error_gain=(X_SERIES_ENCODER_COUNTS_PER_REV / (2 * math.pi))
            / (X_SERIES_KP_DIVISOR * X_SERIES_PWM_LIMIT),
            max_pwm=1.0,
        )

    def get_extra_inertia(self) -> float:
        return self.model.armature.value


class Xm430W350Actuator(_XmActuator):
    """Dynamixel XM430-W350 populated from a fitted BAM JSON file."""

    def initialize(self) -> None:
        self.model.kt = Parameter(1.8, 0.1, 4.0)
        self.model.R = Parameter(5.2, 1.0, 15.0)
        self.model.armature = Parameter(0.01, 0.0001, 0.1)


class Xm540W270Actuator(_XmActuator):
    """Dynamixel XM540-W270 populated from a fitted BAM JSON file."""

    def initialize(self) -> None:
        self.model.kt = Parameter(2.4, 0.5, 6.0)
        self.model.R = Parameter(2.7, 0.5, 10.0)
        self.model.armature = Parameter(0.03, 0.0001, 0.3)


def register_xm_actuators() -> None:
    """Register Swenoid's XM models if the installed BAM release lacks them."""
    pendulum = cast(Testbench, Pendulum)
    actuators.setdefault("xm430_w350", lambda: Xm430W350Actuator(pendulum))
    actuators.setdefault("xm540_w270", lambda: Xm540W270Actuator(pendulum))
