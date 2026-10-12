"""Onboard sensor channels that are not part of the privileged observation.

An accelerometer at the centre of mass measures the specific force of the body:

    ax_imu = F_body_x / m,     ay_imu = F_body_y / m     [m/s^2]

The privileged observation holds ax = dvx/dt = ax_imu + vy yaw_rate instead, which contains the unmeasured lateral
velocity. The channels are computed from the diagnostics of the step that produced the observation, so at reset
(diagnostics all zero) they are zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.model.state import SetupVector
from contextual_bicycle_pacejka.spec.constants import IMU_DIM


def imu_specific_force(diag: dict[str, Any], setup: SetupVector) -> NDArray[np.float64]:
    """Return [ax_imu, ay_imu] [m/s^2] from the body-frame resultants of the step diagnostics."""
    imu = np.empty(IMU_DIM, dtype=np.float64)
    imu[0] = float(diag["F_body_x"]) / float(setup.m)
    imu[1] = float(diag["F_body_y"]) / float(setup.m)
    return imu
