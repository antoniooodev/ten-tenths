"""Student observation, sensor channels and traction cost.

Covers the layouts of the levels and of the diagnostic oracle, its offset and its clip, the accelerometer channels
against the plant equations, the traction cost and its reward weight, the frame history, the absence of setup
information in S1/S2, and the bit-exact agreement of the student observation built online (``env/student.py``) with
the one built offline from logged channels, for the default and the long history and with the oracle.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import REPO

from contextual_bicycle_pacejka.control.teacher import MATv0
from contextual_bicycle_pacejka.env import make_env, make_student_env, randomized_setup
from contextual_bicycle_pacejka.learning.policy import policy_env
from contextual_bicycle_pacejka.model.reward import (
    OFFLINE_REWARD_WEIGHTS,
    RewardWeights,
    compute_reward,
    compute_traction_cost,
)
from contextual_bicycle_pacejka.model.state import VehicleState
from contextual_bicycle_pacejka.model.student import (
    previous_actions,
    stack_frames,
    student_frames,
    student_observations,
    student_oracle,
    student_previews,
)
from contextual_bicycle_pacejka.spec.constants import (
    DT,
    MU_HI,
    MU_LO,
    OBS_DIM,
    TRACTION_SLIP_MARGIN,
    VREF_REFERENCE_MU,
)
from contextual_bicycle_pacejka.spec.layout import OBS, STUDENT_PREVIEW_DIM, StudentLayout, student_layout

MONZA_R0 = REPO / "data/tracks/monza/racing_line_R0.npz"
ROLLOUT_STEPS = 400
MU, SEED, START_S, START_VX = 1.1, 43, 935.35833801, 6.6


def _reset_options(mu: float = MU, seed: int = SEED) -> dict:
    return {"setup": randomized_setup(mu, np.random.default_rng(seed)), "initial_s": START_S, "initial_vx": START_VX}


def _teacher_rollout(level: str | StudentLayout) -> dict[str, np.ndarray]:
    """Teacher lap start on Monza R0 in the student environment of ``level``; logs every channel."""
    env = make_student_env(MONZA_R0, student_layout(level) if isinstance(level, str) else level)
    teacher = MATv0()
    student, info = env.reset(seed=SEED, options=_reset_options())
    log: dict[str, list] = {"student": [], "obs": [], "imu": [], "exec": [], "diag_ax": [], "state": []}
    for _ in range(ROLLOUT_STEPS):
        action = teacher.act(info["privileged_obs"])
        log["student"].append(student)
        log["obs"].append(info["privileged_obs"])
        log["imu"].append(info["imu"])
        log["exec"].append(np.clip(action, env.action_space.low, env.action_space.high))
        log["state"].append(env.unwrapped.state.to_array())
        student, _, terminated, truncated, info = env.step(action)
        log["diag_ax"].append(info["diag"]["ax"])
        assert not (terminated or truncated)
    return {key: np.asarray(value) for key, value in log.items()}


# --------------------------------------------------------------------------------------------- layouts
def test_student_layouts() -> None:
    s0, s1, s2 = (student_layout(level) for level in ("S0", "S1", "S2"))
    assert s0.privileged and s0.dim == OBS_DIM
    assert (s1.frame_dim, s2.frame_dim) == (12, 9)
    assert s1.channels[:9] == s2.channels  # S2 is contained in S1
    assert (s1.dim, s2.dim) == (12 * 8 + STUDENT_PREVIEW_DIM, 9 * 8 + STUDENT_PREVIEW_DIM)
    assert s2.window == 36
    slices = s2.preview_slices()
    assert slices["vref"].start == s2.dynamic_dim and slices["ax_preview"].stop == s2.dim
    with pytest.raises(ValueError):
        student_layout("S3")


def test_oracle_layout() -> None:
    """The oracle entries follow the preview; the long history keeps the frame size and widens the window."""
    s1, oracle = student_layout("S1"), student_layout("S1", oracle=("mu",))
    assert oracle.dim == s1.dim + 1 and oracle.preview_slices() == s1.preview_slices()
    assert student_layout("S2", stride=50).window == 351 and student_layout("S2", stride=50).dim == s1.dim - 24
    for bad in ({"oracle": ("vy",)}, {"oracle": ("mu", "mu")}, {"n_frames": 0}):
        with pytest.raises(ValueError):
            student_layout("S1", **bad)
    with pytest.raises(ValueError):
        student_layout("S0", oracle=("mu",))
    rows = student_observations(np.zeros((3, OBS_DIM)), np.zeros((3, 2)), np.zeros((3, 2)), MU, oracle)
    np.testing.assert_array_equal(rows[:, -1], np.float32(MU))


def test_oracle_offset() -> None:
    """An offset changes only the oracle entry, online as offline; offsets of absent entries are rejected."""
    layout = student_layout("S2", oracle=("mu",))
    env = make_student_env(MONZA_R0, layout, oracle_offset={"mu": -0.1})
    biased, _ = env.reset(seed=SEED, options=_reset_options())
    exact, _ = make_student_env(MONZA_R0, layout).reset(seed=SEED, options=_reset_options())
    np.testing.assert_array_equal(biased[:-1], exact[:-1])
    assert biased[-1] == np.float32(MU - 0.1) and exact[-1] == np.float32(MU)
    for bad_layout, offset in ((student_layout("S2"), {"mu": 0.1}), (layout, {"vy": 0.1})):
        with pytest.raises(ValueError):
            make_student_env(MONZA_R0, bad_layout, oracle_offset=offset)
    with pytest.raises(ValueError):
        policy_env(MONZA_R0, student_layout("S0"), oracle_offset={"mu": 0.1})


def test_oracle_clip() -> None:
    """With the clip an offset entry stays in the range an estimate can take; without it, it does not."""
    layout = student_layout("S2", oracle=("mu",))
    np.testing.assert_array_equal(student_oracle(2, 1.15, layout, {"mu": 0.10}), np.float32(1.25))
    np.testing.assert_array_equal(student_oracle(2, 1.15, layout, {"mu": 0.10}, clip=True), np.float32(MU_HI))
    np.testing.assert_array_equal(student_oracle(1, 0.85, layout, {"mu": -0.10}, clip=True), np.float32(MU_LO))
    np.testing.assert_array_equal(student_oracle(1, 0.95, layout, {"mu": 0.10}, clip=True), np.float32(1.05))
    env = make_student_env(MONZA_R0, layout, oracle_offset={"mu": 0.5}, oracle_clip=True)
    clipped, info = env.reset(seed=SEED, options=_reset_options())
    assert clipped[-1] == np.float32(MU_HI) and info["oracle"]["mu"] == pytest.approx(MU_HI)
    with pytest.raises(ValueError):
        policy_env(MONZA_R0, student_layout("S0"), oracle_clip=True)


# --------------------------------------------------------------------------------------------- sensors and cost
def test_imu_is_the_specific_force() -> None:
    """dvx/dt (observation ax) = ax_imu + vy yaw_rate; the IMU channels are zero at reset."""
    log = _teacher_rollout("S0")
    np.testing.assert_array_equal(log["imu"][0], [0.0, 0.0])
    vy, yaw_rate = log["state"][:, 4], log["state"][:, 5]
    np.testing.assert_allclose(log["diag_ax"][:-1], log["imu"][1:, 0] + vy[:-1] * yaw_rate[:-1], rtol=0, atol=1e-9)


def test_traction_cost() -> None:
    sat = (0.12, 0.13)
    assert compute_traction_cost(0.5 * sat[0], -0.5 * sat[1], sat) == 0.0
    assert compute_traction_cost(TRACTION_SLIP_MARGIN * sat[0], 0.0, sat) == 0.0
    assert compute_traction_cost(-1.1 * sat[0], 1.0 * sat[1], sat) == pytest.approx(
        (1.1 - TRACTION_SLIP_MARGIN) + (1.0 - TRACTION_SLIP_MARGIN))


def test_traction_weight_and_offline_reward() -> None:
    state = VehicleState(s=10.0, ey=0.5, vx=30.0)
    flags = dict.fromkeys(("track_excursion", "slip_saturation", "low_speed"), False)
    base = compute_reward(state, 30.0, 5.0, DT, RewardWeights(), False, flags, False, traction_cost=0.4)
    assert base == compute_reward(state, 30.0, 5.0, DT, RewardWeights(), False, flags, False)
    weighted = compute_reward(state, 30.0, 5.0, DT, RewardWeights(w_traction=100.0), False, flags, False, 0.4)
    assert weighted == pytest.approx(base - 100.0 * 0.4 * DT)
    offline = compute_reward(state, 30.0, 5.0, DT, OFFLINE_REWARD_WEIGHTS, False, flags, False, 0.4)
    assert offline == pytest.approx(30.0 * DT - 12.0 * DT)  # no lateral and no traction term


def test_env_info_channels() -> None:
    env = make_env(MONZA_R0)
    _, info = env.reset(seed=SEED, options=_reset_options())
    np.testing.assert_array_equal(info["imu"], [0.0, 0.0])
    _, _, _, _, info = env.step(np.array([0.0, 0.3], dtype=np.float32))
    assert info["traction_cost"] >= 0.0 and info["imu"].shape == (2,)


# --------------------------------------------------------------------------------------------- history
def test_frame_history() -> None:
    """Frames at t, t - stride, ...; frames before the episode start repeat the first frame."""
    layout = student_layout("S2", n_frames=3, stride=2)
    frames = np.arange(6, dtype=np.float32)[:, None] * np.ones((1, layout.frame_dim), dtype=np.float32)
    stacks = stack_frames(frames, layout)[:, ::layout.frame_dim]
    np.testing.assert_array_equal(stacks, [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0], [4, 2, 0], [5, 3, 1]])


def test_previous_actions() -> None:
    executed = np.array([[0.1, 0.5], [0.2, -0.3], [0.0, 1.0]], dtype=np.float32)
    expected = np.array([[0.0, 0.0], [0.1, 0.5], [0.2, -0.3]], dtype=np.float32)
    np.testing.assert_array_equal(previous_actions(executed), expected)


# --------------------------------------------------------------------------------------------- no setup information
def test_s1_s2_carry_no_setup_information() -> None:
    """Changing the setup entries, the force-use features, ax and the friction scaling of v_ref leaves the student
    frames unchanged and the v_ref preview unchanged up to float32 rounding."""
    log = _teacher_rollout("S0")
    obs, imu = log["obs"][200:210], log["imu"][200:210]
    prev = previous_actions(log["exec"])[200:210]
    other_mu = 0.85
    altered = obs.copy()
    altered[:, OBS.XI_START:OBS.XI_END] = -0.3
    altered[:, [OBS.RHO_LAT, OBS.RHO_LON, OBS.FY_F, OBS.FY_R, OBS.AX]] = 0.77
    reference_vref = obs[:, OBS.VREF_START:OBS.VREF_END].astype(np.float64) / math.sqrt(MU / VREF_REFERENCE_MU)
    altered[:, OBS.VREF_START:OBS.VREF_END] = reference_vref * math.sqrt(other_mu / VREF_REFERENCE_MU)
    for level in ("S1", "S2"):
        layout = student_layout(level)
        np.testing.assert_array_equal(student_frames(obs, imu, prev, layout),
                                      student_frames(altered, imu, prev, layout))
    np.testing.assert_allclose(student_previews(obs, MU), student_previews(altered, other_mu), rtol=1e-6, atol=0)


def test_s2_carries_no_tyre_state() -> None:
    log = _teacher_rollout("S0")
    obs, imu, prev = log["obs"][:5], log["imu"][:5], previous_actions(log["exec"])[:5]
    altered = obs.copy()
    altered[:, [OBS.VY, OBS.ALPHA_F, OBS.ALPHA_R]] = 0.5
    s2, s1 = student_layout("S2"), student_layout("S1")
    np.testing.assert_array_equal(student_frames(obs, imu, prev, s2), student_frames(altered, imu, prev, s2))
    assert not np.array_equal(student_frames(obs, imu, prev, s1), student_frames(altered, imu, prev, s1))


# --------------------------------------------------------------------------------------------- online = offline
ONLINE_LAYOUTS = {"S0": student_layout("S0"), "S1": student_layout("S1"), "S2": student_layout("S2"),
                  "S1_mu": student_layout("S1", oracle=("mu",)), "S2_h50": student_layout("S2", stride=50)}


@pytest.mark.parametrize("name", ONLINE_LAYOUTS)
def test_online_student_observation_equals_offline(name: str) -> None:
    """The wrapper's observations equal those rebuilt from the logged channels, bit for bit."""
    layout = ONLINE_LAYOUTS[name]
    log = _teacher_rollout(layout)
    offline = student_observations(log["obs"], log["imu"], previous_actions(log["exec"]), MU, layout)
    assert offline.shape == (ROLLOUT_STEPS, layout.dim) and offline.dtype == np.float32
    np.testing.assert_array_equal(log["student"], offline)
