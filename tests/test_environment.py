"""Racing environment: layouts, stages, action integration, top speed, reward, failures, starts, the Oracle input and
reproducibility."""

from __future__ import annotations

import itertools

import gymnasium as gym
import numpy as np
import pytest
from conftest import REPO

from contextual_bicycle_pacejka import ENV_ID

from contextual_bicycle_pacejka.env import environment, held_out_bands, make_env, make_stage, sample_setup
from contextual_bicycle_pacejka.model.observation import vehicle_world_position
from contextual_bicycle_pacejka.model.state import AdmissibilityError, VehicleState
from contextual_bicycle_pacejka.model.termination import check_failure
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import (
    ACTION_DIM,
    DECISION_DT,
    DELTA_MAX,
    FREE_STAGE_SIZING_MPS,
    G,
    N_PREVIEW,
    OBS_DIM,
    SPIN_EPSI_RAD,
    STAGE_TOP_SPEEDS_MPS,
    STALL_SPEED_MPS,
    START_SPEED_MIN_MPS,
)
from contextual_bicycle_pacejka.spec.layout import OBS

CATALUNYA = REPO / "data/tracks/catalunya/reference.npz"
MAIN_STRAIGHT_S = -200.0  # [m] 980 m of straight ahead (the main straight runs from s = -215 m to s = 780 m)
TRACK = RealTrack(str(CATALUNYA))


def follow_centreline(state: np.ndarray, throttle: float) -> np.ndarray:
    """Deterministic test input: kinematic steering for the curvature with feedback on e_y and e_psi."""
    s, ey, epsi = state[0], state[1], state[2]
    delta = 2.6 * TRACK.kappa(s) - 0.05 * ey - 0.5 * epsi
    return np.array([np.clip(delta / DELTA_MAX, -1.0, 1.0), throttle], dtype=np.float32)


def test_layouts() -> None:
    assert (OBS.MU, OBS.KAPPA_START, OBS.W_LEFT_START, OBS.W_RIGHT_START, OBS.W_RIGHT_END) == (9, 10, 40, 70, 100)
    assert OBS_DIM == 10 + 3 * N_PREVIEW == 100 and OBS.B == OBS_DIM
    nominal, oracle = make_env(CATALUNYA), make_env(CATALUNYA, oracle=True)
    assert nominal.observation_space.shape == (OBS_DIM,) and oracle.observation_space.shape == (OBS_DIM + 1,)
    assert nominal.action_space.shape == (ACTION_DIM,)
    assert np.all(nominal.action_space.low == -1.0) and np.all(nominal.action_space.high == 1.0)
    assert nominal.unwrapped.edge_margin == 0.0 and make_env(CATALUNYA, edge_margin=0.3).unwrapped.edge_margin == 0.3


def test_registered_gymnasium_id() -> None:
    env = gym.make(ENV_ID, track=TRACK, stage=make_stage(2, TRACK), oracle=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM + 1,)


def test_track_edges_are_the_centreline_widths() -> None:
    """ey > 0 (left) is bounded by the left width of the centreline, ey < 0 by the right one, with no floor; the
    world point at ey = +w_left lies on the left edge."""
    with np.load(CATALUNYA) as reference:
        i = int(np.argmin(reference["w_tr_left_m"]))  # the narrowest left side of the lap (about 4.2 m)
        s, w_left = float(reference["s_m"][i]), float(reference["w_tr_left_m"][i])
        assert TRACK.boundary_left(s) == pytest.approx(w_left, abs=1e-9) and w_left < 5.0
        assert not TRACK.is_out_of_bounds(0.99 * w_left, s) and TRACK.is_out_of_bounds(1.01 * w_left, s)
        right = TRACK.boundary_right(s)
        assert not TRACK.is_out_of_bounds(-0.99 * right, s) and TRACK.is_out_of_bounds(-1.01 * right, s)

        x, y = vehicle_world_position(VehicleState(s=s, ey=w_left), TRACK)
        psi = reference["psi_rad"][i]
        offset = (x - reference["x_m"][i]) * -np.sin(psi) + (y - reference["y_m"][i]) * np.cos(psi)
        assert offset == pytest.approx(w_left, abs=0.1)


@pytest.mark.parametrize("s", [3400.0, TRACK.track_length - 100.0])
def test_mean_curvature_is_the_integral_of_the_plant_curvature(s: float) -> None:
    """Over each preview interval, also across the end of the lap."""
    distances = np.array([4.6, 40.0, 80.0, 160.0])
    edges = np.concatenate([[0.0], distances])
    expected = [np.trapezoid(TRACK.kappa_many(np.linspace(s + a, s + b, 20001)), dx=(b - a) / 20000) / (b - a)
                for a, b in itertools.pairwise(edges)]
    np.testing.assert_allclose(TRACK.mean_curvature(s, distances), expected, atol=1e-7)


@pytest.mark.parametrize("number", range(1, len(STAGE_TOP_SPEEDS_MPS) + 1))
def test_stage_rules(number: int) -> None:
    stage = make_stage(number, TRACK)
    v = stage.sizing_speed
    assert v == (STAGE_TOP_SPEEDS_MPS[number - 1] or FREE_STAGE_SIZING_MPS)
    assert len(stage.preview) == N_PREVIEW and stage.preview[0] == pytest.approx(v * DECISION_DT)
    assert stage.preview[-1] == pytest.approx(v * v / (2.0 * G) + v)
    assert np.allclose(np.diff(stage.preview), np.diff(stage.preview)[0])
    assert stage.start_speed == (START_SPEED_MIN_MPS, STAGE_TOP_SPEEDS_MPS[max(number - 2, 0)])
    assert stage.scales.speed == v and stage.scales.curvature == pytest.approx(1.0 / 22.8, rel=0.01)


def test_longitudinal_command_integrates_its_rate() -> None:
    env = make_env(CATALUNYA, stage=3).unwrapped
    env.reset(seed=0, options={"initial_s": MAIN_STRAIGHT_S, "initial_vx": 10.0})
    for expected in (0.25, 0.5, 0.75, 1.0, 1.0):
        env.step(np.array([0.0, 1.0], dtype=np.float32))
        assert env.tau == pytest.approx(expected)
    env.step(np.array([0.0, -0.4], dtype=np.float32))
    assert env.tau == pytest.approx(0.9) and env.delta_ref == 0.0


def test_longitudinal_command_does_not_wind_up_at_top_speed() -> None:
    """Full throttle at the top speed leaves the command near zero, so one braking decision brakes at once."""
    env = make_env(CATALUNYA, stage=3).unwrapped
    env.reset(seed=0, options={"initial_s": MAIN_STRAIGHT_S, "initial_vx": 19.0})
    for _ in range(100):  # 5 s of full throttle down the main straight
        env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert env.state.vx == pytest.approx(20.0, abs=0.4) and 0.0 <= env.tau < 0.3
    env.step(np.array([0.0, -1.0], dtype=np.float32))
    assert env.tau < 0.0


@pytest.mark.parametrize("number", range(1, len(STAGE_TOP_SPEEDS_MPS)))
def test_top_speed_bound(number: int) -> None:
    """Full throttle down the main straight for up to 30 s: vx reaches the top speed of the stage and stays within
    2 % of it."""
    top = STAGE_TOP_SPEEDS_MPS[number - 1]
    env = make_env(CATALUNYA, stage=number)
    _, info = env.reset(seed=0, options={"initial_s": MAIN_STRAIGHT_S, "initial_vx": max(3.0, 0.9 * top)})
    speeds = []
    for _ in range(600):
        _, _, terminated, _, info = env.step(follow_centreline(info["state"], 1.0))
        speeds.append(info["state"][3])
        if terminated:
            break
    assert 0.97 * top <= max(speeds) <= 1.02 * top


def test_reward_is_the_normalised_progress() -> None:
    env = make_env(CATALUNYA, stage=4)
    _, info = env.reset(seed=0, options={"initial_s": MAIN_STRAIGHT_S, "initial_vx": 20.0})
    _, reward, terminated, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert not terminated and info["progress"] == pytest.approx(20.0 * DECISION_DT, rel=1e-3)
    assert info["reward_terms"]["progress"] == pytest.approx(info["progress"] / (30.0 * DECISION_DT))
    assert info["reward_terms"]["action_rate"] == 0.0
    assert reward == pytest.approx(0.1 * info["reward_terms"]["progress"])
    _, reward, _, _, info = env.step(np.array([0.5, -1.0], dtype=np.float32))
    assert info["reward_terms"]["action_rate"] == pytest.approx(-0.01 * (0.25 ** 2 + 1.0))


def test_failures_of_a_state() -> None:
    s = MAIN_STRAIGHT_S
    assert check_failure(VehicleState(s=s, ey=TRACK.boundary_left(s) + 0.01, vx=10.0), TRACK) == "off_track"
    assert check_failure(VehicleState(s=s, epsi=-SPIN_EPSI_RAD - 0.01, vx=10.0), TRACK) == "spin"
    assert check_failure(VehicleState(s=s, vx=STALL_SPEED_MPS - 0.01), TRACK) == "stall"
    assert check_failure(VehicleState(s=s, vx=10.0), TRACK) is None
    inside = VehicleState(s=s, ey=TRACK.boundary_left(s) - 0.2, vx=10.0)
    assert check_failure(inside, TRACK) is None and check_failure(inside, TRACK, edge_margin=0.3) == "off_track"


def test_failures_end_the_episode_with_the_penalty() -> None:
    env = make_env(CATALUNYA, stage=3)
    # off the track: full steering to the left from near the left edge
    _, info = env.reset(seed=0, options={"initial_s": MAIN_STRAIGHT_S, "initial_vx": 15.0,
                                         "initial_ey": TRACK.boundary_left(MAIN_STRAIGHT_S) - 0.3})
    for _ in range(40):
        _, reward, terminated, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        if terminated:
            break
    assert terminated and info["failure"] == "off_track" and info["reward_terms"]["failure"] == -1.0
    # stall: full braking from 3 m/s
    env.reset(seed=0, options={"initial_s": MAIN_STRAIGHT_S, "initial_vx": 3.0})
    for _ in range(40):
        _, reward, terminated, _, info = env.step(np.array([0.0, -1.0], dtype=np.float32))
        if terminated:
            break
    assert terminated and info["failure"] == "stall" and reward < -0.09


def test_inadmissible_plant_state_ends_the_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env(CATALUNYA, stage=3)
    _, info = env.reset(seed=0, options={"initial_s": MAIN_STRAIGHT_S, "initial_vx": 15.0})

    def inadmissible(*args, **kwargs):
        raise AdmissibilityError("test")

    monkeypatch.setattr(environment, "step_plant", inadmissible)
    _, _, terminated, _, after = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert terminated and after["failure"] == "inadmissible" and np.array_equal(after["state"], info["state"])


def test_starts_and_oracle_input() -> None:
    env = make_env(CATALUNYA, stage=5, b_range=(-0.2, 0.1), oracle=True)
    for seed in range(50):
        obs, info = env.reset(seed=seed)
        s, ey, epsi, vx = info["state"][:4]
        b = info["setup"][-1]
        assert 0.0 <= s < TRACK.track_length and ey == 0.0 and epsi == 0.0 and START_SPEED_MIN_MPS <= vx <= 30.0
        assert -0.2 <= b <= 0.1 and obs[OBS.B] == pytest.approx(b / 0.2, rel=1e-6)
    nominal_obs, info = make_env(CATALUNYA, stage=5).reset(seed=0)
    assert nominal_obs.shape == (OBS_DIM,) and info["setup"][-1] == 0.0


def test_held_out_bands_are_never_drawn() -> None:
    """b is drawn over its range but never in the open bands of half-width 0.01 around the held-out values."""
    bands = held_out_bands((-0.10, 0.09))
    np.testing.assert_allclose(bands, [(-0.11, -0.09), (0.08, 0.10)])
    rng = np.random.default_rng(0)
    draws = np.array([sample_setup(rng, (-0.20, 0.17), bands).b for _ in range(20_000)])
    assert draws.min() >= -0.20 and draws.max() <= 0.17
    assert not np.any((np.abs(draws + 0.10) < 0.01) | (np.abs(draws - 0.09) < 0.01))
    assert np.mean(draws < -0.11) == pytest.approx(0.09 / 0.33, abs=0.02)  # uniform over the rest of the range
    env = make_env(CATALUNYA, stage=3, b_range=(-0.20, 0.17), b_exclude=bands)
    for seed in range(200):
        _, info = env.reset(seed=seed)
        assert not any(lo < info["setup"][-1] < hi for lo, hi in bands)
    with pytest.raises(ValueError):
        sample_setup(rng, (-0.105, -0.095), bands)


def test_reproducible_from_the_seed() -> None:
    runs = []
    for _ in range(2):
        env = make_env(CATALUNYA, stage=4, b_range=(-0.2, 0.17))
        obs, _ = env.reset(seed=7)
        rng = np.random.default_rng(7)
        trace = [obs]
        for _ in range(50):
            obs, reward, terminated, truncated, _ = env.step(rng.uniform(-1.0, 1.0, 2).astype(np.float32))
            trace.append(np.append(obs, reward))
            if terminated or truncated:
                obs, _ = env.reset()
        runs.append(np.concatenate(trace))
    np.testing.assert_array_equal(runs[0], runs[1])
