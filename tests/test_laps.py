"""Protocol laps, the point-mass yardstick and the stage criterion."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import REPO

from contextual_bicycle_pacejka.env import make_env, make_stage
from contextual_bicycle_pacejka.evaluation.corners import CORNER_WINDOWS_M, CornerRecorder
from contextual_bicycle_pacejka.evaluation.laps import (
    LAP_STARTS,
    drive_lap,
    longitudinal_limits,
    point_mass_lap_time,
    protocol_starts,
)
from contextual_bicycle_pacejka.evaluation.limit import lap_gaps, limit_criterion
from contextual_bicycle_pacejka.evaluation.stage_gate import stage_gate, two_level_bootstrap
from contextual_bicycle_pacejka.spec.constants import DELTA_MAX

CATALUNYA = REPO / "data/tracks/catalunya/reference.npz"


def centreline_follower(env):
    """Deterministic test input: kinematic steering with feedback, full throttle (the stage bounds the speed)."""
    def policy(obs: np.ndarray) -> np.ndarray:
        state = env.state
        delta = 2.6 * env.track.kappa(state.s) - 0.05 * state.ey - 0.5 * state.epsi
        return np.array([np.clip(delta / DELTA_MAX, -1.0, 1.0), 1.0], dtype=np.float32)
    return policy


def test_a_centreline_follower_completes_the_first_stage() -> None:
    env = make_env(CATALUNYA, stage=1).unwrapped
    starts = protocol_starts(env.track)
    assert len(starts) == LAP_STARTS and starts[1] == pytest.approx(env.track.track_length / LAP_STARTS)
    record = drive_lap(centreline_follower(env), env, starts[3])
    assert record.completed and record.failure is None
    assert record.lap_time == pytest.approx(env.track.track_length / 5.0, rel=0.01)


def test_timeout_and_failure() -> None:
    env = make_env(CATALUNYA, stage=1).unwrapped
    record = drive_lap(lambda obs: np.array([0.0, 0.0], dtype=np.float32), env, 0.0, timeout_s=5.0)
    assert not record.completed and record.failure == "timeout" and record.decisions == 100
    record = drive_lap(lambda obs: np.array([1.0, 0.0], dtype=np.float32), env, 0.0)
    assert not record.completed and record.failure == "off_track" and record.failure_s is not None


def test_point_mass_yardstick() -> None:
    env = make_env(CATALUNYA, stage=1).unwrapped
    traction, braking = longitudinal_limits()
    assert traction == pytest.approx(5.606, abs=1e-3) and braking == pytest.approx(8.349, abs=1e-3)
    length = env.track.track_length
    assert point_mass_lap_time(env.track, make_stage(1, env.track)) == pytest.approx(length / 5.0, rel=1e-6)
    free = point_mass_lap_time(env.track, make_stage(7, env.track))
    assert 120.0 < free < point_mass_lap_time(env.track, make_stage(6, env.track))
    third = make_stage(3, env.track)
    assert point_mass_lap_time(env.track, third, lateral_ratio=1.0) == point_mass_lap_time(env.track, third)
    assert point_mass_lap_time(env.track, third, lateral_ratio=0.808) == pytest.approx(235.8, abs=0.1)


def test_corner_measurements_of_a_lap() -> None:
    """The centreline follower at 5 m/s from s = 1395 m drives through every window, the last ones after the start
    line; it never brakes, and its slips stay well below the peak."""
    env = make_env(CATALUNYA, stage=1).unwrapped
    recorder = CornerRecorder()
    record = drive_lap(centreline_follower(env), env, 1395.0, observer=recorder)
    assert record.completed and len(recorder.samples) == record.decisions + 1  # after the reset and every decision
    corners = recorder.corners(env.track.track_length)
    assert [tuple(c["window"]) for c in corners] == list(CORNER_WINDOWS_M)  # in the order of the windows
    for corner in corners:
        assert 4.8 < corner["min_speed"] <= corner["entry_speed"] <= 5.1 and corner["braking_s"] is None
        assert corner["window"][0] <= corner["min_speed_s"] <= corner["window"][1]
        assert 0.0 < corner["peak_slip_front"] < 0.5 and 0.0 < corner["peak_slip_rear"] < 0.5
        assert corner["near_peak_front"] == corner["beyond_peak_rear"] == 0.0 and abs(corner["apex_ey"]) < 1.0


def test_stage_criterion() -> None:
    times = np.full((5, 10), 100.0)
    passed = stage_gate(np.ones((5, 10), dtype=bool), times)
    assert passed["passed"] and passed["seeds_passing"] == 5 and passed["completion"]["interval"] == [1.0, 1.0]
    assert passed["lap_time"]["mean"] == 100.0
    completed = np.ones((5, 10), dtype=bool)
    completed[0] = False  # one seed never completes a lap, another misses one lap: 4 of 5 seeds pass
    completed[1, 3] = False
    one_seed_down = stage_gate(completed, np.where(completed, times, np.nan))
    assert one_seed_down["passed"] and one_seed_down["seeds_passing"] == 4
    completed[2, :2] = False  # a third seed misses two laps: 3 of 5
    failed = stage_gate(completed, np.where(completed, times, np.nan))
    assert not failed["passed"] and failed["seeds_passing"] == 3 and failed["completion"]["rate"] == 0.74
    assert failed["per_seed"]["completed"] == [0, 9, 8, 10, 10] and failed["per_seed"]["mean_lap_time"][0] is None
    assert two_level_bootstrap(np.arange(50.0).reshape(5, 10)) == two_level_bootstrap(np.arange(50.0).reshape(5, 10))


def test_distance_from_the_limit_and_its_criterion() -> None:
    """Distance from the minimum-time lap of each start; a seed is at the limit with 9 of 10 laps and a median distance
    of at most 5 %; the policy with 4 of 5 seeds."""
    reference = np.linspace(129.0, 134.0, 10)
    times = np.tile(reference * 1.04, (5, 1))
    times[0, :3] = np.nan  # 7 of 10 laps
    times[1, :] = reference * 1.06  # too far from the limit
    gaps = lap_gaps(times, reference)
    assert np.nanmax(np.abs(gaps[2:] - 0.04)) < 1e-12
    result = limit_criterion(~np.isnan(times), gaps)
    assert result["seeds_at_limit"] == [False, False, True, True, True] and not result["passed"]
    times[0, :3] = reference[:3] * 1.03
    times[0, 3] = np.nan  # 9 of 10 laps
    result = limit_criterion(~np.isnan(times), lap_gaps(times, reference))
    assert result["seeds_at_limit"] == [True, False, True, True, True] and result["passed"]
    assert result["median_gap"][1] == pytest.approx(0.06) and result["completed"] == [9, 10, 10, 10, 10]
