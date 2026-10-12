"""Environment, plant, observation and reward: exact agreement with the recorded reference rollouts."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import REPO, golden_json, golden_npz, repo_path

from contextual_bicycle_pacejka.control.safety_filter import safe_action_set
from contextual_bicycle_pacejka.env import make_env, randomized_setup
from contextual_bicycle_pacejka.model.dynamics import DIAGNOSTIC_KEYS
from contextual_bicycle_pacejka.model.observation import vehicle_world_position
from contextual_bicycle_pacejka.model.state import SetupVector, VehicleMemory, VehicleState
from contextual_bicycle_pacejka.model.tires import compute_saturation_threshold
from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.layout import OBS

CASES = golden_json("env_cases")


@pytest.mark.parametrize("name", sorted(CASES))
def test_rollout_matches_reference(name: str) -> None:
    """Replaying the recorded teacher actions reproduces observations, rewards, states and diagnostics bit-exactly."""
    case = CASES[name]
    ref = golden_npz(name)
    env = make_env(repo_path(case["raceline"]))
    setup = randomized_setup(case["mu"], np.random.default_rng(case["seed"]))
    np.testing.assert_array_equal(setup, ref["setup"])
    obs, info = env.reset(seed=case["seed"], options={"setup": setup, "initial_s": case["start_s"],
                                                      "initial_vx": max(1.0, case["start_vx"])})
    assert (info["alpha_sat_f"], info["alpha_sat_r"]) == tuple(ref["alpha_sat"])
    np.testing.assert_array_equal(obs, ref["obs"][0])
    for t, action in enumerate(ref["action"]):
        obs, reward, terminated, _, info = env.step(action)
        np.testing.assert_array_equal(obs, ref["obs"][t + 1], err_msg=f"observation at step {t}")
        assert reward == ref["reward"][t], f"reward at step {t}"
        assert terminated == ref["terminated"][t]
        np.testing.assert_array_equal(info["state"], ref["state"][t])
        np.testing.assert_array_equal([info["diag"][k] for k in DIAGNOSTIC_KEYS], ref["diag"][t])
        events = [info["event_flags"][k] for k in ("track_excursion", "slip_saturation", "low_speed")]
        assert events == list(ref["events"][t])


def test_observation_layout() -> None:
    """Index layout of the 116-entry observation."""
    assert (OBS.VREF_START, OBS.VREF_END, OBS.TRAJ_START, OBS.TRAJ_END, OBS.AX) == (6, 18, 18, 78, 78)
    assert (OBS.XI_START, OBS.XI_END, OBS.RHO_LAT, OBS.RHO_LON) == (79, 86, 86, 87)
    assert (OBS.KAPPA_START, OBS.KAPPA_END, OBS.AX_PREVIEW_START, OBS.AX_PREVIEW_END) == (88, 100, 100, 112)
    assert (OBS.ALPHA_F, OBS.ALPHA_R, OBS.FY_F, OBS.FY_R) == (112, 113, 114, 115)


def test_track_boundaries_follow_the_sign_of_ey() -> None:
    """ey > 0 (left) is bounded by the left edge, ey < 0 by the right one, in the rule and in the safety filter.

    Monza R0 at s = 3268 m: the racing line runs 2.7 m left of the centreline, 1.0 m from the left edge and 6.7 m from
    the right one. The world point at ey = +w_left lies on the left edge of the centreline reference.
    """
    track = RealTrack(str(REPO / "data/tracks/monza/racing_line_R0.npz"))
    s = 3268.0
    left, right = track.boundary_left(s), track.boundary_right(s)
    assert left < 1.5 and right > 6.0
    assert not track.is_out_of_bounds(0.99 * left, s) and track.is_out_of_bounds(1.01 * left, s)
    assert not track.is_out_of_bounds(-0.99 * right, s) and track.is_out_of_bounds(-1.01 * right, s)

    with np.load(REPO / "data/tracks/monza/reference.npz") as reference:
        x, y = vehicle_world_position(VehicleState(s=s, ey=left), track)
        i = int(np.argmin(np.hypot(reference["x_m"] - x, reference["y_m"] - y)))
        psi = reference["psi_rad"][i]
        offset = (x - reference["x_m"][i]) * -np.sin(psi) + (y - reference["y_m"][i]) * np.cos(psi)
        assert offset == pytest.approx(reference["w_tr_left_m"][i], abs=0.1)

    setup = SetupVector.from_array(randomized_setup(1.0, np.random.default_rng(0)))
    alpha_sat = (compute_saturation_threshold(setup, "front"), compute_saturation_threshold(setup, "rear"))
    for ey, room in ((-2.0, True), (2.0, False)):
        state = VehicleState(s=s, ey=ey, vx=40.0)
        safe = safe_action_set(state, VehicleMemory.zeros(), setup, track, alpha_sat, prev_delta_cmd=0.0)
        assert safe.empty is not room, f"ey = {ey}"
