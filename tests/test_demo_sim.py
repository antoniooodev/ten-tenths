"""The simulator of the web demo (docs/demo/js/sim.js) against the Python plant, environment and policies.

The browser port is run in Node through scripts/demo/check_sim.mjs on the exported data of docs/demo/data; the tests
are skipped when Node is not installed. The track queries and an open-loop drive must agree with Python to rounding;
a closed-loop lap of a policy (its float32 weights run in float64 on both sides, the observation rounded to float32)
must follow the same trajectory and give the same lap times.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
from conftest import REPO

from contextual_bicycle_pacejka.env.environment import RacingEnv
from contextual_bicycle_pacejka.env.stages import make_stage
from contextual_bicycle_pacejka.model.track import RealTrack

DATA = REPO / "docs" / "demo" / "data"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None or not (DATA / "demo.json").exists(),
                                reason="needs Node and the exported demo data")
sys.path.insert(0, str(REPO / "scripts" / "figures"))


@pytest.fixture(scope="module")
def track() -> RealTrack:
    return RealTrack(str(REPO / "data" / "tracks" / "catalunya" / "reference.npz"))


def run_js(tmp_path, **request) -> dict:
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"data": str(DATA), **request}))
    result = subprocess.run([NODE, str(REPO / "scripts" / "demo" / "check_sim.mjs"), str(path)], capture_output=True,
                            text=True, check=True, timeout=600)
    return json.loads(result.stdout)


class NumpyPolicy:
    """The exported actor of a policy, run in float64 like the browser."""

    def __init__(self, tag: str) -> None:
        manifest = json.loads((DATA / "demo.json").read_text())
        blob = (DATA / manifest["files"]["policies"].split("?")[0]).read_bytes()
        spec = manifest["policies"][tag]

        def array(entry):
            values = np.frombuffer(blob, dtype="<f4", count=entry["length"], offset=entry["offset"])
            return values.reshape(entry["shape"]).astype(np.float64)

        self.oracle = spec["oracle"]
        self.layers = [(array(layer["weight"]), array(layer["bias"]), layer["activation"]) for layer in spec["layers"]]

    def mean(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float64)
        for weight, bias, activation in self.layers:
            x = weight @ x + bias
            if activation == "leaky_relu":
                x = np.where(x < 0, 0.2 * x, x)
        return np.clip(x, -1.0, 1.0)


def state_row(env: RacingEnv) -> list[float]:
    st = env.state
    return [st.s, st.ey, st.epsi, st.vx, st.vy, st.yaw_rate, st.delta, env.tau]


def test_track_queries(tmp_path, track: RealTrack) -> None:
    from thesis_plots import frenet_to_world

    s_values = [0.0, 1.234, 821.5, 2134.7, 3499.99, 4649.0, track.track_length + 17.3]
    js = run_js(tmp_path, track=s_values)["track"]
    preview = make_stage(7, track).preview
    for s, row in zip(s_values, js, strict=True):
        assert row["kappa"] == track.kappa(s)
        assert row["w_left"] == track.boundary_left(s)
        assert row["w_right"] == track.boundary_right(s)
        np.testing.assert_allclose(row["mean_curvature"], track.mean_curvature(s, preview), rtol=0, atol=1e-15)
        x, y, heading = frenet_to_world(np.array([s]), np.array([2.0]), np.array([0.1]), real=track)
        np.testing.assert_allclose(row["pose"][:2], [x[0], y[0]], rtol=0, atol=1e-9)
        assert abs(np.angle(np.exp(1j * (row["pose"][2] - heading[0])))) < 1e-12


def test_open_loop_drive(tmp_path, track: RealTrack) -> None:
    rng = np.random.default_rng(3)
    k = np.arange(400)
    throttle = np.clip(0.8 * np.cos(k / 40.0) + 0.1 * rng.normal(size=k.size), -1, 1)
    actions = np.stack([0.25 * np.sin(k / 15.0), throttle], axis=1).astype(np.float32)  # float32, as the policies'
    request = {"b": 0.03, "s0": 400.0, "vx0": 25.0, "actions": actions.astype(np.float64).tolist()}
    js = run_js(tmp_path, open_loop=request)["open_loop"]
    env = RacingEnv(track, make_stage(7, track), b_range=(0.03, 0.03), oracle=True)
    obs, _ = env.reset(options={"initial_s": 400.0, "initial_vx": 25.0})
    np.testing.assert_array_equal(np.float32(js["observations"][0]), obs)
    failure = None
    for i, action in enumerate(actions[:len(js["states"])]):
        obs, _, terminated, _, info = env.step(action)
        np.testing.assert_allclose(js["states"][i], state_row(env), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.float32(js["observations"][i + 1]), obs, rtol=1e-6, atol=1e-7)
        if terminated:
            failure = info["failure"]
            break
    assert js["failure"] == failure


@pytest.mark.parametrize(("tag", "b"), [("oracle_limit_w1", 0.0), ("randomised_limit_w1", -0.11)])
def test_closed_loop_lap(tmp_path, track: RealTrack, tag: str, b: float) -> None:
    js = run_js(tmp_path, closed_loop={"policy": tag, "b": b, "s0": 0.0, "vx0": 3.0, "decisions": 6000})
    js = js["closed_loop"]
    policy = NumpyPolicy(tag)
    env = RacingEnv(track, make_stage(7, track), b_range=(b, b), oracle=policy.oracle)
    obs, _ = env.reset(options={"initial_s": 0.0, "initial_vx": 3.0})
    rows, laps, failure = [], [], None
    finish = [track.track_length, 2.0 * track.track_length]
    for _ in range(len(js["states"])):
        s_before = env.state.s
        obs, _, terminated, _, info = env.step(policy.mean(obs).astype(np.float32))
        rows.append(state_row(env))
        if terminated:
            failure = info["failure"]
            break
        if len(laps) < 2 and env.state.s >= finish[len(laps)]:
            fraction = (finish[len(laps)] - s_before) / (env.state.s - s_before)
            laps.append((len(rows) - 1 + fraction) * 0.05 - sum(laps))
    assert failure is None and js["failure"] is None
    assert len(laps) == len(js["laps"]) == 2
    np.testing.assert_allclose(js["laps"], laps, atol=0.02)
    # one decision of matrix products summed in another order differs by rounding: the first lap must coincide
    first_lap = int(laps[0] / 0.05)
    np.testing.assert_allclose(np.array(js["states"])[:first_lap], np.array(rows)[:first_lap], rtol=0, atol=1e-3)
