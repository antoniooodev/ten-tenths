"""Vectorised environment with several environments per subprocess: the same results as one ``DummyVecEnv``."""

from __future__ import annotations

import numpy as np
from conftest import REPO

from contextual_bicycle_pacejka.env import make_vec_env

CATALUNYA = REPO / "data/tracks/catalunya/reference.npz"


def test_batched_workers_equal_one_dummy_vec_env() -> None:
    """7 environments in 3 workers (chunks of 2, 3 and 2) against the same 7 in this process, with random actions
    over failures and automatic resets."""
    batched = make_vec_env(CATALUNYA, 7, seed=3, stage=4, b_range=(-0.2, 0.17), oracle=True, edge_margin=0.3, workers=3)
    local = make_vec_env(CATALUNYA, 7, seed=3, stage=4, b_range=(-0.2, 0.17), oracle=True, edge_margin=0.3, workers=0)
    try:
        assert [len(chunk) for chunk in batched.chunks] == [2, 3, 2]
        np.testing.assert_array_equal(batched.reset(), local.reset())
        rng = np.random.default_rng(0)
        resets = 0
        for _ in range(150):
            actions = rng.uniform(-1.0, 1.0, size=(7, 2)).astype(np.float32)
            obs_b, rew_b, done_b, info_b = batched.step(actions)
            obs_l, rew_l, done_l, info_l = local.step(actions)
            np.testing.assert_array_equal(obs_b, obs_l)
            np.testing.assert_array_equal(rew_b, rew_l)
            np.testing.assert_array_equal(done_b, done_l)
            for b, loc in zip(info_b, info_l, strict=True):
                assert b["failure"] == loc["failure"] and b["TimeLimit.truncated"] == loc["TimeLimit.truncated"]
                if "terminal_observation" in loc:
                    np.testing.assert_array_equal(b["terminal_observation"], loc["terminal_observation"])
            resets += int(done_l.sum())
        assert resets > 7
        assert batched.get_attr("oracle", [1, 4]) == [True, True] and batched.get_attr("edge_margin", [6]) == [0.3]
        np.testing.assert_array_equal(np.stack(batched.get_attr("b_range")), np.stack(local.get_attr("b_range")))
        assert batched.env_method("get_wrapper_attr", "oracle", indices=[6]) == [True]
    finally:
        batched.close()
        local.close()
