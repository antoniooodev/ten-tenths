"""Export what the web demo (docs/index.html) needs: the track, three policies and the recorded laps.

    python scripts/demo/export_demo_data.py
    python scripts/demo/export_demo_data.py --output docs/demo/data

Writes <output>/demo.json, a manifest, and three binary files of little-endian arrays that it indexes:
- track.bin:    the periodic cubic splines of the circuit (centreline x and y, curvature and its integral, the two edge
                distances) as breakpoints and polynomial coefficients in float64, so that the browser evaluates the
                same piecewise polynomials as ``RealTrack``; the drawn edges every metre in float32;
- policies.bin: the actor of each policy (two hidden layers of 256 with LeakyReLU and the linear mean of the
                action) and its log standard deviation, in float32;
- laps.bin:     the recorded runs of scripts/figures/record_laps.py that the replay shows, one sample per decision,
                in float32 (the times in float64).

The manifest also holds the constants of the plant, of the free stage (preview distances, observation scales) and the
peak slip angles, and the metadata of every run (policy, seed, b, lap times, failure). The binary files are written
only when their content changes, so that an unchanged export leaves the repository clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from thesis_plots import CATALUNYA, LABELS, load, load_track  # noqa: E402

from contextual_bicycle_pacejka.env.setup_sampling import sample_setup  # noqa: E402
from contextual_bicycle_pacejka.env.stages import make_stage  # noqa: E402
from contextual_bicycle_pacejka.model.tires import compute_saturation_threshold, pacejka_coefficients  # noqa: E402
from contextual_bicycle_pacejka.model.track import RealTrack  # noqa: E402
from contextual_bicycle_pacejka.spec import constants as C  # noqa: E402

POLICIES = ("nominal_limit_cont", "randomised_limit_w1", "oracle_limit_w1")
SEED = 12
BALANCE_VALUES = (-0.21, -0.11, 0.0, 0.03, 0.05)
FLYING_TAGS = ("nominal_limit", "randomised_limit_w1", "oracle_limit_w1")
CHANNELS = ("s", "ey", "epsi", "vx", "vy", "r", "delta", "tau", "ax", "ay", "slip_f", "slip_r")
CONSTANTS = ("DT", "LF", "LR", "IZ", "G", "A_MAX", "D_MAX", "BETA_BRAKE", "AIR_DENSITY", "DRAG_AREA_M2",
             "WHEEL_POWER_W", "POWER_SPEED_FLOOR_MPS", "DELTA_MAX", "EPS_D", "V_MIN", "MASS_NOM", "MU_NOM",
             "HCG_NOM", "TAU_D_NOM", "B_MIN", "B_MAX", "DECISION_STEPS", "DECISION_DT", "TAU_RATE_MAX",
             "SPIN_EPSI_RAD", "STALL_SPEED_MPS", "B_OBS_SCALE", "N_PREVIEW", "OBS_DIM")


class Packer:
    """Concatenates arrays into one binary blob and records where each one is."""

    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.offset = 0

    def add(self, array: np.ndarray, dtype: str) -> dict:
        data = np.ascontiguousarray(array, dtype=np.dtype(dtype).newbyteorder("<"))
        pad = (-self.offset) % 8  # every array starts 8-byte aligned, so a typed array can view it directly
        if pad:
            self.parts.append(b"\0" * pad)
            self.offset += pad
        entry = {"offset": self.offset, "length": int(data.size), "dtype": dtype, "shape": list(data.shape)}
        self.parts.append(data.tobytes())
        self.offset += data.nbytes
        return entry

    def write(self, path: Path) -> str:
        blob = b"".join(self.parts)
        if not path.exists() or path.read_bytes() != blob:
            path.write_bytes(blob)
        return hashlib.sha256(blob).hexdigest()[:12]


def spline(packer: Packer, poly) -> dict:
    """Breakpoints and coefficients (highest power first, one column per interval) of a scipy piecewise polynomial."""
    return {"breaks": packer.add(poly.x, "float64"), "coefs": packer.add(poly.c, "float64")}


def export_track(real: RealTrack, packer: Packer) -> dict:
    drawn = load_track(1.0)
    stage = make_stage(7, real)
    setup = sample_setup(np.random.default_rng(0), (0.0, 0.0))
    return {
        "length": real.track_length,
        "turning_per_lap": real._turning_per_lap,
        "splines": {"x": spline(packer, real._x), "y": spline(packer, real._y), "kappa": spline(packer, real._kappa),
                    "kappa_integral": spline(packer, real._kappa_integral), "w_left": spline(packer, real._w_left),
                    "w_right": spline(packer, real._w_right)},
        "drawn": {"step": 1.0, "left": packer.add(drawn.left, "float32"), "right": packer.add(drawn.right, "float32"),
                  "centre": packer.add(drawn.centre, "float32")},
        "stage": {"preview": stage.preview.tolist(), "scales": stage.scales._asdict(),
                  "sizing_speed": stage.sizing_speed},
        "alpha_sat": {"front": compute_saturation_threshold(setup, "front"),
                      "rear": compute_saturation_threshold(setup, "rear")},
        "pacejka": {axle: dict(zip("BCE", pacejka_coefficients(axle), strict=True)) for axle in ("front", "rear")},
    }


def export_policy(path: Path, packer: Packer) -> dict:
    model = PPO.load(str(path), device="cpu")
    state = {k: v.detach().to(torch.float32).numpy() for k, v in model.policy.state_dict().items()}
    layers = [("mlp_extractor.policy_net.0", "leaky_relu"), ("mlp_extractor.policy_net.2", "leaky_relu"),
              ("action_net", "identity")]
    return {
        "inputs": int(model.observation_space.shape[0]),
        "oracle": int(model.observation_space.shape[0]) == C.OBS_DIM + 1,
        "negative_slope": 0.2,
        "layers": [{"weight": packer.add(state[f"{name}.weight"], "float32"),
                    "bias": packer.add(state[f"{name}.bias"], "float32"), "activation": act} for name, act in layers],
        "log_std": packer.add(state["log_std"], "float32"),
    }


def export_run(name: str, packer: Packer, kind: str, **meta) -> dict:
    run = load(name)
    entry = {"name": name, "kind": kind, "samples": int(run["t"].size), "t": packer.add(run["t"], "float64"),
             "channels": {key: packer.add(run[key], "float32") for key in CHANNELS if key in run}, **meta}
    if "lap_times" in run:
        entry["lap_times"] = [float(v) for v in np.atleast_1d(run["lap_times"])]
    if "failure" in run:
        failure = str(run["failure"])
        entry["failure"] = None if failure == "None" else failure
    if "lap_time" in run:
        entry["lap_time"] = float(run["lap_time"])
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, default=Path("outputs/models"))
    parser.add_argument("--output", type=Path, default=Path("docs/demo/data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    real = RealTrack(CATALUNYA)

    track_pack, policy_pack, lap_pack = Packer(), Packer(), Packer()
    track = export_track(real, track_pack)
    policies = {tag: {"label": LABELS[tag], "seed": SEED,
                      **export_policy(args.models / tag / f"s{SEED}" / "stage7.zip", policy_pack)}
                for tag in POLICIES}
    runs = []
    for b in BALANCE_VALUES:
        runs.append(export_run(f"reference_b{b:+.2f}", lap_pack, "reference", tag="minimum_time", b=b))
    for tag in FLYING_TAGS:
        runs.append(export_run(f"flying_{tag}_s{SEED}", lap_pack, "flying", tag=tag, seed=SEED, b=0.0))
    for tag in POLICIES:
        for b in BALANCE_VALUES:
            runs.append(export_run(f"balance_{tag}_s{SEED}_b{b:+.2f}", lap_pack, "balance", tag=tag, seed=SEED, b=b))
    runs.append(export_run("stochastic_nominal_limit_cont_s14_mean", lap_pack, "stochastic", tag="nominal_limit_cont",
                           seed=14, b=0.0, deterministic=True))
    for k in range(5):
        runs.append(export_run(f"stochastic_nominal_limit_cont_s14_noise{k}", lap_pack, "stochastic",
                               tag="nominal_limit_cont", seed=14, b=0.0, deterministic=False))

    manifest = {
        "constants": {name: getattr(C, name) for name in CONSTANTS},
        "labels": LABELS,
        "track": track,
        "policies": policies,
        "runs": runs,
        "files": {"track": f"track.bin?v={track_pack.write(args.output / 'track.bin')}",
                  "policies": f"policies.bin?v={policy_pack.write(args.output / 'policies.bin')}",
                  "laps": f"laps.bin?v={lap_pack.write(args.output / 'laps.bin')}"},
    }
    (args.output / "demo.json").write_text(json.dumps(manifest, separators=(",", ":"), allow_nan=False))
    sizes = {p.name: f"{p.stat().st_size / 1e6:.2f} MB" for p in sorted(args.output.iterdir())}
    print(f"wrote {args.output}: {sizes}")


if __name__ == "__main__":
    main()
