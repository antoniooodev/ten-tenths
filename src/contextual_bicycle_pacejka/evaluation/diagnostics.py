"""Per-episode tracking and failure diagnostics.

At every step the MATv0 teacher is evaluated on the same observation as the controller (it is not applied);
its speed target v_target and its action are the reference of the diagnostics:

    speed_error_mean_abs_mps   mean |vx - v_target|
    speed_error_mean_mps       mean (vx - v_target)
    frac_overspeed             fraction of steps with vx > 1.03 v_target
    ey_rms_m                   root-mean-square lateral error
    rear_slip_ratio_p99        99th percentile of |alpha_r| / alpha_sat_r
    teacher_tau_mae, teacher_delta_mae   mean |executed - teacher| of the two action components

For an episode that does not complete, also:

    terminal_zone        corner zone containing the final position (or "straight/other")
    terminal_state       final state, executed action, v_target, curvature and slip ratios
    pre_failure_last5s   for overspeed, front slip >= alpha_sat and rear slip >= alpha_sat: first occurrence in
                         the last 5 s (position, seconds before the end, number of steps), or None
"""

from __future__ import annotations

from typing import Any

import numpy as np

from contextual_bicycle_pacejka.model.track import RealTrack
from contextual_bicycle_pacejka.spec.constants import DT

OVERSPEED_RATIO = 1.03  # [-] vx above this multiple of the teacher target counts as overspeed
PRE_FAILURE_STEPS = 500  # [steps] last 5 s
COLUMNS = ("s", "ey", "epsi", "vx", "vy", "yaw_rate", "delta", "delta_exec", "tau_exec", "v_target", "kappa",
           "alpha_f", "alpha_r", "teacher_delta", "teacher_tau")
_C = {name: i for i, name in enumerate(COLUMNS)}


class EpisodeRecorder:
    """Collects one row per step and summarises it at the end of the episode."""

    def __init__(self, track_length: float, alpha_sat: tuple[float, float], mu: float,
                 corner_zones: list[dict] | None) -> None:
        self.length = float(track_length)
        self.alpha_sat = alpha_sat
        self.mu = mu
        self.corner_zones = corner_zones or []
        self.rows: list[tuple[float, ...]] = []

    def record(self, state: np.ndarray, action: np.ndarray, teacher_action: np.ndarray, v_target: float,
               kappa: float, diag: dict[str, Any], track: RealTrack) -> None:
        """Add the step that ended in ``state`` (``v_target`` and ``teacher_action`` from the pre-step observation)."""
        del track
        self.rows.append((float(state[0]) % self.length, *(float(x) for x in state[1:7]), float(action[0]),
                          float(action[1]), float(v_target), kappa, float(diag["alpha_f"]), float(diag["alpha_r"]),
                          float(teacher_action[0]), float(teacher_action[1])))

    def summary(self, completed: bool) -> dict[str, Any]:
        """Diagnostics of the episode (see the module docstring)."""
        rows = np.asarray(self.rows, dtype=np.float64).reshape(-1, len(COLUMNS))
        vx, vt = rows[:, _C["vx"]], rows[:, _C["v_target"]]
        ar = np.abs(rows[:, _C["alpha_r"]]) / self.alpha_sat[1]
        af = np.abs(rows[:, _C["alpha_f"]]) / self.alpha_sat[0]
        out: dict[str, Any] = {
            "speed_error_mean_abs_mps": float(np.mean(np.abs(vx - vt))) if rows.size else None,
            "speed_error_mean_mps": float((vx - vt).mean()) if rows.size else None,
            "frac_overspeed": float((vx > OVERSPEED_RATIO * vt).mean()) if rows.size else None,
            "ey_rms_m": float(np.sqrt(np.mean(rows[:, _C["ey"]] ** 2))) if rows.size else None,
            "rear_slip_ratio_p99": float(np.percentile(ar, 99)) if ar.size else None,
            "teacher_tau_mae": float(np.mean(np.abs(rows[:, _C["tau_exec"]] - rows[:, _C["teacher_tau"]]))),
            "teacher_delta_mae": float(np.mean(np.abs(rows[:, _C["delta_exec"]] - rows[:, _C["teacher_delta"]]))),
        }
        if completed or not rows.size:
            return out
        s = rows[:, _C["s"]]
        terminal_s = float(s[-1])
        out["terminal_zone"] = next((z["zone"] for z in self.corner_zones if z["s_m"][0] <= terminal_s <= z["s_m"][1]),
                                    "straight/other")
        tail = slice(max(0, len(rows) - PRE_FAILURE_STEPS), len(rows))
        onset: dict[str, Any] = {}
        for name, flag in (("rear_slip_saturated", ar >= 1.0), ("front_slip_saturated", af >= 1.0),
                           ("overspeed", vx > OVERSPEED_RATIO * vt)):
            idx = np.nonzero(flag[tail])[0]
            onset[name] = None if idx.size == 0 else {
                "first_s_m": float(s[tail][idx[0]]),
                "seconds_before_end": float((len(rows) - (tail.start + idx[0])) * DT),
                "steps_in_last_5s": int(idx.size)}
        out["pre_failure_last5s"] = onset
        last = rows[-1]
        out["terminal_state"] = {
            "s": float(last[_C["s"]]), "ey": float(last[_C["ey"]]), "epsi": float(last[_C["epsi"]]),
            "vx": float(last[_C["vx"]]), "vy": float(last[_C["vy"]]), "yaw_rate": float(last[_C["yaw_rate"]]),
            "delta": float(last[_C["delta"]]), "tau": float(last[_C["tau_exec"]]),
            "delta_cmd": float(last[_C["delta_exec"]]), "v_target": float(last[_C["v_target"]]),
            "kappa": float(last[_C["kappa"]]), "alpha_f": float(last[_C["alpha_f"]]),
            "alpha_r": float(last[_C["alpha_r"]]), "rear_slip_ratio": float(ar[-1]), "front_slip_ratio": float(af[-1]),
        }
        return out
