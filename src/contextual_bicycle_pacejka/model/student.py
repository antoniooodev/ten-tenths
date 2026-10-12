"""Student observation of levels S0, S1 and S2 (layouts in ``spec/layout.py``).

The student observation of one episode is built from three channels logged at every step t:

    o_t        privileged observation (float32, as returned by the environment)
    imu_t      accelerometer specific force [ax_imu, ay_imu] (``model/sensors.py``) of the step that produced o_t
    a_{t-1}    action executed at the previous step (zeros after a reset)

and from the episode friction mu, used only to undo the friction scaling of the v_ref preview: the student reads
the preview at the reference friction, so mu does not reach it. Every channel is rounded to float32 first, so a
student observation built offline from a dataset equals the one built online by ``env/student.py`` bit for bit.

    frame_t     dynamic channels of the level at step t
    stack_t     [frame_t, frame_{t - stride}, ..., frame_{t - (n_frames - 1) stride}]; a frame before the start of
                the episode is replaced by the first frame
    preview_t   [v_ref / sqrt(mu / 0.9), racing-line points, curvature, reference acceleration]
    student_t   [stack_t, preview_t]                (+ oracle_t, the oracle entries of a diagnostic layout)

For S0 the student observation is o_t itself.

What the levels withhold, besides the setup xi: the friction scaling of the v_ref preview, the force use and the
normalised lateral forces (divided by mu and m), and ax = dvx/dt (which contains vy; the accelerometer channels
replace it). S2 also withholds vy and the slip angles.

The coach observation (level C, ``coach_observation``) is the privileged observation without the plan: the entries
``COACH_INDEX`` of o_t, without the v_ref and reference-acceleration previews.

A plan scale c multiplies the v_ref preview the student reads (``student_previews``): the speed plan of a plan coach
(``learning/plan_coach.py``) or a constant scale of the teacher's plan. The datasets never use it.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from contextual_bicycle_pacejka.spec.constants import IMU_DIM, MU_HI, MU_LO, OBS_DIM, VREF_REFERENCE_MU
from contextual_bicycle_pacejka.spec.layout import COACH_INDEX, OBS, STUDENT_PREVIEW_DIM, StudentLayout

ACTION_DIM: int = 2
ORACLE_RANGES: dict[str, tuple[float, float]] = {"mu": (MU_LO, MU_HI)}  # range an estimate can take


def previous_actions(executed: NDArray[np.floating]) -> NDArray[np.float32]:
    """Action executed at the previous step of every step of one episode (zeros at the first step)."""
    executed = np.asarray(executed, dtype=np.float32).reshape(-1, ACTION_DIM)
    previous = np.zeros_like(executed)
    previous[1:] = executed[:-1]
    return previous


def student_frames(
    obs: NDArray[np.floating], imu: NDArray[np.floating], prev_actions: NDArray[np.floating], layout: StudentLayout,
) -> NDArray[np.float32]:
    """Dynamic channels of ``layout`` for every row, shape (T, frame_dim)."""
    return channel_frames(obs, imu, prev_actions, layout.channels)


def channel_frames(
    obs: NDArray[np.floating], imu: NDArray[np.floating], prev_actions: NDArray[np.floating],
    channels: tuple[str, ...],
) -> NDArray[np.float32]:
    """The named dynamic channels (``STUDENT_CHANNELS["S1"]``) for every row, shape (T, len(channels))."""
    obs, imu, prev_actions = _rows(obs, imu, prev_actions)
    columns = {
        "vx": obs[:, OBS.VX], "yaw_rate": obs[:, OBS.YAW_RATE], "delta": obs[:, OBS.DELTA],
        "ax_imu": imu[:, 0], "ay_imu": imu[:, 1], "ey": obs[:, OBS.EY], "epsi": obs[:, OBS.EPSI],
        "prev_delta_cmd": prev_actions[:, 0], "prev_tau": prev_actions[:, 1],
        "vy": obs[:, OBS.VY], "alpha_f": obs[:, OBS.ALPHA_F], "alpha_r": obs[:, OBS.ALPHA_R],
    }
    return np.column_stack([columns[name] for name in channels]).astype(np.float32)


def student_previews(obs: NDArray[np.floating], mu: float, plan_scale: float = 1.0) -> NDArray[np.float32]:
    """Previews of every row, shape (T, STUDENT_PREVIEW_DIM), with v_ref at the reference friction, multiplied by
    ``plan_scale``."""
    obs = np.asarray(obs, dtype=np.float32).reshape(-1, OBS_DIM)
    vref = obs[:, OBS.VREF_START:OBS.VREF_END].astype(np.float64) / math.sqrt(float(mu) / VREF_REFERENCE_MU)
    vref = vref * float(plan_scale)
    previews = np.concatenate([
        vref.astype(np.float32),
        obs[:, OBS.TRAJ_START:OBS.TRAJ_END],
        obs[:, OBS.KAPPA_START:OBS.KAPPA_END],
        obs[:, OBS.AX_PREVIEW_START:OBS.AX_PREVIEW_END],
    ], axis=1)
    if previews.shape[1] != STUDENT_PREVIEW_DIM:
        raise RuntimeError(f"preview width {previews.shape[1]} != {STUDENT_PREVIEW_DIM}")
    return previews


def student_oracle(n_rows: int, mu: float, layout: StudentLayout, offset: dict[str, float] | None = None,
                   clip: bool = False) -> NDArray[np.float32]:
    """Oracle entries of ``layout`` for ``n_rows`` steps of one episode, shape (n_rows, len(layout.oracle)).

    ``offset`` adds a constant error to named entries (evaluation of the sensitivity to an estimated value); the
    datasets never use it. With ``clip`` every entry is clipped to the range an estimate of it can take
    (``ORACLE_RANGES``): a friction estimate never leaves [MU_LO, MU_HI], so the clipped offset tests only errors
    that an estimator can make.
    """
    offset = dict(offset or {})
    unknown = set(offset) - set(layout.oracle)
    if unknown:
        raise ValueError(f"offset for {sorted(unknown)} but the layout's oracle is {layout.oracle}")
    values = {"mu": float(mu)}
    entries = [values[name] + float(offset.get(name, 0.0)) for name in layout.oracle]
    if clip:
        entries = [min(max(value, ORACLE_RANGES[name][0]), ORACLE_RANGES[name][1])
                   for name, value in zip(layout.oracle, entries, strict=True)]
    row = np.array(entries, dtype=np.float32)
    return np.broadcast_to(row, (n_rows, len(layout.oracle))).copy()


def stack_frames(frames: NDArray[np.float32], layout: StudentLayout) -> NDArray[np.float32]:
    """Frame stacks of every step of one episode, shape (T, dynamic_dim); the current frame comes first."""
    frames = np.asarray(frames, dtype=np.float32).reshape(-1, layout.frame_dim)
    return stack_window(frames, layout.n_frames, layout.stride)


def stack_window(frames: NDArray[np.float32], n_frames: int, stride: int) -> NDArray[np.float32]:
    """Stacks of ``n_frames`` frames ``stride`` steps apart for every row of (T, width) frames, shape
    (T, n_frames * width); the current frame comes first and frames before the start repeat the first one."""
    frames = np.asarray(frames, dtype=np.float32)
    steps = np.arange(len(frames))
    lags = np.arange(n_frames) * stride
    index = np.maximum(steps[:, None] - lags[None, :], 0)
    return frames[index].reshape(len(frames), n_frames * frames.shape[1])


def coach_observation(obs: NDArray[np.floating]) -> NDArray[np.float32]:
    """Coach observation of one or several privileged observations: the entries without the plan (``COACH_INDEX``)."""
    return np.asarray(obs, dtype=np.float32)[..., COACH_INDEX]


def student_observations(
    obs: NDArray[np.floating], imu: NDArray[np.floating], prev_actions: NDArray[np.floating], mu: float,
    layout: StudentLayout,
) -> NDArray[np.float32]:
    """Student observations of every step of one episode, shape (T, layout.dim)."""
    if layout.privileged:
        return np.asarray(obs, dtype=np.float32).reshape(-1, OBS_DIM).copy()
    if layout.coach:
        return coach_observation(np.asarray(obs).reshape(-1, OBS_DIM))
    stacks = stack_frames(student_frames(obs, imu, prev_actions, layout), layout)
    return np.concatenate([stacks, student_previews(obs, mu), student_oracle(len(stacks), mu, layout)], axis=1)


def _rows(
    obs: NDArray[np.floating], imu: NDArray[np.floating], prev_actions: NDArray[np.floating],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """The three channels as float32 rows of equal length."""
    obs = np.asarray(obs, dtype=np.float32).reshape(-1, OBS_DIM)
    imu = np.asarray(imu, dtype=np.float32).reshape(-1, IMU_DIM)
    prev_actions = np.asarray(prev_actions, dtype=np.float32).reshape(-1, ACTION_DIM)
    if not len(obs) == len(imu) == len(prev_actions):
        raise ValueError(f"channel lengths differ: obs {len(obs)}, imu {len(imu)}, actions {len(prev_actions)}")
    return obs, imu, prev_actions
