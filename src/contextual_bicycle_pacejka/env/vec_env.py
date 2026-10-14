"""Vectorised environment with several environments per subprocess.

``SubprocVecEnv`` of Stable-Baselines3 runs one environment per process and exchanges two messages per decision with
each: for this environment the messages, not the plant, bound the throughput. ``BatchedSubprocVecEnv`` splits the
environments into contiguous chunks, one per worker process; each worker runs its chunk as a ``DummyVecEnv`` and
exchanges one message per chunk and decision. Automatic reset, ``terminal_observation``, ``TimeLimit.truncated``,
seeds and options are those of ``DummyVecEnv``, so the result equals a ``DummyVecEnv`` of all the environments.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnvIndices, VecEnvObs, VecEnvStepReturn


def _worker(remote: Any, parent_remote: Any, env_fns: CloudpickleWrapper) -> None:
    """Serve the commands of ``BatchedSubprocVecEnv`` for one chunk of environments."""
    parent_remote.close()
    venv = DummyVecEnv(env_fns.var)
    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break
        if cmd == "step":
            obs, rewards, dones, infos = venv.step(data)
            remote.send((obs, rewards, dones, infos, venv.reset_infos))
        elif cmd == "reset":
            venv._seeds, venv._options = data  # consumed and cleared by DummyVecEnv.reset
            obs = venv.reset()
            remote.send((obs, venv.reset_infos))
        elif cmd == "get_spaces":
            remote.send((venv.observation_space, venv.action_space))
        elif cmd == "get_attr":
            remote.send(venv.get_attr(data[0], data[1]))
        elif cmd == "set_attr":
            remote.send(venv.set_attr(data[0], data[1], data[2]))
        elif cmd == "env_method":
            remote.send(venv.env_method(data[0], *data[1], indices=data[3], **data[2]))
        elif cmd == "is_wrapped":
            remote.send(venv.env_is_wrapped(data[0], data[1]))
        elif cmd == "close":
            venv.close()
            remote.close()
            break
        else:
            raise NotImplementedError(f"`{cmd}` is not implemented in the worker")


class BatchedSubprocVecEnv(VecEnv):
    """``n_workers`` subprocesses, each running a contiguous chunk of ``env_fns`` as a ``DummyVecEnv``.

    Parameters
    ----------
    env_fns : list of callables
        Constructors of the environments; they must be picklable by cloudpickle.
    n_workers : int
        Number of subprocesses (at most ``len(env_fns)``); the chunks differ in size by at most one.
    start_method : str, optional
        ``multiprocessing`` start method; ``forkserver`` where available, ``spawn`` otherwise (as ``SubprocVecEnv``).
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]], n_workers: int, start_method: str | None = None) -> None:
        if not 1 <= n_workers <= len(env_fns):
            raise ValueError(f"n_workers={n_workers} outside 1..{len(env_fns)}")
        if start_method is None:
            start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        bounds = np.linspace(0, len(env_fns), n_workers + 1).round().astype(int)
        self.chunks = [range(int(lo), int(hi)) for lo, hi in itertools.pairwise(bounds)]
        self.remotes, self.processes = [], []
        for chunk in self.chunks:
            remote, work_remote = ctx.Pipe()
            fns = CloudpickleWrapper([env_fns[i] for i in chunk])
            process = ctx.Process(target=_worker, args=(work_remote, remote, fns), daemon=True)
            process.start()
            work_remote.close()
            self.remotes.append(remote)
            self.processes.append(process)
        self.waiting = False
        self.closed = False
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(len(env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, chunk in zip(self.remotes, self.chunks, strict=True):
            remote.send(("step", actions[chunk.start:chunk.stop]))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        self.reset_infos = [info for result in results for info in result[4]]
        return (np.concatenate([r[0] for r in results]), np.concatenate([r[1] for r in results]),
                np.concatenate([r[2] for r in results]), [info for r in results for info in r[3]])

    def reset(self) -> VecEnvObs:
        for remote, chunk in zip(self.remotes, self.chunks, strict=True):
            remote.send(("reset", ([self._seeds[i] for i in chunk], [self._options[i] for i in chunk])))
        results = [remote.recv() for remote in self.remotes]
        self.reset_infos = [info for result in results for info in result[1]]
        self._reset_seeds()
        self._reset_options()
        return np.concatenate([result[0] for result in results])

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        return self._call("get_attr", indices, lambda local: (attr_name, local))

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        self._call("set_attr", indices, lambda local: (attr_name, value, local))

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        return self._call("env_method", indices, lambda local: (method_name, method_args, method_kwargs, local))

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        return self._call("is_wrapped", indices, lambda local: (wrapper_class, local))

    def _call(self, cmd: str, indices: VecEnvIndices, payload: Callable[[list[int]], Any]) -> list[Any]:
        """Send ``cmd`` to the workers that hold ``indices``, with their local indices; return the results in order."""
        wanted = list(self._get_indices(indices))
        calls: list[tuple[Any, Sequence[int]]] = []
        for remote, chunk in zip(self.remotes, self.chunks, strict=True):
            local = [i - chunk.start for i in wanted if i in chunk]
            if local:
                remote.send((cmd, payload(local)))
                calls.append((remote, [chunk.start + j for j in local]))
        by_index: dict[int, Any] = {}
        for remote, global_indices in calls:
            result = remote.recv()
            for i, value in zip(global_indices, result if result is not None else [None] * len(global_indices),
                                strict=True):
                by_index[i] = value
        return [by_index[i] for i in wanted]
