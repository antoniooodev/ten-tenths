"""Model-free reinforcement learning for a simulated race car on the Catalunya circuit.

Sub-packages: ``spec`` (constants and vector layouts), ``model`` (vehicle plant, track, observation, reward,
failures), ``env`` (Gymnasium environment, speed stages, vectorised environments), ``tracks`` (centreline reference of
a circuit), ``learning`` (PPO and the speed curriculum), ``reference`` (minimum-time lap) and ``evaluation`` (protocol
laps, statistics of the results, steady-state check of the balance index).

Importing the package registers the Gymnasium ID ``racing/ContextualBicyclePacejka-v0`` (constructor arguments:
``track``, ``stage``, and optionally ``b_range`` and ``oracle``; see ``env.environment.RacingEnv``).
"""

import gymnasium as gym

ENV_ID = "racing/ContextualBicyclePacejka-v0"

if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="contextual_bicycle_pacejka.env.environment:RacingEnv")
