"""Learning-based lap completion for a simulated race car.

Sub-packages: ``spec`` (constants and vector layouts), ``model`` (vehicle plant, track and student observation),
``env`` (Gymnasium environment and student wrapper), ``tracks`` (racing-line preparation), ``control`` (teacher and
safety filter), ``learning`` (demonstrations, datasets, behaviour cloning, friction estimator, offline RL, DAgger,
PPO fine-tuning) and ``evaluation``.

Importing the package registers the Gymnasium ID ``racing/ContextualBicyclePacejka-v0``
(constructor argument: ``track``).
"""

import gymnasium as gym

ENV_ID = "racing/ContextualBicyclePacejka-v0"

if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="contextual_bicycle_pacejka.env.environment:ContextualBicyclePacejkaEnv")
