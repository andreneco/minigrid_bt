# minigrid_bt/env_initialization.py

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from minigrid_bt.utils import ExtendedFlatObsWrapper

def initialize_env(env_id, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)
    env = FullyObsWrapper(env)
    image_shape = env.observation_space['image'].shape if isinstance(env.observation_space, gym.spaces.Dict) else env.observation_space.shape
    env = ExtendedFlatObsWrapper(env)
    return env, image_shape
