# minigrid_bt/policy.py

from stable_baselines3.common.policies import BasePolicy
from minigrid_bt.utils import update_tree_obs, ReconstructObsWrapper, current_action

class BehaviorTreePolicy(BasePolicy):
    def __init__(self, observation_space, action_space, env, image_shape, tree_creation_func, reconstruct_obs_wrapper_class, features_extractor=None, *args, **kwargs):
        super(BehaviorTreePolicy, self).__init__(observation_space, action_space, features_extractor)
        self.env = env
        self.image_shape = image_shape
        self.tree_creation_func = tree_creation_func
        self.reconstruct_obs_wrapper_class = reconstruct_obs_wrapper_class
        self.obs = None
        self.tree = None
        self._initialize_env_and_tree()

    def _initialize_env_and_tree(self):
        self.obs = self.env.reset()
        self.reconstruct_obs_wrapper = self.reconstruct_obs_wrapper_class(self.image_shape)
        self.obs = self.reconstruct_obs_wrapper.reconstruct_observation(self.obs[0])
        self.tree = self.tree_creation_func(self.env, self.obs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['tree'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._initialize_env_and_tree()

    def _predict(self, observation, deterministic=False):
        if isinstance(observation, tuple):
            observation = observation[0]
        if observation.ndim > 1:
            observation = np.squeeze(observation)
        update_tree_obs(self.tree, self.reconstruct_obs_wrapper.reconstruct_observation(observation))
        self.tree.tick()
        action = current_action["action"]
        return action

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        action = self._predict(observation, deterministic)
        if action is not None and 0 <= action < self.action_space.n:
            return action, None
        else:
            return self.action_space.sample(), None
