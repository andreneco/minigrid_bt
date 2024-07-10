import gymnasium as gym
from minigrid_bt.policy import BehaviorTreePolicy
from minigrid_bt.env_initialization import initialize_env
from minigrid_bt.main import create_ObstructedMaze_bt, create_Empty_bt
from minigrid_bt.utils import ExtendedFlatObsWrapper, ReconstructObsWrapper
from minigrid.wrappers import FullyObsWrapper

# List of environment IDs
env_ids = [
    "MiniGrid-ObstructedMaze-1Dlhb-v0", #0 ok (suboptimal)
    "MiniGrid-Empty-6x6-v0", #1 ok
    "MiniGrid-Empty-8x8-v0", #2 ok
    "MiniGrid-Empty-Random-5x5-v0", #3 ok
    "MiniGrid-Empty-Random-6x6-v0", #4 ok
    "BabyAI-GoToRedBallNoDists-v0", #5
    "MiniGrid-DistShift2-v0", #6 ok
    "MiniGrid-LavaGapS7-v0", #7 ok
    "MiniGrid-FourRooms-v0", #8 ok
    "MiniGrid-MultiRoom-N6-v0", #9
    "MiniGrid-SimpleCrossingS11N5-v0", #10 ok
    "MiniGrid-LavaCrossingS11N5-v0", #11 ok
    "MiniGrid-Unlock-v0", #12
    "MiniGrid-DoorKey-8x8-v0", #13
    "MiniGrid-UnlockPickup-v0" #14
]

# Choose the environment ID and the corresponding behavior tree creation function
env_id = env_ids[6]  # or any other from the list
tree_creation_func = create_Empty_bt  # or create_another_behaviour_tree

# Initialize the environment and capture the image shape
env, image_shape = initialize_env(env_id, render_mode='human')

# Get observation space and action space by initializing the environment once
observation_space = env.observation_space
action_space = env.action_space

# Create the behavior tree policy
policy = BehaviorTreePolicy(
    observation_space=observation_space,
    action_space=action_space,
    env=env,
    image_shape=image_shape,
    tree_creation_func=tree_creation_func,
    reconstruct_obs_wrapper_class=ReconstructObsWrapper
)

# Reset the environment and get the initial observation
obs = policy.env.reset()

# Accessing width and height attributes properly
width = policy.env.unwrapped.width if hasattr(policy.env.unwrapped, 'width') else policy.env.get_wrapper_attr('width')
height = policy.env.unwrapped.height if hasattr(policy.env.unwrapped, 'height') else policy.env.get_wrapper_attr('height')

done = False

while not done:
    # Predict the next action using the behavior tree policy
    action, _ = policy.predict(obs)

    if action is not None:
        # Step the environment using the action
        obs, reward, done, info, *_ = policy.env.step(action)

        # Render the updated state of the environment
        policy.env.render()
    else:
        print("Waiting for action decision...")

print(f"Final reward: {reward}")
policy.env.close()