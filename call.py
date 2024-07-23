import gymnasium as gym
from minigrid_bt.policy import BehaviorTreePolicy
from minigrid_bt.env_initialization import initialize_env
from minigrid_bt.main import create_ObstructedMaze_bt, create_UnlockPickup_bt, create_DoorKey_bt, create_Unlock_bt, create_MultiRoom_bt, create_Empty_bt, create_BabyAI_bt
from minigrid_bt.utils import ReconstructObsWrapper
from minigrid.manual_control import ManualControl

# List of environment IDs
env_ids = [
    "MiniGrid-ObstructedMaze-1Dlhb-v0", #0 create_ObstructedMaze_bt ok
    "MiniGrid-Empty-6x6-v0", #1 create_Empty_bt ok
    "MiniGrid-Empty-8x8-v0", #2 create_Empty_bt ok
    "MiniGrid-Empty-Random-5x5-v0", #3 create_Empty_bt ok
    "MiniGrid-Empty-Random-6x6-v0", #4 create_Empty_bt ok
    "BabyAI-GoToRedBallNoDists-v0", #5 create_BabyAI_bt ok
    "MiniGrid-DistShift2-v0", #6 create_Empty_bt ok
    "MiniGrid-LavaGapS7-v0", #7 create_Empty_bt ok
    "MiniGrid-FourRooms-v0", #8 create_Empty_bt ok
    "MiniGrid-MultiRoom-N6-v0", #9 PROBLEM--------------------------------
    "MiniGrid-SimpleCrossingS11N5-v0", #10 create_Empty_bt ok
    "MiniGrid-LavaCrossingS11N5-v0", #11 create_Empty_bt ok
    "MiniGrid-Unlock-v0", #12 create_Unlock_bt ok
    "MiniGrid-DoorKey-8x8-v0", #13 create_DoorKey_bt ok
    "MiniGrid-UnlockPickup-v0", #14 PROBLEM-------------------------------
    "MiniGrid-Empty-16x16-v0", #15 create_Empty_bt ok
    "MiniGrid-LavaCrossingS9N1-v0", #16 create_Empty_bt ok
    "MiniGrid-LavaCrossingS9N2-v0", #17 create_Empty_bt ok
    "MiniGrid-LavaCrossingS9N3-v0", #18 create_Empty_bt ok
    "MiniGrid-SimpleCrossingS9N1-v0", #19 create_Empty_bt ok
    "MiniGrid-SimpleCrossingS9N2-v0", #20 create_Empty_bt ok
    "MiniGrid-SimpleCrossingS9N3-v0", #21 create_Empty_bt ok
    "MiniGrid-DistShift1-v0", #22 create_Empty_bt ok
    "MiniGrid-DoorKey-5x5-v0", #23 create_DoorKey_bt ok
    "MiniGrid-DoorKey-6x6-v0", #24 create_DoorKey_bt ok
    "MiniGrid-DoorKey-16x16-v0", #25 create_DoorKey_bt ok
    "MiniGrid-Dynamic-Obstacles-5x5-v0", #26 PROBLEM----------------------
    "MiniGrid-Dynamic-Obstacles-Random-5x5-v0", #27 PROBLEM---------------
    "MiniGrid-Dynamic-Obstacles-6x6-v0", #28 PROBLEM----------------------
    "MiniGrid-Dynamic-Obstacles-Random-6x6-v0", #29 PROBLEM---------------
    "MiniGrid-Dynamic-Obstacles-8x8-v0", #30 PROBLEM----------------------
    "MiniGrid-Dynamic-Obstacles-16x16-v0", #31 PROBLEM--------------------
    "MiniGrid-Empty-5x5-v0", #32 create_Empty_bt ok
    "MiniGrid-LavaGapS5-v0", #33 create_Empty_bt ok
    "MiniGrid-LavaGapS6-v0", #34 create_Empty_bt ok
    "MiniGrid-LavaGapS7-v0", #35 create_Empty_bt ok
    "MiniGrid-MultiRoom-N2-S4-v0", #36 PROBLEM----------------------------
    "MiniGrid-MultiRoom-N4-S5-v0", #37 PROBLEM----------------------------
    "MiniGrid-ObstructedMaze-Full-v0", #38 PROBLEM------------------------
    "MiniGrid-BlockedUnlockPickup-v0", #39 PROBLEM------------------------
    "MiniGrid-KeyCorridorS3R1-v0", #40 PROBLEM----------------------------
    "MiniGrid-KeyCorridorS3R2-v0", #41 PROBLEM----------------------------
    "MiniGrid-KeyCorridorS3R3-v0", #42 PROBLEM----------------------------
    "MiniGrid-KeyCorridorS4R3-v0", #43 PROBLEM----------------------------
    "MiniGrid-KeyCorridorS5R3-v0", #44 PROBLEM----------------------------
    "MiniGrid-KeyCorridorS6R3-v0", #45 PROBLEM----------------------------
    "MiniGrid-RedBlueDoors-6x6-v0", #46 PROBLEM---------------------------
    "MiniGrid-RedBlueDoors-8x8-v0" #47 PROBLEM----------------------------
]

'''OMITTED

TEXTUAL
MiniGrid-Fetch-5x5-N2-v0
MiniGrid-Fetch-6x6-N2-v0
MiniGrid-Fetch-8x8-N3-v0
MiniGrid-GoToDoor-5x5-v0
MiniGrid-GoToDoor-6x6-v0
MiniGrid-GoToDoor-8x8-v0
MiniGrid-GoToObject-6x6-N2-v0
MiniGrid-GoToObject-8x8-N2-v0
MiniGrid-LockedRoom-v0
MiniGrid-PutNear-6x6-N2-v0
MiniGrid-PutNear-8x8-N3-v0

MEMORY
MiniGrid-MemoryS17Random-v0
MiniGrid-MemoryS13Random-v0
MiniGrid-MemoryS13-v0
MiniGrid-MemoryS11-v0
'''

# Choose the environment ID and the corresponding behavior tree creation function
env_id = env_ids[0]  # or any other from the list
tree_creation_func = create_ObstructedMaze_bt  # or create_another_behaviour_tree

# Initialize the environment and capture the image shape
env, image_shape = initialize_env(env_id, render_mode='human')
# env, image_shape = initialize_env(env_id)

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
obs = policy.env.reset(seed=1)
input("Press Enter to continue...")

# manual_control = ManualControl(env, seed=4)
# manual_control.start()

# input("Press Enter to continue...")

# Accessing width and height attributes properly
width = policy.env.unwrapped.width if hasattr(policy.env.unwrapped, 'width') else policy.env.get_wrapper_attr('width')
height = policy.env.unwrapped.height if hasattr(policy.env.unwrapped, 'height') else policy.env.get_wrapper_attr('height')

done = False
step_count = 0

while not done:
    # Predict the next action using the behavior tree policy
    action, _ = policy.predict(obs)
    # print("ACTION: ", action)
    # input("Press...")
    if action is not None:
        # Step the environment using the action
        obs, reward, done, info, *_ = policy.env.step(action)

        # Render the updated state of the environment
        policy.env.render()

        # Increment the step count
        step_count += 1
    else:
        print("Waiting for action decision...")

# Episode finished, record the episode length
print(f"Episode length: {step_count}")

print(f"Final reward: {reward}")


# Close the environment
policy.env.close()
