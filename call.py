import gymnasium as gym
from minigrid_bt.policy import BehaviorTreePolicy
from minigrid_bt.env_initialization import initialize_env
from minigrid_bt.main import (
    create_KeyCorridor_bt, create_RedBlueDoors_bt, create_ObstructedMaze_bt, 
    create_BlockedUnlockPickup_bt, create_UnlockPickup_bt, create_DoorKey_bt, 
    create_Unlock_bt, create_MultiRoom_bt, create_Empty_bt, create_BabyAI_bt
)
from minigrid_bt.utils import ReconstructObsWrapper
from minigrid.manual_control import ManualControl
import pandas as pd

'''UNSUPPORTED
MiniGrid-ObstructedMaze-Full-v0

DYNAMIC
MiniGrid-Dynamic-Obstacles-5x5-v0
MiniGrid-Dynamic-Obstacles-Random-5x5-v0
MiniGrid-Dynamic-Obstacles-6x6-v0
MiniGrid-Dynamic-Obstacles-Random-6x6-v0
MiniGrid-Dynamic-Obstacles-8x8-v0
MiniGrid-Dynamic-Obstacles-16x16-v0
    
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

# List of environment IDs
env_ids = [
    "MiniGrid-MultiRoom-N2-S4-v0", #30 create_MultiRoom_bt ok
    "MiniGrid-MultiRoom-N4-S5-v0", #31 create_MultiRoom_bt ok    
    "MiniGrid-MultiRoom-N6-v0", #9 create_MultiRoom_bt ok
    "MiniGrid-ObstructedMaze-1Dlhb-v0", #0 create_ObstructedMaze_bt ok
    "MiniGrid-Empty-6x6-v0", #1 create_Empty_bt ok
    "MiniGrid-Empty-8x8-v0", #2 create_Empty_bt ok
    "MiniGrid-Empty-Random-5x5-v0", #3 create_Empty_bt ok
    "MiniGrid-Empty-Random-6x6-v0", #4 create_Empty_bt ok
    "BabyAI-GoToRedBallNoDists-v0", #5 create_BabyAI_bt ok
    "MiniGrid-DistShift2-v0", #6 create_Empty_bt ok
    "MiniGrid-LavaGapS7-v0", #7 create_Empty_bt ok
    "MiniGrid-FourRooms-v0", #8 create_Empty_bt ok
    "MiniGrid-SimpleCrossingS11N5-v0", #10 create_Empty_bt ok
    "MiniGrid-LavaCrossingS11N5-v0", #11 create_Empty_bt ok
    "MiniGrid-Unlock-v0", #12 create_Unlock_bt ok
    "MiniGrid-DoorKey-8x8-v0", #13 create_DoorKey_bt ok
    "MiniGrid-UnlockPickup-v0", #14 create_UnlockPickup_bt ok
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
    "MiniGrid-Empty-5x5-v0", #26 create_Empty_bt ok
    "MiniGrid-LavaGapS5-v0", #27 create_Empty_bt ok
    "MiniGrid-LavaGapS6-v0", #28 create_Empty_bt ok
    "MiniGrid-LavaGapS7-v0", #29 create_Empty_bt ok
    "MiniGrid-RedBlueDoors-8x8-v0", #32 create_RedBlueDoors_bt ok
    "MiniGrid-RedBlueDoors-6x6-v0", #33 create_RedBlueDoors_bt ok
    "MiniGrid-BlockedUnlockPickup-v0", #34 create_BlockedUnlockPickup_bt ok
    "MiniGrid-KeyCorridorS3R1-v0", #35 create_KeyCorridor_bt ok
    "MiniGrid-KeyCorridorS3R2-v0", #36 create_KeyCorridor_bt ok
    "MiniGrid-KeyCorridorS3R3-v0", #37 create_KeyCorridor_bt ok
    "MiniGrid-KeyCorridorS4R3-v0", #38 create_KeyCorridor_bt ok
    "MiniGrid-KeyCorridorS5R3-v0", #39 create_KeyCorridor_bt ok
    "MiniGrid-KeyCorridorS6R3-v0" #40 create_KeyCorridor_bt ok
]

env_to_bt_mapping = {
    "MiniGrid-ObstructedMaze-1Dlhb-v0": create_ObstructedMaze_bt,
    "MiniGrid-Empty-6x6-v0": create_Empty_bt,
    "MiniGrid-Empty-8x8-v0": create_Empty_bt,
    "MiniGrid-Empty-Random-5x5-v0": create_Empty_bt,
    "MiniGrid-Empty-Random-6x6-v0": create_Empty_bt,
    "BabyAI-GoToRedBallNoDists-v0": create_BabyAI_bt,
    "MiniGrid-DistShift2-v0": create_Empty_bt,
    "MiniGrid-LavaGapS7-v0": create_Empty_bt,
    "MiniGrid-FourRooms-v0": create_Empty_bt,
    "MiniGrid-MultiRoom-N6-v0": create_MultiRoom_bt,
    "MiniGrid-SimpleCrossingS11N5-v0": create_Empty_bt,
    "MiniGrid-LavaCrossingS11N5-v0": create_Empty_bt,
    "MiniGrid-Unlock-v0": create_Unlock_bt,
    "MiniGrid-DoorKey-8x8-v0": create_DoorKey_bt,
    "MiniGrid-UnlockPickup-v0": create_UnlockPickup_bt,
    "MiniGrid-Empty-16x16-v0": create_Empty_bt,
    "MiniGrid-LavaCrossingS9N1-v0": create_Empty_bt,
    "MiniGrid-LavaCrossingS9N2-v0": create_Empty_bt,
    "MiniGrid-LavaCrossingS9N3-v0": create_Empty_bt,
    "MiniGrid-SimpleCrossingS9N1-v0": create_Empty_bt,
    "MiniGrid-SimpleCrossingS9N2-v0": create_Empty_bt,
    "MiniGrid-SimpleCrossingS9N3-v0": create_Empty_bt,
    "MiniGrid-DistShift1-v0": create_Empty_bt,
    "MiniGrid-DoorKey-5x5-v0": create_DoorKey_bt,
    "MiniGrid-DoorKey-6x6-v0": create_DoorKey_bt,
    "MiniGrid-DoorKey-16x16-v0": create_DoorKey_bt,
    "MiniGrid-Empty-5x5-v0": create_Empty_bt,
    "MiniGrid-LavaGapS5-v0": create_Empty_bt,
    "MiniGrid-LavaGapS6-v0": create_Empty_bt,
    "MiniGrid-LavaGapS7-v0": create_Empty_bt,
    "MiniGrid-MultiRoom-N2-S4-v0": create_MultiRoom_bt,
    "MiniGrid-MultiRoom-N4-S5-v0": create_MultiRoom_bt,
    "MiniGrid-RedBlueDoors-8x8-v0": create_RedBlueDoors_bt,
    "MiniGrid-RedBlueDoors-6x6-v0": create_RedBlueDoors_bt,
    "MiniGrid-BlockedUnlockPickup-v0": create_BlockedUnlockPickup_bt,
    "MiniGrid-KeyCorridorS3R1-v0": create_KeyCorridor_bt,
    "MiniGrid-KeyCorridorS3R2-v0": create_KeyCorridor_bt,
    "MiniGrid-KeyCorridorS3R3-v0": create_KeyCorridor_bt,
    "MiniGrid-KeyCorridorS4R3-v0": create_KeyCorridor_bt,
    "MiniGrid-KeyCorridorS5R3-v0": create_KeyCorridor_bt,
    "MiniGrid-KeyCorridorS6R3-v0": create_KeyCorridor_bt
}

def run_experiment(env_id, seed, results=[], render=False):

    # Choose the environment ID and the corresponding behavior tree creation function
    tree_creation_func = env_to_bt_mapping.get(env_id)

    if tree_creation_func is None:
        raise ValueError(f"No behavior tree creation function found for environment ID: {env_id}")

    # Initialize the environment and capture the image shape
    if render:
        env, image_shape = initialize_env(env_id, render_mode='human')
    else:
        env, image_shape = initialize_env(env_id)

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
    obs = policy.env.reset(seed=seed)
    # input("Press Enter to continue...")

    # manual_control = ManualControl(env, seed=4)
    # manual_control.start()

    # input("Press Enter to continue...")

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

    # Episode finished, record the episode length and reward
    results.append({
        "env_id": env_id,
        "seed": seed,
        "episode_length": step_count,
        "final_reward": reward
    })

    # Close the environment
    policy.env.close()

# run_experiment(env_ids[2], results=[], render=True)

# List of seeds to use for each environment
seeds = [0, 1, 2, 3, 4]

# Run experiments for all environments with multiple seeds
results = []
for env_id in env_ids:
    for seed in seeds:
        print(f"Running experiment for env_id: {env_id} with seed: {seed}")
        run_experiment(env_id, seed, results, render=False)

results_df = pd.DataFrame(results)
results_df.to_csv('experiment_results.csv', index=False)
print(results_df)