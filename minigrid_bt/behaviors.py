import py_trees
import numpy as np
from minigrid_bt.utils import (
    extract_grid_and_direction, extract_positions, extract_multiple_positions,
    align_direction_to_target, find_path_to_adjacent, find_any_ball_position,
    prepare_action_sequence, find_ball_position, find_safe_drop_location,
    astar_pathfinding, find_door_position, find_path_to_adjacent_doors,
    prepare_action_sequence_doors, find_locked_door_position,
    extract_door_positions,
    astar_pathfinding_to_goal, ACTION_TO_IDX, direction_map, current_action
)

class GoToGoal(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, goal_object="goal", debug=False):
        super(GoToGoal, self).__init__(name)
        self.env = env
        self.obs = obs
        self.goal_object = goal_object
        self.action_sequence = []
        self.current_action_index = 0
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        agent_pos = positions.get('agent')
        goal_pos = positions.get(self.goal_object)

        if goal_pos:
            path_to_goal = astar_pathfinding_to_goal(grid[:, :, 0], agent_pos, goal_pos, agent_dir)
            if path_to_goal:
                self.action_sequence = prepare_action_sequence(path_to_goal, agent_dir, goal_pos)
            else:
                self.feedback_message = "Cannot determine path to goal."
                self.action_sequence = []
        else:
            self.feedback_message = "Goal position not found."

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                return py_trees.common.Status.FAILURE
            
class GoToGoalDoors(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, goal_object="goal", debug=False):
        super(GoToGoalDoors, self).__init__(name)
        self.env = env
        self.obs = obs
        self.goal_object = goal_object
        self.action_sequence = []
        self.current_action_index = 0
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        agent_pos = positions.get('agent')
        goal_pos = positions.get(self.goal_object)

        if goal_pos:
            path_to_goal = find_path_to_adjacent_doors(grid, agent_pos, goal_pos, agent_dir)
            if path_to_goal:
                path_to_goal = [(x[0], x[1]) for x in path_to_goal]
                self.action_sequence = prepare_action_sequence(path_to_goal, agent_dir, goal_pos)
                self.action_sequence.append(ACTION_TO_IDX["move_forward"])
            else:
                self.feedback_message = "Cannot determine path to goal."
                self.action_sequence = []
        else:
            self.feedback_message = "Goal position not found."

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                return py_trees.common.Status.FAILURE            

class GoNextToGoal(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, goal_object="goal", goal_color=None, debug=False):
        super(GoNextToGoal, self).__init__(name)
        self.env = env
        self.obs = obs
        self.goal_object = goal_object
        self.goal_color = goal_color
        self.action_sequence = []
        self.current_action_index = 0
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        agent_pos = positions.get('agent')
        
        if self.goal_object == "ball" and self.goal_color:
            goal_pos = find_ball_position(self.obs, self.goal_color)
        else:
            goal_pos = positions.get(self.goal_object)

        if goal_pos:
            path_to_goal = astar_pathfinding(grid[:, :, 0], agent_pos, goal_pos, agent_dir)
            if path_to_goal:
                self.action_sequence = prepare_action_sequence(path_to_goal, agent_dir, goal_pos)
            else:
                self.feedback_message = "Cannot determine path to goal."
                self.action_sequence = []
        else:
            self.feedback_message = f"Goal position for {self.goal_object} not found."

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                return py_trees.common.Status.FAILURE

class PickUp(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, goal_object="ball", goal_color=None, debug=False):
        super(PickUp, self).__init__(name)
        self.env = env
        self.obs = obs
        self.goal_object = goal_object
        self.goal_color = goal_color
        self.action_sequence = []
        self.current_action_index = 0
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        agent_pos = positions.get('agent')

        if self.goal_object == "ball" and self.goal_color:
            goal_pos = find_ball_position(self.obs, self.goal_color)
        else:
            goal_pos = positions.get(self.goal_object)

        if not goal_pos:
            self.feedback_message = f"Cannot find {self.goal_object} position."
            return

        agent_neighbors = [
            (agent_pos[0] + 1, agent_pos[1]),
            (agent_pos[0], agent_pos[1] + 1),
            (agent_pos[0], agent_pos[1] - 1),
            (agent_pos[0] - 1, agent_pos[1]),
        ]
        if goal_pos in agent_neighbors:
            self.feedback_message = "Already a neighbor, maybe needs to adjust direction."
            self.action_sequence = align_direction_to_target(agent_pos, goal_pos, agent_dir)
            self.action_sequence.append(ACTION_TO_IDX["pick_up"])
        else:
            path_to_goal = astar_pathfinding(grid[:, :, 0], agent_pos, goal_pos, agent_dir)
            if path_to_goal:
                self.action_sequence = prepare_action_sequence(path_to_goal, agent_dir, goal_pos)
                if len(path_to_goal) >= 2:
                    last_position = path_to_goal[-1]
                    second_last_position = path_to_goal[-2]
                    move_vector = (second_last_position[0] - last_position[0], second_last_position[1] - last_position[1])
                else:
                    move_vector = (0, 0)
                current_direction = direction_map.get(move_vector)
                if current_direction is None:
                    raise ValueError(f"Invalid move vector {move_vector}. Cannot determine direction from start={agent_pos} to end={goal_pos}.")
                self.action_sequence.append(ACTION_TO_IDX["pick_up"])
            else:
                self.feedback_message = "Cannot determine path to goal."
                self.action_sequence = []

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                return py_trees.common.Status.FAILURE

class PickUpObstacle(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, goal_object="ball", goal_color="green", debug=False):
        super(PickUpObstacle, self).__init__(name)
        self.env = env
        self.obs = obs
        self.goal_object = goal_object
        self.goal_color = goal_color
        self.action_sequence = []
        self.current_action_index = 0
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        agent_pos = positions.get('agent')

        if self.goal_object == "ball":
            goal_pos = find_ball_position(self.obs, self.goal_color)
        else:
            goal_pos = positions.get(self.goal_object)

        if not goal_pos:
            self.feedback_message = f"Cannot find {self.goal_object} position."
            return

        self.action_sequence = align_direction_to_target((agent_pos[0], agent_pos[1]), goal_pos, agent_dir)
        self.action_sequence.append(ACTION_TO_IDX["pick_up"])

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                return py_trees.common.Status.FAILURE

class PickUpGoal(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, goal_object="ball", goal_color=None, debug=False):
        super(PickUpGoal, self).__init__(name)
        self.env = env
        self.obs = obs
        self.goal_object = goal_object
        self.goal_color = goal_color
        self.action_sequence = []
        self.current_action_index = 0
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        agent_pos = positions.get('agent')

        if self.goal_object == "ball":
            if self.goal_color:
                goal_pos = find_ball_position(self.obs, self.goal_color)
            else:
                goal_pos = find_any_ball_position(self.obs)
        else:
            goal_pos = positions.get(self.goal_object)

        if not goal_pos:
            self.feedback_message = f"Cannot find {self.goal_object} position."
            return

        # Check if the goal position is adjacent to a door
        door_positions = extract_door_positions(self.obs)
        adjacent_to_door = any(
            (goal_pos[0], goal_pos[1] + dy) == (door[0], door[1]) or
            (goal_pos[0] + dx, goal_pos[1]) == (door[0], door[1])
            for door in door_positions
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        )

        if adjacent_to_door:
            self.feedback_message = "Goal is adjacent to a door, using fixed action sequence."
            self.action_sequence = [2, 1, 1, 4, 0, 0, 3]
            return
        else:
            self.action_sequence.append(ACTION_TO_IDX["move_forward"])
            self.action_sequence.append(ACTION_TO_IDX["move_forward"])

            # Calculate the position two moves forward from the agent's position
            if agent_dir == 0:  # Facing right
                target_pos = (agent_pos[0], agent_pos[1] + 2)
                drop_pos = (agent_pos[0], agent_pos[1] + 3) 
            elif agent_dir == 1:  # Facing down
                target_pos = (agent_pos[0] + 2, agent_pos[1])
                drop_pos = (agent_pos[0] + 3, agent_pos[1]) 
            elif agent_dir == 2:  # Facing left
                target_pos = (agent_pos[0], agent_pos[1] - 2)
                drop_pos = (agent_pos[0], agent_pos[1] - 3) 
            elif agent_dir == 3:  # Facing up
                target_pos = (agent_pos[0] - 2, agent_pos[1])
                drop_pos = (agent_pos[0] - 3, agent_pos[1]) 

            if goal_pos == drop_pos:
                if grid[target_pos[0], target_pos[1] - 1, 0] == 1:
                    self.action_sequence += [1, 4, 0, 3]
                else:
                    self.action_sequence += [0, 4, 1, 3]
            else:        
                self.action_sequence.append(ACTION_TO_IDX["drop"])
                grid[:, :, 0][drop_pos[1], drop_pos[0]] = 5

        agent_neighbors = [
            (agent_pos[0], agent_pos[1] - 1), 
            (agent_pos[0] + 1, agent_pos[1]), 
            (agent_pos[0], agent_pos[1] + 1), 
            (agent_pos[0] - 1, agent_pos[1]), 
        ]
        if goal_pos in agent_neighbors:
            self.feedback_message = "Already a neighbor, maybe needs to adjust direction."
            self.action_sequence += align_direction_to_target(agent_pos, goal_pos, agent_dir)
            self.action_sequence.append(ACTION_TO_IDX["pick_up"])
        else:
            path_to_goal = find_path_to_adjacent(grid[:,:,0], target_pos, goal_pos, agent_dir)
            if path_to_goal:
                self.action_sequence += prepare_action_sequence(path_to_goal, agent_dir, goal_pos)
                self.action_sequence.append(ACTION_TO_IDX["pick_up"])
            else:
                self.feedback_message = f"Cannot determine path to {self.goal_object}."
                self.action_sequence = []

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                return py_trees.common.Status.FAILURE

class EnterRoom(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, debug=False):
        super(EnterRoom, self).__init__(name)
        self.env = env
        self.obs = obs
        self.action_sequence = []
        self.current_action_index = 0
        self.initialized = False  # Flag to check if initialize has been called
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        if not self.initialized:
            grid, agent_dir = extract_grid_and_direction(self.obs)
            positions = extract_positions(self.obs)
            door_pos = positions.get('door')
            agent_pos = positions.get('agent')

            agent_neighbors = [
                (agent_pos[0], agent_pos[1] - 1), 
                (agent_pos[0] + 1, agent_pos[1]), 
                (agent_pos[0], agent_pos[1] + 1), 
                (agent_pos[0] - 1, agent_pos[1]), 
            ]
            if door_pos:
                if door_pos in agent_neighbors:
                    self.feedback_message = "Already a neighbor, maybe needs to adjust direction."
                    self.action_sequence = align_direction_to_target(agent_pos, door_pos, agent_dir)
                    self.action_sequence.append(ACTION_TO_IDX["open"])
                    self.action_sequence.append(ACTION_TO_IDX["move_forward"])
                else:
                    path_to_door = find_path_to_adjacent(grid[:,:,0], agent_pos, door_pos, agent_dir)
                    if path_to_door:
                        self.action_sequence = prepare_action_sequence(path_to_door, agent_dir, door_pos)
                        self.action_sequence.append(ACTION_TO_IDX["open"])
                        self.action_sequence.append(ACTION_TO_IDX["move_forward"])
                    else:
                        self.feedback_message = "Cannot determine path."
                        self.action_sequence = []
                        print("Cannot determine path.")
            else:
                self.feedback_message = "Cannot determine door position."
                print("Cannot determine door position.")
            
            self.initialized = True  # Mark the behaviour as initialized

    def update_obs(self, new_obs):
        self.obs = new_obs
        self.initialized = False  # Reset initialized flag to allow reinitialization with new observation

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                self.initialized = False  # Allow reinitialization for next tick
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                self.initialized = False  # Allow reinitialization for next tick
                return py_trees.common.Status.FAILURE

class OpenLockedDoor(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, door_color="red", debug=False):
        super(OpenLockedDoor, self).__init__(name)
        self.env = env
        self.obs = obs
        self.door_color = door_color
        self.action_sequence = []
        self.current_action_index = 0
        self.initialized = False  # Flag to check if initialize has been called
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        if not self.initialized:
            grid, agent_dir = extract_grid_and_direction(self.obs)
            positions = extract_positions(self.obs)
            door_pos = find_locked_door_position(self.obs)  # Updated to find locked door
            agent_pos = positions.get('agent')

            agent_neighbors = [
                (agent_pos[0], agent_pos[1] - 1), 
                (agent_pos[0] + 1, agent_pos[1]), 
                (agent_pos[0], agent_pos[1] + 1), 
                (agent_pos[0] - 1, agent_pos[1]), 
            ]
            if door_pos:
                if door_pos in agent_neighbors:
                    self.feedback_message = "Already a neighbor, maybe needs to adjust direction."
                    self.action_sequence = align_direction_to_target(agent_pos, door_pos, agent_dir)
                    self.action_sequence.append(ACTION_TO_IDX["open"])
                else:
                    path_to_door = find_path_to_adjacent_doors(grid, agent_pos, (door_pos[0],door_pos[1]), agent_dir)  # Updated to consider doors
                    if path_to_door:
                        path_to_door = [(x[0], x[1]) for x in path_to_door]
                        self.action_sequence = prepare_action_sequence(path_to_door, agent_dir, door_pos)
                        self.action_sequence.append(ACTION_TO_IDX["open"])
                    else:
                        self.feedback_message = "Cannot determine path."
                        self.action_sequence = []
                        print("Cannot determine path.")
            else:
                self.feedback_message = f"Cannot determine locked {self.door_color} door position."
                print(f"Cannot determine locked {self.door_color} door position.")
            
            self.initialized = True  # Mark the behaviour as initialized

    def update_obs(self, new_obs):
        self.obs = new_obs
        self.initialized = False  # Reset initialized flag to allow reinitialization with new observation

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                self.initialized = False  # Allow reinitialization for next tick
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                self.initialized = False  # Allow reinitialization for next tick
                return py_trees.common.Status.FAILURE

class OpenDoor(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, door_color="red", debug=False):
        super(OpenDoor, self).__init__(name)
        self.env = env
        self.obs = obs
        self.door_color = door_color
        self.action_sequence = []
        self.current_action_index = 0
        self.initialized = False  # Flag to check if initialize has been called
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        if not self.initialized:
            grid, agent_dir = extract_grid_and_direction(self.obs)
            positions = extract_positions(self.obs)
            door_pos = find_door_position(self.obs, self.door_color)
            agent_pos = positions.get('agent')

            agent_neighbors = [
                (agent_pos[0], agent_pos[1] - 1), 
                (agent_pos[0] + 1, agent_pos[1]), 
                (agent_pos[0], agent_pos[1] + 1), 
                (agent_pos[0] - 1, agent_pos[1]), 
            ]
            if door_pos:
                if door_pos in agent_neighbors:
                    self.feedback_message = "Already a neighbor, maybe needs to adjust direction."
                    self.action_sequence = align_direction_to_target(agent_pos, door_pos, agent_dir)
                    self.action_sequence.append(ACTION_TO_IDX["open"])
                else:
                    path_to_door = find_path_to_adjacent(grid[:,:,0], agent_pos, door_pos, agent_dir)
                    if path_to_door:
                        self.action_sequence = prepare_action_sequence(path_to_door, agent_dir, door_pos)
                        self.action_sequence.append(ACTION_TO_IDX["open"])
                    else:
                        self.feedback_message = "Cannot determine path."
                        self.action_sequence = []
                        print("Cannot determine path.")
            else:
                self.feedback_message = f"Cannot determine {self.door_color} door position."
                print(f"Cannot determine {self.door_color} door position.")
            
            self.initialized = True  # Mark the behaviour as initialized

    def update_obs(self, new_obs):
        self.obs = new_obs
        self.initialized = False  # Reset initialized flag to allow reinitialization with new observation

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                self.initialized = False  # Allow reinitialization for next tick
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                self.initialized = False  # Allow reinitialization for next tick
                return py_trees.common.Status.FAILURE

class EnterStep(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, debug=False):
        super(EnterStep, self).__init__(name)
        self.env = env
        self.obs = obs
        self.action_sequence = [ACTION_TO_IDX["move_forward"], ACTION_TO_IDX["move_forward"]]
        self.current_action_index = 0
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        self.feedback_message = "Moving forward."
        SharedStatus.has_entered_room = False  # Reset status at the start

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            if self.current_action_index == len(self.action_sequence):
                SharedStatus.has_entered_room = True
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING
        else:
            self.current_action_index = 0
            return py_trees.common.Status.FAILURE

class OpenAllDoors(py_trees.behaviour.Behaviour):
    def __init__(self, name="OpenDoors", env=None, obs=None, debug=False):
        super(OpenAllDoors, self).__init__(name)
        self.env = env
        self.obs = obs
        self.action_sequence = []
        self.current_action_index = 0
        self.closed_doors = []
        self.debug = debug

    def update_obs(self, new_obs):
        self.obs = new_obs

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_multiple_positions(self.obs)
        agent_pos = positions.get('agent', None)
        door_positions = positions.get('door', [])

        if not isinstance(door_positions, list):
            door_positions = [door_positions]

        if agent_pos is None:
            self.feedback_message = "Agent position not found."
            if self.debug:
                print(self.feedback_message)
            return
        
        # Select the first agent position if there are multiple positions
        if isinstance(agent_pos[0], tuple):
            agent_pos = agent_pos[0]

        # Filter closed doors and their distances from the agent
        self.closed_doors = []
        for pos in door_positions:
            if isinstance(pos, tuple) and len(pos) == 2:
                x, y = int(pos[0]), int(pos[1])
                door_status = grid[y, x, 2]
                if door_status == 1:  # Assuming 1 represents a closed door
                    distance = np.abs(x - agent_pos[0]) + np.abs(y - agent_pos[1])
                    self.closed_doors.append(((x, y), distance))

        # Sort doors by distance
        self.closed_doors.sort(key=lambda x: x[1])

        self.action_sequence = []
        self.current_action_index = 0

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
        if not self.action_sequence and self.closed_doors:
            grid, agent_dir = extract_grid_and_direction(self.obs)
            positions = extract_multiple_positions(self.obs)
            agent_pos = positions.get('agent', None)

            if agent_pos is None:
                self.feedback_message = "Agent position not found."
                if self.debug:
                    print(self.feedback_message)
                return py_trees.common.Status.FAILURE

            # Select the first agent position if there are multiple positions
            if isinstance(agent_pos[0], tuple):
                agent_pos = agent_pos[0]

            door_pos, _ = self.closed_doors.pop(0)
            path_to_door = find_path_to_adjacent(grid[:, :, 0], agent_pos, door_pos, agent_dir)
            if path_to_door:
                agent_neighbors = [
                    (agent_pos[0], agent_pos[1] - 1),
                    (agent_pos[0] + 1, agent_pos[1]),
                    (agent_pos[0], agent_pos[1] + 1),
                    (agent_pos[0] - 1, agent_pos[1]),
                ]
                if door_pos in agent_neighbors:
                    self.feedback_message = "Already a neighbor, maybe needs to adjust direction."
                    self.action_sequence = align_direction_to_target(agent_pos, door_pos, agent_dir)
                    self.action_sequence.append(ACTION_TO_IDX["open"])
                    self.action_sequence.append(ACTION_TO_IDX["move_forward"])
                else:
                    self.action_sequence = prepare_action_sequence(path_to_door, agent_dir, door_pos)
                    self.action_sequence.append(ACTION_TO_IDX["open"])
                    self.action_sequence.append(ACTION_TO_IDX["move_forward"])
            else:
                self.feedback_message = f"Cannot determine path to door at {door_pos}."
                return py_trees.common.Status.FAILURE

        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            if self.current_action_index >= len(self.action_sequence):
                self.action_sequence = []
                self.current_action_index = 0
            return py_trees.common.Status.RUNNING
        elif not self.closed_doors:
            self.feedback_message = "All doors are open."
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING

class PickUpKey(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, debug=False):
        super(PickUpKey, self).__init__(name)
        self.env = env
        self.obs = obs
        self.action_sequence = []
        self.current_action_index = 0
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        key_pos = positions.get('key')
        agent_pos = positions.get('agent')

        agent_neighbors = [
            (agent_pos[0], agent_pos[1] - 1), 
            (agent_pos[0] + 1, agent_pos[1]), 
            (agent_pos[0], agent_pos[1] + 1), 
            (agent_pos[0] - 1, agent_pos[1]), 
        ]
        if key_pos:
            if key_pos in agent_neighbors:
                self.action_sequence += align_direction_to_target(agent_pos, key_pos, agent_dir)
                self.action_sequence.append(ACTION_TO_IDX["pick_up"]) 
            else:
                path_to_key = find_path_to_adjacent(grid[:,:,0], agent_pos, key_pos, agent_dir)
                if path_to_key:
                    self.action_sequence += prepare_action_sequence(path_to_key, 0, key_pos)
                    self.action_sequence.append(ACTION_TO_IDX["pick_up"]) 
                else:
                    self.feedback_message = "Cannot determine path."
                    self.action_sequence = []
        else:
            self.feedback_message = "Cannot determine key position."

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]

            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                return py_trees.common.Status.FAILURE

class PickUpKeyDoors(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, debug=False):
        super(PickUpKeyDoors, self).__init__(name)
        self.env = env
        self.obs = obs
        self.action_sequence = []
        self.current_action_index = 0
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        key_pos = positions.get('key')
        agent_pos = positions.get('agent')

        if key_pos:
            path_to_key = find_path_to_adjacent_doors(grid, agent_pos, key_pos, agent_dir)
            if path_to_key:
                self.action_sequence = prepare_action_sequence_doors(path_to_key, agent_dir, key_pos)
                self.action_sequence.append(ACTION_TO_IDX["pick_up"]) 
            else:
                self.feedback_message = "Cannot determine path."
                self.action_sequence = []
        else:
            self.feedback_message = "Cannot determine key position."

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                return py_trees.common.Status.FAILURE

class DropPickUpKeyBox(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, debug=False):
        super(DropPickUpKeyBox, self).__init__(name)
        self.env = env
        self.obs = obs
        self.action_sequence = []
        self.current_action_index = 0
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        box_pos = positions.get('box')
        agent_pos = positions.get('agent')
        ball_pos = find_ball_position(self.obs, 'green')
        door_pos = positions.get('door')

        agent_neighbors = [
            (agent_pos[0], agent_pos[1] - 1), 
            (agent_pos[0] + 1, agent_pos[1]), 
            (agent_pos[0], agent_pos[1] + 1), 
            (agent_pos[0] - 1, agent_pos[1]), 
        ]
        if not ball_pos:
            drop_pos = find_safe_drop_location(grid[:,:,0], door_pos, agent_pos, agent_dir)
            if drop_pos:
                self.action_sequence = align_direction_to_target(agent_pos, drop_pos, agent_dir)
                self.action_sequence.append(ACTION_TO_IDX["drop"])         
            else:
                self.action_sequence.append(ACTION_TO_IDX["move_forward"])    
                self.action_sequence.append(ACTION_TO_IDX["drop"])
                if agent_dir == 1:
                    agent_pos = (agent_pos[0]+1, agent_pos[1])
                    drop_pos = (agent_pos[0]+1, agent_pos[1])
                else:
                    agent_pos = (agent_pos[0]-1, agent_pos[1])
                    drop_pos = (agent_pos[0]-1, agent_pos[1])
        else:
            drop_pos = ball_pos
        if box_pos:
            grid[:,:,0][drop_pos[1],drop_pos[0]]=6
            agent_neighbors = [
                (agent_pos[0], agent_pos[1] - 1), 
                (agent_pos[0] + 1, agent_pos[1]), 
                (agent_pos[0], agent_pos[1] + 1), 
                (agent_pos[0] - 1, agent_pos[1]), 
            ]
            if box_pos in agent_neighbors:
                self.feedback_message = "Already a neighbor, maybe needs to adjust direction."
                move_vector = (drop_pos[0] - agent_pos[0], drop_pos[1] - agent_pos[1])
                current_direction = direction_map.get(move_vector)
                self.action_sequence += align_direction_to_target(agent_pos, box_pos, current_direction)
                self.action_sequence.append(ACTION_TO_IDX["open"]) 
                self.action_sequence.append(ACTION_TO_IDX["pick_up"]) 
            else:
                path_to_box = find_path_to_adjacent(grid[:,:,0], agent_pos, box_pos, agent_dir)
                if path_to_box:
                    move_vector = (drop_pos[0] - agent_pos[0], drop_pos[1] - agent_pos[1])
                    current_direction = direction_map.get(move_vector)
                    self.action_sequence += prepare_action_sequence(path_to_box, current_direction, box_pos)
                    self.action_sequence.append(ACTION_TO_IDX["open"]) 
                    self.action_sequence.append(ACTION_TO_IDX["pick_up"]) 
                else:
                    self.feedback_message = "Cannot determine path."
                    self.action_sequence = []
        else:
            self.feedback_message = "Cannot determine door or ball position."

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                return py_trees.common.Status.FAILURE

class DropKey(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, goal_object="ball", goal_color=None, debug=False):
        super(DropKey, self).__init__(name)
        self.env = env
        self.obs = obs
        self.goal_object = goal_object
        self.goal_color = goal_color
        self.path_to_ball = []
        self.path_to_drop = []
        self.action_sequence = []
        self.current_action_index = 0
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        agent_pos = positions.get('agent')
        door_pos = positions.get('door')

        if self.goal_object == "ball" and self.goal_color:
            goal_pos = find_ball_position(self.obs, self.goal_color)
        else:
            goal_pos = positions.get(self.goal_object)

        if not goal_pos:
            self.feedback_message = f"Cannot find {self.goal_object} position."
            return

        if goal_pos == (agent_pos[0], agent_pos[1] + 1):
            self.action_sequence = [1, 1, 4, 0, 0, 3]
        else:
            self.action_sequence.append(ACTION_TO_IDX["move_forward"])
            if goal_pos == (agent_pos[0], agent_pos[1] + 2):
                if grid[agent_pos[0] + 1, agent_pos[1] - 1, 0] == 1:
                    self.action_sequence += [1, 4, 0, 3]
                else:
                    self.drop_pos = (agent_pos[0] + 1, agent_pos[1] + 1)
                    self.action_sequence += [0, 4, 1, 3]
            self.action_sequence.append(ACTION_TO_IDX["drop"])
            
    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                return py_trees.common.Status.FAILURE

class DropObstacle(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, debug=False):
        super(DropObstacle, self).__init__(name)
        self.env = env
        self.obs = obs
        self.path_to_ball = []
        self.path_to_drop = []
        self.action_sequence = []
        self.current_action_index = 0
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        agent_pos = positions.get('agent')
        door_pos = positions.get('door')
        ball_pos = find_ball_position(self.obs, 'green')

        if not ball_pos:
            drop_pos = find_safe_drop_location(grid[:,:,0], door_pos, agent_pos, agent_dir)
            if drop_pos:
                self.action_sequence = align_direction_to_target(agent_pos, drop_pos, agent_dir)
                self.action_sequence.append(ACTION_TO_IDX["drop"])         
            else:
                self.action_sequence.append(ACTION_TO_IDX["move_forward"])    
                self.action_sequence.append(ACTION_TO_IDX["drop"])
                if agent_dir == 1:
                    agent_pos = (agent_pos[0]+1, agent_pos[1])
                    drop_pos = (agent_pos[0]+1, agent_pos[1])
                else:
                    agent_pos = (agent_pos[0]-1, agent_pos[1])
                    drop_pos = (agent_pos[0]-1, agent_pos[1])
            
    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                return py_trees.common.Status.FAILURE
            
class PickUpKey(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, debug=False):
        super(PickUpKey, self).__init__(name)
        self.env = env
        self.obs = obs
        self.action_sequence = []
        self.current_action_index = 0
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        key_pos = positions.get('key')
        agent_pos = positions.get('agent')

        agent_neighbors = [
            (agent_pos[0], agent_pos[1] - 1), 
            (agent_pos[0] + 1, agent_pos[1]), 
            (agent_pos[0], agent_pos[1] + 1), 
            (agent_pos[0] - 1, agent_pos[1]), 
        ]
        if key_pos:
            if key_pos in agent_neighbors:
                self.action_sequence += align_direction_to_target(agent_pos, key_pos, agent_dir)
                self.action_sequence.append(ACTION_TO_IDX["pick_up"]) 
            else:
                path_to_key = find_path_to_adjacent(grid[:,:,0], agent_pos, key_pos, agent_dir)
                if path_to_key:
                    self.action_sequence += prepare_action_sequence(path_to_key, 0, key_pos)
                    self.action_sequence.append(ACTION_TO_IDX["pick_up"]) 
                else:
                    self.feedback_message = "Cannot determine path."
                    self.action_sequence = []
        else:
            self.feedback_message = "Cannot determine key position."

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                return py_trees.common.Status.FAILURE

class PickUpKeyBox(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, debug=False):
        super(PickUpKeyBox, self).__init__(name)
        self.env = env
        self.obs = obs
        self.action_sequence = []
        self.current_action_index = 0
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        box_pos = positions.get('box')
        agent_pos = positions.get('agent')
        ball_pos = find_ball_position(self.obs, 'green')
        door_pos = positions.get('door')

        agent_neighbors = [
            (agent_pos[0], agent_pos[1] - 1), 
            (agent_pos[0] + 1, agent_pos[1]), 
            (agent_pos[0], agent_pos[1] + 1), 
            (agent_pos[0] - 1, agent_pos[1]), 
        ]
        if box_pos:
            if box_pos in agent_neighbors:
                self.feedback_message = "Already a neighbor, maybe needs to adjust direction."
                self.action_sequence += align_direction_to_target(agent_pos, box_pos, agent_dir)
                self.action_sequence.append(ACTION_TO_IDX["open"]) 
                self.action_sequence.append(ACTION_TO_IDX["pick_up"]) 
            else:
                path_to_box = find_path_to_adjacent(grid[:,:,0], agent_pos, box_pos, agent_dir)
                if path_to_box:
                    self.action_sequence += prepare_action_sequence(path_to_box, agent_dir, box_pos)
                    self.action_sequence.append(ACTION_TO_IDX["open"]) 
                    self.action_sequence.append(ACTION_TO_IDX["pick_up"]) 
                else:
                    self.feedback_message = "Cannot determine path."
                    self.action_sequence = []
        else:
            self.feedback_message = "Cannot determine door or ball position."

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                return py_trees.common.Status.FAILURE

class DropObstacle(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, object_type="ball", object_color="green", debug=False):
        super(DropObstacle, self).__init__(name)
        self.env = env
        self.obs = obs
        self.goal_object = object_type
        self.goal_color = object_color
        self.path_to_goal = []
        self.path_to_drop = []
        self.action_sequence = []
        self.current_action_index = 0
        self.debug = debug

    def initialise(self):
        if self.debug:
            print(f"Initializing {self.name}")
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        agent_pos = positions.get('agent')
        door_pos = positions.get('door')

        if self.goal_object == "ball":
            goal_pos = find_ball_position(self.obs, self.goal_color)
        else:
            goal_pos = positions.get(self.goal_object)

        if not goal_pos:
            drop_pos = find_safe_drop_location(grid[:, :, 0], door_pos, agent_pos, agent_dir)
            if drop_pos:
                self.action_sequence = align_direction_to_target(agent_pos, drop_pos, agent_dir)
                self.action_sequence.append(ACTION_TO_IDX["drop"])
            else:
                self.action_sequence.append(ACTION_TO_IDX["move_forward"])
                self.action_sequence.append(ACTION_TO_IDX["drop"])
                if agent_dir == 1:
                    agent_pos = (agent_pos[0] + 1, agent_pos[1])
                    drop_pos = (agent_pos[0] + 1, agent_pos[1])
                else:
                    agent_pos = (agent_pos[0] - 1, agent_pos[1])
                    drop_pos = (agent_pos[0] - 1, agent_pos[1])

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.debug:
            print(f"{self.name} is active.")
            print(self.action_sequence)
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            return py_trees.common.Status.RUNNING
        else:
            if self.current_action_index == len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            else:
                self.current_action_index = 0
                return py_trees.common.Status.FAILURE
