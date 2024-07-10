# minigrid_bt/behaviors.py

import py_trees
import numpy as np
from minigrid_bt.utils import (
    extract_grid_and_direction, extract_positions,
    align_direction_to_target, find_path_to_adjacent,
    prepare_action_sequence, find_ball_position, find_safe_drop_location,
    astar_pathfinding,
    astar_pathfinding_to_goal, ACTION_TO_IDX, direction_map, current_action
)

class GoToGoal(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, goal_object="goal"):
        super(GoToGoal, self).__init__(name)
        self.env = env
        self.obs = obs
        self.goal_object = goal_object
        self.action_sequence = []
        self.current_action_index = 0

    def initialise(self):
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
        if 0 <= self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            if self.current_action_index >= len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING
        else:
            self.current_action_index = 0
            return py_trees.common.Status.FAILURE

class PickUpGoal(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs):
        super(PickUpGoal, self).__init__(name)
        self.env = env
        self.obs = obs
        self.action_sequence = []
        self.current_action_index = 0

    def initialise(self):
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        agent_pos = positions.get('agent')
        ball_pos = find_ball_position(self.obs, 'blue')
        key_pos = positions.get('key')

        if not key_pos:
            if ball_pos == (agent_pos[0], agent_pos[1]+1):
                self.action_sequence = [1, 1, 4, 0, 0, 3]
            else:
                self.action_sequence.append(ACTION_TO_IDX["move_forward"])
                if ball_pos == (agent_pos[0], agent_pos[1]+2):
                    if grid[agent_pos[0] + 1, agent_pos[1] - 1, 0] == 1:
                        self.action_sequence += [1, 4, 0, 3]
                    else:
                        self.drop_pos = (agent_pos[0] + 1, agent_pos[1] + 1)
                        self.action_sequence += [0, 4, 1, 3]
            self.action_sequence.append(ACTION_TO_IDX["drop"])

            agent_neighbors = [
                (agent_pos[0], agent_pos[1]), 
                (agent_pos[0] + 1, agent_pos[1]+1), 
                (agent_pos[0], agent_pos[1] + 2), 
                (agent_pos[0] - 1, agent_pos[1]+1), 
            ]
            if ball_pos:
                if ball_pos in agent_neighbors:
                    self.feedback_message = "Already a neighbor, maybe needs to adjust direction."
                    self.action_sequence += align_direction_to_target((agent_pos[0], agent_pos[1]+1), ball_pos, 0)
                    self.action_sequence.append(ACTION_TO_IDX["pick_up"])

                grid[:,:,0][agent_pos[1] + 2, agent_pos[0]] = 5
                grid[:,:,0][agent_pos[1] + 1, agent_pos[0]] = 10
                grid[:,:,0][agent_pos[1], agent_pos[0]] = 4
                path_to_goal = find_path_to_adjacent(grid[:,:,0], (agent_pos[0], agent_pos[1]+1), ball_pos)
                if path_to_goal:
                    self.action_sequence += prepare_action_sequence(path_to_goal, 0, ball_pos)
                    self.action_sequence.append(ACTION_TO_IDX["pick_up"])
                else:
                    self.feedback_message = "Cannot determine path."
        else:
            self.feedback_message = "Cannot determine door or ball position."

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if 0 <= self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            if self.current_action_index >= len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING
        else:
            self.current_action_index = 0
            return py_trees.common.Status.FAILURE

class EnterRoom(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs):
        super(EnterRoom, self).__init__(name)
        self.env = env
        self.obs = obs
        self.action_sequence = []
        self.current_action_index = 0

    def initialise(self):
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
                path_to_door = find_path_to_adjacent(grid[:,:,0], agent_pos, door_pos)
                if path_to_door:
                    self.action_sequence = prepare_action_sequence(path_to_door, agent_dir, door_pos)
                    self.action_sequence.append(ACTION_TO_IDX["open"])
                    self.action_sequence.append(ACTION_TO_IDX["move_forward"])
                else:
                    self.feedback_message = "Cannot determine path."
                    self.action_sequence = []
        else:
            self.feedback_message = "Cannot determine door position."
            
    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        global current_action
        if self.current_action_index <= len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            if self.current_action_index > len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING
        else:
            self.current_action_index = 0
            return py_trees.common.Status.FAILURE

class PickUpKey(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs):
        super(PickUpKey, self).__init__(name)
        self.env = env
        self.obs = obs
        self.action_sequence = []
        self.current_action_index = 0

    def initialise(self):
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
                path_to_box = find_path_to_adjacent(grid[:,:,0], agent_pos, box_pos)
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
        if self.current_action_index <= len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            if self.current_action_index > len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING
        else:
            self.current_action_index = 0
            return py_trees.common.Status.FAILURE

class ClearPath(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs):
        super(ClearPath, self).__init__(name)
        self.env = env
        self.obs = obs
        self.path_to_ball = []
        self.path_to_drop = []
        self.action_sequence = []
        self.current_action_index = 0

    def initialise(self):
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        agent_pos = positions.get('agent')
        ball_pos = find_ball_position(self.obs, 'green')

        agent_neighbors = [
            (agent_pos[0], agent_pos[1] - 1), 
            (agent_pos[0] + 1, agent_pos[1]), 
            (agent_pos[0], agent_pos[1] + 1), 
            (agent_pos[0] - 1, agent_pos[1]), 
        ]
        if ball_pos:
            if ball_pos in agent_neighbors:
                self.feedback_message = "Already a neighbor, maybe needs to adjust direction."
                self.action_sequence = align_direction_to_target(agent_pos, ball_pos, agent_dir)
                self.action_sequence.append(ACTION_TO_IDX["pick_up"]) 
            else:
                path_to_ball = find_path_to_adjacent(grid[:,:,0], agent_pos, ball_pos)
                if path_to_ball:
                    self.action_sequence = prepare_action_sequence(path_to_ball, agent_dir, find_ball_position(self.obs, 'green'))
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
        if self.current_action_index <= len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            if self.current_action_index > len(self.action_sequence):
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING
        else:
            self.current_action_index = 0
            return py_trees.common.Status.FAILURE
