# minigrid_bt/utils.py

import numpy as np
import operator
from functools import reduce
import heapq
from gymnasium import ObservationWrapper, spaces

# Constants from the MiniGrid environment
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

# Global dictionary to store actions decided by the behaviour tree
current_action = {'action': None}

# Global map for encoding direction definitions from the environment
direction_map = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}

# Global map for encoding action definitions from the environment
ACTION_TO_IDX = {"turn_left": 0, "turn_right": 1, "move_forward": 2, "pick_up": 3, "drop": 4, "open": 5}

# Utility Functions
def extract_grid_and_direction(obs):
    if isinstance(obs, dict) and 'image' in obs and 'direction' in obs:
        return obs['image'], obs['direction']
    elif isinstance(obs, tuple) and 'image' in obs[0] and 'direction' in obs[0]:
        return obs[0]['image'], obs[0]['direction']
    else:
        raise ValueError("Unexpected observation structure")

def extract_positions(obs):
    image = obs['image'] if 'image' in obs else obs[0]['image']
    positions = {}
    for object_name, object_idx in OBJECT_TO_IDX.items():
        mask = image[:, :, 0] == object_idx
        y, x = np.where(mask)
        if y.size > 0:
            positions[object_name] = (x[0], y[0])
    return positions

def align_direction_to_target(current_position, target_position, current_direction):
    goal_vector = (target_position[0] - current_position[0], target_position[1] - current_position[1])

    if goal_vector == (0, 0):
        return []  # No need to align if already at the goal position

    goal_direction = direction_map.get(goal_vector)

    if goal_direction is None:
        raise ValueError(f"Invalid move vector {goal_vector}. Cannot determine direction from start={current_position} to end={target_position}.")

    # Calculate turns needed to face the target direction
    turns_needed = calculate_turn_direction(current_direction, goal_direction)
    action_sequence = []
    if turns_needed == 1:
        action_sequence.append(ACTION_TO_IDX["turn_left"])
    elif turns_needed == 2:
        action_sequence.extend([ACTION_TO_IDX["turn_left"], ACTION_TO_IDX["turn_left"]])
    elif turns_needed == 3:
        action_sequence.append(ACTION_TO_IDX["turn_right"])

    return action_sequence

def calculate_turn_direction(current_direction, target_direction):
    if current_direction is None or target_direction is None:
        return 0
    direction_diff = np.int8(current_direction) - np.int8(target_direction)
    normalized_diff = direction_diff % 4
    return normalized_diff

def find_path_to_adjacent(grid, start, target):
    path = astar_pathfinding(grid, start, target)
    if path:
        return path
    return []

def astar_pathfinding(grid, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def goal_reached(current, goal):
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            adj = (goal[0] + dx, goal[1] + dy)
            if current == adj and grid[adj[1]][adj[0]] == 1:
                return True
        return False

    def get_neighbors(node):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            x, y = node[0] + dx, node[1] + dy
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] and grid[y][x] == 1:
                neighbors.append((x, y))
        return neighbors

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    gscore = {start: 0}

    while open_set:
        _, g, current = heapq.heappop(open_set)
        if goal_reached(current, goal):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor in get_neighbors(current):
            tentative_g_score = g + 1
            if neighbor not in gscore or tentative_g_score < gscore[neighbor]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

    return []

def astar_pathfinding_to_goal(grid, start, goal, current_direction):
    def heuristic(a, b, direction):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + (direction != current_direction)

    def get_neighbors(node, direction):
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for i, (dx, dy) in enumerate(directions):
            x, y = node[0] + dx, node[1] + dy
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] and (grid[y][x] == 1 or grid[y][x] == 8):
                cost = 1 if i == direction else 2
                neighbors.append(((x, y), i, cost))
        
        return neighbors

    if grid[goal[1]][goal[0]] != 1 and grid[goal[1]][goal[0]] != 8:
        print("Goal cell is not navigable.")
        return []

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal, current_direction), 0, start, current_direction))
    came_from = {}
    gscore = {start: 0}

    while open_set:
        _, g, current, direction = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current][0]
                path.append(current)
            path.reverse()
            return path

        for neighbor, new_direction, cost in get_neighbors(current, direction):
            tentative_g_score = g + cost
            if neighbor not in gscore or tentative_g_score < gscore[neighbor]:
                came_from[neighbor] = (current, new_direction)
                gscore[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal, new_direction)
                heapq.heappush(open_set, (f_score, tentative_g_score, neighbor, new_direction))

    print("No path found")  # Debug statement
    return []

def prepare_action_sequence(path, current_direction, goal_pos):
    action_sequence = []
    if path:
        for start, end in zip(path[:-1], path[1:]):
            actions = calculate_actions_to_move(start, end, current_direction)
            action_sequence.extend(actions)
            move_vector = (end[0] - start[0], end[1] - start[1])
            current_direction = direction_map.get(move_vector)
        dir_correction = align_direction_to_target(path[-1], goal_pos, current_direction)
        if dir_correction:
            action_sequence += dir_correction
    return action_sequence

def calculate_actions_to_move(start, end, current_direction):
    if not (isinstance(start, tuple) and isinstance(end, tuple) and len(start) == 2 and len(end) == 2):
        raise ValueError(f"Start and end must be tuples of two integers. Received start={start}, end={end}.")
    actions = []
    move_vector = (end[0] - start[0], end[1] - start[1])
    target_direction = direction_map.get(move_vector)
    if target_direction is None:
        raise ValueError(f"Invalid move vector {move_vector}. Cannot determine direction from start={start} to end={end}.")
    turns_needed = calculate_turn_direction(current_direction, target_direction)
    turn_action = "turn_left" if turns_needed <= 2 else "turn_right"
    turns_needed = 1 if turn_action == "turn_right" else turns_needed
    for _ in range(abs(turns_needed)):
        actions.append(ACTION_TO_IDX[turn_action])
    actions.append(ACTION_TO_IDX["move_forward"])
    return actions

def find_ball_position(obs, color):
    grid, agent_dir = extract_grid_and_direction(obs)
    object_layer = grid[:,:,0]
    color_layer = grid[:,:,1]
    ball_idx = OBJECT_TO_IDX['ball']
    green_idx = COLOR_TO_IDX[color]
    positions = np.where((object_layer == ball_idx) & (color_layer == green_idx))
    if len(positions[0]) > 0:
        return (positions[1][0], positions[0][0])
    return None

def find_safe_drop_location(grid, door_pos, agent_pos, agent_dir):
    if agent_dir % 2 == 0:
        prioritized_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        prioritized_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    door_adjacent_cells = [(door_pos[0] + dx, door_pos[1] + dy) for dx, dy in prioritized_offsets]
    for offset in prioritized_offsets:
        nx, ny = agent_pos[0] + offset[0], agent_pos[1] + offset[1]
        if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
            if (nx, ny) not in door_adjacent_cells and grid[ny, nx] == OBJECT_TO_IDX['empty']:
                return (nx, ny)
    return None

class ExtendedFlatObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + 1,),  # +1 for the direction
            dtype="uint8",
        )

    def observation(self, obs):
        image = obs["image"].flatten()
        direction = np.array([obs["direction"]], dtype="uint8")
        return np.concatenate((image, direction))

class ReconstructObsWrapper:
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.img_size = reduce(operator.mul, img_shape, 1)

    def reconstruct_observation(self, flat_obs):
        img_flat = flat_obs[:-1]
        direction = flat_obs[-1]
        image = img_flat.reshape(self.img_shape).astype(np.float32)
        return {'image': image, 'direction': direction}

def update_tree_obs(tree, new_obs):
    for node in tree.root.iterate():
        if hasattr(node, 'update_obs'):
            node.update_obs(new_obs)
