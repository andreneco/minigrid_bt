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
    """
    Extracts the grid and direction from the observation.

    Args:
        obs (dict or tuple): The observation containing the grid image and direction.

    Returns:
        tuple: A tuple containing the grid image and direction.

    Raises:
        ValueError: If the observation structure is unexpected.
    """
    if isinstance(obs, dict) and 'image' in obs and 'direction' in obs:
        return obs['image'], obs['direction']
    elif isinstance(obs, tuple) and 'image' in obs[0] and 'direction' in obs[0]:
        return obs[0]['image'], obs[0]['direction']
    else:
        raise ValueError("Unexpected observation structure")

def extract_positions(obs):
    """
    Extracts the positions of all objects from the observation.

    Args:
        obs (dict or tuple): The observation containing the grid image.

    Returns:
        dict: A dictionary with object names as keys and their positions as values.
    """
    image = obs['image'] if 'image' in obs else obs[0]['image']
    positions = {}
    for object_name, object_idx in OBJECT_TO_IDX.items():
        mask = image[:, :, 0] == object_idx
        y, x = np.where(mask)
        if y.size > 0:
            positions[object_name] = (x[0], y[0])
    return positions

def extract_multiple_positions(obs):
    """
    Extracts the positions of all instances of each object from the observation.

    Args:
        obs (dict or tuple): The observation containing the grid image.

    Returns:
        dict: A dictionary with object names as keys and lists of positions as values.
    """
    image = obs['image'] if 'image' in obs else obs[0]['image']
    positions = {}
    for object_name, object_idx in OBJECT_TO_IDX.items():
        mask = image[:, :, 0] == object_idx
        y, x = np.where(mask)
        if y.size > 0:
            positions[object_name] = list(zip(x, y))  # Store all positions as list of tuples
    return positions

def find_ball_position_any_color(obs):
    """
    Finds the position of any ball in the observation, regardless of color.

    Args:
        obs (dict or tuple): The observation containing the grid image.

    Returns:
        tuple or None: The position of the ball, or None if no ball is found.
    """
    positions = extract_positions(obs)
    for key in positions:
        if key.startswith('ball'):
            return positions[key]
    return None

def align_direction_to_target(current_position, target_position, current_direction):
    """
    Aligns the agent's direction to face the target position.

    Args:
        current_position (tuple): The agent's current position.
        target_position (tuple): The target position to face.
        current_direction (int): The agent's current direction.

    Returns:
        list: A list of actions required to align the direction.

    Raises:
        ValueError: If the move vector is invalid.
    """
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
    """
    Calculates the number of turns needed to face the target direction.

    Args:
        current_direction (int): The agent's current direction.
        target_direction (int): The target direction.

    Returns:
        int: The number of turns needed to face the target direction.
    """
    if current_direction is None or target_direction is None:
        return 0
    direction_diff = np.int8(current_direction) - np.int8(target_direction)
    normalized_diff = direction_diff % 4
    return normalized_diff

def find_path_to_adjacent(grid, start, target, current_direction):
    """
    Finds a path to a position adjacent to the target using A* pathfinding.

    Args:
        grid (ndarray): The grid representing the environment.
        start (tuple): The starting position.
        target (tuple): The target position.
        current_direction (int): The agent's current direction.

    Returns:
        list: A list of positions representing the path.
    """
    path = astar_pathfinding(grid, start, target, current_direction)
    if path:
        return path
    return []

def find_path_to_adjacent_doors(grid, start, target, current_direction):
    """
    Finds a path to a position adjacent to the target, considering doors, using A* pathfinding.

    Args:
        grid (ndarray): The grid representing the environment.
        start (tuple): The starting position.
        target (tuple): The target position.
        current_direction (int): The agent's current direction.

    Returns:
        list: A list of positions representing the path.
    """
    path = astar_pathfinding_doors(grid, start, target, current_direction)
    if path:
        return path
    return []

def find_door_position(obs, color):
    """
    Finds the position of a door with a specific color in the observation.

    Args:
        obs (dict or tuple): The observation containing the grid image.
        color (str): The color of the door to find.

    Returns:
        tuple or None: The position of the door, or None if no door is found.
    """
    positions = extract_multiple_positions(obs)
    color_idx = COLOR_TO_IDX[color]
    image = obs['image'] if 'image' in obs else obs[0]['image']
    
    for key, values in positions.items():
        if key == 'door':
            for pos in values:
                x, y = pos
                if image[y, x, 1] == color_idx:  # Check the color channel
                    return (np.int64(x), np.int64(y))
    return None

def astar_pathfinding(grid, start, goal, initial_direction):
    """
    A* pathfinding algorithm to find a path to a position adjacent to the goal.

    Args:
        grid (ndarray): The grid representing the environment.
        start (tuple): The starting position.
        goal (tuple): The goal position.
        initial_direction (int): The agent's initial direction.

    Returns:
        list: A list of positions representing the path, or an empty list if no path is found.
    """
    def heuristic(a, b, direction):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + (direction != initial_direction)

    def goal_reached(current, goal):
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            adj = (goal[0] + dx, goal[1] + dy)
            if current == adj and grid[adj[1]][adj[0]] == 1:
                return True
        return False

    def get_neighbors(node, direction):
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for i, (dx, dy) in enumerate(directions):
            x, y = node[0] + dx, node[1] + dy
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] and grid[y][x] == 1:
                cost = 1 if i == direction else 2
                neighbors.append(((x, y), i, cost))
        
        return neighbors

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal, initial_direction), 0, start, initial_direction))
    came_from = {}
    gscore = {start: 0}

    while open_set:
        _, g, current, direction = heapq.heappop(open_set)
        if goal_reached(current, goal):
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

    return []

def astar_pathfinding_doors(grid, start, goal, initial_direction):
    """
    A* pathfinding algorithm to find a path to a position adjacent to the goal, considering doors.

    Args:
        grid (ndarray): The grid representing the environment.
        start (tuple): The starting position.
        goal (tuple): The goal position.
        initial_direction (int): The agent's initial direction.

    Returns:
        list: A list of positions representing the path, or an empty list if no path is found.
    """
    def heuristic(a, b, direction):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + (direction != initial_direction)

    def goal_reached(current, goal):
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            adj = (goal[0] + dx, goal[1] + dy)
            if current[:2] == adj and (grid[adj[1], adj[0], 0] == 1 or (grid[adj[1], adj[0], 0] == 4 and grid[adj[1], adj[0], 2] in [0, 1])):
                return True
        return False

    def get_neighbors(node, direction):
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for i, (dx, dy) in enumerate(directions):
            x, y = node[0] + dx, node[1] + dy
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                if grid[y, x, 0] == 1 or (grid[y, x, 0] == 4 and grid[y, x, 2] in [0, 1]):
                    cost = 1 if i == direction else 2
                    is_door = grid[y, x, 0] == 4
                    door_state = grid[y, x, 2] if is_door else None
                    neighbors.append(((x, y, is_door, door_state), i, cost))
        
        return neighbors

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal, initial_direction), 0, (start[0], start[1], False, None), initial_direction))
    came_from = {}
    gscore = {(start[0], start[1], False, None): 0}

    while open_set:
        _, g, current, direction = heapq.heappop(open_set)
        if goal_reached(current, goal):
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

    return []

def astar_pathfinding_to_goal(grid, start, goal, current_direction):
    """
    A* pathfinding algorithm to find a path to the goal.

    Args:
        grid (ndarray): The grid representing the environment.
        start (tuple): The starting position.
        goal (tuple): The goal position.
        current_direction (int): The agent's current direction.

    Returns:
        list: A list of positions representing the path, or an empty list if no path is found.
    """
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

def find_any_ball_position(obs):
    """
    Finds the position of any ball in the observation.

    Args:
        obs (dict or tuple): The observation containing the grid image.

    Returns:
        tuple or None: The position of the ball, or None if no ball is found.
    """
    positions = extract_positions(obs)
    return positions.get('ball')

def extract_door_positions(obs):
    """
    Extracts the positions of all doors in the observation.

    Args:
        obs (dict or tuple): The observation containing the grid image.

    Returns:
        list: A list of tuples representing the positions of doors along with their color and state.
    """
    image = obs['image'] if 'image' in obs else obs[0]['image']
    door_positions = []
    door_idx = OBJECT_TO_IDX['door']
    mask = image[:, :, 0] == door_idx
    y, x = np.where(mask)
    if y.size > 0:
        door_positions = list(zip(x, y, image[y, x, 1], image[y, x, 2]))  # Store all positions as list of tuples with color and state
    return door_positions

def find_locked_door_position(obs):
    """
    Finds the position of a locked door in the observation.

    Args:
        obs (dict or tuple): The observation containing the grid image.

    Returns:
        tuple or None: The position of the locked door, or None if no locked door is found.
    """
    door_positions = extract_door_positions(obs)
    for pos in door_positions:
        x, y, door_color, door_state = pos
        if door_state == 2:  # Check if door is locked
            return (x, y, door_color, door_state)
    return None

def prepare_action_sequence(path, current_direction, goal_pos):
    """
    Prepares a sequence of actions to follow the given path to the goal position.

    Args:
        path (list): The path to follow.
        current_direction (int): The agent's current direction.
        goal_pos (tuple): The goal position.

    Returns:
        list: A list of actions to follow the path and align with the goal.
    """
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

def prepare_action_sequence_doors(path, current_direction, goal_pos):
    """
    Prepares a sequence of actions to follow the given path to the goal position, considering doors.

    Args:
        path (list): The path to follow.
        current_direction (int): The agent's current direction.
        goal_pos (tuple): The goal position.

    Returns:
        list: A list of actions to follow the path, open doors, and align with the goal.
    """
    action_sequence = []
    if path:
        for start, end in zip(path[:-1], path[1:]):
            
            # Check if end position is a door and add open action if needed
            if end[2]:
                dir_correction = align_direction_to_target((start[0], start[1]), (end[0], end[1]), current_direction)
                if dir_correction:
                    action_sequence.extend(dir_correction)
                    current_direction = direction_map.get((end[0] - start[0], end[1] - start[1]))
                action_sequence.append(ACTION_TO_IDX["open"])

            actions = calculate_actions_to_move((start[0], start[1]), (end[0], end[1]), current_direction)
            action_sequence.extend(actions)
            move_vector = (end[0] - start[0], end[1] - start[1])
            current_direction = direction_map.get(move_vector)
                
        dir_correction = align_direction_to_target((path[-1][0], path[-1][1]), goal_pos, current_direction)
        if dir_correction:
            action_sequence += dir_correction
    return action_sequence

def calculate_actions_to_move(start, end, current_direction):
    """
    Calculates the actions needed to move from the start position to the end position.

    Args:
        start (tuple): The starting position.
        end (tuple): The ending position.
        current_direction (int): The agent's current direction.

    Returns:
        list: A list of actions needed to move to the end position.

    Raises:
        ValueError: If start or end positions are invalid.
    """
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
    """
    Finds the position of a ball of a specific color in the observation.

    Args:
        obs (dict or tuple): The observation containing the grid image.
        color (str): The color of the ball to find.

    Returns:
        tuple or None: The position of the ball, or None if no ball is found.
    """
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
    """
    Finds a safe location to drop an item near the door.

    Args:
        grid (ndarray): The grid representing the environment.
        door_pos (tuple): The position of the door.
        agent_pos (tuple): The agent's current position.
        agent_dir (int): The agent's current direction.

    Returns:
        tuple or None: The safe drop location, or None if no safe location is found.
    """
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

def update_tree_obs(tree, new_obs):
    """
    Updates the observation in each node of the behavior tree.

    Args:
        tree (Tree): The behavior tree.
        new_obs (dict): The new observation to update in the tree.
    """
    for node in tree.root.iterate():
        if hasattr(node, 'update_obs'):
            node.update_obs(new_obs)

class ExtendedFlatObsWrapper(ObservationWrapper):
    """
    A wrapper that converts the observation to a flat representation including the direction.
    """
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
        """
        Converts the observation to a flat array including the direction.

        Args:
            obs (dict): The original observation.

        Returns:
            ndarray: The flattened observation with the direction.
        """
        image = obs["image"].flatten()
        direction = np.array([obs["direction"]], dtype="uint8")
        return np.concatenate((image, direction))

class ReconstructObsWrapper:
    """
    A class to reconstruct the original observation from the flat representation.
    """
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.img_size = reduce(operator.mul, img_shape, 1)

    def reconstruct_observation(self, flat_obs):
        """
        Reconstructs the original observation from the flat representation.

        Args:
            flat_obs (ndarray): The flattened observation.

        Returns:
            dict: The reconstructed observation with the image and direction.
        """
        img_flat = flat_obs[:-1]
        direction = flat_obs[-1]
        image = img_flat.reshape(self.img_shape).astype(np.float32)
        return {'image': image, 'direction': direction}