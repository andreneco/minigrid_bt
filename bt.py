import py_trees
import gymnasium as gym
import numpy as np
import heapq
import functools

# Constants from the MiniGrid environment
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

# Wrappers from the MiniGrid environment
from minigrid.wrappers import FullyObsWrapper

# Global dictionary to store actions decided by the behaviour tree
current_action = {'action': None}

# Global map for encoding direction definitions from the environment
direction_map = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}

# Global map for encoding action definitions from the environment
ACTION_TO_IDX = {"turn_left": 0, "turn_right": 1, "move_forward": 2, "pick_up": 3, "drop": 4, "open": 5}

# Utility Functions
def extract_grid_and_direction(obs):
    """
    Extracts the grid and agent's direction from the observation.

    Parameters:
    - obs: The observation which can be a dict with 'image' and 'direction' keys or a tuple where the first element is a dict.

    Returns:
    - tuple: A tuple containing the grid and the agent's direction.
    """
    if isinstance(obs, dict) and 'image' in obs and 'direction' in obs:
        return obs['image'], obs['direction']
    elif isinstance(obs, tuple) and 'image' in obs[0] and 'direction' in obs[0]:
        return obs[0]['image'], obs[0]['direction']
    else:
        raise ValueError("Unexpected observation structure")

def extract_positions(obs):
    """
    Extracts the positions of all recognizable objects within the observation grid.

    Parameters:
    - obs: The current observation, which includes the 'image' field with grid information.

    Returns:
    - dict: A dictionary mapping object names to their (x, y) positions within the grid.
    """
    image = obs['image'] if 'image' in obs else obs[0]['image']
    positions = {}
    for object_name, object_idx in OBJECT_TO_IDX.items():
        y, x = np.where(image[:, :, 0] == object_idx)
        if y.size > 0:
            positions[object_name] = (x[0], y[0])

    return positions

def align_direction_to_target(current_position, target_position, current_direction):
    """
    Determines the sequence of turn actions required for an agent to face a target from its current position and direction.

    Parameters:
    - current_position (tuple): The current (x, y) position of the agent.
    - target_position (tuple): The (x, y) position of the target.
    - current_direction (int): The current facing direction of the agent.

    Returns:
    - list: A sequence of actions (as integers) to align the agent's direction towards the target.
    """
    goal_vector = (target_position[0] - current_position[0], target_position[1] - current_position[1])            
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

def find_path_to_adjacent(grid, start, target):
    """
    Finds a path to a cell adjacent to the target position that is navigable.

    Parameters:
    - grid (array): The grid representation of the environment.
    - start (tuple): The starting (x, y) position.
    - target (tuple): The target (x, y) position.

    Returns:
    - list of tuples: A path as a list of (x, y) positions, or an empty list if no path is found.
    """
    # Adjust the pathfinding goal to be a passable cell adjacent to the target
    # if the target itself is non-passable.
    path = astar_pathfinding(grid, start, target)
    if path:
        return path  # Return the first successful path found

    return []  # Return an empty path if no adjacent passable cell is found or reachable

def prepare_action_sequence(path, current_direction, goal_pos):
    """
    Generates a sequence of actions to navigate from the current position to a goal position
    along a specified path, and performs an action upon reaching the goal.

    The function calculates the necessary turns and moves to follow the path,
    updating the agent's direction as it proceeds. Once at the goal position,
    it performs a final alignment if necessary.

    Parameters:
    - path (list of tuples): The path to follow as a list of (x, y) positions. 
    - current_direction (int): The current direction the agent is facing.
    - goal_pos (tuple): The goal position where an action needs to be performed.

    Returns:
    - list: A sequence of actions (as integers) to reach the goal and perform the final action.
    """
    action_sequence = []

    if path:
        # Iterate through the path to generate a sequence of actions for each step
        for start, end in zip(path[:-1], path[1:]):
            # Calculate the actions required to move from start to end, including any necessary turns
            actions = calculate_actions_to_move(start, end, current_direction)
            # Add these actions to the overall action sequence
            action_sequence.extend(actions)
            
            # Update the agent's direction based on the movement from start to end
            # This assumes the final action in 'actions' correctly aligns the agent's direction towards 'end'
            move_vector = (end[0] - start[0], end[1] - start[1])
            current_direction = direction_map.get(move_vector)

        # Once at the final position before the goal, adjust the agent's direction to face the goal if needed
        # This is necessary for actions like 'pick up' which require the agent to be facing the object
        dir_correction = align_direction_to_target(path[-1], goal_pos, current_direction)
        # If direction adjustment is needed, append the corresponding turn actions
        if dir_correction:
            action_sequence += dir_correction

    # Return the complete sequence of actions including any final action required at the goal
    return action_sequence

def find_safe_drop_location(grid, door_pos, agent_pos, agent_dir):
    """
    Identifies a safe location to drop an object within the grid, avoiding the door's adjacent cells.

    Parameters:
    - grid (array): The grid representation of the environment.
    - door_pos (tuple): The (x, y) position of the door.
    - agent_pos (tuple): The current (x, y) position of the agent.
    - agent_dir (int): The current facing direction of the agent.

    Returns:
    - tuple or None: A (x, y) position that is safe for dropping the object, or None if no suitable location is found.
    """
    # Prioritize directions based on the agent's current direction to minimize turns
    if agent_dir % 2 == 0:
        prioritized_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up and down prioritized
    else:
        prioritized_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)] # Left and right prioritized
    
    door_adjacent_cells = [(door_pos[0] + dx, door_pos[1] + dy) for dx, dy in prioritized_offsets]
    for offset in prioritized_offsets:
        nx, ny = agent_pos[0] + offset[0], agent_pos[1] + offset[1]
        if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:  # Check grid bounds
            if (nx, ny) not in door_adjacent_cells and grid[ny, nx] == OBJECT_TO_IDX['empty']:
                # Found a suitable empty cell not adjacent to the door
                return (nx, ny)
    
    return None  # No suitable location found

def calculate_turn_direction(current_direction, target_direction):
    """
    Calculates the number of turns required for the agent to face a specific direction.

    Parameters:
    - current_direction (int): The agent's current direction.
    - target_direction (int): The desired direction to face.

    Returns:
    - int: The minimum number of turns needed to face the target direction.
    """    
    if current_direction is None or target_direction is None:
        # Handle the error, for example, by returning 0 turns needed or another default value
        return 0

    # Calculate the difference in directions
    direction_diff = current_direction - target_direction
    # Normalize the difference within the range [0, 4)
    normalized_diff = direction_diff % 4

    return normalized_diff

def calculate_actions_to_move(start, end, current_direction):
    """
    Calculates the optimal sequence of actions required for an agent to navigate from a starting position
    to a target ending position, taking into account the agent's current orientation.

    This function first determines the direction the agent must face to move directly towards the target position.
    It then calculates the minimum number of turns required to face this direction and appends the corresponding
    turn actions to the action sequence. Once the agent is correctly oriented, a move forward action is added to
    the sequence to simulate movement towards the target.

    Parameters:
    - start (tuple): The starting (x, y) coordinates of the agent.
    - end (tuple): The target (x, y) coordinates to reach.
    - current_direction (int): The current orientation of the agent, where 0 represents up, 1 is right, 2 is down, and 3 is left.

    Returns:
    - list: A sequence of action IDs (as integers) that the agent should execute to move from start to end. This includes any necessary turns followed by a single move forward action.

    Raises:
    - ValueError: If the start or end positions are not valid tuples of two integers, or if the calculated move vector does not correspond to a valid direction in `direction_map`.
    """
    # Validate input coordinates
    if not (isinstance(start, tuple) and isinstance(end, tuple) and len(start) == 2 and len(end) == 2):
        raise ValueError(f"Start and end must be tuples of two integers. Received start={start}, end={end}.")

    actions = []
    move_vector = (end[0] - start[0], end[1] - start[1])

    # Determine the direction the agent must face to proceed directly towards the target
    target_direction = direction_map.get(move_vector)
    if target_direction is None:
        raise ValueError(f"Invalid move vector {move_vector}. Cannot determine direction from start={start} to end={end}.")

    # Calculate the minimal number of turns needed to face the target direction
    turns_needed = calculate_turn_direction(current_direction, target_direction)

    # Choose the most efficient turn action based on the number of turns needed
    turn_action = "turn_left" if turns_needed <= 2 else "turn_right"
    # Adjust turns_needed for right turn to ensure the loop below correctly interprets the action
    turns_needed = 1 if turn_action == "turn_right" else turns_needed

    # Add the required turn actions to the sequence
    for _ in range(abs(turns_needed)):
        actions.append(ACTION_TO_IDX[turn_action])

    # Finally, add a move forward action to simulate movement towards the target
    actions.append(ACTION_TO_IDX["move_forward"])

    return actions

def find_ball_position(obs, color):
    """
    Searches for the position of a ball of a specific color within the observation grid.

    Parameters:
    - obs: The current observation, which includes the 'image' field with grid information.
    - color (str): The color of the ball to find.

    Returns:
    - tuple or None: The (x, y) position of the ball if found, otherwise None.
    """
    grid, agent_dir = extract_grid_and_direction(obs)
       
    object_layer = grid[:,:,0]  # Layer with object types
    color_layer = grid[:,:,1]  # Layer with color information

    # Indices for the ball object and green color (these are example values, check your environment's documentation)
    ball_idx = OBJECT_TO_IDX['ball']
    green_idx = COLOR_TO_IDX[color]

    # Find positions where the object is a ball and its color is green
    positions = np.where((object_layer == ball_idx) & (color_layer == green_idx))

    # If a green ball is found, return its position
    if len(positions[0]) > 0:
        return (positions[1][0], positions[0][0])  # Note: NumPy returns positions in (row, column) format
    return None

def astar_pathfinding(grid, start, goal):
    """
    Implements the A* pathfinding algorithm to calculate the shortest path from a start position to a goal position
    within a grid. This method uses a heuristic to estimate the distance to the goal, checks for goal reachability,
    and identifies navigable neighbors to efficiently find the path.

    Parameters:
    - grid (array): A 2D array representing the environment, with 1 indicating navigable space and 0 indicating obstacles.
    - start (tuple): The starting coordinates (x, y) in the grid.
    - goal (tuple): The goal coordinates (x, y) in the grid.

    Returns:
    - list of tuples: A list representing the path from start to goal, including both the start and goal positions. Each
      element of the list is a tuple (x, y) indicating a step on the path. Returns an empty list if no path is found.
    """
    # Heuristic function to estimate the distance from a node to the goal using Manhattan distance
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Checks if the goal has been reached by verifying adjacency to the goal position
    def goal_reached(current, goal):
        # Consider all four adjacent positions to the goal
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            adj = (goal[0] + dx, goal[1] + dy)
            if current == adj and grid[adj[1]][adj[0]] == 1:  # Check if adjacent cell is navigable
                return True
        return False

    # Finds all navigable neighbors for a given node within the grid
    def get_neighbors(node):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            x, y = node[0] + dx, node[1] + dy
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] and grid[y][x] == 1:  # Check bounds and navigability
                neighbors.append((x, y))
        return neighbors

    open_set = []  # Initialize the open set as a priority queue for nodes to explore
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))  # Add start node with initial F and G scores

    came_from = {}  # Tracks the most efficient path to a node
    gscore = {start: 0}  # Cost from start to a node

    while open_set:
        _, g, current = heapq.heappop(open_set)  # Node with the lowest F-score

        if goal_reached(current, goal):  # If goal is reached, reconstruct and return the path
            path = [current]
            while current in came_from:  # Trace back from goal to start
                current = came_from[current]
                path.append(current)
            path.reverse()  # Reverse the path to start-to-goal order
            return path

        # Explore neighbors of the current node
        for neighbor in get_neighbors(current):
            tentative_g_score = g + 1  # Assume uniform cost between nodes
            if neighbor not in gscore or tentative_g_score < gscore[neighbor]:  # Check if new path is more efficient
                came_from[neighbor] = current  # Update path to this neighbor
                gscore[neighbor] = tentative_g_score  # Update G-score
                f_score = tentative_g_score + heuristic(neighbor, goal)  # Calculate F-score
                heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))  # Add neighbor to open set

    return []  # If no path is found, return an empty list

def update_tree_obs(tree, new_obs):
    """
    Updates the observation data for all nodes within the behavior tree that have the 'update_obs' method. This ensures
    that each node's decision-making is based on the most current observation of the environment.

    Parameters:
    - tree: The behavior tree containing nodes that may require updated observations.
    - new_obs: The latest observation data from the environment.

    This function iterates through all nodes in the tree, updating them with new observation data where applicable.
    """
    for node in tree.root.iterate():
        if hasattr(node, 'update_obs'):
            node.update_obs(new_obs)

def post_tick_handler(snapshot_visitor, behaviour_tree):
    """
    Prints the behaviour tree's structure after each tick.

    This function is intended for debugging purposes, allowing the visualization of
    the behaviour tree's state at each tick.

    Parameters:
    - snapshot_visitor: Visitor object that tracks the nodes visited during the tick.
    - behaviour_tree: The behaviour tree instance being debugged.
    """
    print(py_trees.display.unicode_tree(behaviour_tree.root, visited=snapshot_visitor.visited, previously_visited=snapshot_visitor.visited))

# ACTIONS
class PickUpGoal(py_trees.behaviour.Behaviour):
    """
    This behavior node aims to navigate the agent towards a specific goal (e.g., picking up a ball) within the environment.
    It evaluates the environment's state through the current observation and plans an action sequence to reach and interact
    with the target object. This node is versatile and can adjust the planned actions based on the agent's possession status
    regarding key objects necessary for achieving the goal, such as keys for unlocking doors.

    Parameters:
    - name (str): Identifier for the behavior, facilitating debugging and visualization.
    - env: The environment object where the agent operates, allowing access to perform actions and receive feedback.
    - obs: The current observation from the environment, providing the state based on which the node plans its actions.

    Attributes:
    - action_sequence (list): Dynamically generated sequence of actions the agent must execute to reach and pick up the goal object.
    - current_action_index (int): Tracks the execution progress within the `action_sequence`.

    The behavior starts by assessing whether the agent already possesses necessary objects (e.g., a key) and then plans a path
    towards the goal. It considers environmental constraints like obstacles and dynamically generates a sequence of actions, including
    movement and interaction with objects, to successfully achieve the goal. The sequence execution is managed through subsequent
    updates, reacting to the evolving state of the environment.
    """
    def __init__(self, name, env, obs):
        super(PickUpGoal, self).__init__(name)
        self.env = env
        self.obs = obs
        self.action_sequence = []  # Sequence of actions to be filled based on observation.
        self.current_action_index = 0  # Starting point of action execution.

    def initialise(self):
        # Extract relevant information from the current observation.
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        agent_pos = positions.get('agent')  # Agent's current position.
        ball_pos = find_ball_position(self.obs, 'blue')  # Position of the goal object (e.g., blue ball).
        key_pos = positions.get('key')  # Position of the key, if present.

        # Planning action sequence based on the agent's current status and goal object's location.
        # Includes conditions for when the agent holds a key and needs to unlock a door or directly pick up the goal object.
        # Generates action sequence for navigating towards the goal, adjusting direction if needed, and performing the pick-up action.
        # Utilizes `prepare_action_sequence` for path planning and action generation.

        if not key_pos: # It is holding the key
            if ball_pos == (agent_pos[0], agent_pos[1]+1): # Goal is by the door -> hard coded
                self.action_sequence = [1,1,4,0,0,3] # turn_right, turn_right, drop, turn_left, turn_left, pick_up
            else: # Enter room
                self.action_sequence.append(ACTION_TO_IDX["move_forward"])
                if ball_pos == (agent_pos[0], agent_pos[1]+2): # Goal is in front -> hard coded, but checking empty cell for dropping key
                    if grid[agent_pos[0] + 1, agent_pos[1] - 1, 0] == 1:
                        self.action_sequence += [1,4,0,3] # turn_right, drop, turn_left, pick_up
                    else:
                        self.drop_pos = (agent_pos[0] + 1, agent_pos[1] + 1)
                        self.action_sequence += [0,4,1,3] # turn_left, drop, turn_right, pick_up
            self.action_sequence.append(ACTION_TO_IDX["drop"])

            # First, check if the agent is already in an adjacent position to the blue ball
            agent_neighbors = [
                (agent_pos[0], agent_pos[1]),  # Up
                (agent_pos[0] + 1, agent_pos[1]+1),  # Right
                (agent_pos[0], agent_pos[1] + 2),  # Down
                (agent_pos[0] - 1, agent_pos[1]+1),  # Left
            ]
            if ball_pos:
                if ball_pos in agent_neighbors:
                    self.feedback_message = "Already a neighbor, maybe needs to adjust direction."
                    self.action_sequence += align_direction_to_target((agent_pos[0], agent_pos[1]+1), ball_pos, 0)
                    self.action_sequence.append(ACTION_TO_IDX["pick_up"])

                # Corrections to the grid consider that the key will be dropped and where the agent will be
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
        if self.current_action_index < len(self.action_sequence):
            current_action['action'] = self.action_sequence[self.current_action_index]
            self.current_action_index += 1
            if self.current_action_index == len(self.action_sequence):
                # Once all actions are set, reset for potential future re-use
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING
        else:
            return py_trees.common.Status.FAILURE

class EnterRoom(py_trees.behaviour.Behaviour):
    """
    A behavior node designed to navigate an agent through a door into a room. It encompasses identifying the door's location,
    moving towards it, and performing actions to open the door and enter the room. This node dynamically generates an action
    sequence based on the current environmental observation, aiming to successfully navigate the agent into the room.

    Parameters:
    - name (str): The name of the behavior, useful for debugging and visualization.
    - env: The environment object where the agent operates, providing access to environmental data and actions.
    - obs: The current observation from the environment, used to identify the door's position and plan the sequence of actions.

    Attributes:
    - action_sequence (list): A dynamically generated list of actions (integers) that the agent must execute to enter the room.
    - current_action_index (int): The index of the current action within `action_sequence`, tracking execution progress.

    Upon initialization, the behavior node evaluates the environment's state and plans a sequence of actions to open the door
    and enter the room. It then executes these actions step-by-step, updating the agent's state in the environment.
    """
    def __init__(self, name, env, obs):
        super(EnterRoom, self).__init__(name)
        self.env = env
        self.obs = obs
        self.action_sequence = []  # Initially empty, to be filled based on environmental observation.
        self.current_action_index = 0  # Start executing from the first action.

    def initialise(self):
        # Extract relevant information from the observation.
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        door_pos = positions.get('door')  # Position of the door.
        agent_pos = positions.get('agent')  # Current position of the agent.

        # First, check if the agent is already in an adjacent position to the green ball
        agent_neighbors = [
            (agent_pos[0], agent_pos[1] - 1),  # Up
            (agent_pos[0] + 1, agent_pos[1]),  # Right
            (agent_pos[0], agent_pos[1] + 1),  # Down
            (agent_pos[0] - 1, agent_pos[1]),  # Left
        ]
        # Determine if the agent is adjacent to the door.
        # If so, it may need to adjust its direction to interact with the door.
        # Otherwise, find a path to the door and generate the corresponding action sequence.
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
                # Once all actions are set, reset for potential future re-use
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING
        else:
            return py_trees.common.Status.FAILURE

class PickUpKey(py_trees.behaviour.Behaviour):
    """
    This behavior node aims to guide the agent to locate and pick up a key within the environment. It assesses the current
    state through observation to plan a series of actions that lead to acquiring the key, considering environmental obstacles
    and the key's location. This node demonstrates dynamic decision-making based on the agent's current position, the key's location,
    and any obstacles that may require navigating around or interacting with objects like boxes.

    Parameters:
    - name (str): Identifier for the behavior, useful for debugging and visualization.
    - env: The environment object, enabling the node to interact with the environment.
    - obs: The current observation from the environment, providing the state information necessary for planning actions.

    Attributes:
    - action_sequence (list): Dynamically generated list of actions for the agent to execute in order to pick up the key.
    - current_action_index (int): Tracks the current position in the `action_sequence` to manage execution progress.

    The behavior initiates by determining the agent's proximity to the key or any intermediary objects like boxes that might
    contain the key. It plans a path to the key or box, including actions to navigate towards, potentially move objects, and
    finally pick up the key. The sequence of actions is executed progressively, with the ability to adjust based on the changing
    state of the environment or the discovery of the key's precise location.
    """
    def __init__(self, name, env, obs):
        super(PickUpKey, self).__init__(name)
        self.env = env
        self.obs = obs
        self.action_sequence = []  # To be filled with actions based on current observation.
        self.current_action_index = 0  # Start of action execution.

    def initialise(self):
        # Extract relevant information from the observation.
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_positions(self.obs)
        box_pos = positions.get('box')
        agent_pos = positions.get('agent')
        ball_pos = find_ball_position(self.obs, 'green')
        door_pos = positions.get('door')

        # Utilizes the current observation to extract necessary environmental data.
        # Determines the agent's position, the key's (or box's) location, and plans the action sequence.
        # Considers if the agent already holds the key or needs to interact with objects to find it.
        # Utilizes `prepare_action_sequence` for efficient path planning and action generation towards the key or box.

        # First, check if the agent is already in an adjacent position to the box
        agent_neighbors = [
            (agent_pos[0], agent_pos[1] - 1),  # Up
            (agent_pos[0] + 1, agent_pos[1]),  # Right
            (agent_pos[0], agent_pos[1] + 1),  # Down
            (agent_pos[0] - 1, agent_pos[1]),  # Left
        ]
        if not ball_pos: # It is holding the green ball
            drop_pos = find_safe_drop_location(grid[:,:,0], door_pos, agent_pos, agent_dir)
            if drop_pos:
                self.action_sequence = align_direction_to_target(agent_pos, drop_pos, agent_dir)
                self.action_sequence.append(ACTION_TO_IDX["drop"])         
            else: # agent cornered by box, just move forward
                self.action_sequence.append(ACTION_TO_IDX["move_forward"])    
                self.action_sequence.append(ACTION_TO_IDX["drop"])
                if agent_dir == 1: # Agent pointing down 
                    agent_pos = (agent_pos[0]+1, agent_pos[1])
                    drop_pos = (agent_pos[0]+1, agent_pos[1])
                else: # Agent pointing up 
                    agent_pos = (agent_pos[0]-1, agent_pos[1])
                    drop_pos = (agent_pos[0]-1, agent_pos[1])

        if box_pos:
            # Correction to consider that the ball will be dropped
            grid[:,:,0][drop_pos[1],drop_pos[0]]=6

            # First, check if the agent is already in an adjacent position to the box
            agent_neighbors = [
                (agent_pos[0], agent_pos[1] - 1),  # Up
                (agent_pos[0] + 1, agent_pos[1]),  # Right
                (agent_pos[0], agent_pos[1] + 1),  # Down
                (agent_pos[0] - 1, agent_pos[1]),  # Left
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
                # Once all actions are set, reset for potential future re-use
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING
        else:
            return py_trees.common.Status.FAILURE

class ClearPath(py_trees.behaviour.Behaviour):
    """
    A behavior node designed to clear a path towards a goal. It determines if there is an obstacle
    blocking the path to the goal and, if so, executes a sequence of actions to remove the obstacle.

    Parameters:
    - name (str): The name of the behavior.
    - env: The environment in which the agent operates.
    - obs: The current observation received from the environment.

    Attributes:
    - path_to_ball (list): A list of coordinates representing the path to the ball (goal).
    - path_to_drop (list): A list of coordinates representing where to drop any carried objects to clear the path.
    - action_sequence (list): A sequence of actions generated based on the current state to clear the path.
    - current_action_index (int): The index of the current action in the sequence that the agent is executing.
    """
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

        # First, check if the agent is already in an adjacent position to the green ball
        agent_neighbors = [
            (agent_pos[0], agent_pos[1] - 1),  # Up
            (agent_pos[0] + 1, agent_pos[1]),  # Right
            (agent_pos[0], agent_pos[1] + 1),  # Down
            (agent_pos[0] - 1, agent_pos[1]),  # Left
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
                # Once all actions are set, reset for potential future re-use
                self.current_action_index = 0
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING
        else:
            return py_trees.common.Status.FAILURE

# CONDITIONS
class HasKey(py_trees.behaviour.Behaviour):
    """
    Determines whether the agent currently possesses the key. This is inferred by checking the absence
    of the key in the observable environment, assuming that if the key is not visible, the agent holds it.

    Parameters:
    - name (str): Name of the behavior, defaulting to "HasKey".
    - env: The environment instance, unused but included for interface consistency.
    - obs: The current observation from the environment, used to determine key possession.

    This condition succeeds if the agent has the key and fails otherwise.
    """
    def __init__(self, name="HasKey", env=None, obs=None):
        """
        :param name: The name of the behaviour
        :param env: The environment object (not used in this example but may be useful for more complex checks)
        :param obs: The current observation
        """
        super(HasKey, self).__init__(name)
        self.env = env
        self.obs = obs

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        """
        Update the behaviour status. Check if the key is present in the observation.
        """
        positions = extract_positions(self.obs)
        box_pos = 0
        key_pos = 0

        # Check if both 'agent' and 'door' positions are available
        if 'box' in positions:
            box_pos = positions['box']        
        if 'key' in positions:
            key_pos = positions['key']

        # If a key is not visible, but a box is, assume the key is inside the box and the agent does not have the key
        if key_pos == 0 and box_pos != 0:
            self.feedback_message = "Key is not visible but a box is, assuming the key is inside the box."
            return py_trees.common.Status.FAILURE
        elif key_pos == 0:
            # No key visible and no box implies the agent has the key
            self.feedback_message = "Key is not visible, assuming the agent has the key."
            return py_trees.common.Status.SUCCESS
        else:
            # Key is visible, which means the agent does not have it
            self.feedback_message = "Key is visible, agent does not have the key."
            return py_trees.common.Status.FAILURE

class IsInsideRoom(py_trees.behaviour.Behaviour):
    """
    Verifies if the agent has successfully entered a room by comparing its position relative to the door. 
    The agent is considered inside if its position is beyond the door in the environment's layout.

    Parameters:
    - name (str): Name of the behavior, defaulting to "IsInsideRoom".
    - env: The environment instance, unused but included for interface consistency.
    - obs: The current observation from the environment, used to determine the agent's location relative to the door.

    This condition succeeds if the agent is inside the room and fails otherwise.
    """
    def __init__(self, name="IsInsideRoom", env=None, obs=None):
        """
        :param name: The name of the behaviour
        :param env: The environment object (not used in this example but may be useful for more complex checks)
        :param obs: The current observation
        """
        super(IsInsideRoom, self).__init__(name)
        self.env = env
        self.obs = obs

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        """
        Update the behaviour status. Checks if the agent has moved past the door in the y-coordinate.
        """
        positions = extract_positions(self.obs)
        
        # Check if both 'agent' and 'door' positions are available
        door_pos = 0
        if 'door' in positions:
            door_pos = positions['door'] 
        agent_pos = positions['agent']

        # Assuming the door's y-coordinate determines the "inside" threshold
        if not door_pos:
            # Agent's y-coordinate is greater than the door's, meaning it has passed the door
            self.feedback_message = "Agent is passing the door and is considered inside the room."
            return py_trees.common.Status.SUCCESS
        elif agent_pos[1] >= door_pos[1]:
            # Agent's y-coordinate is greater than the door's, meaning it has passed the door
            self.feedback_message = "Agent has passed the door."
            return py_trees.common.Status.SUCCESS
        else:
            # Agent is not past the door yet
            self.feedback_message = "Agent has not passed the door, considered outside the room."
            return py_trees.common.Status.FAILURE

class IsPathClear(py_trees.behaviour.Behaviour):
    """
    Checks if there is a clear path for the agent to reach the door without encountering obstacles. It utilizes
    the A* pathfinding algorithm to determine the presence of a navigable path from the agent's current position to the door.

    Parameters:
    - name (str): Name of the behavior, defaulting to "IsPathClear".
    - env: The environment instance, unused but included for interface consistency.
    - obs: The current observation from the environment, used to analyze the grid for pathfinding.

    This condition succeeds if a clear path exists and fails if the path is obstructed.
    """
    def __init__(self, name="IsPathClear", env=None, obs=None):
        super(IsPathClear, self).__init__(name)
        self.env = env
        self.obs = obs

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        positions = extract_positions(self.obs)

        grid, agent_dir = extract_grid_and_direction(self.obs)
        
        if 'agent' in positions and 'door' in positions:
            agent_pos = positions['agent']
            door_pos = positions['door']

            # First check if agent is already next to the door
            agent_neighbors = [
                (agent_pos[0], agent_pos[1] - 1),  # Up
                (agent_pos[0] + 1, agent_pos[1]),  # Right
                (agent_pos[0], agent_pos[1] + 1),  # Down
                (agent_pos[0] - 1, agent_pos[1]),  # Left
            ]
            if door_pos in agent_neighbors:
                self.feedback_message = "Agent is next to door."
                return py_trees.common.Status.SUCCESS

            # Adjust the target to be adjacent to the door rather than the door itself
            path = astar_pathfinding(grid[:,:,0], agent_pos, door_pos)
            
            # If A* returns a non-empty path, the path to the door is considered clear
            if path:
                self.feedback_message = "Path to the door is clear."
                return py_trees.common.Status.SUCCESS
            else:
                self.feedback_message = "Path to the door is blocked."
                return py_trees.common.Status.FAILURE
        else:
            self.feedback_message = "Cannot determine the path to the door."
            return py_trees.common.Status.FAILURE

# Main Function and Behavior Tree Construction
def create_behaviour_tree(env, obs):
    """
    Constructs and returns a backward chained behaviour tree for the agent.

    Parameters:
    - env: The environment instance.
    - obs: The initial observation from the environment.

    Returns:
    - An instance of py_trees.trees.BehaviourTree configured for the agent.
    """
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    clear_path_selector = py_trees.composites.Selector(name="Clear path selector", memory=False)
    clear_path_selector.add_child(IsPathClear(name="Is path clear", env=env, obs=obs))
    clear_path_selector.add_child(ClearPath(name="Clear path", env=env, obs=obs))

    obtain_key_sequence = py_trees.composites.Sequence(name="Pick up key sequence", memory=False)
    obtain_key_sequence.add_child(clear_path_selector)
    obtain_key_sequence.add_child(PickUpKey(name="Pick up key", env=env, obs=obs))

    has_key_selector = py_trees.composites.Selector(name="Has key selector", memory=False)
    has_key_selector.add_child(HasKey(name="Has key", env=env, obs=obs))
    has_key_selector.add_child(obtain_key_sequence)

    open_door_sequence = py_trees.composites.Sequence(name="Open door sequence", memory=False)
    open_door_sequence.add_child(has_key_selector)
    open_door_sequence.add_child(EnterRoom(name="Open door", env=env, obs=obs))

    inside_locked_room_selector = py_trees.composites.Selector(name="Door open selector", memory=False)
    inside_locked_room_selector.add_child(IsInsideRoom(name="Is door open", env=env, obs=obs))
    inside_locked_room_selector.add_child(open_door_sequence)

    root.add_child(inside_locked_room_selector)
    root.add_child(PickUpGoal(name="Pick up goal", env=env, obs=obs))

    return py_trees.trees.BehaviourTree(root)

def main():
    """
    Main function to initialize the environment, create the behavior tree, and execute the simulation loop. This function sets up the
    environment for the MiniGrid-ObstructedMaze-1Dlhb-v0 game, wraps it for full observability, initializes the behavior tree with the
    starting observation, and iteratively ticks the behavior tree to make decisions based on the current state of the environment.

    The loop continues until the termination condition of the environment is met, which typically occurs when the agent successfully
    completes the maze or fails. The environment's visual state is rendered at each step for observation.

    Key steps include:
    - Initializing the environment and wrapping it for full observability.
    - Creating and visualizing the initial behavior tree based on the starting environment state.
    - Entering a loop where the tree is ticked to progress the agent's behavior, with the environment rendered after each action.
    - Handling user input to proceed through each tick for step-by-step execution.
    - Closing the environment upon completion of the simulation.
    """
    # Environment setup and behavior tree initialization
    env = gym.make("MiniGrid-ObstructedMaze-1Dlhb-v0", render_mode='human')
    env = FullyObsWrapper(env)  # Wrapping for full observability
    obs = env.reset()  # Initial environment observation
    tree = create_behaviour_tree(env, obs)  # Behavior tree creation
    py_trees.display.render_dot_tree(tree.root)  # Optional: Visualize the tree structure

    visualize = False # Visualization of the BT (every tick)
    if visualize:
        snapshot_visitor = py_trees.visitors.SnapshotVisitor()
        tree.add_post_tick_handler(
            functools.partial(post_tick_handler,
                        snapshot_visitor))
        tree.visitors.append(snapshot_visitor)

    done = False

    while not done:
        update_tree_obs(tree, obs)  # Update the tree with the latest observation
        tree.tick()  # Tick the behavior tree
        action = current_action["action"]
        if action is not None:
            obs, reward, done, info, probs = env.step(action)  # Execute the chosen action in the environment
        else:
            print("Waiting for action decision...")

    print(f"Final reward: {reward}")
    env.close()

if __name__ == "__main__":
    main()