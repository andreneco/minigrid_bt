import py_trees
from minigrid_bt.utils import extract_door_positions, extract_multiple_positions, find_ball_position, extract_positions, astar_pathfinding, extract_grid_and_direction, find_door_position

class HasKeyBox(py_trees.behaviour.Behaviour):
    def __init__(self, name="HasKey", env=None, obs=None, debug=False):
        super(HasKeyBox, self).__init__(name)
        self.env = env
        self.obs = obs
        self.debug = debug

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        positions = extract_positions(self.obs)
        key_pos = positions.get('key', 0)
        box_pos = positions.get('box', 0)
        
        if key_pos == 0 and box_pos == 0:
            self.feedback_message = "Key is not visible and no box present, assuming the agent has the key."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.SUCCESS
        elif key_pos == 0 and box_pos != 0:
            self.feedback_message = "Key is not visible, but a box is present. Key might be inside the box."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.FAILURE
        else:
            self.feedback_message = "Key is visible, agent does not have the key."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.FAILURE
        
class HasKey(py_trees.behaviour.Behaviour):
    def __init__(self, name="HasKey", env=None, obs=None, debug=False):
        super(HasKey, self).__init__(name)
        self.env = env
        self.obs = obs
        self.debug = debug

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        positions = extract_positions(self.obs)
        key_pos = positions.get('key', 0)
        
        if key_pos == 0:
            self.feedback_message = "Key is not visible, assuming the agent has the key."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.SUCCESS
        else:
            self.feedback_message = "Key is visible, agent does not have the key."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.FAILURE        

class HasObstacle(py_trees.behaviour.Behaviour):
    def __init__(self, name="HasObstacle", env=None, obs=None, object_type="ball", object_color=None, debug=False):
        super(HasObstacle, self).__init__(name)
        self.env = env
        self.obs = obs
        self.goal_object = object_type
        self.goal_color = object_color
        self.debug = debug

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        if self.goal_object == "ball":
            if self.goal_color is None:
                # Check for the ball's position regardless of color
                object_pos = find_ball_position_any_color(self.obs)
            else:
                object_pos = find_ball_position(self.obs, self.goal_color)
        else:
            positions = extract_positions(self.obs)
            object_pos = positions.get(self.goal_object, None)

        if object_pos is None:
            if self.goal_color:
                feedback = f"{self.goal_color.capitalize()} {self.goal_object} is not visible, assuming the agent has the obstacle."
            else:
                feedback = f"{self.goal_object} is not visible, assuming the agent has the obstacle."
            self.feedback_message = feedback
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.SUCCESS
        else:
            if self.goal_color:
                feedback = f"{self.goal_color.capitalize()} {self.goal_object} is visible, agent does not have the obstacle."
            else:
                feedback = f"{self.goal_object} is visible, agent does not have the obstacle."
            self.feedback_message = feedback
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.FAILURE

def find_ball_position_any_color(obs):
    positions = extract_positions(obs)
    for key in positions:
        if key.startswith('ball'):
            return positions[key]
    return None

class AllDoorsUnlocked(py_trees.behaviour.Behaviour):
    def __init__(self, name="AllDoorsUnlocked", env=None, obs=None, debug=False):
        super(AllDoorsUnlocked, self).__init__(name)
        self.env = env
        self.obs = obs
        self.debug = debug
        
    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        door_positions = extract_door_positions(self.obs)
        all_unlocked = True
        for pos in door_positions:
            x, y, color, state = pos
            if state == 2:  # Assuming 2 represents a locked door
                all_unlocked = False
                break

        if all_unlocked:
            self.feedback_message = "All doors are unlocked."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.SUCCESS
        else:
            self.feedback_message = "Some doors are locked."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.FAILURE

class IsInsideRoom(py_trees.behaviour.Behaviour):
    def __init__(self, name="IsInsideRoom", env=None, obs=None, debug=False):
        super(IsInsideRoom, self).__init__(name)
        self.env = env
        self.obs = obs
        self.debug = debug
        
    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        positions = extract_positions(self.obs)
        door_pos = positions.get('door', 0)
        agent_pos = positions['agent']
        if not door_pos or agent_pos[1] >= door_pos[1]:
            self.feedback_message = "Agent has passed the door."
            if self.debug:
                print("Inside")
            return py_trees.common.Status.SUCCESS
        else:
            self.feedback_message = "Agent has not passed the door, considered outside the room."
            if self.debug:
                print("Outside")
            return py_trees.common.Status.FAILURE
        
class IsNearObject(py_trees.behaviour.Behaviour):
    def __init__(self, name="IsNearObject", env=None, obs=None, object_type='door', object_color=None, debug=False):
        super(IsNearObject, self).__init__(name)
        self.env = env
        self.obs = obs
        self.object_type = object_type
        self.object_color = object_color
        self.debug = debug

    def update_obs(self, new_obs):
        self.obs = new_obs

    def find_object_position(self):
        positions = extract_positions(self.obs)
        if self.object_type == "ball" and self.object_color:
            return find_ball_position(self.obs, self.object_color)
        return positions.get(self.object_type)

    def update(self):
        positions = extract_positions(self.obs)
        object_pos = self.find_object_position()
        agent_pos = positions['agent']

        if not object_pos:
            self.feedback_message = f"{self.object_type.capitalize()} position not found."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.FAILURE

        agent_neighbors = [
            (agent_pos[0], agent_pos[1] - 1), 
            (agent_pos[0] + 1, agent_pos[1]), 
            (agent_pos[0], agent_pos[1] + 1), 
            (agent_pos[0] - 1, agent_pos[1]), 
        ]

        if object_pos in agent_neighbors:
            self.feedback_message = f"Agent is near the {self.object_type}."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.SUCCESS
        else:
            self.feedback_message = f"Agent is not near the {self.object_type}."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.FAILURE

class DoorOpen(py_trees.behaviour.Behaviour):
    def __init__(self, name="AnyDoorOpen", env=None, obs=None, door_color="red", debug=False):
        super(DoorOpen, self).__init__(name)
        self.env = env
        self.obs = obs
        self.door_color = door_color
        self.debug = debug

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        grid, agent_dir = extract_grid_and_direction(self.obs)
        door_position = find_door_position(self.obs, self.door_color)

        if not door_position:
            self.feedback_message = f"No doors of color {self.door_color} found."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.FAILURE

        any_open = False
        x, y = int(door_position[0]), int(door_position[1])
        door_status = grid[y, x, 2]
        if door_status == 0:  # Assuming 0 represents an open door
            any_open = True

        if any_open:
            self.feedback_message = f"At least one {self.door_color} door is open."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.SUCCESS
        else:
            self.feedback_message = f"No {self.door_color} doors are open."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.FAILURE


class AllDoorsOpen(py_trees.behaviour.Behaviour):
    def __init__(self, name="AllDoorsOpen", env=None, obs=None, debug=False):
        super(AllDoorsOpen, self).__init__(name)
        self.env = env
        self.obs = obs
        self.debug = debug

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        grid, agent_dir = extract_grid_and_direction(self.obs)
        positions = extract_multiple_positions(self.obs)
        door_positions = positions.get('door', [])

        # Ensure door_positions is a list of tuples
        if not isinstance(door_positions, list):
            door_positions = [door_positions]

        all_open = True
        for pos in door_positions:
            # Convert np.int64 to standard Python int
            if isinstance(pos, tuple) and len(pos) == 2:
                x, y = int(pos[0]), int(pos[1])
                door_status = grid[y, x, 2]
                if door_status == 2 or door_status == 1:  # Assuming 1 represents a closed door
                    all_open = False
                    break
            else:
                print(f"Invalid position: {pos}")

        if all_open:
            self.feedback_message = "All doors are open."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.SUCCESS
        else:
            self.feedback_message = "Not all doors are open."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.FAILURE

class IsPathClear(py_trees.behaviour.Behaviour):
    def __init__(self, name="IsPathClear", env=None, obs=None, debug=False):
        super(IsPathClear, self).__init__(name)
        self.env = env
        self.obs = obs
        self.debug = debug

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        positions = extract_positions(self.obs)
        grid, agent_dir = extract_grid_and_direction(self.obs)
        if 'agent' in positions and 'door' in positions:
            agent_pos = positions['agent']
            door_pos = positions['door']
            agent_neighbors = [
                (agent_pos[0], agent_pos[1] - 1), (agent_pos[0] + 1, agent_pos[1]),
                (agent_pos[0], agent_pos[1] + 1), (agent_pos[0] - 1, agent_pos[1])
            ]
            if door_pos in agent_neighbors:
                self.feedback_message = "Agent is next to door."
                if self.debug:
                    print(self.feedback_message)
                return py_trees.common.Status.SUCCESS
            path = astar_pathfinding(grid[:, :, 0], agent_pos, door_pos, agent_dir)
            if path:
                self.feedback_message = "Path to the door is clear."
                if self.debug:
                    print(self.feedback_message)
                return py_trees.common.Status.SUCCESS
            else:
                self.feedback_message = "Path to the door is blocked."
                if self.debug:
                    print(self.feedback_message)
                return py_trees.common.Status.FAILURE
        else:
            self.feedback_message = "Cannot determine the path to the door."
            if self.debug:
                print(self.feedback_message)
            return py_trees.common.Status.FAILURE
