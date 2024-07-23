import py_trees
from minigrid_bt.utils import find_ball_position, extract_positions, astar_pathfinding, extract_grid_and_direction
from minigrid_bt.shared_status import SharedStatus  # Import SharedStatus from the separate module

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
        positions = extract_positions(self.obs)
        door_positions = positions.get('door', [])

        all_open = True
        for pos in door_positions:
            door_pos = (pos[0], pos[1])
            door_status = grid[door_pos[1], door_pos[0], 2]
            if door_status == 1:  # Assuming 1 represents a closed door
                all_open = False
                break

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
        
class InFinalRoom(py_trees.behaviour.Behaviour):
    def __init__(self, name, env, obs, debug=False):
        super(InFinalRoom, self).__init__(name)
        self.env = env
        self.obs = obs
        self.debug = debug

    def update(self):
        if SharedStatus.has_entered_room:
            if self.debug:
                print("Agent has entered the final room.")
            return py_trees.common.Status.SUCCESS
        else:
            if self.debug:
                print("Agent has not entered the final room.")
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
