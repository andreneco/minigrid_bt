import py_trees
from minigrid_bt.utils import extract_positions, astar_pathfinding, extract_grid_and_direction

class HasKey(py_trees.behaviour.Behaviour):
    def __init__(self, name="HasKey", env=None, obs=None):
        super(HasKey, self).__init__(name)
        self.env = env
        self.obs = obs

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        positions = extract_positions(self.obs)
        box_pos = positions.get('box', 0)
        key_pos = positions.get('key', 0)
        if key_pos == 0 and box_pos != 0:
            self.feedback_message = "Key is not visible but a box is, assuming the key is inside the box."
            return py_trees.common.Status.FAILURE
        elif key_pos == 0:
            self.feedback_message = "Key is not visible, assuming the agent has the key."
            return py_trees.common.Status.SUCCESS
        else:
            self.feedback_message = "Key is visible, agent does not have the key."
            return py_trees.common.Status.FAILURE

class IsInsideRoom(py_trees.behaviour.Behaviour):
    def __init__(self, name="IsInsideRoom", env=None, obs=None):
        super(IsInsideRoom, self).__init__(name)
        self.env = env
        self.obs = obs

    def update_obs(self, new_obs):
        self.obs = new_obs

    def update(self):
        positions = extract_positions(self.obs)
        door_pos = positions.get('door', 0)
        agent_pos = positions['agent']
        if not door_pos or agent_pos[1] >= door_pos[1]:
            self.feedback_message = "Agent has passed the door."
            return py_trees.common.Status.SUCCESS
        else:
            self.feedback_message = "Agent has not passed the door, considered outside the room."
            return py_trees.common.Status.FAILURE

class IsPathClear(py_trees.behaviour.Behaviour):
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
            agent_neighbors = [
                (agent_pos[0], agent_pos[1] - 1), (agent_pos[0] + 1, agent_pos[1]),
                (agent_pos[0], agent_pos[1] + 1), (agent_pos[0] - 1, agent_pos[1])
            ]
            if door_pos in agent_neighbors:
                self.feedback_message = "Agent is next to door."
                return py_trees.common.Status.SUCCESS
            path = astar_pathfinding(grid[:, :, 0], agent_pos, door_pos)
            if path:
                self.feedback_message = "Path to the door is clear."
                return py_trees.common.Status.SUCCESS
            else:
                self.feedback_message = "Path to the door is blocked."
                return py_trees.common.Status.FAILURE
        else:
            self.feedback_message = "Cannot determine the path to the door."
            return py_trees.common.Status.FAILURE
