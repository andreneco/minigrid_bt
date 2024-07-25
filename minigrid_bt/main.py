# minigrid_bt/main.py

import py_trees
from minigrid_bt.behaviors import GoToGoalDoors, OpenLockedDoor, PickUpKeyDoors, OpenDoor, OpenAllDoors, GoNextToGoal, PickUpKeyBox, PickUpGoal, EnterRoom, DropObstacle, GoToGoal, PickUp
from minigrid_bt.conditions import AllDoorsUnlocked, DoorOpen, HasObstacle, IsNearObject, HasKey, IsPathClear, AllDoorsOpen, HasKeyBox

# Define the global debug variable
debug = False

def create_ObstructedMaze_bt(env, obs):
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    neighbouring_ball_selector = py_trees.composites.Selector(name="Near obstable selector", memory=False)
    neighbouring_ball_selector.add_child(IsNearObject(name="Is near obstacle", env=env, obs=obs, object_type='ball', object_color='green', debug=debug))
    neighbouring_ball_selector.add_child(GoNextToGoal(name="Go to obstacle", env=env, obs=obs, goal_object="ball", goal_color="green", debug=debug))

    pickup_obstable_sequence = py_trees.composites.Sequence(name="Clear path sequence", memory=False)
    pickup_obstable_sequence.add_child(neighbouring_ball_selector)
    pickup_obstable_sequence.add_child(PickUp(name="Pick up obstacle", env=env, obs=obs, goal_object="ball", goal_color="green", debug=debug))

    clear_path_selector = py_trees.composites.Selector(name="Clear path selector", memory=False)
    clear_path_selector.add_child(IsPathClear(name="Is path clear", env=env, obs=obs, debug=debug))
    clear_path_selector.add_child(pickup_obstable_sequence)

    obtain_key_sequence = py_trees.composites.Sequence(name="Pick up key sequence", memory=False)
    obtain_key_sequence.add_child(clear_path_selector)
    obtain_key_sequence.add_child(PickUpKeyBox(name="Pick up key", env=env, obs=obs, debug=debug))

    holding_obstable_selector = py_trees.composites.Selector(name="Near obstable selector", memory=False)
    holding_obstable_selector.add_child(HasObstacle(name="Picked up obstacle", env=env, obs=obs, object_type='ball', object_color='green', debug=debug))
    holding_obstable_selector.add_child(obtain_key_sequence)

    drop_obstacle_sequence = py_trees.composites.Sequence(name="Pick up key sequence", memory=False)
    drop_obstacle_sequence.add_child(holding_obstable_selector)
    drop_obstacle_sequence.add_child(DropObstacle(name="Drop obstable", env=env, obs=obs, debug=debug))

    has_key_selector = py_trees.composites.Selector(name="Has key selector", memory=False)
    has_key_selector.add_child(HasKeyBox(name="Has key", env=env, obs=obs, debug=debug))
    has_key_selector.add_child(drop_obstacle_sequence)

    open_door_sequence = py_trees.composites.Sequence(name="Open door sequence", memory=False)
    open_door_sequence.add_child(has_key_selector)
    open_door_sequence.add_child(EnterRoom(name="Open door", env=env, obs=obs, debug=debug))

    inside_locked_room_selector = py_trees.composites.Selector(name="Door open selector", memory=False)
    inside_locked_room_selector.add_child(AllDoorsOpen(name="Is door open", env=env, obs=obs, debug=debug))
    inside_locked_room_selector.add_child(open_door_sequence)

    root.add_child(inside_locked_room_selector)
    root.add_child(PickUpGoal(name="Pick up goal", env=env, obs=obs, goal_object="ball", goal_color="blue", debug=debug))

    return py_trees.trees.BehaviourTree(root)

def create_Empty_bt(env, obs):
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    root.add_child(GoToGoal(name="Go to Goal", env=env, obs=obs, debug=debug))

    return py_trees.trees.BehaviourTree(root)

def create_BabyAI_bt(env, obs):
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    root.add_child(GoNextToGoal(name="Go to Goal", env=env, obs=obs, goal_object="ball", goal_color="red", debug=debug))

    return py_trees.trees.BehaviourTree(root)

def create_Unlock_bt(env, obs):
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    has_key_selector = py_trees.composites.Selector(name="Has key selector", memory=False)
    has_key_selector.add_child(HasKey(name="Has key", env=env, obs=obs, debug=debug))
    has_key_selector.add_child(PickUp(name="Pick up key", env=env, obs=obs, goal_object="key", debug=debug))

    root.add_child(has_key_selector)
    root.add_child(EnterRoom(name="Open door", env=env, obs=obs, debug=debug))

    return py_trees.trees.BehaviourTree(root)

def create_KeyCorridor_bt(env, obs):
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    has_key_selector = py_trees.composites.Selector(name="Has key selector", memory=False)
    has_key_selector.add_child(HasKey(name="Has key", env=env, obs=obs, debug=debug))
    has_key_selector.add_child(PickUpKeyDoors(name="Pick up key", env=env, obs=obs, debug=debug))

    open_door_sequence = py_trees.composites.Sequence(name="Open door sequence", memory=False)
    open_door_sequence.add_child(has_key_selector)
    open_door_sequence.add_child(OpenLockedDoor(name="Open door", env=env, obs=obs, debug=debug))

    inside_locked_room_selector = py_trees.composites.Selector(name="Door open selector", memory=False)
    inside_locked_room_selector.add_child(AllDoorsUnlocked(name="Is door open", env=env, obs=obs, debug=debug))
    inside_locked_room_selector.add_child(open_door_sequence)

    root.add_child(inside_locked_room_selector)
    root.add_child(PickUpGoal(name="Pick up goal", env=env, obs=obs, goal_object="ball", goal_color=None, debug=debug))

    return py_trees.trees.BehaviourTree(root)

def create_RedBlueDoors_bt(env, obs):
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    red_door_open_selector = py_trees.composites.Selector(name="Red door open selector", memory=False)
    red_door_open_selector.add_child(DoorOpen(name="Is red door open", env=env, obs=obs, door_color="red", debug=debug))
    red_door_open_selector.add_child(OpenDoor(name="Open red door", env=env, obs=obs, door_color="red", debug=debug))

    root.add_child(red_door_open_selector)
    root.add_child(OpenDoor(name="Open blue door", env=env, obs=obs, door_color="blue", debug=debug))

    return py_trees.trees.BehaviourTree(root)

def create_BlockedUnlockPickup_bt(env, obs):
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    neighbouring_ball_selector = py_trees.composites.Selector(name="Near obstable selector", memory=False)
    neighbouring_ball_selector.add_child(IsNearObject(name="Is near obstacle", env=env, obs=obs, object_type='ball', object_color='grey', debug=debug))
    neighbouring_ball_selector.add_child(GoNextToGoal(name="Go to obstacle", env=env, obs=obs, goal_object="ball", goal_color="grey", debug=debug))

    pickup_obstable_sequence = py_trees.composites.Sequence(name="Clear path sequence", memory=False)
    pickup_obstable_sequence.add_child(neighbouring_ball_selector)
    pickup_obstable_sequence.add_child(PickUp(name="Pick up obstacle", env=env, obs=obs, goal_object="ball", goal_color=None, debug=debug))

    clear_path_selector = py_trees.composites.Selector(name="Clear path selector", memory=False)
    clear_path_selector.add_child(IsPathClear(name="Is path clear", env=env, obs=obs, debug=debug))
    clear_path_selector.add_child(pickup_obstable_sequence)

    obtain_key_sequence = py_trees.composites.Sequence(name="Pick up key sequence", memory=False)
    obtain_key_sequence.add_child(clear_path_selector)
    obtain_key_sequence.add_child(PickUp(name="Pick up key", env=env, obs=obs, goal_object="key", debug=debug))

    holding_obstable_selector = py_trees.composites.Selector(name="Near obstable selector", memory=False)
    holding_obstable_selector.add_child(HasObstacle(name="Picked up obstacle", env=env, obs=obs, object_type='ball', object_color=None, debug=debug))
    holding_obstable_selector.add_child(obtain_key_sequence)

    drop_obstacle_sequence = py_trees.composites.Sequence(name="Pick up key sequence", memory=False)
    drop_obstacle_sequence.add_child(holding_obstable_selector)
    drop_obstacle_sequence.add_child(DropObstacle(name="Drop obstable", env=env, obs=obs, object_color='grey', debug=debug))

    has_key_selector = py_trees.composites.Selector(name="Has key selector", memory=False)
    has_key_selector.add_child(HasKey(name="Has key", env=env, obs=obs, debug=debug))
    has_key_selector.add_child(drop_obstacle_sequence)

    open_door_sequence = py_trees.composites.Sequence(name="Open door sequence", memory=False)
    open_door_sequence.add_child(has_key_selector)
    open_door_sequence.add_child(EnterRoom(name="Open door", env=env, obs=obs, debug=debug))

    inside_locked_room_selector = py_trees.composites.Selector(name="Door open selector", memory=False)
    inside_locked_room_selector.add_child(AllDoorsUnlocked(name="Is door open", env=env, obs=obs, debug=debug))
    inside_locked_room_selector.add_child(open_door_sequence)

    root.add_child(inside_locked_room_selector)
    root.add_child(PickUpGoal(name="Pick up goal", env=env, obs=obs, goal_object="box", debug=debug))

    return py_trees.trees.BehaviourTree(root)

def create_UnlockPickup_bt(env, obs):
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    has_key_selector = py_trees.composites.Selector(name="Has key selector", memory=False)
    has_key_selector.add_child(HasKey(name="Has key", env=env, obs=obs, debug=debug))
    has_key_selector.add_child(PickUp(name="Pick up key", env=env, obs=obs, goal_object="key", debug=debug))

    open_door_sequence = py_trees.composites.Sequence(name="Open door sequence", memory=False)
    open_door_sequence.add_child(has_key_selector)
    open_door_sequence.add_child(EnterRoom(name="Open door", env=env, obs=obs, debug=debug))

    inside_locked_room_selector = py_trees.composites.Selector(name="Door open selector", memory=False)
    inside_locked_room_selector.add_child(AllDoorsUnlocked(name="Is door open", env=env, obs=obs, debug=debug))
    inside_locked_room_selector.add_child(open_door_sequence)

    root.add_child(inside_locked_room_selector)
    root.add_child(PickUpGoal(name="Pick up goal", env=env, obs=obs, goal_object="box", debug=debug))

    return py_trees.trees.BehaviourTree(root)

def create_DoorKey_bt(env, obs):
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    has_key_selector = py_trees.composites.Selector(name="Has key selector", memory=False)
    has_key_selector.add_child(HasKey(name="Has key", env=env, obs=obs, debug=debug))
    has_key_selector.add_child(PickUp(name="Pick up key", env=env, obs=obs, goal_object="key", debug=debug))

    open_door_sequence = py_trees.composites.Sequence(name="Open door sequence", memory=False)
    open_door_sequence.add_child(has_key_selector)
    open_door_sequence.add_child(EnterRoom(name="Open door", env=env, obs=obs, debug=debug))

    inside_locked_room_selector = py_trees.composites.Selector(name="Door open selector", memory=False)
    inside_locked_room_selector.add_child(AllDoorsUnlocked(name="Is door open", env=env, obs=obs, debug=debug))
    inside_locked_room_selector.add_child(open_door_sequence)

    root.add_child(inside_locked_room_selector)
    root.add_child(GoToGoalDoors(name="Go to Goal", env=env, obs=obs, debug=debug))

    return py_trees.trees.BehaviourTree(root)

def create_MultiRoom_bt(env, obs):
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    open_doors_selector = py_trees.composites.Selector(name="Doors open selector", memory=False)
    open_doors_selector.add_child(AllDoorsOpen(name="Are doors open", env=env, obs=obs, debug=debug))
    open_doors_selector.add_child(OpenAllDoors(name="Open doors", env=env, obs=obs, debug=debug))

    root.add_child(open_doors_selector)
    root.add_child(GoToGoalDoors(name="Go to Goal", env=env, obs=obs, debug=debug))

    return py_trees.trees.BehaviourTree(root)
