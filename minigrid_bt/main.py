# minigrid_bt/main.py

import py_trees
from minigrid_bt.behaviors import EnterStep, OpenDoors, GoNextToGoal, PickUpKeyBox, PickUpGoal, EnterRoom, PickUpKey, ClearPath, GoToGoal, PickUp
from minigrid_bt.conditions import InFinalRoom, HasKey, IsInsideRoom, IsPathClear, AllDoorsOpen

def create_ObstructedMaze_bt(env, obs):
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    clear_path_selector = py_trees.composites.Selector(name="Clear path selector", memory=False)
    clear_path_selector.add_child(IsPathClear(name="Is path clear", env=env, obs=obs))
    clear_path_selector.add_child(ClearPath(name="Clear path", env=env, obs=obs))

    obtain_key_sequence = py_trees.composites.Sequence(name="Pick up key sequence", memory=False)
    obtain_key_sequence.add_child(clear_path_selector)
    obtain_key_sequence.add_child(PickUpKeyBox(name="Pick up key", env=env, obs=obs))

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
    root.add_child(PickUpGoal(name="Pick up goal", env=env, obs=obs, goal_object="ball", goal_color="blue"))

    return py_trees.trees.BehaviourTree(root)

def create_Empty_bt(env, obs):
    root = py_trees.composites.Sequence(name="Another Root sequence", memory=False)

    root.add_child(GoToGoal(name="Go to Goal", env=env, obs=obs, goal_object="goal"))

    return py_trees.trees.BehaviourTree(root)

def create_BabyAI_bt(env, obs):
    root = py_trees.composites.Sequence(name="Another Root sequence", memory=False)

    root.add_child(GoNextToGoal(name="Go to Goal", env=env, obs=obs, goal_object="ball"))

    return py_trees.trees.BehaviourTree(root)

def create_Unlock_bt(env, obs):
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    has_key_selector = py_trees.composites.Selector(name="Has key selector", memory=False)
    has_key_selector.add_child(HasKey(name="Has key", env=env, obs=obs))
    has_key_selector.add_child(PickUp(name="Pick up key", env=env, obs=obs, goal_object="key"))

    root.add_child(has_key_selector)
    root.add_child(EnterRoom(name="Open door", env=env, obs=obs))

    return py_trees.trees.BehaviourTree(root)

def create_UnlockPickup_bt(env, obs):
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    has_key_selector = py_trees.composites.Selector(name="Has key selector", memory=False)
    has_key_selector.add_child(HasKey(name="Has key", env=env, obs=obs))
    has_key_selector.add_child(PickUp(name="Pick up key", env=env, obs=obs, goal_object="key"))

    open_door_sequence = py_trees.composites.Sequence(name="Open door sequence", memory=False)
    open_door_sequence.add_child(has_key_selector)
    open_door_sequence.add_child(EnterRoom(name="Open door", env=env, obs=obs))

    inside_locked_room_selector = py_trees.composites.Selector(name="Door open selector", memory=False)
    inside_locked_room_selector.add_child(IsInsideRoom(name="Is door open", env=env, obs=obs))
    inside_locked_room_selector.add_child(open_door_sequence)

    root.add_child(inside_locked_room_selector)
    root.add_child(PickUpGoal(name="Pick up goal", env=env, obs=obs, goal_object="box"))

    return py_trees.trees.BehaviourTree(root)

def create_DoorKey_bt(env, obs):
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    has_key_selector = py_trees.composites.Selector(name="Has key selector", memory=False)
    has_key_selector.add_child(HasKey(name="Has key", env=env, obs=obs))
    has_key_selector.add_child(PickUp(name="Pick up key", env=env, obs=obs, goal_object="key"))

    open_door_sequence = py_trees.composites.Sequence(name="Open door sequence", memory=False)
    open_door_sequence.add_child(has_key_selector)
    open_door_sequence.add_child(EnterRoom(name="Open door", env=env, obs=obs))

    inside_locked_room_selector = py_trees.composites.Selector(name="Door open selector", memory=False)
    inside_locked_room_selector.add_child(IsInsideRoom(name="Is door open", env=env, obs=obs))
    inside_locked_room_selector.add_child(open_door_sequence)

    root.add_child(inside_locked_room_selector)
    root.add_child(GoToGoal(name="Go to Goal", env=env, obs=obs, goal_object="goal"))

    return py_trees.trees.BehaviourTree(root)

def create_MultiRoom_bt(env, obs):
    root = py_trees.composites.Sequence(name="Root sequence", memory=False)

    open_doors_selector = py_trees.composites.Selector(name="Doors open selector", memory=False)
    open_doors_selector.add_child(AllDoorsOpen(name="Are doors open", env=env, obs=obs))
    open_doors_selector.add_child(OpenDoors(name="Open doors", env=env, obs=obs))
    
    enter_room_sequence = py_trees.composites.Sequence(name="Enter final room", memory=False)
    enter_room_sequence.add_child(open_doors_selector)
    enter_room_sequence.add_child(EnterStep(name="Enter room", env=env, obs=obs))

    inside_locked_room_selector = py_trees.composites.Selector(name="Inside room", memory=False)
    inside_locked_room_selector.add_child(InFinalRoom(name="Is in final room", env=env, obs=obs))
    inside_locked_room_selector.add_child(enter_room_sequence)

    root.add_child(inside_locked_room_selector)
    root.add_child(GoToGoal(name="Go to Goal", env=env, obs=obs, goal_object="goal"))

    return py_trees.trees.BehaviourTree(root)