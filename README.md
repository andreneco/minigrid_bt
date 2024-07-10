# Minigrid BT

Minigrid BT is a Python package for creating behavior trees to solve minigrid environments. This package uses the `py_trees` library to implement behavior trees and the `gymnasium` library for the minigrid environments.

## Installation

To install the package, navigate to the root directory (where `setup.py` is located) and run:

```bash
pip install -e .
```

## Usage

You can run the package using the console script defined in `setup.py`:

```bash
minigrid_bt
```

## Directory Structure

```
minigrid_bt/
├── minigrid_bt/
│   ├── __init__.py
│   ├── behaviors.py
│   ├── conditions.py
│   ├── utils.py
│   ├── policy.py
│   └── main.py
├── setup.py
└── README.md
```

### Modules

- `minigrid_bt/behaviors.py`: Contains the behavior classes for the behavior tree.
- `minigrid_bt/conditions.py`: Contains the condition classes for the behavior tree.
- `minigrid_bt/utils.py`: Contains utility functions and constants used across the package.
- `minigrid_bt/policy.py`: Contains the implementation of the `BehaviorTreePolicy` class.
- `minigrid_bt/main.py`: Contains the main function to run the behavior tree and create the behavior tree structure.

### Example Code

Here's a brief overview of what each part of the code does:

#### behaviors.py

Defines the behavior nodes for the behavior tree, such as `PickUpGoal`, `EnterRoom`, `PickUpKey`, and `ClearPath`.

#### conditions.py

Defines the condition nodes for the behavior tree, such as `HasKey`, `IsInsideRoom`, and `IsPathClear`.

#### utils.py

Contains utility functions for extracting information from observations, pathfinding, and action preparation.

#### policy.py

Defines the `BehaviorTreePolicy` class that integrates the behavior tree with the stable-baselines3 policy interface.

#### main.py

Creates and runs the behavior tree within a minigrid environment. This is the entry point of the package.

## Contributing

Feel free to contribute by submitting issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

This `README.md` provides a comprehensive overview of the package, including installation instructions, usage, and a description of the directory structure and module contents.