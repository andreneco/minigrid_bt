from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="minigrid_bt",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'minigrid_bt = minigrid_bt.main:main',
        ],
    }
)
