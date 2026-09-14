from glob import glob

from setuptools import setup

package_name = "turtlebot3_drl_nav"

setup(
    name=package_name,
    version="1.0.2",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/drl_training.launch.py"]),
        ("share/" + package_name + "/config", glob("config/*.yaml") + glob("config/*.csv")),
        ("share/" + package_name + "/worlds", glob("worlds/*.world")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Asha Barua",
    maintainer_email="ashabarua.ab@gmail.com",
    description="TurtleBot3 Burger Phase-1 discrete navigation, random-initial-state arm: Double DQN package.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "drl_environment = turtlebot3_drl_nav.drl_environment_node:main",
            "doubledqn_agent = turtlebot3_drl_nav.doubledqn_agent_node:main",
            "dynamic_obstacles = turtlebot3_drl_nav.dynamic_obstacle_node:main",
        ],
    },
)
