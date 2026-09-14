"""Training / evaluation launch: environment node + the package's agent node, fail-fast.

All argument forwarding is done by turtlebot3_drl_nav.launch_config so it is
unit-tested without ROS. The obstacle node is started with the simulator
(scripts/run_sim_headless.sh), because it belongs to the world.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from turtlebot3_drl_nav.launch_config import LAUNCH_ARGUMENTS, build_node_parameters, package_algorithm


def _nodes(context, *args, **kwargs):
    values = {name: LaunchConfiguration(name).perform(context) for name in LAUNCH_ARGUMENTS}
    params = build_node_parameters(values)
    package_root = params["environment"]["package_root"]
    _, config_name, executable = package_algorithm(package_root)
    common_config = os.path.join(package_root, "config", "common_environment.yaml")
    algorithm_config = os.path.join(package_root, "config", config_name)
    environment_node = Node(
        package="turtlebot3_drl_nav", executable="drl_environment", name="drl_environment", output="screen",
        parameters=[common_config, params["environment"]],
    )
    agent_node = Node(
        package="turtlebot3_drl_nav", executable=executable, name=executable, output="screen",
        parameters=[algorithm_config, params["agent"]],
    )
    fail_fast = [
        RegisterEventHandler(OnProcessExit(target_action=node, on_exit=[EmitEvent(event=Shutdown(reason=f"{label} exited"))]))
        for node, label in ((environment_node, "environment"), (agent_node, "agent"))
    ]
    return [environment_node, agent_node] + fail_fast


def generate_launch_description():
    arguments = [DeclareLaunchArgument(name, default_value="") for name in LAUNCH_ARGUMENTS]
    return LaunchDescription(arguments + [OpaqueFunction(function=_nodes)])
