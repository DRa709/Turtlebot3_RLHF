"""End-to-end run of the two ROS adapters (environment node, agent node) and
the obstacle node on in-memory rclpy stubs, driven by the fake simulator:
readiness announcement, transactional reset through the service clients,
message encoding on the wire, recording, checkpoints, tier-1 evaluation and
completion markers. Mock-backed evidence for Gates 3, 4, 7 and 10; it does not
certify ROS or Gazebo behaviour."""

import importlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

import ros_stubs  # noqa: E402

ros_stubs.install()

from fake_sim import FakeSim  # noqa: E402

import yaml  # noqa: E402

from turtlebot3_drl_nav.env_config import build_arena, load_common_parameters  # noqa: E402
from turtlebot3_drl_nav.identity import build_identity, config_digest, release_digest, sha256_file, shared_layer_digest, write_identity  # noqa: E402
from turtlebot3_drl_nav.launch_config import build_node_parameters, package_algorithm  # noqa: E402
from turtlebot3_drl_nav.recorder import read_stream  # noqa: E402
from turtlebot3_drl_nav.state import quaternion_to_yaw  # noqa: E402
from turtlebot3_drl_nav.validator import validate_run  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = {"world_seed": 7000, "initialization_seed": 7001, "dynamic_obstacle_seed": 7002, "evaluation_seed": 9001}


def tiny_package(tmp: str) -> str:
    root = os.path.join(tmp, "pkg")
    shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    common_path = os.path.join(root, "config", "common_environment.yaml")
    with open(common_path, encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    document["drl_environment"]["ros__parameters"]["episode_max_steps"] = 6
    document["evaluation_protocol"]["phases"]["pilot"].update({
        "environment_budget": 40, "checkpoint_interval_steps": 20, "full_checkpoint_interval_steps": 40,
        "tier2_checkpoint_steps": [40], "tier1_episodes_per_checkpoint": 2, "learning_seeds": [101],
    })
    with open(common_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=False)
    _, config_name, executable = package_algorithm(root)
    algo_path = os.path.join(root, "config", config_name)
    with open(algo_path, encoding="utf-8") as stream:
        algo = yaml.safe_load(stream)
    algo[executable]["ros__parameters"].update({
        "warmup_steps": 8, "target_update_steps": 5, "batch_size": 4,
        "hidden_size": 8, "atoms": 5, "v_min": -2.0, "v_max": 2.0,
    })
    with open(algo_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(algo, stream, sort_keys=False)
    # regenerate manifests for the copy so the shared-layer digest matches its own files
    import subprocess
    subprocess.run([sys.executable, os.path.join(root, "scripts", "make_manifests.py")], check=True, capture_output=True)
    return root


class MockRosRunTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is required for the ROS-agent mock run")
    def test_full_training_run_through_the_ros_adapters(self):
        ros_stubs.BUS.reset()
        with tempfile.TemporaryDirectory() as tmp:
            pkg = tiny_package(tmp)
            run_dir = os.path.join(tmp, "run")
            os.makedirs(run_dir)
            algorithm, _, executable = package_algorithm(pkg)
            with open(os.path.join(pkg, "VERSION"), encoding="utf-8") as version_stream:
                package_version = version_stream.read().strip()
            identity = build_identity("phase1-random-arm", algorithm, package_version, "random", 101, "phase1_mixed", 7000, 7001, 7002, 9001,
                                      config_digest(pkg), "0" * 64, shared_layer_digest(pkg), release_digest(pkg),
                                      package_version, "training", "pilot", "mock")
            write_identity(os.path.join(run_dir, "run_identity.json"), identity)
            params = build_node_parameters({"package_root": pkg, "run_dir": run_dir, "mode": "train", "learning_seed": "101",
                                            **{k: str(v) for k, v in SEEDS.items()}, "phase_label": "pilot", "checkpoint_path": ""})
            ros_stubs.Node.overrides = {"drl_environment": params["environment"], executable: params["agent"], "dynamic_obstacles": {}}

            common = load_common_parameters(os.path.join(pkg, "config"))
            common.update(SEEDS)
            arena = build_arena(common, os.path.join(pkg, "worlds", "phase1_mixed.world"))
            sim = FakeSim(arena, float(common["control_period"]))

            # simulator services on the bus
            def reset_world(request):
                sim.robot = [0.0, 0.0, 0.0]
                sim.cmd = (0.0, 0.0)
                sim.schedule = None
                for d in arena.dynamic_obstacles:
                    sim.obstacles[d.name] = [d.shape.cx, d.shape.cy]
                return ros_stubs.Empty.Response()

            def set_entity_state(request):
                q = request.state.pose.orientation
                sim.robot = [request.state.pose.position.x, request.state.pose.position.y, quaternion_to_yaw(q.x, q.y, q.z, q.w)]
                response = ros_stubs.SetEntityState.Response()
                response.success = request.state.name == "burger"
                return response

            def get_entity_state(request):
                response = ros_stubs.GetEntityState.Response()
                if request.name == "burger":
                    x, y, yaw = sim.robot
                else:
                    x, y = sim.obstacles[request.name]
                    yaw = 0.0
                response.state.pose.position.x, response.state.pose.position.y = x, y
                response.state.pose.orientation.z, response.state.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
                response.success = True
                return response

            def pause_physics(request):
                sim.paused = True
                return ros_stubs.Empty.Response()

            def unpause_physics(request):
                sim.paused = False
                return ros_stubs.Empty.Response()

            ros_stubs.BUS.services.update({
                "/reset_world": reset_world,
                "/set_entity_state": set_entity_state,
                "/get_entity_state": get_entity_state,
                "/pause_physics": pause_physics,
                "/unpause_physics": unpause_physics,
            })
            # sensor publishers the simulator would own
            scan_pub = ros_stubs._Publisher("/scan", False)
            odom_pub = ros_stubs._Publisher("/odom", False)
            obs_pubs = {d.name: ros_stubs._Publisher(f"/{d.name}/odom", False) for d in arena.dynamic_obstacles}
            contact_pub = ros_stubs._Publisher("/bumper_states", False)
            ros_stubs.BUS.subscriptions.setdefault("/cmd_vel", []).append(lambda m: setattr(sim, "cmd", (m.linear.x, m.angular.z)))

            env_mod = importlib.import_module("turtlebot3_drl_nav.drl_environment_node")
            agent_mod = importlib.import_module(f"turtlebot3_drl_nav.{executable}_node")
            obstacle_mod = importlib.import_module("turtlebot3_drl_nav.dynamic_obstacle_node")
            env = env_mod.DRLEnvironmentNode()
            obstacles = obstacle_mod.DynamicObstacleNode()
            agent = agent_mod.AGENT_NODE_CLASS()

            # obstacle node drives the fake obstacles through its cmd_vel publishers
            def obstacle_cmd(name):
                def cb(msg):
                    sim.obstacle_cmd = getattr(sim, "obstacle_cmd", {})
                    sim.obstacle_cmd[name] = (msg.linear.x, msg.linear.y)
                return cb
            for d in arena.dynamic_obstacles:
                ros_stubs.BUS.subscriptions.setdefault(f"/{d.name}/cmd_vel", []).append(obstacle_cmd(d.name))

            def obstacle_control(msg):
                payload = json.loads(msg.data)
                if payload.get("cmd") == "hold":
                    sim.schedule = None
                elif payload.get("cmd") == "start":
                    sim.schedule = {
                        "t0": sim.sim_time,
                        "signs": payload["signs"],
                        "offsets": payload["offsets"],
                        "speed": payload["speed"],
                        "half_period": payload["half_period"],
                    }

            ros_stubs.BUS.subscriptions.setdefault("/drl/obstacle_control", []).append(obstacle_control)

            def publish_sensors():
                scan = sim.scan()
                msg = ros_stubs.LaserScan()
                msg.ranges = list(scan.ranges)
                msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max = scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max
                msg.header.stamp.sec, msg.header.stamp.nanosec = int(sim.sim_time), int((sim.sim_time % 1) * 1e9)
                scan_pub.publish(msg)
                odom = ros_stubs.Odometry()
                odom.pose.pose.position.x, odom.pose.pose.position.y = sim.robot[0], sim.robot[1]
                odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(sim.robot[2] / 2), math.cos(sim.robot[2] / 2)
                odom.header.stamp.sec, odom.header.stamp.nanosec = msg.header.stamp.sec, msg.header.stamp.nanosec
                odom_pub.publish(odom)
                for name, pub in obs_pubs.items():
                    o = ros_stubs.Odometry()
                    o.pose.pose.position.x, o.pose.pose.position.y = sim.obstacles[name]
                    o.header.stamp.sec, o.header.stamp.nanosec = msg.header.stamp.sec, msg.header.stamp.nanosec
                    pub.publish(o)
                collision, static, dynamic = sim.contact()
                contacts = ros_stubs.ContactsState()
                if collision:
                    contacts.states.append(ros_stubs._Contact("burger::base_link::base_collision", "static_box_1::link::static_collision" if static else "dynamic_obstacle_1::link::dynamic_collision"))
                contact_pub.publish(contacts)

            for _ in range(20000):
                if not ros_stubs.BUS.ok:
                    break
                # physics: the simulator advances only while unpaused
                sim.step_physics()
                ros_stubs.BUS.sim_time_ns = int(sim.sim_time * 1e9)
                publish_sensors()
                for timer in list(ros_stubs.BUS.timers):
                    timer.callback()
            self.assertFalse(ros_stubs.BUS.ok, "the run did not finish")
            self.assertTrue(os.path.isfile(os.path.join(run_dir, "AGENT_DONE")), ros_stubs.BUS.log[-5:])
            self.assertFalse(os.path.exists(os.path.join(run_dir, "AGENT_FATAL")))
            self.assertFalse(os.path.exists(os.path.join(run_dir, "ENV_FATAL")))
            env.close()
            agent.close()
            manifest = dict(identity)
            manifest.update({"environment_budget": 40, "slurm_job_id": "1", "slurm_array_job_id": "1",
                             "slurm_array_task_id": "0", "hostname": "mock-node", "started_utc": "2026-09-03T00:00:00Z"})
            with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            sim_dir = os.path.join(run_dir, "sim")
            os.makedirs(sim_dir, exist_ok=True)
            runtime_world = os.path.join(sim_dir, "phase1_mixed.world")
            runtime_robot = os.path.join(sim_dir, "turtlebot3_burger_with_contact.sdf")
            shutil.copyfile(os.path.join(pkg, "worlds", "phase1_mixed.world"), runtime_world)
            with open(runtime_robot, "w", encoding="utf-8") as stream:
                stream.write("<sdf version='1.6'><model name='burger'/></sdf>\n")
            for payload, sidecar in ((runtime_world, "runtime_world.sha256"), (runtime_robot, "runtime_robot_model.sha256")):
                with open(os.path.join(sim_dir, sidecar), "w", encoding="utf-8") as stream:
                    stream.write(f"{sha256_file(payload)}  {payload}\n")
            with open(os.path.join(run_dir, "runtime_manifest.txt"), "w", encoding="utf-8") as stream:
                stream.write("Python 3.8.10\nnumpy=1.24.4\ntorch=2.4.1+cpu\nros_distro=foxy\nGazebo multi-robot simulator, version 11\nros_domain_id=1 gazebo_master_uri=http://127.0.0.1:11345\n")
            report = validate_run(pkg, run_dir)
            self.assertEqual(report.failures, [])
            evaluation = read_stream(os.path.join(run_dir, "evaluation.csv"))
            self.assertEqual(len(evaluation), 4)
            transitions = read_stream(os.path.join(run_dir, "transitions.csv"))
            training = [r for r in transitions if r["phase"] == "training" and r["step_in_episode"] != "0"]
            self.assertEqual(len(training), 40)
            self.assertTrue(any("environment ready" in line for line in ros_stubs.BUS.log))


if __name__ == "__main__":
    unittest.main()
