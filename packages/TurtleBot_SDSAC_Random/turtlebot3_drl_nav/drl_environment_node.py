"""ROS 2 (Foxy) adapter of the Phase-1 environment. Shared layer.

All decisions live in :mod:`episode_engine`; this node only converts ROS
messages and simulator services into engine inputs and executes the effects the
engine returns. It records ``transitions.csv`` and ``episodes.csv``.
"""

import math
import os
import sys
import time
from typing import Dict, List, Optional

import rclpy
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Empty

from . import episode_engine as ee
from .collision import classify_collision_names
from .env_config import (
    ENV_NODE_NAME,
    build_arena,
    build_environment_config,
    build_law,
    load_common_parameters,
    load_evaluation_protocol,
)
from .identity import config_digest, read_identity, release_digest, shared_layer_digest
from .initialization import Sampler, read_scenarios
from .protocol import EpisodeSpec, decode_action, decode_json, encode_json
from .recorder import open_stream
from .ros_qos import latched_qos
from .state import quaternion_to_yaw

EXIT_FATAL = 3


def _stamp(msg_header) -> float:
    return float(msg_header.stamp.sec) + float(msg_header.stamp.nanosec) * 1e-9


class DRLEnvironmentNode(Node):
    def __init__(self) -> None:
        super().__init__(ENV_NODE_NAME)
        self.declare_parameter("package_root", "")
        self.declare_parameter("run_dir", "")
        package_root = os.path.expanduser(str(self.get_parameter("package_root").value))
        run_dir = os.path.expanduser(str(self.get_parameter("run_dir").value))
        if not package_root or not run_dir:
            raise ValueError("package_root and run_dir are required")
        config_dir = os.path.join(package_root, "config")
        defaults = load_common_parameters(config_dir)
        for name, value in defaults.items():
            if name == "use_sim_time":
                continue
            self.declare_parameter(name, value)
        params: Dict[str, object] = {}
        for name in defaults:
            if name == "use_sim_time":
                params[name] = True
                continue
            value = self.get_parameter(name).value
            params[name] = list(value) if isinstance(value, (list, tuple)) else value
        self.params = params
        if not bool(self.get_parameter("use_sim_time").value):
            raise ValueError("use_sim_time must be true: the control tick runs on the simulation clock")

        self.identity = read_identity(os.path.join(run_dir, "run_identity.json"))
        if self.identity["action_space"] != "discrete" or self.identity["arm"] != "random":
            raise ValueError("this environment package accepts only the discrete random-initial-state arm")
        for name in ("world_seed", "initialization_seed", "dynamic_obstacle_seed", "evaluation_seed"):
            if int(params[name]) != int(self.identity[name]):
                raise ValueError(f"parameter {name}={params[name]} disagrees with run_identity.json ({self.identity[name]})")
        actual_config = config_digest(package_root)
        if actual_config != self.identity["config_sha256"]:
            raise ValueError("configuration digest of the package differs from run_identity.json")
        actual_shared = shared_layer_digest(package_root)
        if actual_shared != self.identity["shared_layer_sha256"]:
            raise ValueError("shared-layer digest of the package differs from run_identity.json")
        if release_digest(package_root) != self.identity["release_sha256"]:
            raise ValueError("release digest of the package differs from run_identity.json")

        world_path = os.path.join(package_root, "worlds", f"{params['world_id']}.world")
        self.config = build_environment_config(params)
        arena = build_arena(params, world_path)
        sampler = Sampler(build_law(params), arena)
        protocol = load_evaluation_protocol(config_dir)
        scenarios = read_scenarios(os.path.join(config_dir, protocol["scenario_file"]))
        self.engine = ee.EpisodeEngine(self.config, arena, sampler, scenarios, wall_clock=time.monotonic)

        self.run_dir = run_dir
        self.transitions = open_stream(run_dir, "transitions", self.identity)
        self.episodes = open_stream(run_dir, "episodes", self.identity)
        self.exit_code = 0

        self.latest_scan: Optional[ee.ScanSample] = None
        self.latest_odom: Optional[ee.OdomSample] = None
        self.obstacle_samples: Dict[str, ee.ObstacleSample] = {}
        self.pending_service: Optional[tuple] = None  # (kind, future, entity)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.state_pub = self.create_publisher(Float32MultiArray, "/drl/state", 10)
        self.step_pub = self.create_publisher(Float32MultiArray, "/drl/step", 10)
        self.summary_pub = self.create_publisher(String, "/drl/episode_summary", 10)
        self.obstacle_pub = self.create_publisher(String, "/drl/obstacle_control", 10)
        self.ready_pub = self.create_publisher(String, "/drl/env_ready", latched_qos())

        self.reset_client = self.create_client(Empty, "/reset_world")
        self.set_client = self.create_client(SetEntityState, "/set_entity_state")
        self.get_client = self.create_client(GetEntityState, "/get_entity_state")
        self.pause_client = self.create_client(Empty, "/pause_physics")
        self.unpause_client = self.create_client(Empty, "/unpause_physics")
        for client, name in (
            (self.reset_client, "/reset_world"), (self.set_client, "/set_entity_state"),
            (self.get_client, "/get_entity_state"), (self.pause_client, "/pause_physics"),
            (self.unpause_client, "/unpause_physics"),
        ):
            if not client.wait_for_service(timeout_sec=60.0):
                raise RuntimeError(f"simulator service {name} unavailable; is libgazebo_ros_state.so loaded?")

        self.create_subscription(LaserScan, "/scan", self._scan_callback, 10)
        self.create_subscription(Odometry, "/odom", self._odom_callback, 10)
        for index, name in enumerate(self.config.obstacle_names):
            self.create_subscription(Odometry, f"/{name}/odom", self._make_obstacle_callback(name), 10)
        self.create_subscription(ContactsState, str(params["collision_topic"]), self._contact_callback, 10)
        self.create_subscription(Float32MultiArray, "/drl/action", self._action_callback, 10)
        self.create_subscription(String, "/drl/episode_control", self._control_callback, 10)
        self.create_subscription(String, "/drl/obstacle_status", self._obstacle_status_callback, latched_qos())
        self.contact_topic = str(params["collision_topic"])
        self.contact_seen = False
        self.start_wall = time.monotonic()
        self.obstacle_ready = False
        # The supervisory timer must continue while Gazebo physics is paused;
        # all transition durations are still measured from simulation stamps.
        self._latest_sim_time = 0.0
        self.supervisor_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.002, self._tick, clock=self.supervisor_clock)
        self.ready_published = False
        self.get_logger().info("Phase-1 environment (shared layer 3.0.3) initialized; waiting for all live inputs")

    # ---------------------------------------------------------- callbacks

    def _update_sim_time(self, stamp: float) -> None:
        if stamp > self._latest_sim_time:
            self._latest_sim_time = stamp

    def _now(self) -> float:
        ros_clock = self.get_clock().now().nanoseconds * 1e-9
        if ros_clock < self._latest_sim_time - 1.0:
            self._latest_sim_time = ros_clock
        return max(ros_clock, self._latest_sim_time)

    def _scan_callback(self, msg: LaserScan) -> None:
        s = _stamp(msg.header)
        self._update_sim_time(s)
        self.latest_scan = ee.ScanSample(tuple(msg.ranges), msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max, s)

    def _odom_callback(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        s = _stamp(msg.header)
        self._update_sim_time(s)
        self.latest_odom = ee.OdomSample(msg.pose.pose.position.x, msg.pose.pose.position.y, quaternion_to_yaw(q.x, q.y, q.z, q.w), s)

    def _make_obstacle_callback(self, name: str):
        def callback(msg: Odometry) -> None:
            s = _stamp(msg.header)
            self._update_sim_time(s)
            self.obstacle_samples[name] = ee.ObstacleSample(name, msg.pose.pose.position.x, msg.pose.pose.position.y, s)
        return callback

    def _contact_callback(self, msg: ContactsState) -> None:
        self.contact_seen = True
        for contact in msg.states:
            collision, static, dynamic = classify_collision_names((contact.collision1_name, contact.collision2_name))
            if collision and not (static or dynamic):
                self._execute([ee.Fatal(
                    "contact involved an unclassified collision name; static/dynamic exhaustiveness is required"
                )])
                return
            self.engine.note_contact(collision, static, dynamic)

    def _action_callback(self, msg: Float32MultiArray) -> None:
        try:
            action = decode_action(list(msg.data))
        except ValueError as error:
            self._execute([ee.Fatal(f"malformed action: {error}")])
            return
        self._execute(self.engine.receive_action(action))

    def _control_callback(self, msg: String) -> None:
        try:
            payload = decode_json(msg.data)
            if payload.get("cmd") != "start_episode":
                return
            spec = EpisodeSpec.from_payload(payload)
        except (ValueError, KeyError) as error:
            self._execute([ee.Fatal(f"malformed episode control: {error}")])
            return
        self._execute(self.engine.start_episode(spec, self._now()))

    def _obstacle_status_callback(self, msg: String) -> None:
        try:
            payload = decode_json(msg.data)
        except ValueError as error:
            self._execute([ee.Fatal(f"malformed obstacle status: {error}")])
            return
        if payload.get("cmd") == "ready":
            self.obstacle_ready = True
            return
        self._execute(self.engine.obstacle_ack(payload))

    # --------------------------------------------------------------- tick

    def _tick(self) -> None:
        if self.engine.state == ee.EpisodeEngine.FATAL:
            return
        if not self.ready_published:
            if self.count_publishers(self.contact_topic) == 0:
                if time.monotonic() - self.start_wall > 60.0:
                    self._execute([ee.Fatal(f"no publisher on {self.contact_topic}")])
                return
            if self.latest_scan is None or self.latest_odom is None or len(self.obstacle_samples) < 2 or not self.obstacle_ready:
                if time.monotonic() - self.start_wall > 60.0:
                    self._execute([ee.Fatal("startup inputs did not become live within 60 seconds")])
                return
            self.ready_pub.publish(String(data=encode_json({
                "cmd": "env_ready", "config_sha256": self.identity["config_sha256"],
                "shared_layer_sha256": self.identity["shared_layer_sha256"], "run_id": self.identity["run_id"],
            })))
            self.ready_published = True
            self.get_logger().info("environment ready; announced identity to the agent")
        self._poll_service()
        obstacles = [self.obstacle_samples[n] for n in self.config.obstacle_names if n in self.obstacle_samples]
        self._execute(self.engine.tick(self._now(), self.latest_scan, self.latest_odom, obstacles))

    def _poll_service(self) -> None:
        if self.pending_service is None:
            return
        kind, future, entity = self.pending_service
        if not future.done():
            return
        self.pending_service = None
        try:
            result = future.result()
        except Exception as error:  # noqa: BLE001 - any transport failure is a failed reset
            self._execute(self.engine.service_result(kind, False, None, self._now()))
            self.get_logger().error(f"service {kind} raised: {error}")
            return
        if kind in ("reset_world", "pause_physics", "unpause_physics"):
            self._execute(self.engine.service_result(kind, True, None, self._now()))
        elif kind == "set_entity_state":
            self._execute(self.engine.service_result(kind, bool(result.success), None, self._now()))
        else:
            q = result.state.pose.orientation
            payload = {
                "name": entity, "x": result.state.pose.position.x, "y": result.state.pose.position.y,
                "yaw": quaternion_to_yaw(q.x, q.y, q.z, q.w),
            }
            self._execute(self.engine.service_result(kind, bool(result.success), payload, self._now()))

    # ------------------------------------------------------------ effects

    def _execute(self, effects: List[object]) -> None:
        for effect in effects:
            if isinstance(effect, ee.PublishVelocity):
                twist = Twist()
                twist.linear.x = float(effect.linear)
                twist.angular.z = float(effect.angular)
                self.cmd_pub.publish(twist)
            elif isinstance(effect, ee.PublishState):
                self.state_pub.publish(Float32MultiArray(data=effect.values))
            elif isinstance(effect, ee.PublishStep):
                self.step_pub.publish(Float32MultiArray(data=effect.values))
            elif isinstance(effect, ee.PublishEpisodeSummary):
                self.summary_pub.publish(String(data=encode_json(effect.payload)))
            elif isinstance(effect, ee.ObstacleControl):
                self.obstacle_pub.publish(String(data=encode_json(effect.payload)))
            elif isinstance(effect, ee.CallService):
                self._call(effect)
            elif isinstance(effect, ee.RecordTransition):
                self.transitions.write(effect.row)
            elif isinstance(effect, ee.RecordEpisode):
                self.episodes.write(effect.row)
                self.get_logger().info(
                    f"training episode {effect.row['training_episode']} {effect.row['outcome']} "
                    f"len={effect.row['length']} return={float(effect.row['return']):.3f} env_step={effect.row['end_env_step']}"
                )
            elif isinstance(effect, ee.Log):
                self.get_logger().info(effect.text)
            elif isinstance(effect, ee.Fatal):
                self._fatal(effect.reason)
            else:
                self._fatal(f"unknown effect {effect!r}")

    def _call(self, call: ee.CallService) -> None:
        if self.pending_service is not None:
            self._fatal("a simulator service call is already pending")
            return
        if call.kind == "reset_world":
            future = self.reset_client.call_async(Empty.Request())
        elif call.kind == "pause_physics":
            future = self.pause_client.call_async(Empty.Request())
        elif call.kind == "unpause_physics":
            future = self.unpause_client.call_async(Empty.Request())
        elif call.kind == "set_entity_state":
            request = SetEntityState.Request()
            request.state.name = call.entity
            x, y, yaw = call.pose
            request.state.pose.position.x = float(x)
            request.state.pose.position.y = float(y)
            request.state.pose.position.z = 0.01
            request.state.pose.orientation.z = math.sin(yaw / 2.0)
            request.state.pose.orientation.w = math.cos(yaw / 2.0)
            request.state.reference_frame = "world"
            request.state.twist.linear.x = 0.0
            request.state.twist.linear.y = 0.0
            request.state.twist.linear.z = 0.0
            request.state.twist.angular.x = 0.0
            request.state.twist.angular.y = 0.0
            request.state.twist.angular.z = 0.0
            future = self.set_client.call_async(request)
        elif call.kind == "get_entity_state":
            request = GetEntityState.Request()
            request.name = call.entity
            request.reference_frame = "world"
            future = self.get_client.call_async(request)
        else:
            self._fatal(f"unknown service kind {call.kind}")
            return
        self.pending_service = (call.kind, future, call.entity)

    def _fatal(self, reason: str) -> None:
        self.get_logger().fatal(reason)
        stop = Twist()
        self.cmd_pub.publish(stop)
        with open(os.path.join(self.run_dir, "ENV_FATAL.tmp"), "w", encoding="utf-8") as stream:
            stream.write(reason + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(os.path.join(self.run_dir, "ENV_FATAL.tmp"), os.path.join(self.run_dir, "ENV_FATAL"))
        self.exit_code = EXIT_FATAL
        self.close()
        if rclpy.ok():
            rclpy.shutdown()
        import os as _os
        _os._exit(EXIT_FATAL)

    def close(self) -> None:
        self.transitions.close()
        self.episodes.close()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[DRLEnvironmentNode] = None
    exit_code = 0
    try:
        node = DRLEnvironmentNode()
        rclpy.spin(node)
    except KeyboardInterrupt:  # launch shutdown delivers SIGINT; an orderly stop is not a failure
        pass
    except Exception as error:  # noqa: BLE001 - any startup failure is fatal and must be visible
        print(f"drl_environment fatal: {error}", file=sys.stderr)
        exit_code = EXIT_FATAL
    finally:
        if node is not None:
            exit_code = max(exit_code, node.exit_code)
            try:
                node.cmd_pub.publish(Twist())
            except Exception:  # noqa: BLE001
                pass
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        import os as _os
        _os._exit(EXIT_FATAL)
    sys.exit(exit_code)
