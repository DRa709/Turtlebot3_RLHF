"""ROS 2 (Foxy) Double DQN agent node. Algorithm-specific file.

All scheduling lives in the shared :mod:`orchestrator`; the learner is
:class:`doubledqn.DoubleDQNAgent`. This node converts messages, executes effects, records
``updates.csv``, ``evaluation.csv`` and ``checkpoints.csv``, and handles the
walltime warning signal forwarded by the ARC wrapper.
"""

import json
import os
import signal
import sys
import time
from typing import Dict, List, Optional

import rclpy
import torch
import yaml
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from . import orchestrator as orc
from .doubledqn import ALGORITHM, DoubleDQNAgent, DoubleDQNConfig, GreedyPolicy, _torch_load, parameter_digest, validate_checkpoint_payload
from .env_config import action_map_from, load_common_parameters, load_evaluation_protocol
from .identity import config_digest, read_identity, release_digest, sha256_file, shared_layer_digest
from .initialization import read_scenarios
from .protocol import decode_json, decode_state, decode_step, encode_action
from .recorder import open_stream
from .ros_qos import latched_qos
from .state import OBSERVATION_DIM

EXIT_FATAL = 3
EXIT_INTERRUPTED = 4
AGENT_CONFIG_NAME = "phase1_doubledqn.yaml"
NODE_NAME = "doubledqn_agent"


class DoubleDQNAgentNode(Node):
    def __init__(self) -> None:
        super().__init__(NODE_NAME)
        self.declare_parameter("package_root", "")
        self.declare_parameter("run_dir", "")
        self.declare_parameter("mode", "train")
        self.declare_parameter("learning_seed", -1)
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("phase_label", "")
        package_root = os.path.expanduser(str(self.get_parameter("package_root").value))
        run_dir = os.path.expanduser(str(self.get_parameter("run_dir").value))
        if not package_root or not run_dir:
            raise ValueError("package_root and run_dir are required")
        config_dir = os.path.join(package_root, "config")
        with open(os.path.join(config_dir, AGENT_CONFIG_NAME), encoding="utf-8") as stream:
            defaults = dict(yaml.safe_load(stream)[NODE_NAME]["ros__parameters"])
        for name, value in defaults.items():
            if name == "use_sim_time":
                continue
            self.declare_parameter(name, value)
        params: Dict[str, object] = {name: self.get_parameter(name).value for name in defaults if name != "use_sim_time"}
        self.mode = str(self.get_parameter("mode").value).strip().lower()
        if self.mode not in ("train", "eval"):
            raise ValueError("mode must be train or eval")
        learning_seed = int(self.get_parameter("learning_seed").value)
        if learning_seed < 0:
            raise ValueError("learning_seed is unset")
        if str(params["algorithm"]) != ALGORITHM:
            raise ValueError(f"configuration algorithm {params['algorithm']} is not {ALGORITHM}")

        self.identity = read_identity(os.path.join(run_dir, "run_identity.json"))
        if int(self.identity["learning_seed"]) != learning_seed or str(self.identity["algorithm"]) != ALGORITHM:
            raise ValueError("run_identity.json disagrees with the agent's seed or algorithm")
        expected_phase_type = "training" if self.mode == "train" else "evaluation"
        with open(os.path.join(package_root, "VERSION"), encoding="utf-8") as version_stream:
            package_version = version_stream.read().strip()
        if str(self.identity["algorithm_version"]) != str(params["algorithm_version"]):
            raise ValueError("algorithm_version differs between run identity and Double DQN configuration")
        if str(self.identity["package_version"]) != package_version:
            raise ValueError("package_version differs between run identity and VERSION")
        if self.identity["action_space"] != "discrete" or self.identity["arm"] != "random":
            raise ValueError("this package accepts only the discrete random-initial-state arm")
        if self.identity["phase_type"] != expected_phase_type:
            raise ValueError("run identity phase_type disagrees with agent mode")
        phase_label = str(self.get_parameter("phase_label").value).strip()
        if phase_label != str(self.identity["phase_label"]):
            raise ValueError("phase_label parameter disagrees with run_identity.json")
        if config_digest(package_root) != self.identity["config_sha256"]:
            raise ValueError("configuration digest of the package differs from run_identity.json")
        if shared_layer_digest(package_root) != self.identity["shared_layer_sha256"]:
            raise ValueError("shared-layer digest of the package differs from run_identity.json")
        if release_digest(package_root) != self.identity["release_sha256"]:
            raise ValueError("release digest of the package differs from run_identity.json")

        common = load_common_parameters(config_dir)
        protocol = load_evaluation_protocol(config_dir)
        phase = dict(protocol["phases"][phase_label])
        scenarios = read_scenarios(os.path.join(config_dir, protocol["scenario_file"]))
        action_map = action_map_from(common)
        commands = action_map.commands()

        device_name = str(params["device"])
        if device_name != "cpu":
            self.get_logger().warn("controlled runs are CPU-pinned; a non-CPU device is a configuration change")
        self.device = torch.device(device_name)
        learner_config = DoubleDQNConfig(
            gamma=float(params["gamma"]), learning_rate=float(params["learning_rate"]), batch_size=int(params["batch_size"]),
            replay_capacity=int(params["replay_capacity"]), warmup_steps=int(params["warmup_steps"]),
            target_update_steps=int(params["target_update_steps"]), epsilon_start=float(params["epsilon_start"]),
            epsilon_end=float(params["epsilon_end"]), epsilon_decay_steps=int(params["epsilon_decay_steps"]),
            hidden_size=int(params["hidden_size"]), gradient_clip_norm=float(params["gradient_clip_norm"]),
            loss=str(params["loss"]), torch_threads=int(params["torch_threads"]),
        )
        self.learner = DoubleDQNAgent(OBSERVATION_DIM, len(commands), learner_config, self.device, learning_seed)
        budget = int(phase["environment_budget"])
        if learning_seed not in [int(v) for v in phase["learning_seeds"]]:
            raise ValueError(f"learning seed {learning_seed} is not preregistered for phase {phase_label}")

        checkpoint_path = os.path.expanduser(str(self.get_parameter("checkpoint_path").value))
        eval_step = eval_index = 0
        if self.mode == "eval":
            if not checkpoint_path or not os.path.isfile(checkpoint_path):
                raise ValueError("eval mode requires an existing checkpoint_path")
            with open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8") as stream:
                eval_manifest = json.load(stream)
            if (
                os.path.realpath(checkpoint_path) != os.path.realpath(str(eval_manifest.get("checkpoint_path", "")))
                or sha256_file(checkpoint_path) != str(eval_manifest.get("checkpoint_sha256", ""))
            ):
                raise ValueError("evaluation checkpoint path or digest differs from run_manifest.json")
            payload = _torch_load(checkpoint_path, self.device)
            saved_config = validate_checkpoint_payload(payload)
            if saved_config != learner_config or int(payload["seed"]) != learning_seed:
                raise ValueError("evaluation checkpoint configuration or learning seed differs from this run")
            if payload.get("extra", {}).get("action_map") != [list(command) for command in commands]:
                raise ValueError("evaluation checkpoint action map differs from the frozen package action map")
            eval_step = int(payload["env_steps"])
            if eval_step != int(eval_manifest.get("checkpoint_step", -1)):
                raise ValueError("evaluation checkpoint step differs from run_manifest.json")
            eval_index = eval_step // int(phase["checkpoint_interval_steps"])
        elif checkpoint_path:
            raise ValueError("training runs start from scratch by policy; checkpoint_path is not accepted in train mode")

        self.run_dir = run_dir
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        orch_config = orc.OrchestratorConfig(
            algorithm=ALGORITHM, mode=self.mode, environment_budget=budget,
            checkpoint_interval=int(phase["checkpoint_interval_steps"]),
            full_checkpoint_interval=int(phase["full_checkpoint_interval_steps"]),
            tier1_episodes=int(phase["tier1_episodes_per_checkpoint"]),
            tier2_e3_episodes=int(protocol["tier2_e3_episodes"]), tier2_conditions=tuple(protocol["tier2_conditions"]),
            checkpoint_dir=checkpoint_dir, learning_seed=learning_seed, action_commands=commands,
            config_sha256=str(self.identity["config_sha256"]), eval_checkpoint_path=checkpoint_path,
            eval_checkpoint_step=eval_step, eval_checkpoint_index=eval_index, policy_modes=("greedy",),
        )
        self.orchestrator = orc.Orchestrator(
            orch_config, self.learner, lambda path: GreedyPolicy(path, self.device), scenarios,
            sha256_file, parameter_digest, wall_clock=time.monotonic,
        )
        self.streams = {}
        for name in ("updates", "evaluation", "checkpoints"):
            if self.mode == "train" or name == "evaluation":
                self.streams[name] = open_stream(run_dir, name, self.identity)
        self.exit_code = 0
        self.finished_reason: Optional[str] = None
        self.interrupt_requested = False

        self.action_pub = self.create_publisher(Float32MultiArray, "/drl/action", 10)
        self.control_pub = self.create_publisher(String, "/drl/episode_control", 10)
        self.create_subscription(String, "/drl/env_ready", self._ready_callback, latched_qos())
        self.create_subscription(Float32MultiArray, "/drl/state", self._state_callback, 10)
        self.create_subscription(Float32MultiArray, "/drl/step", self._step_callback, 10)
        self.create_subscription(String, "/drl/episode_summary", self._summary_callback, 10)
        # Housekeeping runs on wall time so an interrupted, stalled simulator cannot block the emergency checkpoint.
        try:
            self.create_timer(1.0, self._housekeeping, clock=Clock(clock_type=ClockType.SYSTEM_TIME))
        except TypeError:  # rclpy without the clock keyword
            self.create_timer(1.0, self._housekeeping)
        signal.signal(signal.SIGUSR1, self._on_walltime_warning)
        self.get_logger().info(
            f"{ALGORITHM} agent mode={self.mode} seed={learning_seed} budget={budget} device={self.device} "
            f"phase={phase_label} warmup={learner_config.warmup_steps} checkpoint_interval={phase['checkpoint_interval_steps']}"
        )

    # ---------------------------------------------------------- callbacks

    def _ready_callback(self, msg: String) -> None:
        try:
            payload = decode_json(msg.data)
        except ValueError as error:
            self._fatal(f"malformed env_ready: {error}")
            return
        self._execute(self.orchestrator.on_env_ready(payload))

    def _state_callback(self, msg: Float32MultiArray) -> None:
        try:
            state = decode_state(list(msg.data))
        except ValueError as error:
            self._fatal(f"malformed state: {error}")
            return
        self._execute(self.orchestrator.on_state(state))

    def _step_callback(self, msg: Float32MultiArray) -> None:
        try:
            step = decode_step(list(msg.data))
        except ValueError as error:
            self._fatal(f"malformed step: {error}")
            return
        self._execute(self.orchestrator.on_step(step))

    def _summary_callback(self, msg: String) -> None:
        try:
            payload = decode_json(msg.data)
        except ValueError as error:
            self._fatal(f"malformed episode summary: {error}")
            return
        if payload.get("cmd") != "episode_summary":
            return
        self._execute(self.orchestrator.on_episode_summary(payload))

    def _on_walltime_warning(self, signum, frame) -> None:  # noqa: ARG002
        self.interrupt_requested = True

    def _housekeeping(self) -> None:
        if self.interrupt_requested and self.finished_reason is None:
            self.interrupt_requested = False
            path = os.path.join(self.run_dir, "checkpoints", f"doubledqn_seed{self.learner.seed}_step{self.learner.env_steps}_interrupted_full.pt")
            try:
                self.learner.save(path, kind="full")
            except Exception as error:  # noqa: BLE001
                self.get_logger().error(f"emergency checkpoint failed: {error}")
            self._write_marker("INTERRUPTED", f"walltime signal at env_step={self.learner.env_steps}; checkpoint {path}")
            self.exit_code = EXIT_INTERRUPTED
            self.finished_reason = "interrupted"
            self.close()
            if rclpy.ok():
                rclpy.shutdown()

    # ------------------------------------------------------------ effects

    def _execute(self, effects: List[object]) -> None:
        for effect in effects:
            if isinstance(effect, orc.SendEpisodeControl):
                self.control_pub.publish(String(data=effect.spec.to_json()))
            elif isinstance(effect, orc.SendAction):
                m = effect.message
                self.action_pub.publish(Float32MultiArray(data=encode_action(m.sequence, m.action_index, m.linear, m.angular, m.final_transition)))
            elif isinstance(effect, orc.RecordUpdate):
                self.streams["updates"].write(effect.row)
            elif isinstance(effect, orc.RecordEvaluation):
                for row in effect.rows:
                    self.streams["evaluation"].write(row)
                self.get_logger().info(f"evaluation block recorded: {len(effect.rows)} episodes")
            elif isinstance(effect, orc.RecordCheckpoint):
                self.streams["checkpoints"].write(effect.row)
                self.get_logger().info(f"checkpoint {effect.row['kind']} at env_step {effect.row['env_step']}")
            elif isinstance(effect, orc.Progress):
                self._write_progress(effect.payload)
            elif isinstance(effect, orc.Finish):
                self.finished_reason = effect.reason
                self._write_marker("AGENT_DONE", effect.reason)
                self.get_logger().info(f"finished: {effect.reason}")
                self.close()
                if rclpy.ok():
                    rclpy.shutdown()
            elif isinstance(effect, orc.Fatal):
                self._fatal(effect.reason)
            else:
                self._fatal(f"unknown effect {effect!r}")

    def _write_progress(self, payload: Dict[str, object]) -> None:
        path = os.path.join(self.run_dir, "progress.json")
        with open(path + ".tmp", "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
        os.replace(path + ".tmp", path)

    def _write_marker(self, name: str, text: str) -> None:
        path = os.path.join(self.run_dir, name)
        with open(path + ".tmp", "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(path + ".tmp", path)

    def _fatal(self, reason: str) -> None:
        self.get_logger().fatal(reason)
        self._write_marker("AGENT_FATAL", reason)
        self.exit_code = EXIT_FATAL
        self.finished_reason = "fatal"
        self.close()
        if rclpy.ok():
            rclpy.shutdown()

    def close(self) -> None:
        for stream in self.streams.values():
            stream.close()


AGENT_NODE_CLASS = DoubleDQNAgentNode


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[DoubleDQNAgentNode] = None
    exit_code = 0
    try:
        node = DoubleDQNAgentNode()
        rclpy.spin(node)
    except KeyboardInterrupt:  # launch shutdown delivers SIGINT; an orderly stop is not a failure
        pass
    except Exception as error:  # noqa: BLE001
        print(f"{NODE_NAME} fatal: {error}", file=sys.stderr)
        exit_code = EXIT_FATAL
    finally:
        if node is not None:
            exit_code = max(exit_code, node.exit_code)
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(exit_code)
