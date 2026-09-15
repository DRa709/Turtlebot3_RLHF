import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from fake_sim import FakeSim  # noqa: E402

from turtlebot3_drl_nav.env_config import (  # noqa: E402
    build_arena,
    build_environment_config,
    build_law,
    load_common_parameters,
)
from turtlebot3_drl_nav.episode_engine import CallService, EpisodeEngine, ObstacleControl, OdomSample, PublishVelocity  # noqa: E402
from turtlebot3_drl_nav.initialization import Sampler, generate_scenarios  # noqa: E402
from turtlebot3_drl_nav.protocol import ActionMessage, EpisodeSpec, decode_state, decode_step  # noqa: E402
from turtlebot3_drl_nav.recorder import EPISODES_COLUMNS, TRANSITIONS_COLUMNS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config")
WORLD = os.path.join(ROOT, "worlds", "phase1_mixed.world")


def make_engine(seeds=(7000, 7001, 7002, 9001), **overrides):
    params = load_common_parameters(CONFIG)
    params.update({"world_seed": seeds[0], "initialization_seed": seeds[1], "dynamic_obstacle_seed": seeds[2], "evaluation_seed": seeds[3]})
    params.update(overrides)
    cfg = build_environment_config(params)
    arena = build_arena(params, WORLD)
    sampler = Sampler(build_law(params), arena)
    scenarios = generate_scenarios(sampler, seeds[3], 12, 3, cfg.obstacle_half_period)
    sim = FakeSim(arena, cfg.control_period)
    engine = EpisodeEngine(cfg, arena, sampler, scenarios, wall_clock=lambda: sim.wall)
    return engine, sim, cfg, scenarios


def run_reset(engine, sim, spec):
    sim.apply_effects(engine, engine.start_episode(spec, sim.sim_time))
    for _ in range(200):
        if engine.state in (EpisodeEngine.AWAIT_ACTION, EpisodeEngine.FATAL):
            break
        sim.tick(engine)
    return engine.state


def act(engine, sim, action_index, final=False):
    cfg = engine.cfg
    state_values = sim.states[-1] if engine.current.step_index == 0 else None
    sequence = engine.current.sequence
    linear, angular = cfg.action_map.commands()[action_index]
    sim.apply_effects(engine, engine.receive_action(ActionMessage(sequence, action_index, linear, angular, final)))
    sim.tick(engine)  # applies at this tick
    assert engine.state == EpisodeEngine.HOLD, engine.state
    prior_steps = len(sim.steps)
    for _ in range(5):
        sim.tick(engine)
        if len(sim.steps) > prior_steps or engine.state == EpisodeEngine.FATAL:
            break
    return decode_step(sim.steps[-1])


class ResetTransactionTests(unittest.TestCase):
    def test_invalid_timing_configuration_is_refused(self):
        params = load_common_parameters(CONFIG)
        params.update({"world_seed": 1, "initialization_seed": 2, "dynamic_obstacle_seed": 3, "evaluation_seed": 4})
        params["sensor_max_age_sim_seconds"] = float(params["control_period"]) + 0.01
        with self.assertRaisesRegex(ValueError, "sensor_max_age"):
            build_environment_config(params)

    def test_reset_sequence_awaits_each_service_and_verifies(self):
        engine, sim, cfg, _ = make_engine()
        state = run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy"))
        self.assertEqual(state, EpisodeEngine.AWAIT_ACTION)
        self.assertEqual([k for k, _ in sim.services], [
            "reset_world", "set_entity_state", "unpause_physics", "pause_physics",
            "get_entity_state", "get_entity_state", "get_entity_state",
        ])
        self.assertEqual([e for k, e in sim.services if k in ("set_entity_state", "get_entity_state")],
                         ["burger", "burger", "dynamic_obstacle_1", "dynamic_obstacle_2"])
        self.assertTrue(sim.paused)
        row = sim.transitions[0]
        self.assertEqual(row["step_in_episode"], 0)
        self.assertEqual(row["training_episode"], 1)
        state_msg = decode_state(sim.states[-1])
        self.assertEqual(state_msg.step_index, 0)
        self.assertEqual(len(state_msg.observation), 41)
        # the requested pose is the seeded draw and the realized pose matches it
        init = engine.current.init_row
        self.assertAlmostEqual(init["init_pos_error"], 0.0, places=9)
        self.assertEqual(init["reset_status"], "ok")
        self.assertTrue(init["init_tolerance_ok"])
        # obstacles were told to hold before the reset and to start after the initial state
        self.assertIsNotNone(sim.schedule)
        self.assertEqual(sim.schedule["signs"], list(engine.current.phases.signs))

    def test_episode_one_is_randomized_not_spawn_pose(self):
        engine, sim, _, _ = make_engine()
        run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy"))
        self.assertGreater(math.hypot(engine.current.start.x, engine.current.start.y), 0.5)

    def test_set_entity_state_failure_is_fatal_not_fallback(self):
        engine, sim, _, _ = make_engine()
        sim.set_state_ok = False
        state = run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy"))
        self.assertEqual(state, EpisodeEngine.FATAL)
        self.assertIn("set_entity_state", sim.fatal)
        self.assertEqual(sim.states, [])

    def test_nonfinite_realized_or_sensor_pose_fails_closed(self):
        engine, sim, _, _ = make_engine(sensor_timeout_wall_seconds=0.1)
        effects = engine.start_episode(EpisodeSpec(phase="training", policy_mode="noisy"), sim.sim_time)
        sim.apply_effects(engine, effects)
        # Drive to reset capture, but supply non-finite odometry until the bounded wall timeout.
        for _ in range(20):
            sim.step_physics()
            bad = OdomSample(float("nan"), 0.0, 0.0, sim.sim_time)
            sim.wall += 0.02
            sim.apply_effects(engine, engine.tick(sim.sim_time, sim.scan(), bad, sim.obstacle_samples()))
            if engine.state == EpisodeEngine.FATAL:
                break
        self.assertEqual(engine.state, EpisodeEngine.FATAL)
        self.assertIn("sensor", sim.fatal)

    def test_realized_pose_outside_tolerance_is_fatal(self):
        engine, sim, _, _ = make_engine()
        sim.pose_perturbation = (0.1, 0.0, 0.0)
        self.assertEqual(run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy")), EpisodeEngine.FATAL)
        self.assertIn("deviates", sim.fatal)

    def test_odometry_frame_mismatch_is_fatal(self):
        engine, sim, _, _ = make_engine()
        sim.odom_offset = (0.3, 0.0)
        self.assertEqual(run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy")), EpisodeEngine.FATAL)
        self.assertIn("odometry_source", sim.fatal)

    def test_contact_during_reset_is_fatal(self):
        engine, sim, _, _ = make_engine()
        sim.force_contact = True
        self.assertEqual(run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy")), EpisodeEngine.FATAL)
        self.assertIn("contact", sim.fatal)

    def test_service_timeout_is_fatal(self):
        engine, sim, _, _ = make_engine(service_timeout_wall_seconds=0.5)
        effects = engine.start_episode(EpisodeSpec(phase="training", policy_mode="noisy"), sim.sim_time)
        hold = next(e for e in effects if isinstance(e, ObstacleControl))
        effects = engine.obstacle_ack({"cmd": hold.payload["cmd"], "episode_key": hold.payload["episode_key"], "status": "ok"})
        self.assertIsInstance(effects[-1], CallService)  # reset_world is never answered
        sim.wall += 1.0
        sim.apply_effects(engine, engine.tick(sim.sim_time, sim.scan(), sim.odom(), sim.obstacle_samples()))
        self.assertEqual(engine.state, EpisodeEngine.FATAL)

    def test_obstacle_ack_timeout_is_fatal(self):
        engine, sim, _, _ = make_engine(obstacle_ack_timeout_wall_seconds=0.5)
        engine.start_episode(EpisodeSpec(phase="training", policy_mode="noisy"), sim.sim_time)
        sim.wall += 1.0
        sim.apply_effects(engine, engine.tick(sim.sim_time, sim.scan(), sim.odom(), sim.obstacle_samples()))
        self.assertEqual(engine.state, EpisodeEngine.FATAL)
        self.assertIn("acknowledgement", sim.fatal)

    def test_realized_pose_must_remain_inside_support(self):
        engine, sim, _, _ = make_engine(seeds=(7000, 3, 7002, 9001))
        sim.pose_perturbation = (0.029, 0.0, 0.0)
        self.assertEqual(run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy")), EpisodeEngine.FATAL)
        self.assertIn("support", sim.fatal)

    def test_odometry_position_must_remain_inside_support(self):
        engine, sim, _, _ = make_engine(seeds=(7000, 3, 7002, 9001))
        sim.odom_offset = (0.029, 0.0)
        state = run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="epsilon_greedy"))
        start = engine.current.start
        self.assertTrue(engine.sampler.admissible(start.x, start.y))
        self.assertFalse(engine.sampler.admissible(start.x + sim.odom_offset[0], start.y))
        self.assertLess(sim.odom_offset[0], engine.cfg.odom_tolerance)
        self.assertEqual(state, EpisodeEngine.FATAL)
        self.assertIn("support", sim.fatal)

    def test_position_tolerance_allows_perturbation_inside_support(self):
        engine, sim, _, _ = make_engine(seeds=(7000, 3, 7002, 9001))
        sim.pose_perturbation = (-0.01, 0.0, 0.0)
        self.assertEqual(run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="epsilon_greedy")), EpisodeEngine.AWAIT_ACTION)
        self.assertAlmostEqual(engine.current.init_row["init_pos_error"], 0.01)
        self.assertTrue(engine.current.init_row["init_support_ok"])

    def test_odometry_yaw_mismatch_is_fatal(self):
        engine, sim, _, _ = make_engine()
        sim.odom_yaw_offset = 0.2
        self.assertEqual(run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy")), EpisodeEngine.FATAL)
        self.assertIn("odom yaw", sim.fatal)


class DeterminismTests(unittest.TestCase):
    def test_equal_seeds_reproduce_the_episode_indexed_sequence(self):
        starts = []
        for _ in range(2):
            engine, sim, _, _ = make_engine()
            seq = []
            for _ in range(3):
                run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy"))
                seq.append((engine.current.start.x, engine.current.start.y, engine.current.start.yaw, engine.current.phases.signs, engine.current.phases.offsets))
                act(engine, sim, 0, final=True)  # end the episode immediately by truncation
            starts.append(seq)
        self.assertEqual(starts[0], starts[1])
        self.assertNotEqual(starts[0][0][:3], starts[0][1][:3])

    def test_different_initialization_seed_changes_starts_but_not_obstacles(self):
        engine_a, sim_a, _, _ = make_engine(seeds=(7000, 7001, 7002, 9001))
        engine_b, sim_b, _, _ = make_engine(seeds=(7000, 7777, 7002, 9001))
        run_reset(engine_a, sim_a, EpisodeSpec(phase="training", policy_mode="noisy"))
        run_reset(engine_b, sim_b, EpisodeSpec(phase="training", policy_mode="noisy"))
        self.assertNotEqual(engine_a.current.start.x, engine_b.current.start.x)
        self.assertEqual(engine_a.current.phases, engine_b.current.phases)

    def test_e1_draws_depend_on_checkpoint_and_episode_only(self):
        engine_a, sim_a, _, _ = make_engine(seeds=(1, 2, 3, 9001))
        engine_b, sim_b, _, _ = make_engine(seeds=(4, 5, 6, 9001))
        spec = EpisodeSpec(phase="evaluation", policy_mode="greedy", condition="E1", checkpoint_step=25000, checkpoint_index=1, evaluation_episode=7)
        run_reset(engine_a, sim_a, spec)
        run_reset(engine_b, sim_b, spec)
        self.assertEqual(engine_a.current.start, engine_b.current.start)
        self.assertEqual(engine_a.current.phases, engine_b.current.phases)


class TransitionTests(unittest.TestCase):
    def test_hold_is_exactly_one_control_period_and_robot_moves(self):
        engine, sim, cfg, _ = make_engine()
        run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy"))
        x0 = engine.current.init_row["odom_x"]
        step = act(engine, sim, 0)
        self.assertAlmostEqual(step.hold_sim_s, cfg.control_period, places=9)
        self.assertAlmostEqual(step.hold_odom_s, cfg.control_period, places=9)
        self.assertEqual(step.step_index, 1)
        self.assertFalse(step.episode_end)
        self.assertAlmostEqual(math.hypot(step.x - x0, step.y - engine.current.init_row["odom_y"]), 0.15 * cfg.control_period, places=6)
        self.assertEqual(sim.cmd, (0.0, 0.0))  # stopped after the hold
        row = sim.transitions[-1]
        self.assertEqual(row["env_step"], 1)
        self.assertAlmostEqual(row["reward_total"], row["r_distance"] + row["r_step"] + row["r_collision"] + row["r_goal"] + row["r_angular"] + row["r_near"], places=9)
        self.assertTrue(set(row) <= set(TRANSITIONS_COLUMNS))
        self.assertAlmostEqual(row["decision_gap_sim_s"], 0.0, places=12)

    def test_action_is_stopped_at_deadline_while_waiting_for_boundary_fresh_sensors(self):
        engine, sim, cfg, _ = make_engine()
        run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy"))
        sequence = engine.current.sequence
        linear, angular = cfg.action_map.commands()[0]
        sim.apply_effects(engine, engine.receive_action(ActionMessage(sequence, 0, linear, angular, False)))
        sim.tick(engine)
        self.assertEqual(engine.state, EpisodeEngine.HOLD)
        stale_scan, stale_odom, stale_obstacles = sim.scan(), sim.odom(), sim.obstacle_samples()
        sim.step_physics()  # exactly the action deadline; supplied sensor messages are still pre-deadline
        effects = engine.tick(sim.sim_time, stale_scan, stale_odom, stale_obstacles)
        self.assertTrue(any(isinstance(effect, PublishVelocity) and effect.linear == 0.0 for effect in effects))
        sim.apply_effects(engine, effects)
        self.assertEqual(sim.cmd, (0.0, 0.0))
        self.assertEqual(engine.state, EpisodeEngine.HOLD)
        # Once boundary-fresh samples arrive, the normal pause/capture closes the transition.
        sim.apply_effects(engine, engine.tick(sim.sim_time, sim.scan(), sim.odom(), sim.obstacle_samples()))
        sim.tick(engine)
        self.assertEqual(len(sim.steps), 1)
        self.assertAlmostEqual(decode_step(sim.steps[-1]).hold_odom_s, cfg.control_period, places=9)

    def test_policy_latency_does_not_advance_simulation(self):
        engine, sim, _, _ = make_engine()
        run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy"))
        state_time = sim.sim_time
        for _ in range(40):
            sim.step_physics()
        self.assertEqual(sim.sim_time, state_time)
        step = act(engine, sim, 0)
        self.assertAlmostEqual(step.decision_gap_sim_s, 0.0, places=12)
        self.assertGreater(step.decision_latency_wall_s, 0.0)

    def test_obstacle_trajectory_departure_is_fatal(self):
        engine, sim, cfg, _ = make_engine()
        run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy"))
        sequence = engine.current.sequence
        linear, angular = cfg.action_map.commands()[0]
        sim.apply_effects(engine, engine.receive_action(ActionMessage(sequence, 0, linear, angular, False)))
        sim.tick(engine)
        sim.obstacles[cfg.obstacle_names[0]][0] += 0.2
        for _ in range(5):
            sim.tick(engine)
            if engine.state == EpisodeEngine.FATAL:
                break
        self.assertEqual(engine.state, EpisodeEngine.FATAL)
        self.assertIn("seeded trajectory", sim.fatal)

    def test_duplicate_or_stale_action_is_ignored_and_invalid_action_is_fatal(self):
        engine, sim, cfg, _ = make_engine()
        run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy"))
        seq = engine.current.sequence
        linear, angular = cfg.action_map.commands()[1]
        engine.receive_action(ActionMessage(seq + 5, 1, linear, angular, False))
        self.assertIsNone(engine.pending_action)
        engine.receive_action(ActionMessage(seq, 1, linear + 0.05, angular, False))
        self.assertEqual(engine.state, EpisodeEngine.FATAL)

    def test_final_transition_flag_truncates_and_ends_the_episode(self):
        engine, sim, _, _ = make_engine()
        run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy"))
        step = act(engine, sim, 3, final=True)
        self.assertTrue(step.truncated and step.episode_end and not step.terminated)
        self.assertEqual(engine.state, EpisodeEngine.IDLE)
        self.assertEqual(len(sim.episodes), 1)
        row = sim.episodes[0]
        self.assertEqual(row["outcome"], "timeout")
        self.assertEqual(row["length"], 1)
        self.assertEqual(row["end_env_step"], 1)
        self.assertTrue(set(row) <= set(EPISODES_COLUMNS))
        self.assertEqual(sim.summaries[-1]["phase"], "training")

    def test_time_limit_is_a_truncation_that_bootstraps(self):
        engine, sim, _, _ = make_engine(episode_max_steps=3)
        run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy"))
        steps = [act(engine, sim, 3) for _ in range(3)]
        self.assertFalse(steps[0].episode_end)
        self.assertTrue(steps[2].truncated and steps[2].episode_end and not steps[2].terminated)

    def test_goal_terminates_with_bonus(self):
        engine, sim, cfg, _ = make_engine()
        run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy"))
        sim.robot = [cfg.goal_x - 0.30, cfg.goal_y, 0.0]  # 0.30 m short of the goal, facing it
        for _ in range(12):
            step = act(engine, sim, 0)
            if step.episode_end:
                break
        self.assertTrue(step.goal and step.terminated and not step.truncated)
        self.assertEqual(sim.episodes[-1]["outcome"], "goal")
        self.assertIsNotNone(sim.episodes[-1]["path_efficiency"])
        self.assertGreater(step.r_goal, 0.0)

    def test_safety_stop_precedes_goal_and_collision_precedes_safety(self):
        engine, sim, cfg, _ = make_engine()
        run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy"))
        # drive straight at static_box_1 from 0.25 m: the beam drops below stop_distance before contact
        sim.robot = [0.8, 0.85 - 0.30 - 0.20, math.pi / 2.0]
        for _ in range(10):
            step = act(engine, sim, 0)
            if step.episode_end:
                break
        self.assertTrue(step.safety and step.terminated and not step.collision)
        engine2, sim2, _, _ = make_engine()
        run_reset(engine2, sim2, EpisodeSpec(phase="training", policy_mode="noisy"))
        sim2.force_contact = False
        sim2.robot = [0.8, 0.85 - 0.30 - 0.05, math.pi / 2.0]
        sim2.footprint = 0.40  # make contact fire in the first step
        step2 = act(engine2, sim2, 0)
        self.assertTrue(step2.collision and step2.terminated and not step2.safety)
        self.assertTrue(step2.static_collision)

    def test_evaluation_episode_records_no_env_step_and_no_episode_row(self):
        engine, sim, _, scenarios = make_engine()
        spec = EpisodeSpec(phase="evaluation", policy_mode="greedy", condition="E2", checkpoint_step=500000, checkpoint_index=20, scenario_id=scenarios[3].scenario_id, evaluation_episode=4)
        run_reset(engine, sim, spec)
        self.assertAlmostEqual(engine.current.start.x, scenarios[3].x)
        step = act(engine, sim, 0, final=True)
        self.assertTrue(step.episode_end)
        self.assertEqual(engine.env_step, 0)
        self.assertEqual(sim.episodes, [])
        self.assertEqual(sim.transitions[-1]["env_step"], None)
        self.assertEqual(sim.transitions[-1]["condition"], "E2")
        self.assertEqual(sim.summaries[-1]["scenario_labels"]["difficulty"], scenarios[3].difficulty)
        self.assertEqual(sim.summaries[-1]["init_block"]["init_kind"], "E2_scenario")

    def test_e3_uses_the_fixed_start(self):
        engine, sim, cfg, _ = make_engine()
        spec = EpisodeSpec(phase="evaluation", policy_mode="greedy", condition="E3", checkpoint_step=500000, checkpoint_index=20, evaluation_episode=1)
        run_reset(engine, sim, spec)
        self.assertEqual((engine.current.start.x, engine.current.start.y, engine.current.start.yaw), cfg.fixed_start)

    def test_initial_state_is_republished_until_an_action_arrives(self):
        engine, sim, _, _ = make_engine()
        run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy"))
        n = len(sim.states)
        sim.tick(engine)
        sim.tick(engine)
        self.assertEqual(len(sim.states), n + 2)
        self.assertEqual(sim.states[-1], sim.states[-2])

    def test_start_episode_outside_idle_is_fatal(self):
        engine, sim, _, _ = make_engine()
        run_reset(engine, sim, EpisodeSpec(phase="training", policy_mode="noisy"))
        effects = engine.start_episode(EpisodeSpec(phase="training", policy_mode="noisy"), sim.sim_time)
        self.assertEqual(engine.state, EpisodeEngine.FATAL)
        self.assertIsInstance(effects[0], PublishVelocity)

    def test_obstacles_hold_during_reset_and_move_during_episode(self):
        engine, sim, _, _ = make_engine()
        effects = engine.start_episode(EpisodeSpec(phase="training", policy_mode="noisy"), sim.sim_time)
        self.assertIsInstance(effects[0], ObstacleControl)
        self.assertEqual(effects[0].payload["cmd"], "hold")
        sim.apply_effects(engine, effects)
        for _ in range(200):
            if engine.state == EpisodeEngine.AWAIT_ACTION:
                break
            sim.tick(engine)
        before = dict((k, list(v)) for k, v in sim.obstacles.items())
        act(engine, sim, 3)
        after = sim.obstacles
        moved = sum(math.hypot(after[k][0] - before[k][0], after[k][1] - before[k][1]) for k in after)
        self.assertGreater(moved, 0.0)


if __name__ == "__main__":
    unittest.main()
