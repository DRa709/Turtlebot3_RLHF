import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PI_DIR = PROJECT_ROOT / "pi"
if str(PI_DIR) not in sys.path:
    sys.path.insert(0, str(PI_DIR))

from local_command_supervisor import SupervisorEngine, SupervisorConfig, SupervisorState


def test_disarmed_suppression():
    engine = SupervisorEngine()
    assert engine.state == SupervisorState.DISARMED
    # Send request while disarmed
    vx, wz, state = engine.process_request(0.15, 0.0, seq_id=1, current_time=1.0)
    assert (vx, wz) == (0.0, 0.0)
    assert state == SupervisorState.DISARMED


def test_command_timeout():
    config = SupervisorConfig(command_timeout_sec=0.20, sensor_timeout_sec=0.50)
    engine = SupervisorEngine(config)
    engine.arm()
    now = 100.0

    # Fresh sensor data
    engine.update_scan([1.5] * 360, current_time=now)
    engine.update_odom(current_time=now)

    # Valid command
    vx, wz, state = engine.process_request(0.15, 0.0, seq_id=1, current_time=now)
    assert state == SupervisorState.ACTIVE
    assert vx == 0.15

    # Advance time by 0.10s -> still active
    vx, wz, state = engine.tick_watchdog(current_time=now + 0.10)
    assert state == SupervisorState.ACTIVE

    # Advance time past 0.20s timeout -> watchdog fires!
    vx, wz, state = engine.tick_watchdog(current_time=now + 0.25)
    assert state == SupervisorState.COMMAND_EXPIRED
    assert (vx, wz) == (0.0, 0.0)


def test_proximity_braking():
    config = SupervisorConfig(hardware_stop_threshold=0.18)
    engine = SupervisorEngine(config)
    engine.arm()
    now = 100.0
    engine.update_odom(now)

    # Obstacle detected at 0.14 m (< 0.18 m)
    safe, min_r = engine.update_scan([1.0] * 180 + [0.14] + [1.0] * 179, current_time=now)
    assert not safe
    assert min_r == 0.14
    assert engine.state == SupervisorState.PROTECTIVE_STOP

    # Any command request must now be blocked and clamped to 0
    vx, wz, state = engine.process_request(0.15, 0.0, seq_id=2, current_time=now)
    assert state == SupervisorState.PROTECTIVE_STOP
    assert (vx, wz) == (0.0, 0.0)


def test_stale_sensor_watchdog():
    config = SupervisorConfig(sensor_timeout_sec=0.25, command_timeout_sec=0.50)
    engine = SupervisorEngine(config)
    engine.arm()
    now = 50.0

    engine.update_scan([2.0] * 360, current_time=now)
    engine.update_odom(current_time=now)

    # Send command after sensor expired (e.g. 0.30s later)
    vx, wz, state = engine.process_request(0.15, 0.0, seq_id=3, current_time=now + 0.30)
    assert state == SupervisorState.SENSOR_STALE
    assert (vx, wz) == (0.0, 0.0)


def test_speed_clamping():
    config = SupervisorConfig(max_linear_speed=0.15, max_angular_speed=1.00)
    engine = SupervisorEngine(config)
    engine.arm()
    now = 10.0
    engine.update_scan([2.0] * 360, current_time=now)
    engine.update_odom(current_time=now)

    # Excessive velocity command
    vx, wz, state = engine.process_request(0.50, 3.0, seq_id=4, current_time=now)
    assert state == SupervisorState.ACTIVE
    assert vx == 0.15
    assert wz == 1.00
