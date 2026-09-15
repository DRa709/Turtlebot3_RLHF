"""Check the implementation, rather than a copied observation/action table."""
import math
import pytest
from turtlebot3_drl_nav.common.state import (
    ActionMap, RewardConfig, build_observation, compute_reward,
)


def test_observation_vector_dimension_and_bounds():
    observation = build_observation(
        [1.75] * 36, distance=2.5, heading_error=-math.pi / 2,
        previous_linear=0.12, previous_angular=-0.6,
        lidar_max=3.5, distance_max=5.0, velocity_max=0.15, angular_max=1.0,
    )
    assert len(observation) == 41
    assert observation[:36] == pytest.approx([0.5] * 36)
    assert observation[36:] == pytest.approx([0.5, -1.0, 0.0, 0.8, -0.6])


def test_discrete_action_velocity_lookup():
    actions = ActionMap()
    assert actions.commands() == (
        (0.15, 0.0), (0.12, 0.6), (0.12, -0.6), (0.0, 1.0), (0.0, -1.0),
    )
    assert actions.v_max == 0.15
    assert actions.omega_max == 1.0


@pytest.mark.parametrize('contact,clearance,distance,event', [
    (True, 0.1, 0.1, 'collision'),
    (False, 0.1, 0.1, 'safety'),
    (False, 1.0, 0.1, 'goal'),
])
def test_contact_stop_and_goal_remain_distinct(contact, clearance, distance, event):
    result = compute_reward(1.0, distance, clearance, 0.0, contact, RewardConfig())
    assert result.event == event
    assert sum([result.collision, result.safety, result.goal]) == 1
