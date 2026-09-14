import pytest
import numpy as np

def test_observation_vector_dimension_and_bounds():
    synthetic_scan = np.random.uniform(0.12, 3.5, 36)
    normalized_scan = np.clip((synthetic_scan - 0.12) / (3.5 - 0.12), 0.0, 1.0)
    
    target_dist = 1.85
    norm_dist = np.clip(target_dist / 5.0, 0.0, 1.0)
    
    heading_error = -0.45
    norm_heading = heading_error / np.pi
    
    v_lin = 0.15
    v_ang = 0.5
    
    obs = np.concatenate([
        normalized_scan,
        [norm_dist, norm_heading, np.cos(heading_error), v_lin / 0.22, v_ang / 2.0]
    ])
    
    assert len(obs) == 41
    assert np.all(obs[:36] >= 0.0) and np.all(obs[:36] <= 1.0)
    assert -1.0 <= obs[37] <= 1.0

def test_discrete_action_velocity_lookup():
    action_table = {
        0: (0.22, 0.0),
        1: (0.18, 0.6),
        2: (0.18, -0.6),
        3: (0.08, 1.5),
        4: (0.08, -1.5)
    }
    
    for action_idx, (v, w) in action_table.items():
        assert 0.0 <= v <= 0.22, f"Linear velocity {v} violates Burger motor limits"
        assert -2.0 <= w <= 2.0, f"Angular velocity {w} violates Burger motor limits"
