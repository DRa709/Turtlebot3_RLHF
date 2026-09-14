import pytest
import numpy as np

def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def decimate_lidar_360_to_36(raw_ranges: np.ndarray) -> np.ndarray:
    assert len(raw_ranges) == 360
    indices = np.arange(0, 360, 10)
    return raw_ranges[indices]

def test_angle_wrapping():
    assert np.isclose(wrap_to_pi(3.0 * np.pi), np.pi) or np.isclose(wrap_to_pi(3.0 * np.pi), -np.pi)
    assert np.isclose(wrap_to_pi(-np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_to_pi(2.5 * np.pi), 0.5 * np.pi)

def test_lidar_resampling_index_alignment():
    raw_scan = np.ones(360) * 2.0
    raw_scan[180] = 0.5
    
    sampled = decimate_lidar_360_to_36(raw_scan)
    assert len(sampled) == 36
    assert sampled[18] == 0.5
