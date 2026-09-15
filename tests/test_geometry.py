"""Geometry checks against the actual sensor transformation functions."""
import math
import pytest
from turtlebot3_drl_nav.common.state import normalize_angle, sample_lidar_nearest


def test_angle_wrapping():
    assert abs(normalize_angle(3 * math.pi)) == pytest.approx(math.pi)
    assert normalize_angle(-math.pi / 2) == pytest.approx(-math.pi / 2)
    assert normalize_angle(2.5 * math.pi) == pytest.approx(0.5 * math.pi)


def test_lidar_resampling_index_alignment():
    # Native scan starts at zero; canonical sample 18 must address that beam.
    ranges = [2.0] * 360
    ranges[0] = 0.5
    ranges[180] = 1.25
    sampled = sample_lidar_nearest(ranges, 0.0, math.pi / 180, 0.12, 3.5)
    assert len(sampled) == 36
    assert sampled[18] == 0.5  # Forward: 0 degrees.
    assert sampled[0] == 1.25  # Rear: -180 degrees.
