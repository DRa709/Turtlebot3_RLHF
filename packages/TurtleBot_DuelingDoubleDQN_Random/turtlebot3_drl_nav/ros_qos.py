"""QoS profiles shared by the ROS nodes. Shared layer."""

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def latched_qos() -> QoSProfile:
    """Reliable, transient-local, depth 1: a late subscriber still receives the
    last message (used for the environment's readiness/identity announcement)."""
    return QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
