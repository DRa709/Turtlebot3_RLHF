"""Collision-name classification shared by Gazebo wiring and unit tests."""

from typing import Iterable, Tuple


STATIC_TOKENS = ("static_", "arena_wall", "ground_plane")
DYNAMIC_TOKENS = ("dynamic_obstacle",)


def classify_collision_names(names: Iterable[str]) -> Tuple[bool, bool, bool]:
    """Return (collision, static_collision, dynamic_collision).

    Gazebo contact names contain scoped model/link/collision identifiers. Contacts
    involving the ground plane are ignored because normal wheel-floor contact is
    not a navigation collision.
    """
    lowered = [str(name).lower() for name in names]
    relevant = [name for name in lowered if "burger" not in name and "ground_plane" not in name]
    static_collision = any(any(token in name for token in STATIC_TOKENS) for name in relevant)
    dynamic_collision = any(any(token in name for token in DYNAMIC_TOKENS) for name in relevant)
    unknown_collision = bool(relevant) and not static_collision and not dynamic_collision
    return static_collision or dynamic_collision or unknown_collision, static_collision, dynamic_collision
