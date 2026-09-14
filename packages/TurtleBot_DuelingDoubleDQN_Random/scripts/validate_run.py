#!/usr/bin/env python3
"""Terminal entry point of the run validator (Gate 7)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turtlebot3_drl_nav.validator import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
