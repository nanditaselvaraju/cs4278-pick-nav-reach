"""
headless.py - import this first to run the environment without a window and without real-time sleeping.

    python -c "import headless, runpy, sys; sys.argv=['x', '3', '0']; runpy.run_path('pick_nav_reach_test.py', run_name='__main__')"

(pick_nav_reach_env.py always opens the PyBullet GUI; this swaps the connection for DIRECT mode and removes
the 1/240 s sleep per simulation substep.  The physics is identical, only much faster.  Used by evaluate.py.)
"""
import time

import pybullet as p

time.sleep = lambda seconds: None
_connect = p.connect
p.connect = lambda mode, *args, **kwargs: _connect(p.DIRECT, *args, **kwargs)