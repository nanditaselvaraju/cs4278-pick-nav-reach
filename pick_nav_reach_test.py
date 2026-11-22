"""
pick_nav_reach_test.py - run the whole pick / navigate / reach task on one object and one maze.

    python pick_nav_reach_test.py [object_idx 0-14] [seed]
"""
import sys

from pick_nav_reach_env import PickNavReachEnv
from pick_nav_reach import run_pick_nav_reach


if __name__ == "__main__":
    object_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    env = PickNavReachEnv(seed=seed, object_idx=object_idx)
    result = run_pick_nav_reach(env)

    print("RESULT", result)
    print("TOTAL_STEPS", env.step_count)
    if sys.stdin.isatty():
        input("\nPress Enter to close...")