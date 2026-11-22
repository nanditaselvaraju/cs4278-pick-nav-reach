"""
pick_nav_reach.py - the full task: pick the object, carry it through the maze, put it on the goal.

    from pick_nav_reach import run_pick_nav_reach
    result = run_pick_nav_reach(env)

Stages
    1. pick_object (pick.py)                  grasp generation, IK, RRT-Connect arm motion, lift, rehearsal
    2. plan_configuration_space_path          A* over (x, y, heading) with physics collision checks,
       (configuration_space_planner.py)       including the carried object
    3. navigate_carrying / align_on_goal      drive the path slowly, then slide the object onto the goal
"""
import time

import numpy as np
import pybullet as p

from pick import pick_object
from configuration_space_planner import plan_configuration_space_path, grid_to_world

STEP_LIMIT = 15000          # simulation steps allowed for the whole task
ALIGN_RESERVE = 1500        # steps kept back for the final alignment
MIN_STEPS_PER_STATE = 40    # fastest allowed drive between two path states
MAX_STEPS_PER_STATE = 200
DROP_DISTANCE = 0.10        # the object counts as dropped if it falls this far below its carrying height


def _rendering(on):
    """Switch the viewer on or off (planning is much faster with it off).  No effect without a viewer."""
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1 if on else 0)
    except Exception:
        pass


def _has_viewer(env):
    try:
        return p.getConnectionInfo(env.pb_physics_client)["connectionMethod"] == p.GUI
    except Exception:
        return False


_NAV_CACHE = {}


def navigation_is_feasible(env):
    """Used while choosing a grasp: is there a maze path for the robot carrying the object the way it is
    held now?  The path is kept so it does not have to be planned again."""
    obs = env._get_obs()
    base_goal = np.asarray(env.goal_pos[:2]) - (np.asarray(obs["object_pos"][:2]) - np.asarray(obs["qpos"][:2]))
    print("  checking that a maze path exists for this grasp (can take a few minutes in the GUI)...", flush=True)
    started = time.time()
    _rendering(False)
    try:
        path, qposes = plan_configuration_space_path(env, goal_position=base_goal)
    except RuntimeError:
        path, qposes = None, None
    _rendering(True)
    print("  path check done in %.0f s: %s" % (time.time() - started, "path found" if path is not None else "no path"), flush=True)
    if path is None:
        _NAV_CACHE.clear()
        return False
    _NAV_CACHE["plan"] = {"qpos": env.agent_qpos.copy(), "path": path, "qposes": qposes}
    return True


def navigate_carrying(env, obs, drop_z, verbose=True):
    """Plan a path for the robot and the carried object, then drive it.  Returns (obs, ok)."""
    carried_offset = np.asarray(obs["object_pos"][:2]) - np.asarray(obs["qpos"][:2])
    base_goal = np.asarray(env.goal_pos[:2]) - carried_offset       # base position that puts the object on the goal

    if verbose:
        print("\nPlanning collision-free navigation path...", flush=True)
    cached = _NAV_CACHE.get("plan")
    if cached is not None and np.allclose(cached["qpos"], env.agent_qpos, atol=1e-5):
        path, qposes = cached["path"], cached["qposes"]             # found while choosing the grasp
    else:
        _rendering(False)
        path, qposes = plan_configuration_space_path(env, goal_position=base_goal)
        _rendering(True)
    _NAV_CACHE.clear()
    if path is None:
        print("No carried-arm collision-free navigation path found.", flush=True)
        return obs, False
    if verbose:
        print("Navigation path found. Number of navigation states:", len(path), flush=True)

    previous = env.agent_qpos.copy()
    previous[13] = previous[14] = 0.0                               # gripper stays closed

    # Drive as slowly as the step limit allows: slower means the object slips less.
    steps_left = STEP_LIMIT - env.step_count - ALIGN_RESERVE
    per_state = int(np.clip(steps_left / len(path), MIN_STEPS_PER_STATE, MAX_STEPS_PER_STATE))

    for index, state in enumerate(path):
        target = np.asarray(qposes[state], dtype=np.float32).copy()
        target[13] = target[14] = 0.0

        distance = np.linalg.norm(target[:2] - previous[:2])
        dyaw = np.arctan2(np.sin(target[2] - previous[2]), np.cos(target[2] - previous[2]))
        n_steps = max(per_state,
                      int(np.ceil(distance / 0.01)),                # never faster than 1 cm per step
                      int(np.ceil(abs(dyaw) / np.deg2rad(0.5))))    # nor 0.5 degrees per step

        for alpha in np.linspace(0.0, 1.0, n_steps):
            action = (1.0 - alpha) * previous + alpha * target
            action[13] = action[14] = 0.0
            obs, _ = env.step(action)
        previous = target.copy()

        if verbose:
            print(f"Navigation state {index + 1}/{len(path)}: position={grid_to_world(state)} "
                  f"steps={n_steps} object height={float(obs['object_pos'][2]):.3f}", flush=True)
        if obs["object_pos"][2] < drop_z:
            print("Object was dropped during navigation.", flush=True)
            return obs, False
    if verbose:
        print("Navigation complete.", flush=True)
    return obs, True


def align_on_goal(env, obs, drop_z):
    """Slide the base so the held object ends on the goal.  Returns (obs, info, ok)."""
    q0 = env.agent_qpos.copy()
    q0[13] = q0[14] = 0.0
    to_goal = np.asarray(env.goal_pos, dtype=np.float32) - obs["object_pos"]
    target = q0.copy()
    target[:2] += to_goal[:2]
    n = max(60, int(np.ceil(np.linalg.norm(to_goal[:2]) / 0.002)))      # at most ~2 mm of motion per step
    info = None
    for alpha in np.linspace(0.0, 1.0, n + 1)[1:]:
        action = (1 - alpha) * q0 + alpha * target
        action[13] = action[14] = 0.0
        obs, info = env.step(action)
        if obs["object_pos"][2] < drop_z:
            print("Object dropped during final alignment.", flush=True)
            return obs, info, False
    for _ in range(50):                                                 # settle while still holding the object
        obs, info = env.step(target)
    return obs, info, True


def run_pick_nav_reach(env, verbose=True, camera=True):
    """Runs the whole task on env.  Returns a dict: success, stage, dist_to_goal, steps."""
    obs = env._get_obs()
    viewer = camera and _has_viewer(env)
    if viewer:
        p.resetDebugVisualizerCamera(1.4, 60, -25, [0, 0, 0.7])         # close-up of the table
    _NAV_CACHE.clear()
    obs, picked = pick_object(env, obs, verbose=verbose, accept=navigation_is_feasible)
    if verbose:
        print("Pick succeeded:", picked, "object z:", obs["object_pos"][2], flush=True)
    result = {"success": False, "stage": "pick", "dist_to_goal": None, "steps": env.step_count}
    if not picked:
        print("Pick failed.", flush=True)
        return result
    if viewer:
        p.resetDebugVisualizerCamera(22, 0, -89, [8, 0, 0])             # whole maze from above

    drop_z = float(obs["object_pos"][2]) - DROP_DISTANCE
    obs, ok = navigate_carrying(env, obs, drop_z, verbose=verbose)
    result["stage"] = "navigation"
    if ok:
        obs, info, ok = align_on_goal(env, obs, drop_z)
        result["stage"] = "reach"
        if ok:
            result["success"] = bool(info["success"])
            result["dist_to_goal"] = float(info["dist_to_goal"])
            print("Final:", info, obs["object_pos"], flush=True)
    result["steps"] = env.step_count
    return result