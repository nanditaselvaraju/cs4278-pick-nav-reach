"""
pick.py - generalised pick: choose a grasp from the point cloud, reach it with IK, lift it.

    from pick import pick_object
    obs, ok = pick_object(env, obs)

Tries side grasps first, then top-down grasps.  Each plan is run with env.step; if the object does
not rise >= 10 cm the simulation is restored and the next plan is tried (up to MAX_PLANS).
"""
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from grasp_planner import generate_antipodal_grasps
from ik_solver import solve_arm_ik, set_joints, gripper_pose
from rrt import plan_arm_path

BASE_DISTANCE = 1.0
PREGRASP_BACKOFF = 0.12
LIFT_HEIGHT = 0.14
PALM_CLEARANCE = 0.03
CENTROID_XY = None
OBJECT_PC = None            # set by pick_object; used to place the pads over the object
PALM_FACE = 0.029           # palm face sits this far behind gripper_link's origin


def palm_offset(mid, approach):
    """How far to put gripper_link's origin behind the grasp midpoint (m, >= 0 means behind).
    The pads are centred on the origin, so we want it as close to the midpoint as the palm allows."""
    if OBJECT_PC is None:
        return PALM_CLEARANCE
    back_depth = float(np.max((mid - OBJECT_PC) @ approach))      # object extent behind the midpoint
    t = PALM_FACE - back_depth - 0.004                           # origin may sit this far AHEAD of the midpoint
    if approach[2] < -0.1:                                       # hand pointing down: fingertips must stay above the table
        t = min(t, (mid[2] - 0.626 - 0.004) / (-approach[2]) - 0.032)
    return float(np.clip(-t, -0.02, PALM_CLEARANCE))
OPEN, CLOSED = 0.05, 0.0
READY_ARM = {3: 0.30, 6: 0.0, 7: -0.5, 8: 0.0, 9: 1.6, 10: 0.0, 11: 0.6, 12: 0.0}
MAX_PLANS = 8
MAX_NAV_REJECTIONS = 2      # grasps that hold but have no maze path are dropped at most this many times


def gripper_pose_from_grasp(grasp):
    """
    Turn two contact points into a full gripper pose.

    Fetch gripper axes:  x = the way the fingers point   (approach)
                         y = the way the fingers close   (between the two contacts)
                         z = x cross y
    We approach horizontally, perpendicular to the closing line, from the robot's side (-x).
    """
    up = np.array([0.0, 0.0, 1.0])
    closing = grasp["contact_b"] - grasp["contact_a"]
    closing /= np.linalg.norm(closing)

    approach = np.cross(closing, up)
    approach /= np.linalg.norm(approach)
    if approach[0] < 0:                       # the robot starts on the -x side of the table
        approach = -approach
    # the gripper is symmetric, so flip the closing axis if needed to keep the hand "upright"
    y_axis = np.cross(up, approach)
    y_axis *= np.sign(np.dot(y_axis, closing))
    z_axis = np.cross(approach, y_axis)

    rotation = np.column_stack([approach, y_axis, z_axis])
    quaternion = Rotation.from_matrix(rotation).as_quat()          # x, y, z, w
    position = grasp["midpoint"] - approach * palm_offset(grasp["midpoint"], approach)
    return position, quaternion, approach




def topdown_pose(grasp, flip):
    """Hand pointing straight down; fingers close along the horizontal line between the contacts."""
    closing = grasp["contact_b"] - grasp["contact_a"]
    closing[2] = 0.0
    n = np.linalg.norm(closing)
    if n < 1e-6:
        return None
    closing /= n
    approach = np.array([0.0, 0.0, -1.0])
    y_axis = closing if not flip else -closing
    z_axis = np.cross(approach, y_axis)
    rotation = np.column_stack([approach, y_axis, z_axis])
    quaternion = Rotation.from_matrix(rotation).as_quat()
    mid = np.array(grasp["midpoint"], dtype=float)
    if CENTROID_XY is not None:                       # slide along the pad faces so the pinch is under the centre of mass
        perp = np.array([-closing[1], closing[0]])
        want = float(np.clip(np.dot(CENTROID_XY - mid[:2], perp), -0.05, 0.05))
        for shift in (want, 0.5 * want):              # keep only a slide that leaves both pads on the object
            ok = True
            for c in (grasp["contact_a"], grasp["contact_b"]):
                moved = np.array([c[0] + perp[0] * shift, c[1] + perp[1] * shift, c[2]])
                if np.min(np.linalg.norm(OBJECT_PC - moved, axis=1)) > 0.012:
                    ok = False
            if ok:
                mid[:2] += perp * shift
                break
    return mid - approach * palm_offset(mid, approach), quaternion, approach


def ramp(env, q_target, gripper, max_arm_step=0.02, max_base_step=0.03):
    """Move all joints in a straight line in joint space from where they are now to q_target."""
    q_from = env.agent_qpos.astype(float)
    q_to = np.array(q_target, dtype=float)
    q_to[13:15] = gripper
    q_from[13:15] = gripper
    dyaw = (q_to[2] - q_from[2] + np.pi) % (2 * np.pi) - np.pi      # shortest way round
    q_to[2] = q_from[2] + dyaw
    d = np.abs(q_to - q_from)
    n = int(np.ceil(max(d[:2].max() / max_base_step, d[2] / max_base_step, d[3:13].max() / max_arm_step, 1)))
    for a in np.linspace(0, 1, n + 1)[1:]:
        obs, info = env.step((1 - a) * q_from + a * q_to)
    for _ in range(8):
        obs, info = env.step(q_to)
    return obs


RRT_STATS = {"calls": 0, "detours": 0, "failed": 0}


def move_arm(env, q_target, gripper, obstacle_ids, max_arm_step=0.02):
    """Plan a collision-free arm motion to q_target (RRT-Connect, straight line when that is free) and run it."""
    q_from = env.agent_qpos.astype(float)
    q_goal = np.array(q_target, dtype=float)
    q_from[13:15] = gripper
    q_goal[13:15] = gripper
    q_goal[:3] = q_from[:3]
    path = plan_arm_path(env, q_from, q_goal, obstacle_ids)
    RRT_STATS["calls"] += 1
    if path is None:                                   # no path found: fall back to the direct motion
        RRT_STATS["failed"] += 1
        return ramp(env, q_target, gripper, max_arm_step=max_arm_step)
    if len(path) > 2:
        RRT_STATS["detours"] += 1
    obs = None
    for w0, w1 in zip(path[:-1], path[1:]):
        n = int(np.ceil(max(np.abs(w1 - w0)[3:13].max() / max_arm_step, 1)))
        for a in np.linspace(0, 1, n + 1)[1:]:
            obs, _ = env.step((1 - a) * w0 + a * w1)
    return ramp(env, q_target, gripper, max_arm_step=max_arm_step)       # settle exactly on q_target


def ready_pose(base):
    q = np.zeros(15)
    q[:3] = base
    for j, v in READY_ARM.items():
        q[j] = v
    q[13:15] = OPEN
    return q


def candidate_poses(g):
    """Yield (mode, position, quaternion, approach, base) for one grasp: side approach and top-down."""
    pos, quat, approach = gripper_pose_from_grasp(g)
    yaw = np.arctan2(approach[1], approach[0])
    if approach[0] >= 0.8 and abs(yaw) >= 0.2:      # head-on side approaches scrape the wrist on the table
        bw = pos[:2] - BASE_DISTANCE * approach[:2]
        yield "side", pos, quat, approach, np.array([bw[0] + 2.0, bw[1], yaw])
    for flip in (False, True):
        out = topdown_pose(g, flip)
        if out is None:
            continue
        pos, quat, approach = out
        for dist in (0.9, 1.0, 0.8):
            yield "top", pos, quat, approach, np.array([pos[0] - dist + 2.0, pos[1], 0.0])




def full(q, base):
    q = np.array(q, dtype=float)
    q[:3] = base
    return q


CARRY_REACH = 0.55          # how far in front of the base the hand is held while carrying (m)
CARRY_MIN_Z = 0.80          # keep the held object well above the table edge


def bring_close_to_body(env, base_pose):
    """After the lift: pull the hand in close to the body (gripper closed, orientation unchanged,
    object raised to the goal height), then drive slowly back to the start base pose."""
    cur = env.agent_qpos.astype(float)
    base_now = cur[:3].copy()
    pos, quat = gripper_pose(env)
    yaw = base_now[2]
    z_goal = max(pos[2] + (float(env.goal_pos[2]) - float(env._get_obs()["object_pos"][2])), CARRY_MIN_Z)
    q_carry = None
    for reach in (CARRY_REACH, 0.65, 0.75, 0.85):                    # as close to the body as IK allows
        target = np.array([base_now[0] - 2.0 + reach * np.cos(yaw), base_now[1] + reach * np.sin(yaw), z_goal])
        q_carry = solve_arm_ik(env, target, quat, base_now, seeds=[cur])
        if q_carry is not None:
            break
    set_joints(env, cur)
    if q_carry is not None:
        ramp(env, full(q_carry, base_now), CLOSED, max_arm_step=0.01)
    cur = env.agent_qpos.astype(float)
    return ramp(env, full(cur, np.asarray(base_pose, dtype=float)), CLOSED, max_base_step=0.008)


def rehearse_first_moves(env, distance=0.5, steps=200):
    """Drive forward and back at the navigation speed.  Grasps that cannot survive this are rejected
    now, while another grasp can still be tried."""
    base = env.agent_qpos.astype(float)
    base[13:15] = CLOSED
    target = base.copy()
    target[0] += distance
    obs = None
    for a in list(np.linspace(0.0, 1.0, steps + 1)[1:]) + list(np.linspace(1.0, 0.0, steps + 1)[1:]):
        obs, _ = env.step((1 - a) * base + a * target)
    for turn in (0.9, -0.9):                               # navigation also turns on the spot
        target = base.copy()
        target[2] += turn
        for a in list(np.linspace(0.0, 1.0, steps + 1)[1:]) + list(np.linspace(1.0, 0.0, steps + 1)[1:]):
            obs, _ = env.step((1 - a) * base + a * target)
    for _ in range(30):
        obs, _ = env.step(base)
    return obs


def edge_pinch_grasps(pc, base_xy, bins=36, depth=0.008, half_width=0.02):
    """Fallback for flat objects wider than the gripper (e.g. a plate): pinch the rim.
    Closing axis = horizontal direction from the object's centre through the edge point."""
    pc = np.asarray(pc, dtype=float)
    c = pc[:, :2].mean(axis=0)
    rel = pc[:, :2] - c
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    rad = np.linalg.norm(rel, axis=1)
    grasps = []
    for k in range(bins):
        lo = -np.pi + 2 * np.pi * k / bins
        sel = np.where((ang >= lo) & (ang < lo + 2 * np.pi / bins))[0]
        if len(sel) == 0:
            continue
        e = pc[sel[np.argmax(rad[sel])]]
        r = np.array([e[0] - c[0], e[1] - c[1], 0.0])
        r /= np.linalg.norm(r)
        near = np.linalg.norm(pc[:, :2] - e[:2], axis=1) < 0.03
        z = float(pc[near, 2].max()) - 0.018
        mid = np.array([e[0], e[1], z]) - r * depth
        grasps.append({"contact_a": mid - r * half_width, "contact_b": mid + r * half_width,
                       "midpoint": mid, "width": 2 * half_width,
                       "_d": float(np.linalg.norm(mid[:2] - base_xy))})
    grasps.sort(key=lambda g: g["_d"])
    return grasps


def pick_object(env, obs, verbose=True, accept=None):
    """Returns (obs, success).  The object ends up held ~14 cm above the table.

    accept(env) -> bool, optional: called on every grasp that holds.  A grasp it rejects (for example because
    no maze path exists for the way the object is held) is dropped and the next grasp is tried."""
    global OBJECT_PC
    start = env.agent_qpos.copy()
    for _ in range(100):                                   # let the object settle
        obs, _ = env.step(start)

    OBJECT_PC = np.asarray(obs["object_pc"], dtype=float)
    global CENTROID_XY
    CENTROID_XY = OBJECT_PC[:, :2].mean(axis=0)
    grasps = generate_antipodal_grasps(obs["object_pc"], obs["object_normals"],
                                       max_gripper_width=0.104, normal_threshold=-0.7, max_candidates=60)
    if len(grasps) < 5:        # thin / flat objects: allow grasps closer to the table, narrower pinches
        grasps = generate_antipodal_grasps(obs["object_pc"], obs["object_normals"],
                                           min_gripper_width=0.01, max_gripper_width=0.104,
                                           normal_threshold=-0.6, max_candidates=60,
                                           min_height_above_table=0.008)
    ext = OBJECT_PC.max(axis=0) - OBJECT_PC.min(axis=0)
    if ext[2] < 0.04 and min(ext[0], ext[1]) > 0.12:       # flat and wider than the gripper: pinch the rim
        grasps = edge_pinch_grasps(OBJECT_PC, np.array([start[0] - 2.0, start[1]])) + list(grasps)
    if verbose:
        print("grasp candidates:", len(grasps), flush=True)

    table_id = [b for b in range(p.getNumBodies())
                if b not in (0, env.robot_id, env.object_id, env.goal_marker_id)][0]
    saved = env.agent_qpos.copy()
    plans = []
    for want_mode, (rank, g) in [(m, rg) for m in ("side", "top") for rg in enumerate(grasps)]:
        for mode, pos, quat, approach, base in candidate_poses(g):
            if mode != want_mode:
                continue
            q_grasp = solve_arm_ik(env, pos, quat, base)
            if q_grasp is None:
                continue
            q_pre = solve_arm_ik(env, pos - approach * PREGRASP_BACKOFF, quat, base, seeds=[q_grasp])
            q_lift = solve_arm_ik(env, pos + np.array([0, 0, LIFT_HEIGHT]), quat, base, seeds=[q_grasp])
            if q_pre is None or q_lift is None:
                continue
            touches_table = False
            for q_check in (q_pre, q_grasp):
                set_joints(env, q_check)
                p.performCollisionDetection()
                near = p.getClosestPoints(env.robot_id, table_id, 0.015)
                if any(c[3] <= 20 for c in near):
                    touches_table = True
            if touches_table:
                continue
            plans.append((mode, rank, base, q_pre, q_grasp, q_lift))
            break
        if len(plans) >= MAX_PLANS:
            break
    set_joints(env, saved)
    if not plans:
        if verbose:
            print("pick: no feasible grasp plan", flush=True)
        return obs, False

    def hold_pose(q, base, n):
        target = np.concatenate([full(q, base)[:13], [CLOSED, CLOSED]])
        o = None
        for _ in range(n):
            o, _ = env.step(target)
        return o

    def try_pick(mode, rank, base, q_pre, q_grasp, q_lift):
        z0 = env.step(env.agent_qpos)[0]["object_pos"][2]
        ramp(env, ready_pose(env.agent_qpos[:3]), OPEN)
        ramp(env, ready_pose(base), OPEN)
        move_arm(env, full(q_pre, base), OPEN, [table_id, env.object_id])
        ramp(env, full(q_grasp, base), OPEN, max_arm_step=0.005)
        ramp(env, full(q_grasp, base), CLOSED, max_arm_step=0.005)
        hold_pose(q_grasp, base, 60)
        ramp(env, full(q_lift, base), CLOSED, max_arm_step=0.005)
        o = hold_pose(q_lift, base, 50)
        if 0.03 < o["object_pos"][2] - z0 < 0.10:           # held but hanging low / tilted: lift higher
            q_high = np.array(q_lift, dtype=float) + 0.7 * (np.array(q_lift, dtype=float) - np.array(q_grasp, dtype=float))
            ramp(env, full(q_high, base), CLOSED, max_arm_step=0.005)
            o = hold_pose(q_high, base, 50)
        return o, o["object_pos"][2] - z0

    state = p.saveState()
    TILT_MAX = 0.05                                              # rad the object may rotate in the grip while rehearsing

    def angle_between(q1, q2):
        return 2.0 * np.arccos(min(1.0, abs(float(np.dot(q1, q2)))))

    def attempt_plan(plan):
        """Pick, tuck, rehearse.  Returns (obs, held, tilt)."""
        obs, rise = try_pick(*plan)
        if verbose:
            print(f"pick attempt: {plan[0]} grasp {plan[1]}: rise {rise:.3f} m", flush=True)
        if rise < 0.10:
            return obs, False, 9.0
        z_lifted = float(obs["object_pos"][2])
        obs = bring_close_to_body(env, start[:3])
        if obs["object_pos"][2] <= z_lifted - 0.08:
            return obs, False, 9.0
        orn0 = p.getBasePositionAndOrientation(env.object_id)[1]
        obs = rehearse_first_moves(env)
        z_start = float(obs["object_pos"][2])
        hold = np.concatenate([env.agent_qpos[:13], [CLOSED, CLOSED]])
        for _ in range(300):                                     # a slowly creeping grip shows up as a drift
            obs, _ = env.step(hold)
        tilt = angle_between(orn0, p.getBasePositionAndOrientation(env.object_id)[1])
        held = obs["object_pos"][2] > z_lifted - 0.05 and obs["object_pos"][2] > z_start - 0.02
        if verbose:
            print(f"  held={held} tilt={tilt:.3f} rad", flush=True)
        return obs, bool(held), tilt

    rejected = 0
    fallback = None                                              # (tilt, snapshot of the simulation while holding)
    PICK_STEP_BUDGET = 3000                                      # keep room for the 15000-step limit: stop retrying once we hold something
    for plan in plans:
        if fallback is not None and env.step_count > PICK_STEP_BUDGET:
            break
        obs, held, tilt = attempt_plan(plan)
        if held and accept is not None and rejected < MAX_NAV_REJECTIONS and not accept(env):
            rejected += 1                   # holds, but no maze path exists for the way it is held: try another grasp
            if verbose:
                print("  grasp holds but is rejected by the navigation check", flush=True)
            held = False
        if held and tilt < TILT_MAX:
            if fallback is not None:
                p.removeState(fallback[1])
            p.removeState(state)
            return obs, True
        if held and (fallback is None or tilt < fallback[0]):
            if fallback is not None:
                p.removeState(fallback[1])
            fallback = (tilt, p.saveState())                     # holds, but rotates in the grip: remember it
        p.restoreState(state)
    if fallback is not None:                                     # nothing steadier: use the best of what held
        p.restoreState(fallback[1])
        p.removeState(fallback[1])
        p.removeState(state)
        return env._get_obs(), True
    p.removeState(state)
    obs = env._get_obs()
    return obs, False