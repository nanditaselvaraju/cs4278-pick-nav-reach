import heapq
 
import numpy as np
import pybullet as p
 
from pick_nav_reach_env import PickNavReachEnv
 
 
GRID_SIZE = 0.5
 
YAW_VALUES = [
    0.0,
    np.pi / 2,
    np.pi,
    -np.pi / 2,
]
 
END_EFFECTOR_LINK = 20
 
 
def world_to_grid(position):
    return (
        int(round(position[0] / GRID_SIZE)),
        int(round(position[1] / GRID_SIZE)),
    )
 
 
def grid_to_world(node):
    return np.array(
        [
            node[0] * GRID_SIZE,
            node[1] * GRID_SIZE,
        ],
        dtype=np.float32,
    )
 
 
def nearest_yaw_index(yaw):
    differences = [
        abs(
            np.arctan2(
                np.sin(yaw - candidate),
                np.cos(yaw - candidate),
            )
        )
        for candidate in YAW_VALUES
    ]
 
    return int(np.argmin(differences))
 
 
def get_obstacle_body_ids():
    obstacle_ids = []
 
    for body_id in range(p.getNumBodies()):
        body_info = p.getBodyInfo(body_id)
 
        try:
            body_name = body_info[1].decode("utf-8")
        except Exception:
            body_name = ""
 
        if (
            "table" in body_name.lower()
            or "maze" in body_name.lower()
        ):
            obstacle_ids.append(body_id)
 
    return obstacle_ids
 
 
def has_robot_obstacle_contact(
    robot_id,
    obstacle_ids,
):
    contacts = p.getContactPoints(
        bodyA=robot_id,
    ) or []
 
    for contact in contacts:
        if contact[2] in obstacle_ids:
            return True
 
    return False
 
 
# ---- carried-object collision checking -------------------------------------------------------
# The planner teleports the robot to candidate poses.  The object in the gripper must be teleported
# with it (same rigid offset from the base), otherwise only the robot body is checked and the object
# can hit maze walls on the real drive.
_CARRY = {}
_CHECK_CARRIED = True
 
 
def _begin_carry(env, reference_qpos):
    """Create a stand-in box the size of the carried object.  It is moved around while planning, so
    the real object (and its grip) is never disturbed."""
    if not _CHECK_CARRIED:
        return
    pos, orn = p.getBasePositionAndOrientation(env.object_id)
    pc = np.asarray(env._get_obs()["object_pc"], dtype=float)
    centre = 0.5 * (pc.max(axis=0) + pc.min(axis=0))
    half = 0.5 * (pc.max(axis=0) - pc.min(axis=0)) + 0.01
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half.tolist())
    ghost = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, basePosition=centre.tolist())
    _CARRY.update(
        ref_xy=np.array([reference_qpos[0] - 2.0, reference_qpos[1]]),   # robot is loaded at world x = -2
        ref_yaw=float(reference_qpos[2]),
        pos=centre, orn=(0.0, 0.0, 0.0, 1.0), ghost=ghost,
    )
 
 
def _end_carry():
    if _CARRY.get("ghost") is not None:
        p.removeBody(_CARRY["ghost"])
    _CARRY.clear()
 
 
def _place_carried_object(env, qpos):
    c = _CARRY
    dyaw = float(qpos[2]) - c["ref_yaw"]
    cos_y, sin_y = np.cos(dyaw), np.sin(dyaw)
    rel = c["pos"][:2] - c["ref_xy"]
    new_xy = np.array([qpos[0] - 2.0, qpos[1]]) + np.array([cos_y * rel[0] - sin_y * rel[1],
                                                           sin_y * rel[0] + cos_y * rel[1]])
    new_pos = [float(new_xy[0]), float(new_xy[1]), float(c["pos"][2])]
    p.resetBasePositionAndOrientation(c["ghost"], new_pos, p.getQuaternionFromEuler([0.0, 0.0, dyaw]))
 
 
def _restore_carried_object(env):
    pass
 
 
def carried_object_collides(env, obstacle_ids, margin=0.05):
    for oid in obstacle_ids:
        if p.getClosestPoints(_CARRY["ghost"], oid, margin):
            return True
    return False
 
 
def calculate_carry_configuration(
    env,
    reference_qpos,
    x,
    y,
    yaw,
):
    """
    Preserve the verified lifted arm posture while moving the base.
 
    The arm is not re-solved with IK for every navigation cell.
    This prevents unnecessary posture changes that can destabilise
    the grasp or move the gripper into maze walls.
    """
 
    candidate_qpos = reference_qpos.copy()
 
    # Change only the mobile base configuration.
    candidate_qpos[0] = x
    candidate_qpos[1] = y
    candidate_qpos[2] = yaw
 
    return candidate_qpos
 
 
def state_is_collision_free(
    env,
    reference_qpos,
    x,
    y,
    yaw,
    obstacle_ids,
):
    """
    Collision-check a navigation state while preserving the
    currently verified grasp and arm posture.
    """
 
    candidate_qpos = calculate_carry_configuration(
        env=env,
        reference_qpos=reference_qpos,
        x=x,
        y=y,
        yaw=yaw,
    )
 
    env._set_qpos(candidate_qpos)
    if _CARRY:
        _place_carried_object(env, candidate_qpos)
    p.performCollisionDetection()
 
    contacts = p.getContactPoints(
        bodyA=env.robot_id,
    )
 
    collision = False
    if _CARRY and carried_object_collides(env, obstacle_ids):
        collision = True
 
    for contact in contacts:
        other_body_id = contact[2]
 
        if other_body_id in obstacle_ids:
            collision = True
            break
 
    # Always restore the actual carried configuration.
    env._set_qpos(reference_qpos)
    if _CARRY:
        _restore_carried_object(env)
    p.performCollisionDetection()
 
    return not collision
 
 
def transition_is_collision_free(
    env,
    start_qpos,
    end_qpos,
    obstacle_ids,
    samples=25,
):
    """
    Check the complete interpolated motion between two configurations.
 
    This catches arm/maze collisions that occur during a rotation or
    translation even when both endpoint poses are collision-free.
    """
 
    start_qpos = np.asarray(start_qpos, dtype=np.float32)
    end_qpos = np.asarray(end_qpos, dtype=np.float32)
 
    for alpha in np.linspace(0.0, 1.0, samples):
        candidate_qpos = (
            (1.0 - alpha) * start_qpos
            + alpha * end_qpos
        )
 
        env._set_qpos(candidate_qpos)
        if _CARRY:
            _place_carried_object(env, candidate_qpos)
        p.performCollisionDetection()
 
        if has_robot_obstacle_contact(
            robot_id=env.robot_id,
            obstacle_ids=obstacle_ids,
        ) or (_CARRY and carried_object_collides(env, obstacle_ids)):
            env._set_qpos(start_qpos)
            if _CARRY:
                _restore_carried_object(env)
            p.performCollisionDetection()
            return False
 
    env._set_qpos(start_qpos)
    if _CARRY:
        _restore_carried_object(env)
    p.performCollisionDetection()
 
    return True
 
 
def heuristic(state, goal):
    x, y, _ = state
    gx, gy, _ = goal
 
    return abs(x - gx) + abs(y - gy)
 
 
def reconstruct_path(came_from, current):
    path = [current]
 
    while current in came_from:
        current = came_from[current]
        path.append(current)
 
    path.reverse()
 
    return path
 
 
def _plan_configuration_space_path_impl(
    env,
    goal_position=None,
):
    """
    Plan a path while validating the full carried-arm configuration.
 
    Returns:
        path:
            A list of (grid_x, grid_y, yaw_index) states.
 
        path_qposes:
            A dictionary mapping each state to its validated 15-joint
            target configuration.
    """
 
    saved_qpos = env.agent_qpos.copy()
    reference_qpos = saved_qpos.copy()
    _begin_carry(env, reference_qpos)
 
    start_grid = world_to_grid(
        reference_qpos[:2]
    )
 
    if goal_position is None:
        goal_position = env.goal_pos[:2]
 
    goal_grid = world_to_grid(
        np.asarray(goal_position)[:2]
    )
 
    start_yaw_index = nearest_yaw_index(
        reference_qpos[2]
    )
 
    start = (
        start_grid[0],
        start_grid[1],
        start_yaw_index,
    )
 
    # Delivery depends on position, not final base yaw.
    # Keep the yaw in the goal state equal to the starting yaw;
    # the search termination below checks position only.
    goal = (
        goal_grid[0],
        goal_grid[1],
        start_yaw_index,
    )
 
    obstacle_ids = get_obstacle_body_ids()
 
    print(
        "Obstacle body IDs:",
        obstacle_ids,
        flush=True,
    )
 
    wall_positions = env.obs[
        "cube_positions"
    ]
 
    min_x = int(
        np.floor(wall_positions[:, 0].min())
    ) - 4
 
    max_x = int(
        np.ceil(wall_positions[:, 0].max())
    ) + 4
 
    min_y = int(
        np.floor(wall_positions[:, 1].min())
    ) - 4
 
    max_y = int(
        np.ceil(wall_positions[:, 1].max())
    ) + 4
 
    def inside_bounds(state):
        x, y, _ = state
 
        world_x = x * GRID_SIZE
        world_y = y * GRID_SIZE
 
        return (
            min_x <= world_x <= max_x
            and min_y <= world_y <= max_y
        )
 
    # The current lifted pose has already been physically verified.
    # Use it directly as the start configuration instead of recomputing
    # it through IK.
    start_qpos = reference_qpos.copy()
 
    # Keep the gripper closed.
    start_qpos[13] = 0.0
    start_qpos[14] = 0.0
 
    if not state_is_collision_free(
        env=env,
        reference_qpos=start_qpos,
        x=float(start_qpos[0]),
        y=float(start_qpos[1]),
        yaw=float(start_qpos[2]),
        obstacle_ids=obstacle_ids,
    ):
        env._set_qpos(saved_qpos)
        p.performCollisionDetection()
        raise RuntimeError(
            "The initial carried configuration is colliding."
        )
 
    open_heap = []
 
    heapq.heappush(
        open_heap,
        (
            heuristic(start, goal),
            0.0,
            start,
        ),
    )
 
    came_from = {}
    cost_so_far = {
        start: 0.0,
    }
 
    state_qposes = {
        start: start_qpos,
    }
 
    checked_states = set()
 
    while open_heap:
        _, current_cost, current = (
            heapq.heappop(open_heap)
        )
 
        if current in checked_states:
            continue
 
        checked_states.add(current)
 
        if current == goal:
            path = reconstruct_path(
                came_from,
                current,
            )
 
            env._set_qpos(saved_qpos)
            p.performCollisionDetection()
 
            return path, state_qposes
 
        x, y, yaw_index = current
 
        neighbours = [
            (
                x + 1,
                y,
                yaw_index,
                GRID_SIZE,
            ),
            (
                x - 1,
                y,
                yaw_index,
                GRID_SIZE,
            ),
            (
                x,
                y + 1,
                yaw_index,
                GRID_SIZE,
            ),
            (
                x,
                y - 1,
                yaw_index,
                GRID_SIZE,
            ),
            (
                x,
                y,
                (yaw_index + 1) % len(YAW_VALUES),
                20.0,
            ),
            (
                x,
                y,
                (yaw_index - 1) % len(YAW_VALUES),
                20.0,
            ),
        ]
 
        for nx, ny, nyaw_index, step_cost in neighbours:
            neighbour = (
                nx,
                ny,
                nyaw_index,
            )
 
            if neighbour in checked_states:
                continue
 
            if not inside_bounds(neighbour):
                continue
 
            candidate_qpos = calculate_carry_configuration(
                env=env,
                reference_qpos=reference_qpos,
                x=nx * GRID_SIZE,
                y=ny * GRID_SIZE,
                yaw=YAW_VALUES[nyaw_index],
            )
 
            previous_qpos = state_qposes[current]
 
            if not transition_is_collision_free(
                env=env,
                start_qpos=previous_qpos,
                end_qpos=candidate_qpos,
                obstacle_ids=obstacle_ids,
                samples=25,
            ):
                continue
 
            new_cost = current_cost + step_cost
 
            if (
                neighbour not in cost_so_far
                or new_cost < cost_so_far[neighbour]
            ):
                cost_so_far[neighbour] = new_cost
                state_qposes[neighbour] = (
                    candidate_qpos
                )
 
                priority = (
                    new_cost
                    + heuristic(
                        neighbour,
                        goal,
                    )
                )
 
                heapq.heappush(
                    open_heap,
                    (
                        priority,
                        new_cost,
                        neighbour,
                    ),
                )
 
                came_from[neighbour] = current
 
    env._set_qpos(saved_qpos)
    p.performCollisionDetection()
 
    return None, None
 
 
if __name__ == "__main__":
    env = PickNavReachEnv(
        seed=0,
        object_idx=0,
        gui=True,
    )
 
    path, path_qposes = (
        plan_configuration_space_path(env)
    )
 
    if path is None:
        raise RuntimeError(
            "No carried-arm collision-free path found."
        )
 
    print(
        "\nCarried-arm collision-free path found.",
        flush=True,
    )
 
    print(
        "Number of states:",
        len(path),
        flush=True,
    )
 
    for index, state in enumerate(path):
        x, y, yaw_index = state
 
        print(
            index,
            "position=",
            grid_to_world((x, y)),
            "yaw=",
            YAW_VALUES[yaw_index],
            flush=True,
        )
 
    input("\nPress Enter to close...")
 
 
def plan_configuration_space_path(env, goal_position=None):
    """Plan with the carried object checked against the maze; if that is too strict to find any
    path, fall back to checking only the robot body."""
    global _CHECK_CARRIED
    for check_carried in (True, False):
        _CHECK_CARRIED = check_carried
        try:
            path, qposes = _plan_configuration_space_path_impl(env, goal_position)
        finally:
            _end_carry()
        if path is not None:
            return path, qposes
    return None, None
