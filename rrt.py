"""
rrt.py - collision-aware arm motion planning (RRT-Connect in joint space).

plan_arm_path(env, q_start, q_goal, obstacle_ids) returns a list of 15-number configurations that
moves the torso and the 7-joint arm from q_start to q_goal without touching the table, the object
or the robot itself.  The base, head and gripper joints keep the values they have in q_start.

If the straight joint-space line between the two configurations is already collision free it is
returned as is, so the planner only changes the motion when the direct motion would hit something.
"""
import numpy as np
import pybullet as p

from ik_solver import ARM_JOINTS, set_joints

STEP = 0.15             # largest joint-space step between tree nodes (rad, max over joints)
CHECK_STEP = 0.04       # spacing of collision checks along an edge (rad)
CLEARANCE = 0.004       # robot closer than this to an obstacle counts as a collision (m)


def _contacts(env, q, obstacle_ids):
    """Set of colliding pairs in configuration q (teleports the robot there)."""
    set_joints(env, q)
    p.performCollisionDetection()
    pairs = set()
    for body in obstacle_ids:
        for c in p.getClosestPoints(env.robot_id, body, CLEARANCE):
            pairs.add((body, c[3]))
    for c in p.getClosestPoints(env.robot_id, env.robot_id, 0.0):
        if abs(c[3] - c[4]) > 1:                       # links that are not neighbours in the chain
            pairs.add(("self", min(c[3], c[4]), max(c[3], c[4])))
    return pairs


class _Space:
    def __init__(self, env, q_start, q_goal, obstacle_ids, seed):
        self.env = env
        self.obstacles = obstacle_ids
        self.template = np.array(q_start, dtype=float)
        self.lo = np.array(env.joint_lower, dtype=float)[ARM_JOINTS]
        self.hi = np.array(env.joint_upper, dtype=float)[ARM_JOINTS]
        self.rng = np.random.default_rng(seed)
        # contacts that already exist at both ends (e.g. fingers resting near the table) are tolerated
        self.allowed = _contacts(env, self.full(self.arm(q_start)), obstacle_ids) | \
                       _contacts(env, self.full(self.arm(q_goal)), obstacle_ids)

    def arm(self, q):
        return np.array(q, dtype=float)[ARM_JOINTS]

    def full(self, a):
        q = self.template.copy()
        q[ARM_JOINTS] = a
        return q

    def free(self, a):
        return not (_contacts(self.env, self.full(a), self.obstacles) - self.allowed)

    def edge_free(self, a, b):
        n = max(int(np.ceil(np.abs(b - a).max() / CHECK_STEP)), 1)
        return all(self.free(a + (b - a) * t) for t in np.linspace(0.0, 1.0, n + 1)[1:])

    def sample(self):
        return self.rng.uniform(self.lo, self.hi)


def _extend(space, tree, target):
    """Grow tree one STEP towards target; return the new node index or None if blocked."""
    nodes = np.array([n for n, _ in tree])
    i = int(np.argmin(np.abs(nodes - target).max(axis=1)))
    near = tree[i][0]
    d = target - near
    m = np.abs(d).max()
    new = target.copy() if m <= STEP else near + d * (STEP / m)
    if not space.edge_free(near, new):
        return None
    tree.append((new, i))
    return len(tree) - 1


def _path(tree, i):
    out = []
    while i is not None:
        out.append(tree[i][0])
        i = tree[i][1]
    return out[::-1]


def _shortcut(space, path, rng, tries=60):
    path = list(path)
    for _ in range(tries):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i > 1 and space.edge_free(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return path


def plan_arm_path(env, q_start, q_goal, obstacle_ids, max_iters=1500, seed=0, verbose=False):
    """Returns a list of 15-number configurations from q_start to q_goal, or None if none was found."""
    saved = env.agent_qpos.copy()
    try:
        space = _Space(env, q_start, q_goal, obstacle_ids, seed)
        a, b = space.arm(q_start), space.arm(q_goal)
        if space.edge_free(a, b):
            return [np.array(q_start, dtype=float), np.array(q_goal, dtype=float)]
        if not space.free(b):
            return None
        tree_a, tree_b = [(a, None)], [(b, None)]
        swapped = False
        for it in range(max_iters):
            target = space.sample()
            ia = _extend(space, tree_a, target)
            if ia is not None:
                while True:                              # greedily connect the other tree to the new node
                    ib = _extend(space, tree_b, tree_a[ia][0])
                    if ib is None:
                        break
                    if np.abs(tree_b[ib][0] - tree_a[ia][0]).max() < 1e-9:
                        pa, pb = _path(tree_a, ia), _path(tree_b, ib)[::-1]
                        arm_path = pa + pb[1:]
                        if swapped:
                            arm_path = arm_path[::-1]
                        arm_path = _shortcut(space, arm_path, space.rng)
                        if verbose:
                            print(f"rrt: path with {len(arm_path)} waypoints after {it + 1} iterations", flush=True)
                        return [space.full(w) for w in arm_path]
            tree_a, tree_b = tree_b, tree_a
            swapped = not swapped
        return None
    finally:
        set_joints(env, saved)