import numpy as np
import pybullet as p
 
END_EFFECTOR_LINK = 20                      # gripper_link
ARM_JOINTS = [3, 6, 7, 8, 9, 10, 11, 12]    # indices in the 15-number action: torso, shoulder_pan,
                                            # shoulder_lift, upperarm_roll, elbow, forearm_roll,
                                            # wrist_flex, wrist_roll
GRIPPER_OPEN = 0.05
 
 
def quat_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
 
 
def set_joints(env, q):
    """Teleport the robot to configuration q (no physics, no motion)."""
    for joint_index, value in zip(env.joint_indices, q):
        p.resetJointState(env.robot_id, joint_index, float(value))
 
 
def gripper_pose(env, q=None):
    """World position and orientation (xyzw) of gripper_link."""
    if q is not None:
        set_joints(env, q)
    state = p.getLinkState(env.robot_id, END_EFFECTOR_LINK, computeForwardKinematics=1)
    return np.array(state[4]), np.array(state[5])
 
 
def pose_error(env, target_pos, target_quat):
    pos, quat = gripper_pose(env)
    position_error = float(np.linalg.norm(pos - target_pos))
    angle_error = 2.0 * np.arccos(min(1.0, abs(float(np.dot(quat, target_quat)))))
    return position_error, angle_error
 
 
def _frame_origin_in_com_frame(env):
    """The Jacobian is computed at a point given in the link's centre-of-mass frame."""
    state = p.getLinkState(env.robot_id, END_EFFECTOR_LINK, computeForwardKinematics=1)
    com_pos, com_quat, frame_pos = np.array(state[0]), state[1], np.array(state[4])
    return (quat_to_matrix(com_quat).T @ (frame_pos - com_pos)).tolist()
 
 
def _orientation_error(R_now, R_target):
    return 0.5 * (np.cross(R_now[:, 0], R_target[:, 0])
                  + np.cross(R_now[:, 1], R_target[:, 1])
                  + np.cross(R_now[:, 2], R_target[:, 2]))
 
 
def default_seeds(env):
    """Starting arm postures. IK is iterative, so it can get stuck - we retry from several."""
    seeds = []
    for torso in (0.3, 0.1):
        for shoulder_lift, elbow in ((-0.5, 1.6), (0.3, 1.0), (0.8, 0.5)):
            q = np.zeros(len(env.joint_indices))
            q[3], q[7], q[9], q[11] = torso, shoulder_lift, elbow, 0.6
            seeds.append(q)
    return seeds
 
 
def solve_arm_ik(env, target_pos, target_quat, base_xy_yaw, seeds=None,
                 pos_tol=0.003, angle_tol=0.03, max_iterations=100):
    """
    Returns a full 15-number joint vector that puts gripper_link at (target_pos, target_quat)
    with the base at base_xy_yaw = (x, y, yaw), or None if no solution was found.
    """
    seeds = default_seeds(env) if seeds is None else seeds
    low = env.joint_lower.astype(float).copy()
    high = env.joint_upper.astype(float).copy()
    for j in (8, 10, 12):                     # the three "roll" joints can spin forever; keep +-pi
        low[j], high[j] = -np.pi, np.pi
    R_target = quat_to_matrix(target_quat)
    local_point = _frame_origin_in_com_frame(env)
    n_dof = len(env.joint_indices)
 
    for seed in seeds:
        q = np.array(seed, dtype=float)
        q[:3] = base_xy_yaw
        q[4:6] = 0.0                          # head
        q[13:15] = GRIPPER_OPEN
        rest = q.copy()
        for _ in range(max_iterations):
            set_joints(env, q)
            pos, quat = gripper_pose(env)
            e_pos = np.asarray(target_pos) - pos
            e_rot = _orientation_error(quat_to_matrix(quat), R_target)
            if np.linalg.norm(e_pos) < 0.3 * pos_tol and np.linalg.norm(e_rot) < 0.3 * angle_tol:
                break
            lin, ang = p.calculateJacobian(env.robot_id, END_EFFECTOR_LINK, local_point,
                                           q.tolist(), [0.0] * n_dof, [0.0] * n_dof)
            J = np.vstack([np.array(lin)[:, ARM_JOINTS], np.array(ang)[:, ARM_JOINTS]])
            error = np.concatenate([e_pos, e_rot])
            J_pinv = J.T @ np.linalg.inv(J @ J.T + 0.03 ** 2 * np.eye(6))
            dq = J_pinv @ error
            # spare freedom (8 joints, 6 constraints): drift gently back to the seed posture
            dq += (np.eye(len(ARM_JOINTS)) - J_pinv @ J) @ (0.05 * (rest[ARM_JOINTS] - q[ARM_JOINTS]))
            biggest = np.max(np.abs(dq))
            if biggest > 0.15:                # never jump more than 0.15 rad per iteration
                dq *= 0.15 / biggest
            q[ARM_JOINTS] = np.clip(q[ARM_JOINTS] + dq, low[ARM_JOINTS], high[ARM_JOINTS])
        set_joints(env, q)
        position_error, angle_error = pose_error(env, target_pos, target_quat)
        if position_error < pos_tol and angle_error < angle_tol:
            return q
    return None