import numpy as np
from scipy.spatial.transform import Rotation

TABLE_TOP_Z = 0.626  # height of the table surface, measured from the env

def grasp_orientation_from_contacts(contact_a, contact_b):
    """
    Construct a gripper orientation from two contact points.

    The gripper's local y-axis is aligned with the direction in which
    the fingers should close. The local z-axis points downward toward
    the object.
    """

    contact_a = np.asarray(contact_a, dtype=np.float32)
    contact_b = np.asarray(contact_b, dtype=np.float32)

    closing_axis = contact_b - contact_a
    closing_axis = closing_axis / np.linalg.norm(closing_axis)

    # The robot is located to the left of the object.
    # Approach horizontally in the positive x direction.
    approach_axis = np.array(
        [0.0, 0.0, -1.0],
        dtype=np.float32,
    )   

    # Build an orthonormal right-handed coordinate frame.
    local_y = closing_axis
    local_z = approach_axis
    local_x = np.cross(local_y, local_z)
    local_x = local_x / np.linalg.norm(local_x)

    # Recompute y to remove numerical non-orthogonality.
    local_y = np.cross(local_z, local_x)
    local_y = local_y / np.linalg.norm(local_y)

    rotation_matrix = np.column_stack(
        [local_x, local_y, local_z]
    )

    quaternion_xyzw = Rotation.from_matrix(
        rotation_matrix
    ).as_quat()

    return quaternion_xyzw

def generate_antipodal_grasps(
    object_pc,
    object_normals,
    min_gripper_width=0.02,
    max_gripper_width=0.10,
    normal_threshold=-0.80,
    max_candidates=20,
    min_height_above_table=0.03,
):
    """
    Generate antipodal grasp candidates from an object point cloud.

    An antipodal grasp uses two contact points whose surface normals
    point approximately toward one another.
    """

    object_pc = np.asarray(object_pc, dtype=np.float32)
    object_normals = np.asarray(object_normals, dtype=np.float32)

    if object_pc.shape != object_normals.shape:
        raise ValueError(
            "object_pc and object_normals must have the same shape"
        )

    num_points = object_pc.shape[0]

    # Normalise the normals in case they are not exactly unit vectors.
    normal_lengths = np.linalg.norm(
        object_normals,
        axis=1,
        keepdims=True,
    )

    normal_lengths = np.maximum(normal_lengths, 1e-8)

    normals = object_normals / normal_lengths

    candidates = []
    object_centre = object_pc.mean(axis=0)

    # object_min_z = object_pc[:, 2].min()
    # object_max_z = object_pc[:, 2].max()
    # object_height = object_max_z - object_min_z

    # minimum_grasp_z = object_min_z + 0.05 * object_height

    # Compare pairs of points.
    for i in range(num_points):
        for j in range(i + 1, num_points):
            contact_a = object_pc[i]
            contact_b = object_pc[j]

            contact_vector = contact_b - contact_a
            width = np.linalg.norm(contact_vector)

            # Reject pairs outside the gripper's usable width.
            if width < min_gripper_width:
                continue

            if width > max_gripper_width:
                continue

            contact_direction = contact_vector / width

            normal_a = normals[i]
            normal_b = normals[j]

            # For an antipodal grasp:
            # normal_a should point roughly toward contact_b
            # normal_b should point roughly toward contact_a
            alignment_a = -np.dot(normal_a, contact_direction)
            alignment_b = np.dot(normal_b, contact_direction)

            if alignment_a < -normal_threshold:
                continue

            if alignment_b < -normal_threshold:
                continue

            # The two normals should point approximately toward one another.
            normal_opposition = np.dot(normal_a, normal_b)

            if normal_opposition > normal_threshold:
                continue

            midpoint = 0.5 * (contact_a + contact_b)

            # the pads are 2.6 cm tall, so keep the grasp well above the table
            if midpoint[2] - TABLE_TOP_Z < min_height_above_table:
                continue

            # prefer grasps near the object's centre (more stable when lifted)
            centre_distance = np.linalg.norm(midpoint - object_centre)

            score = (
                alignment_a
                + alignment_b
                - 0.25 * abs(normal_opposition + 1.0)
                - 4.0 * centre_distance
            )

            candidates.append(
                {
                    "contact_a": contact_a.copy(),
                    "contact_b": contact_b.copy(),
                    "midpoint": midpoint,
                    "width": width,
                    "normal_a": normal_a.copy(),
                    "normal_b": normal_b.copy(),
                    "score": float(score),
                }
            )

    # Highest-scoring grasps first.
    candidates.sort(
        key=lambda candidate: candidate["score"],
        reverse=True,
    )

    return candidates[:max_candidates]


if __name__ == "__main__":
    import pybullet as p

    from pick_nav_reach_env import PickNavReachEnv

    env = PickNavReachEnv(seed=0, object_idx=0)

    # Let the object settle.
    action = env.agent_qpos.copy()
    action[13] = 0.05
    action[14] = 0.05

    for _ in range(100):
        obs, info = env.step(action)

    grasps = generate_antipodal_grasps(
    obs["object_pc"],
    obs["object_normals"],
    max_gripper_width=0.12,
    normal_threshold=-0.70,
    max_candidates=20,
    )

    print("Number of generated grasp candidates:", len(grasps))

    if len(grasps) > 0:
        best_grasp = grasps[0]

        grasp_quaternion = grasp_orientation_from_contacts(
            best_grasp["contact_a"],
            best_grasp["contact_b"],
        )

        print("Best grasp midpoint:", best_grasp["midpoint"])
        print("Best grasp width:", best_grasp["width"])
        print("Best grasp orientation:", grasp_quaternion)

    for index, grasp in enumerate(grasps[:5]):
        print(f"\nGrasp {index + 1}")
        print("Contact A:", grasp["contact_a"])
        print("Contact B:", grasp["contact_b"])
        print("Midpoint:", grasp["midpoint"])
        print("Width:", grasp["width"])
        print("Score:", grasp["score"])

    input("\nPress Enter to close...")