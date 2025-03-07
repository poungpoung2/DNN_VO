import numpy as np


def euler_to_rot(pose):
    # Extract Euler angles
    gamma = pose["orientation"][0]  # X-axis rotation
    beta = pose["orientation"][1]  # Y-axis rotation
    alpha = pose["orientation"][2]  # Z-axis rotation

    # Precompute sine and cosine values
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    cos_beta, sin_beta = np.cos(beta), np.sin(beta)
    cos_gamma, sin_gamma = np.cos(gamma), np.sin(gamma)

    # Construct rotation matrix (ZYX convention)
    R = np.array(
        [
            [
                cos_alpha * cos_beta,
                cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma,
                cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma,
            ],
            [
                sin_alpha * cos_beta,
                sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma,
                sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma,
            ],
            [-sin_beta, cos_beta * sin_gamma, cos_beta * cos_gamma],
        ]
    )

    return R


def pose_to_trans(pose):
    R = euler_to_rot(pose)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [pose["position"][0], pose["position"][1], pose["position"][2]]

    return T


def rot_to_euler(rot_mat):
    roll = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    pitch = np.arctan2(
        -1 * rot_mat[2, 0], np.sqrt(np.square(rot_mat[0, 0]) + np.square(rot_mat[1, 0]))
    )
    yaw = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])

    return np.array([roll, pitch, yaw])


def get_relative_pose(pose1, pose2):
    # Extract position and orientation
    T_1 = pose_to_trans(pose1)
    T_2 = pose_to_trans(pose2)

    T_rel = np.linalg.inv(T_1) @ T_2
    
    rel_trans = T_rel[:3, 3]
    rel_rot = rot_to_euler(T_rel[:3, :3])

    return np.concatenate([rel_trans, rel_rot])





