from pytorch_kinematics import matrix_to_euler_angles
import torch

def compute_error(goal_tf, end_effector_tf):
    goal_pos = goal_tf.get_matrix()[..., :3, 3]
    end_effector_pos = end_effector_tf.get_matrix()[..., :3, 3]
    pos_error = torch.norm(goal_pos - end_effector_pos, dim=-1)

    goal_rot = matrix_to_euler_angles(goal_tf.get_matrix()[..., :3, :3], "XYZ")
    end_effector_rot = matrix_to_euler_angles(end_effector_tf.get_matrix()[..., :3, :3], "XYZ")
    rot_error = torch.norm(goal_rot - end_effector_rot, dim=-1)

    return pos_error, rot_error

