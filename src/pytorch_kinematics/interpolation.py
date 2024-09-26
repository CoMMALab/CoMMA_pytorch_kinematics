import pytorch_kinematics as pk
import torch

def interpolate_poses(start_transforms, end_transforms, n):
    device = start_transforms.device
    
    # Extract start and end transformation matrices
    start_matrices = start_transforms.get_matrix()  # (batch_size, 4, 4)
    end_matrices = end_transforms.get_matrix()      # (batch_size, 4, 4)
    if start_matrices.shape[0] == 1 and end_matrices.shape[0] > 1:
        start_matrices = start_matrices.repeat(end_matrices.shape[0], 1, 1)

    # print(start_matrices.shape)
    # print(end_matrices.shape)
    
    # Extract positions (last column) and rotations (upper 3x3 part) from the transformation matrices
    start_pos = start_matrices[..., :3, 3]  # (batch_size, 3)
    end_pos = end_matrices[..., :3, 3]      # (batch_size, 3)
    
    # print(start_matrices[..., :3, :3].shape)
    # print(end_matrices[..., :3, :3].shape)

    start_rot = pk.matrix_to_quaternion(start_matrices[..., :3, :3])  # (batch_size, 4)
    end_rot = pk.matrix_to_quaternion(end_matrices[..., :3, :3])      # (batch_size, 4)

    interpolated_poses = []

    # print(start_rot.shape)
    # print(end_rot.shape)

    # Interpolation steps
    for t in torch.linspace(0, 1, n, device=device):  # (n,)
        # LERP for position (batch_size, 3)
        interp_pos = (1 - t) * start_pos + t * end_pos

        # SLERP for quaternion rotation (batch_size, 4)
        interp_rot = pk.quaternion_slerp(start_rot, end_rot, t)

        # Reconstruct the rotation matrices from the interpolated quaternions (batch_size, 3, 3)
        interp_rot_matrices = pk.quaternion_to_matrix(interp_rot)

        # Construct the interpolated transformation matrices (batch_size, 4, 4)
        interp_matrices = torch.eye(4, device=device).unsqueeze(0).repeat(start_matrices.shape[0], 1, 1)  # (batch_size, 4, 4)
        interp_matrices[..., :3, :3] = interp_rot_matrices
        interp_matrices[..., :3, 3] = interp_pos

        # Create a batch of interpolated Transform3d from the matrices
        interp_transforms = pk.Transform3d(matrix=interp_matrices)
        interpolated_poses.append(interp_transforms)
    # raise NameError
    return interpolated_poses
