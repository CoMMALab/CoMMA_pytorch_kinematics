import torch


def normalize_quaternion(quat):
    norm = torch.norm(quat, p=2, dim=-1, keepdim=True)
    return quat / norm

def slerp(start_quat, end_quat, t):
    # Ensure quaternions are normalized
    start_quat = normalize_quaternion(start_quat)
    end_quat = normalize_quaternion(end_quat)
    
    # Squeeze dimensions if they are singleton dimensions
    if start_quat.dim() > 1:
        start_quat = start_quat.squeeze(0)
        end_quat = end_quat.squeeze(0)
    
    # Compute the dot product
    dot = torch.sum(start_quat * end_quat, dim=-1)
    
    # Clamp the dot product to avoid numerical issues
    dot = torch.clamp(dot, min=-1.0, max=1.0)
    
    # Compute the angle between the quaternions
    theta_0 = torch.acos(dot)
    theta = theta_0 * t
    
    # Compute sin(theta) and sin(theta_0 - theta)
    sin_theta = torch.sin(theta)
    sin_theta_0 = torch.sin(theta_0)
    
    # Handle the case when sin(theta_0) is zero
    coeff_0 = torch.where(sin_theta_0 < 1e-6, torch.ones_like(sin_theta), (torch.sin(theta_0 - theta) / sin_theta_0))
    coeff_1 = torch.where(sin_theta_0 < 1e-6, torch.zeros_like(sin_theta), (torch.sin(theta) / sin_theta_0))
    
    # Perform SLERP interpolation
    interpolated_quat = (coeff_0.unsqueeze(0) * start_quat) + (coeff_1.unsqueeze(0) * end_quat)
    
    # Normalize the result
    interpolated_quat = normalize_quaternion(interpolated_quat)
    
    # Add back the batch dimension if it was squeezed
    if start_quat.dim() == 1:
        interpolated_quat = interpolated_quat.unsqueeze(0)
    
    return interpolated_quat