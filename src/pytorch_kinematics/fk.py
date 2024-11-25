from functools import lru_cache
from typing import Optional, Sequence

import copy
import numpy as np
import torch

import python_robotics_middleware.transforms as tf
from python_robotics_middleware import SerialChain
from python_robotics_middleware.transforms.rotation_conversions import axis_and_angle_to_matrix_44, axis_and_d_to_pris_matrix


class FKSolution:
    def __init__(self):
        self.temporary_parameter = None

    def forward_kinematics(self, chain, th, frame_indices: Optional = None, end_only: bool = True):
        """
        Compute forward kinematics for the given joint values.

        Args:
            th: A dict, list, numpy array, or torch tensor of joints values. Possibly batched.
            frame_indices: A list of frame indices to compute transforms for. If None, all frames are computed.
                Use `get_frame_indices` to convert from frame names to frame indices.

        Returns:
            A dict of Transform3d objects for each frame.

        """

        if isinstance(chain, SerialChain):
            """ Like the base class, except `th` only needs to contain the joints in the SerialChain, not all joints. """
            frame_indices, th = chain.convert_serial_inputs_to_chain_inputs(th, end_only)


        if frame_indices is None:
            frame_indices = chain.get_all_frame_indices()

        th = chain.ensure_tensor(th)
        th = torch.atleast_2d(th)

        b = th.shape[0]
        axes_expanded = chain.axes.unsqueeze(0).repeat(b, 1, 1)

        # compute all joint transforms at once first
        # in order to handle multiple joint types without branching, we create all possible transforms
        # for all joint types and then select the appropriate one for each joint.
        rev_jnt_transform = axis_and_angle_to_matrix_44(axes_expanded, th)
        pris_jnt_transform = axis_and_d_to_pris_matrix(axes_expanded, th)

        frame_transforms = {}
        b = th.shape[0]
        for frame_idx in frame_indices:
            frame_transform = torch.eye(4).to(th).unsqueeze(0).repeat(b, 1, 1)

            # iterate down the list and compose the transform
            for chain_idx in chain.parents_indices[frame_idx.item()]:
                if chain_idx.item() in frame_transforms:
                    frame_transform = frame_transforms[chain_idx.item()]
                else:
                    link_offset_i = chain.link_offsets[chain_idx]
                    if link_offset_i is not None:
                        frame_transform = frame_transform @ link_offset_i

                    joint_offset_i = chain.joint_offsets[chain_idx]
                    if joint_offset_i is not None:
                        frame_transform = frame_transform @ joint_offset_i

                    jnt_idx = chain.joint_indices[chain_idx]
                    jnt_type = chain.joint_type_indices[chain_idx]
                    if jnt_type == 0:
                        pass
                    elif jnt_type == 1:
                        jnt_transform_i = rev_jnt_transform[:, jnt_idx]
                        frame_transform = frame_transform @ jnt_transform_i
                    elif jnt_type == 2:
                        jnt_transform_i = pris_jnt_transform[:, jnt_idx]
                        frame_transform = frame_transform @ jnt_transform_i

            frame_transforms[frame_idx.item()] = frame_transform

        frame_names_and_transform3ds = {chain.idx_to_frame[frame_idx]: tf.Transform3d(matrix=transform) for
                                        frame_idx, transform in frame_transforms.items()}
        
        if isinstance(chain, SerialChain):
            mat = frame_names_and_transform3ds

            if end_only:
                return mat[chain._serial_frames[-1].name]
            else:
                return mat
            
        return frame_names_and_transform3ds

