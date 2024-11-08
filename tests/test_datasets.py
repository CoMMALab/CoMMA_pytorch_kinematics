import json
import os
from timeit import default_timer as timer

import numpy as np
import torch

import pytorch_kinematics as pk
import pytorch_seed

import pybullet as p
import pybullet_data
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


file_path_1 = 'datasets/demo_raw_data.json'
file_path_2 = 'datasets/motion_benchmaker_raw_data.json'
file_path_3 = 'datasets/mpinets_raw_data.json'

with open(file_path_1, 'r') as json_file_1:
    data_dict_1 = json.load(json_file_1)
with open(file_path_2, 'r') as json_file_2:
    data_dict_2 = json.load(json_file_2)
with open(file_path_3, 'r') as json_file_3:
    data_dict_3 = json.load(json_file_3)



def test_dataset_ik(data, urdf, chain, interpolation_method, M, n=10, visualize = True, delay = True, skip = False):
    # data is a list of dictionaries which contain position and rotation tensors under goal poses. Datasets are loaded from JSON files.

    coalesced_position = []
    coalesced_rotation = []
    for i in range(M):
        # print(data[i]['goal_pose'])
        coalesced_position.append(data[i]['goal_pose']['position_xyz'])
        coalesced_rotation.append(data[i]['goal_pose']['quaternion_wxyz'])



    coalesced_position = torch.tensor(coalesced_position, device=device)
    coalesced_rotation = torch.tensor(coalesced_rotation, device=device)

    # robot frame
    pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    rot = torch.tensor([0.0, 0.0, 0.0], device=device)
    rob_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

    # world frame goal
    # use the joint limits to generate random joint angles
    lim = torch.tensor(chain.get_joint_limits(), device=device)

    # get ee pose (in robot frame)
    # Create non-random end effector goal
    # Stacking dataset poses and feeding them as goals into the ik solver, testing to see if they can reach pre-established goals
    goal_in_rob_frame_tf = pk.Transform3d(pos=coalesced_position, rot=coalesced_position, device=device)

    # transform to world frame for visualization
    goal_tf = rob_tf.compose(goal_in_rob_frame_tf)
    goal = goal_tf.get_matrix()
    goal_pos = goal[..., :3, 3]
    goal_rot = pk.matrix_to_euler_angles(goal[..., :3, :3], "XYZ")

    num_retries = 10
    ik = pk.PseudoInverseIK(chain, max_iterations=30, num_retries=num_retries,
                            joint_limits=lim.T,
                            early_stopping_any_converged=True,
                            early_stopping_no_improvement="all",
                            # line_search=pk.BacktrackingLineSearch(max_lr=0.2),
                            debug=False,
                            lr=0.2)

    # do IK
    timer_start = timer()
    sol = None
    # Only uses interpolation support. TODO: Maybe refactor no interpolation method to have consistent return type
    if interpolation_method == 'parallel':
        sol = ik.parallel_interpolation_solve(rob_tf, goal_in_rob_frame_tf, n)
    elif interpolation_method == 'iterative':
        sol = ik.iterative_interpolation_solve(rob_tf, goal_in_rob_frame_tf, n)
    else: 
        raise AssertionError("invalid interpolation method provided.")

    # Initialize lists to store err_pos and err_rot tensors
    err_pos_list = []
    err_rot_list = []

    # Iterate through each IKSolution and accumulate the tensors
    for s in sol:
        err_pos_list.append(s.err_pos)
        err_rot_list.append(s.err_rot)

    # Stack tensors along a new dimension to accumulate them
    accumulated_err_pos = torch.stack(err_pos_list)
    accumulated_err_rot = torch.stack(err_rot_list)

    # Compute the average position and rotation errors
    average_err_pos = torch.mean(accumulated_err_pos)
    average_err_rot = torch.mean(accumulated_err_rot)

    # Print the results
    print("\n\nAverage Position Error:", average_err_pos.item())
    print("Average Rotation Error:", average_err_rot.item())


    timer_end = timer()


    total_converged = 0
    total_iterations = 0
    total_converged_any = 0
    M_total = 0
    M_total_any = 0

    for s in sol:
        total_converged += s.converged.sum().item()  
        M_total += s.converged.numel() 
        total_iterations += s.iterations
        total_converged_any += s.converged_any.sum().item()
        M_total_any += s.converged_any.numel() 
        

    print("IK took %f seconds" % (timer_end - timer_start))
    print("IK converged number: %d / %d" % (total_converged, M_total))
    print("IK took %d iterations" % total_iterations)
    print("IK solved %d / %d goals" % (total_converged_any, M_total_any))

    # check that solving again produces the same solutions
    # visualize everything
    if visualize:
        p.connect(p.GUI)
        p.setRealTimeSimulation(False)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(search_path)

        yaw = 90
        pitch = -65
        # dist = 1.
        dist = 2.4
        target = np.array([2., 1.5, 0])
        p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

        plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])

        # make 1 per retry with positional offsets
        robots = []
        num_robots = 16
        # 4x4 grid position offset
        offset = 1.0
        m = rob_tf.get_matrix()
        pos = m[0, :3, 3]
        rot = m[0, :3, :3]
        quat = pk.matrix_to_quaternion(rot)
        pos = pos.cpu().numpy()
        rot = pk.wxyz_to_xyzw(quat).cpu().numpy()

        for i in range(num_robots):
            this_offset = np.array([i % 4 * offset, i // 4 * offset, 0])
            armId = p.loadURDF(urdf, basePosition=pos + this_offset, baseOrientation=rot, useFixedBase=True)
            # _make_robot_translucent(armId, alpha=0.6)
            robots.append({"id": armId, "offset": this_offset, "pos": pos})

        show_max_num_retries_per_goal = 10

        goals = []
        # draw cone to indicate pose instead of sphere
        visId = p.createVisualShape(p.GEOM_MESH, fileName="meshes/cone.obj", meshScale=1.0,
                                    rgbaColor=[0., 1., 0., 0.5])
        for i in range(num_robots):
            goals.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=visId))

        try:
            # batch over goals with num_robots
            for j in range(0, M, num_robots):
                this_selection = slice(j, j + num_robots)
                
                r = goal_rot[:,this_selection]
                xyzw = pk.wxyz_to_xyzw(pk.matrix_to_quaternion(pk.euler_angles_to_matrix(r, "XYZ")))

                solutions = [s.solutions[this_selection, :, :] for s in sol]
                converged = torch.cat([s.converged[this_selection, :] for s in sol], dim=0)

                # print how many retries converged for this one

                # TODO: Check that this converged value is right
                print("Goal %d to %d converged %d / %d" % (j, j + num_robots, converged.sum(), converged.numel()))
                
                # outer loop over retries, inner loop over goals (for each robot shown in parallel)
                for ii in range(num_retries):
                    if ii > show_max_num_retries_per_goal:
                        break
                    for jj in range(num_robots):
                        p.resetBasePositionAndOrientation(goals[jj],
                                                            goal_pos[j + jj].cpu().numpy() + robots[jj]["offset"],
                                                            xyzw[jj].cpu().numpy())
                        
                        armId = robots[jj]["id"]
                        for step in range(len(sol)):
                            q = solutions[step][jj, ii, :]
                            for dof in range(q.shape[0]):
                                p.resetJointState(armId, dof, q[dof])
                            if delay:
                                time.sleep(0.05)
                    if skip:
                        input("Press enter to continue")
        except:
            print("error has occurred")
        if not skip:
            p.disconnect()    
        else:
            while True:
                p.stepSimulation()

def test_dataset_fk(data, chain, device):
    # data is a list of dictionaries which contain position and rotation tensors under goal poses. Datasets are loaded from JSON files.
    def quat_pos_from_transform3d(tg):
        m = tg.get_matrix()
        pos = m[:, :3, 3]
        rot = pk.matrix_to_quaternion(m[:, :3, :3])
        return pos, rot
    
    # coalesced_joints = []
    
    # testing if forward kinematics on the provided joints results in the associated pose. Unfortunately there are several joint angles (likely to demonstrate movement) 
    # which makes it difficult to pinpoint which one to use. I chose the last joint angles arbitrarily as i assume they are closest to the end pose
    # but there is still a discrepency with the output pose and computed pose.
    pos_errors = []
    rot_errors = []
    for i in range(M):
        # for j in range(len(data[i]['goal_ik'])):
        # print(data[i]['goal_ik'][-1])
        joints = data[i]['goal_ik'][0]
        ret = chain.forward_kinematics(joints)
        tg = ret
        pos, rot = quat_pos_from_transform3d(tg)
        pos_error = (pos - torch.tensor([data[i]['goal_pose']['position_xyz']], device=device))[0]
        rot_error = (rot - torch.tensor([data[i]['goal_pose']['quaternion_wxyz']], device=device))[0]
        pos_errors.append(pos_error)
        rot_errors.append(rot_error)


    # Convert lists to tensors for averaging
    pos_errors_tensor = torch.stack(pos_errors)
    rot_errors_tensor = torch.stack(rot_errors)

    # Compute the mean (average) for position and rotation errors
    avg_pos_error = pos_errors_tensor.mean(dim=0)
    avg_rot_error = rot_errors_tensor.mean(dim=0)
    print(f"avg pos error: {avg_pos_error}")
    print(f"avg rot error: {avg_rot_error}")
    # print()
    print('------------------------------------------------------------')

    


if __name__ == "__main__":
    search_path = pybullet_data.getDataPath()

    urdf = "kuka_iiwa/model.urdf"
    full_urdf = os.path.join(search_path, urdf)
    chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")
    chain = chain.to(device=device)

    # urdf = "widowx/wx250s.urdf"
    # full_urdf = urdf
    # chain = pk.build_serial_chain_from_urdf(open(full_urdf, "rb").read(), "ee_gripper_link")
    # chain = chain.to(device=device)

    # urdf = "franka/fp3_franka_hand.urdf"
    # full_urdf = urdf
    # chain = pk.build_chain_from_urdf(open(full_urdf, mode="rb").read())
    # chain = pk.SerialChain(chain, "fp3_hand_tcp", "base")
    # chain = chain.to(device=device)

    M = len(data_dict_2['bookshelf_small_panda'])
    data = data_dict_2['bookshelf_small_panda']

    # testing fk on the widow robot doesn't work, likely a mismatch of joints
    test_dataset_fk(data,chain, device)
    test_dataset_ik(data=data, urdf=urdf, chain=chain,interpolation_method='parallel',M=M,n=100)
    

    