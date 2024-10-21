import os
from timeit import default_timer as timer

import numpy as np
import torch

import pytorch_kinematics as pk
import pytorch_seed

import pybullet as p
import pybullet_data
import time

visualize = True


def _make_robot_translucent(robot_id, alpha=0.4):
    def make_transparent(link):
        link_id = link[1]
        rgba = list(link[7])
        rgba[3] = alpha
        p.changeVisualShape(robot_id, link_id, rgbaColor=rgba)

    visual_data = p.getVisualShapeData(robot_id)
    for link in visual_data:
        make_transparent(link)

def test_coalesce_and_reshape():
    # Step 1: Create a sample IKSolution
    dof = 6
    num_problems = 10
    num_retries = 5
    pos_tolerance = 1e-3
    rot_tolerance = 1e-3
    device = "cpu"

    # Initialize a sample IKSolution with random data
    original_solution = pk.IKSolution(dof=dof, num_problems=num_problems, num_retries=num_retries,
                                   pos_tolerance=pos_tolerance, rot_tolerance=rot_tolerance, device=device)
    
    # Fill the original_solution tensors with random data
    original_solution.solutions = torch.rand((num_problems, num_retries, dof), device=device)
    original_solution.remaining = torch.randint(0, 2, (num_problems,), dtype=torch.bool, device=device)
    original_solution.err_pos = torch.rand((num_problems, num_retries), device=device)
    original_solution.err_rot = torch.rand_like(original_solution.err_pos)
    original_solution.converged_pos = torch.randint(0, 2, (num_problems, num_retries), dtype=torch.bool, device=device)
    original_solution.converged_rot = torch.randint(0, 2, (num_problems, num_retries), dtype=torch.bool, device=device)
    original_solution.converged = torch.randint(0, 2, (num_problems, num_retries), dtype=torch.bool, device=device)
    original_solution.converged_pos_any = torch.randint(0, 2, (num_problems,), dtype=torch.bool, device=device)
    original_solution.converged_rot_any = torch.randint(0, 2, (num_problems,), dtype=torch.bool, device=device)
    original_solution.converged_any = torch.randint(0, 2, (num_problems,), dtype=torch.bool, device=device)

    # Step 2: Split the original IKSolution into smaller IKSolutions
    M = 3  # Number of splits
    split_solutions = original_solution.reshape_solutions(M)

    # Step 3: Coalesce the smaller IKSolutions back into one
    coalesced_solution = split_solutions[0]
    print(f"Length of Split Solutions: {len(split_solutions)}")
    for i in range(1,len(split_solutions)):
        coalesced_solution.coalesce_solutions(split_solutions[i])

    # Step 4: Verify that all the data matches the original
    assert torch.equal(original_solution.solutions, coalesced_solution.solutions), "Solutions tensor does not match!"
    assert torch.equal(original_solution.remaining, coalesced_solution.remaining), "Remaining tensor does not match!"
    assert torch.equal(original_solution.err_pos, coalesced_solution.err_pos), "Error position tensor does not match!"
    assert torch.equal(original_solution.err_rot, coalesced_solution.err_rot), "Error rotation tensor does not match!"
    assert torch.equal(original_solution.converged_pos, coalesced_solution.converged_pos), "Converged position does not match!"
    assert torch.equal(original_solution.converged_rot, coalesced_solution.converged_rot), "Converged rotation does not match!"
    assert torch.equal(original_solution.converged, coalesced_solution.converged), "Converged tensor does not match!"
    assert torch.equal(original_solution.converged_pos_any, coalesced_solution.converged_pos_any), "Converged position (any) does not match!"
    assert torch.equal(original_solution.converged_rot_any, coalesced_solution.converged_rot_any), "Converged rotation (any) does not match!"
    assert torch.equal(original_solution.converged_any, coalesced_solution.converged_any), "Converged (any) does not match!"

    print("Test passed: All tensors match after reshaping and coalescing.")


def create_test_chain(robot="kuka_iiwa", device="cpu"):
    if robot == "kuka_iiwa":
        urdf = "kuka_iiwa/model.urdf"
        search_path = pybullet_data.getDataPath()
        full_urdf = os.path.join(search_path, urdf)
        chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")
        chain = chain.to(device=device)
    elif robot == "widowx":
        urdf = "widowx/wx250s.urdf"
        full_urdf = urdf
        chain = pk.build_serial_chain_from_urdf(open(full_urdf, "rb").read(), "ee_gripper_link")
        chain = chain.to(device=device)
    elif robot == "fp3_franka_hand":
        urdf = "franka/fp3_franka_hand.urdf"
        full_urdf = urdf
        chain = pk.build_chain_from_urdf(open(full_urdf, mode="rb").read())
        chain = pk.SerialChain(chain, "fp3_hand_tcp", "base")
        chain = chain.to(device=device)
    else:
        raise NotImplementedError(f"Robot {robot} not implemented")
    return chain, urdf

def test_ik_in_place_no_err(robot="kuka_iiwa"):
    pytorch_seed.seed(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    chain, urdf = create_test_chain(robot=robot, device=device)
    # robot frame
    pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    rot = torch.tensor([0.0, 0.0, 0.0], device=device)
    rob_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

    # goal equal to current configuration
    lim = torch.tensor(chain.get_joint_limits(), device=device)
    cur_q = torch.rand(lim.shape[1], device=device) * (lim[1] - lim[0]) + lim[0]
    M = 1
    goal_q = cur_q.unsqueeze(0).repeat(M, 1)

    # get ee pose (in robot frame)
    goal_in_rob_frame_tf = chain.forward_kinematics(goal_q)

    # transform to world frame for visualization
    goal_tf = rob_tf.compose(goal_in_rob_frame_tf)
    goal = goal_tf.get_matrix()
    goal_pos = goal[..., :3, 3]
    goal_rot = pk.matrix_to_euler_angles(goal[..., :3, :3], "XYZ")

    ik = pk.PseudoInverseIK(chain, max_iterations=30, num_retries=10,
                            joint_limits=lim.T,
                            early_stopping_any_converged=True,
                            early_stopping_no_improvement="all",
                            retry_configs=cur_q.reshape(1, -1),
                            # line_search=pk.BacktrackingLineSearch(max_lr=0.2),
                            debug=False,
                            lr=0.2)

    # do IK
    sol = ik.solve(goal_in_rob_frame_tf)
    assert sol.converged.sum() == M
    assert torch.allclose(sol.solutions[0][0], cur_q)
    assert torch.allclose(sol.err_pos[0], torch.zeros(1, device=device), atol=1e-6)
    assert torch.allclose(sol.err_rot[0], torch.zeros(1, device=device), atol=1e-6)




def test_multiple_robot_ik_jacobian_follower(robot="kuka_iiwa", skip=False,seed=3):
    pytorch_seed.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_path = pybullet_data.getDataPath()
    chain, urdf = create_test_chain(robot=robot, device=device)
    

    # robot frame
    pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    rot = torch.tensor([0.0, 0.0, 0.0], device=device)
    rob_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

    # world frame goal
    M = 1000
    # generate random goal joint angles (so these are all achievable)
    # use the joint limits to generate random joint angles
    lim = torch.tensor(chain.get_joint_limits(), device=device)
    goal_q = torch.rand(M, lim.shape[1], device=device) * (lim[1] - lim[0]) + lim[0]

    # get ee pose (in robot frame)
    goal_in_rob_frame_tf = chain.forward_kinematics(goal_q)

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
    sol = ik.solve(goal_in_rob_frame_tf)

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
    print("IK took %f seconds" % (timer_end - timer_start))
    print("IK converged number: %d / %d" % (sol.converged.sum(), sol.converged.numel()))
    print("IK took %d iterations" % sol.iterations)
    print("IK solved %d / %d goals" % (sol.converged_any.sum(), M))

    # check that solving again produces the same solutions
    sol_again = ik.solve(goal_in_rob_frame_tf)
    assert torch.allclose(sol.solutions, sol_again.solutions)
    assert torch.allclose(sol.converged, sol_again.converged)

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
                r = goal_rot[this_selection]
                xyzw = pk.wxyz_to_xyzw(pk.matrix_to_quaternion(pk.euler_angles_to_matrix(r, "XYZ")))

                solutions = sol.solutions[this_selection, :, :]
                converged = sol.converged[this_selection, :]

                # print how many retries converged for this one
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
                        q = solutions[jj, ii, :]
                        for dof in range(q.shape[0]):
                            p.resetJointState(armId, dof, q[dof])
                    if skip:
                        input("Press enter to continue")
        except:
            print("error has occurred")
        if not skip:
            p.disconnect()    
        else:
            while True:
                p.stepSimulation()

def test_multiple_robot_ik_jacobian_follower_iterative_interpolation(robot="kuka_iiwa", skip=False, n=10, seed=3, delay=False):
    pytorch_seed.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_path = pybullet_data.getDataPath()
    chain, urdf = create_test_chain(robot=robot, device=device)

    # robot frame
    pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    rot = torch.tensor([0.0, 0.0, 0.0], device=device)
    rob_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

    # world frame goal
    M = 1000
    # generate random goal joint angles (so these are all achievable)
    # use the joint limits to generate random joint angles
    lim = torch.tensor(chain.get_joint_limits(), device=device)
    goal_q = torch.rand(M, lim.shape[1], device=device) * (lim[1] - lim[0]) + lim[0]

    # get ee pose (in robot frame)
    goal_in_rob_frame_tf = chain.forward_kinematics(goal_q)

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
    sol = ik.iterative_interpolation_solve(rob_tf, goal_in_rob_frame_tf, n)
    interpolated_tfs = pk.interpolate_poses(rob_tf, goal_in_rob_frame_tf, n)
    timer_end = timer()

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


    total_converged = 0
    total_iterations = 0
    total_converged_any = 0
    M_total = 0

    for s in sol:
        total_converged += s.converged.sum().item()  
        M_total += s.converged.numel() 
        total_iterations += s.iterations
        total_converged_any += s.converged_any.sum().item()
        

    print("IK took %f seconds" % (timer_end - timer_start))
    print("IK converged number: %d / %d" % (total_converged, M_total))
    print("IK took %d iterations" % total_iterations)
    print("IK solved %d / %d goals" % (total_converged_any, M))

    # check that solving again produces the same solutions
    sol_again = ik.iterative_interpolation_solve(rob_tf, goal_in_rob_frame_tf, n)
    assert torch.allclose(sol[-1].solutions, sol_again[-1].solutions)
    assert torch.allclose(sol[-1].converged, sol_again[-1].converged)

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
                            # print(step)
                            q = solutions[step][jj, ii, :]
                            for dof in range(q.shape[0]):
                                p.resetJointState(armId, dof, q[dof])
                            if delay:
                                time.sleep(0.05)
                        # for step in range(len(sol)):
                    if skip:
                        input("Press enter to continue")
        except:
            print("error has occurred")
        if not skip:
            p.disconnect()    
        else:
            while True:
                p.stepSimulation()
        

def test_multiple_robot_ik_jacobian_follower_parallel_interpolation(robot="kuka_iiwa", skip=False, n=10, seed=3, delay=False):
    pytorch_seed.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_path = pybullet_data.getDataPath()
    chain, urdf = create_test_chain(robot=robot, device=device)

    # robot frame
    pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    rot = torch.tensor([0.0, 0.0, 0.0], device=device)
    rob_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

    # world frame goal
    M = 1000
    # generate random goal joint angles (so these are all achievable)
    # use the joint limits to generate random joint angles
    lim = torch.tensor(chain.get_joint_limits(), device=device)
    goal_q = torch.rand(M, lim.shape[1], device=device) * (lim[1] - lim[0]) + lim[0]

    # get ee pose (in robot frame)
    goal_in_rob_frame_tf = chain.forward_kinematics(goal_q)

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
    sol = ik.parallel_interpolation_solve(rob_tf, goal_in_rob_frame_tf, n)
    interpolated_tfs = pk.interpolate_poses(rob_tf, goal_in_rob_frame_tf, n)

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

    for s in sol:
        total_converged += s.converged.sum().item()  
        M_total += s.converged.numel() 
        total_iterations += s.iterations
        total_converged_any += s.converged_any.sum().item()
        

    print("IK took %f seconds" % (timer_end - timer_start))
    print("IK converged number: %d / %d" % (total_converged, M_total))
    print("IK took %d iterations" % total_iterations)
    print("IK solved %d / %d goals" % (total_converged_any, M))

    # check that solving again produces the same solutions
    sol_again = ik.parallel_interpolation_solve(rob_tf, goal_in_rob_frame_tf, n)
    assert torch.allclose(sol[-1].solutions, sol_again[-1].solutions)
    assert torch.allclose(sol[-1].converged, sol_again[-1].converged)

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

def test_single_robot_ik_jacobian_follower(robot="kuka_iiwa", num_retries=10, max_iterations = 10, skip=False, seed=3, delay=False):
    pytorch_seed.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_path = pybullet_data.getDataPath()
    chain, urdf = create_test_chain(robot=robot, device=device)

    # robot frame
    pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    rot = torch.tensor([0.0, 0.0, 0.0], device=device)
    start_tf = pk.Transform3d(pos=pos, rot=rot, device=device)
    

    # world frame goal
    M = 1  # Only testing with a single goal
    lim = torch.tensor(chain.get_joint_limits(), device=device)
    goal_q = torch.rand(M, lim.shape[1], device=device) * (lim[1] - lim[0]) + lim[0]

    # get ee pose (in robot frame)
    goal_in_rob_frame_tf = chain.forward_kinematics(goal_q)

    # transform to world frame for visualization
    goal_tf = start_tf.compose(goal_in_rob_frame_tf)
    goal = goal_tf.get_matrix()
    goal_pos = goal[..., :3, 3]
    goal_rot = pk.matrix_to_euler_angles(goal[..., :3, :3], "XYZ")

    ik = pk.PseudoInverseIK(chain, max_iterations=max_iterations, num_retries=num_retries,
                            joint_limits=lim.T,
                            early_stopping_any_converged=True,
                            early_stopping_no_improvement="all",
                            debug=False,
                            lr=0.2)

    # do IK
    sol = ik.solve(goal_in_rob_frame_tf)

    print("IK converged number: %d / %d" % (sol.converged.sum(),sol.converged.numel()))
    print("IK took %d iterations" % sol.iterations)

    # visualization
    p.connect(p.GUI)
    p.setRealTimeSimulation(False)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(search_path)

    # Setup camera view
    yaw = 90
    pitch = -35
    dist = 1
    target = np.array([0, 0, 0])
    p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

    plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])

    # Load the robot in PyBullet
    m = start_tf.get_matrix()
    pos = m[0, :3, 3]
    rot = m[0, :3, :3]
    quat = pk.matrix_to_quaternion(rot)
    pos = pos.cpu().numpy()
    rot = pk.wxyz_to_xyzw(quat).cpu().numpy()

    armId = p.loadURDF(urdf, basePosition=pos, baseOrientation=rot, useFixedBase=True)

    # visualize goal pose (using a green cone)
    visId = p.createVisualShape(p.GEOM_MESH, fileName="meshes/cone.obj", meshScale=1.0,
                                rgbaColor=[0., 1., 0., 0.5])
    goal_marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visId)

    # Set goal marker position and orientation
    xyzw = pk.wxyz_to_xyzw(pk.matrix_to_quaternion(pk.euler_angles_to_matrix(goal_rot, "XYZ")))
    p.resetBasePositionAndOrientation(goal_marker, goal_pos[0].cpu().numpy(), xyzw[0].cpu().numpy())

    original_joint_states = []
    for dof in range(p.getNumJoints(armId)):
        original_joint_states.append(p.getJointState(armId, dof)[0])  # Save the joint position

    # Apply IK solution to the robot's joints
    solutions = sol.solutions[0, :, :]
    for ii in range(num_retries):
        # Reset to the original joint states
        for dof in range(len(original_joint_states)):
            p.resetJointState(armId, dof, original_joint_states[dof])
        # time.sleep(1)
        # Apply the IK solution
        q = solutions[ii, :]
        for dof in range(q.shape[0]):
            p.resetJointState(armId, dof, q[dof])
        
        # Compute the end-effector pose
        end_effector_tf = chain.forward_kinematics(q.unsqueeze(0))
        end_effector_tf = start_tf.compose(end_effector_tf)
        
        # Compute errors
        pos_error, rot_error = pk.compute_error(goal_tf, end_effector_tf)
        
        print(f"Displaying IK solution attempt {ii+1}/{num_retries}")
        try:
            print(f"Position Error: {pos_error.item():.2f} meters")
            print(f"Rotation Error: {rot_error.item():.2f} radians")
        except:
            print(f"Position Error: {pos_error[0].item():.2f} meters")
            print(f"Rotation Error: {rot_error[0].item():.2f} radians")
        
        # time.sleep(0.02)
        if skip:
            input("Press Enter to continue to the next solution")
    

    if not skip:
        p.disconnect()    
    else:
        while True:
            p.stepSimulation()


def test_single_robot_jacobian_follower_ik_iterative_interpolation(robot="kuka_iiwa", num_retries=10, max_iterations=10, skip=False, n=10, seed=3, delay=False):
    pytorch_seed.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_path = pybullet_data.getDataPath()
    chain, urdf = create_test_chain(robot=robot, device=device)
     
    # robot frame
    pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    rot = torch.tensor([0.0, 0.0, 0.0], device=device)
    start_tf = pk.Transform3d(pos=pos, rot=rot, device=device)
    

    # world frame goal
    M = 1  # Only testing with a single goal
    lim = torch.tensor(chain.get_joint_limits(), device=device)
    goal_q = torch.rand(M, lim.shape[1], device=device) * (lim[1] - lim[0]) + lim[0]

    # get ee pose (in robot frame)
    end_tf = chain.forward_kinematics(goal_q)
    
    print("Start TF:",start_tf)
    interpolated_tfs = pk.interpolate_poses(start_tf, end_tf, n)
    print("Interpolated TFs:")
    for tf in interpolated_tfs:
        print("\t",tf)
    print("End TF:",end_tf)

    all_tfs = interpolated_tfs + [end_tf]

   # Transform to world frame for visualization
    goal_tf = start_tf.compose(end_tf)
    goal = goal_tf.get_matrix()
    goal_pos = goal[..., :3, 3]
    goal_rot = pk.matrix_to_euler_angles(goal[..., :3, :3], "XYZ")

    # Initialize IK solver
    ik = pk.PseudoInverseIK(chain, max_iterations=max_iterations, num_retries=num_retries,
                            joint_limits=lim.T,
                            early_stopping_any_converged=True,
                            early_stopping_no_improvement="all",
                            debug=False,
                            lr=0.2)

    # Solve IK for all interpolated transforms (batch solution)
    sol = ik.iterative_interpolation_solve(start_tf, end_tf, n)
    # Initialize lists to store err_pos and err_rot tensors
    err_pos_list = []
    err_rot_list = []
    accumulated_iterations, converged_sum, converged_numel = 0,0,0

    # Iterate through each IKSolution and accumulate the tensors
    for s in sol:
        err_pos_list.append(s.err_pos)
        err_rot_list.append(s.err_rot)
        accumulated_iterations += s.iterations
        converged_sum += s.converged.sum()
        converged_numel += s.converged.numel()


    # Stack tensors along a new dimension to accumulate them
    accumulated_err_pos = torch.stack(err_pos_list)
    accumulated_err_rot = torch.stack(err_rot_list)
    
    # Compute the average position and rotation errors
    average_err_pos = torch.mean(accumulated_err_pos)
    average_err_rot = torch.mean(accumulated_err_rot)

    # Print the results
    print("\n\nAverage Position Error:", average_err_pos.item())
    print("Average Rotation Error:", average_err_rot.item())

    print("IK converged number: %d / %d" % (converged_sum,converged_numel))
    print("IK took %d iterations" % accumulated_iterations)

    # Visualization setup
    p.connect(p.GUI)
    p.setRealTimeSimulation(False)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(search_path)

    # Setup camera view
    yaw = 90
    pitch = -35
    dist = 1
    target = np.array([0, 0, 0])
    p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

    # Load plane
    plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])

    # Load the robot in PyBullet
    m = start_tf.get_matrix()
    pos = m[0, :3, 3]
    rot = m[0, :3, :3]
    quat = pk.matrix_to_quaternion(rot)
    pos = pos.cpu().numpy()
    rot = pk.wxyz_to_xyzw(quat).cpu().numpy()

    armId = p.loadURDF(urdf, basePosition=pos, baseOrientation=rot, useFixedBase=True)

    # Visualize goal pose (green cone)
    visId = p.createVisualShape(p.GEOM_MESH, fileName="meshes/cone.obj", meshScale=1.0,
                                rgbaColor=[0., 1., 0., 0.5])
    goal_marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visId)

    # Set goal marker position and orientation
    xyzw = pk.wxyz_to_xyzw(pk.matrix_to_quaternion(pk.euler_angles_to_matrix(goal_rot, "XYZ")))
    p.resetBasePositionAndOrientation(goal_marker, goal_pos[0].cpu().numpy(), xyzw[0].cpu().numpy())

    # Original joint states
    original_joint_states = []
    for dof in range(p.getNumJoints(armId)):
        original_joint_states.append(p.getJointState(armId, dof)[0])  # Save the joint position
    # print("Length: ",len(sol))
    # Apply IK solutions to the robot's joints and visualize each
    for ii in range(num_retries):

        # Reset to the original joint states
        for dof in range(len(original_joint_states)):
                p.resetJointState(armId, dof, original_joint_states[dof])
        for step in range(len(sol)):
            if delay:
                time.sleep(0.05)
            # Get IK solution for this transform
            # print(sol[step].solutions.shape)
            solutions = sol[step].solutions[0,:,:]           
            

            # Apply the IK solution
            q = solutions[ii, :]
            for dof in range(q.shape[0]):
                p.resetJointState(armId, dof, q[dof])

            # Compute the end-effector pose
            end_effector_tf = chain.forward_kinematics(q.unsqueeze(0))
            end_effector_tf = start_tf.compose(end_effector_tf)

            # Compute errors

            pos_error, rot_error = pk.compute_error(all_tfs[step], end_effector_tf)

            print(f"Step {step+1}/{len(all_tfs)} | IK solution attempt {ii+1}/{num_retries}")
            try:
                print(f"Position Error: {pos_error.item():.2f} meters")
                print(f"Rotation Error: {rot_error.item():.2f} radians")
            except:
                print(f"Position Error: {pos_error[0].item():.2f} meters")
                print(f"Rotation Error: {rot_error[0].item():.2f} radians")
            # write_to_csv(pos_error.item(), rot_error.item(), n, max_iterations, "interpolation.csv")
            # write_to_csv_with_step(pos_error.item(), rot_error.item(), n, max_iterations, step + 1, "steps.csv")
            # time.sleep(0.02)
            if skip:
                input("Press Enter to continue to the next step")
        if skip:
            input("Press Enter to continue to the next solution")
    if not skip:
        p.disconnect()    
    else:
        while True:
            p.stepSimulation()


def test_single_robot_jacobian_follower_ik_parallel_interpolation(robot="kuka_iiwa", num_retries=10, max_iterations=10, skip=False, n=10, seed=3, delay=False):
    pytorch_seed.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_path = pybullet_data.getDataPath()
    chain, urdf = create_test_chain(robot=robot, device=device)
    chain.print_tree()
     
    # robot frame
    pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    rot = torch.tensor([0.0, 0.0, 0.0], device=device)
    start_tf = pk.Transform3d(pos=pos, rot=rot, device=device)
    

    # world frame goal
    M = 1  # Only testing with a single goal
    lim = torch.tensor(chain.get_joint_limits(), device=device)
    goal_q = torch.rand(M, lim.shape[1], device=device) * (lim[1] - lim[0]) + lim[0]

    # get ee pose (in robot frame)
    end_tf = chain.forward_kinematics(goal_q)

    print("Start TF:",start_tf)
    interpolated_tfs = pk.interpolate_poses(start_tf, end_tf, n)
    print("Interpolated TFs:")
    for tf in interpolated_tfs:
        print("\t",tf)
    print("End TF:",end_tf)

    all_tfs = interpolated_tfs + [end_tf]

   # Transform to world frame for visualization
    goal_tf = start_tf.compose(end_tf)
    goal = goal_tf.get_matrix()
    goal_pos = goal[..., :3, 3]
    goal_rot = pk.matrix_to_euler_angles(goal[..., :3, :3], "XYZ")

    # Initialize IK solver
    ik = pk.PseudoInverseIK(chain, max_iterations=max_iterations, num_retries=num_retries,
                            joint_limits=lim.T,
                            early_stopping_any_converged=True,
                            early_stopping_no_improvement="all",
                            debug=False,
                            lr=0.2)

    err_pos_list = []
    err_rot_list = []
    accumulated_iterations, converged_sum, converged_numel = 0,0,0

    sol = ik.parallel_interpolation_solve(start_tf, end_tf, n)

    # Iterate through each IKSolution and accumulate the tensors
    for s in sol:
        err_pos_list.append(s.err_pos)
        err_rot_list.append(s.err_rot)
        accumulated_iterations += s.iterations
        converged_sum += s.converged.sum()
        converged_numel += s.converged.numel()


    # Stack tensors along a new dimension to accumulate them
    accumulated_err_pos = torch.stack(err_pos_list)
    accumulated_err_rot = torch.stack(err_rot_list)
    
    # Compute the average position and rotation errors
    average_err_pos = torch.mean(accumulated_err_pos)
    average_err_rot = torch.mean(accumulated_err_rot)

    # Print the results
    print("\n\nAverage Position Error:", average_err_pos.item())
    print("Average Rotation Error:", average_err_rot.item())

    print("IK converged number: %d / %d" % (converged_sum,converged_numel))
    print("IK took %d iterations" % accumulated_iterations)

    # Visualization setup
    p.connect(p.GUI)
    p.setRealTimeSimulation(False)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(search_path)
    # print("flag",search_path)

    # Setup camera view
    yaw = 90
    pitch = -35
    dist = 1
    target = np.array([0, 0, 0])
    p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

    # Load plane
    plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])

    # Load the robot in PyBullet
    m = start_tf.get_matrix()
    pos = m[0, :3, 3]
    rot = m[0, :3, :3]
    quat = pk.matrix_to_quaternion(rot)
    pos = pos.cpu().numpy()
    rot = pk.wxyz_to_xyzw(quat).cpu().numpy()

    armId = p.loadURDF(urdf, basePosition=pos, baseOrientation=rot, useFixedBase=True)

    # Visualize goal pose (green cone)
    visId = p.createVisualShape(p.GEOM_MESH, fileName="meshes/cone.obj", meshScale=1.0,
                                rgbaColor=[0., 1., 0., 0.5])
    goal_marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visId)

    # Set goal marker position and orientation
    xyzw = pk.wxyz_to_xyzw(pk.matrix_to_quaternion(pk.euler_angles_to_matrix(goal_rot, "XYZ")))
    p.resetBasePositionAndOrientation(goal_marker, goal_pos[0].cpu().numpy(), xyzw[0].cpu().numpy())

    # Original joint states
    original_joint_states = []
    for dof in range(p.getNumJoints(armId)):
        original_joint_states.append(p.getJointState(armId, dof)[0])  # Save the joint position
    # print("Length: ",len(sol))
    # Apply IK solutions to the robot's joints and visualize each
    
    for ii in range(num_retries):

        # Reset to the original joint states
        for dof in range(len(original_joint_states)):
                p.resetJointState(armId, dof, original_joint_states[dof])
        for step in range(len(sol)):
            if delay:
                time.sleep(0.05)
            # Get IK solution for this transform
            # print(sol[step].solutions.shape)
            solutions = sol[step].solutions[0,:,:]           
            
            # Apply the IK solution
            q = solutions[ii, :]
            for dof in range(q.shape[0]):
                p.resetJointState(armId, dof, q[dof])

            # Compute the end-effector pose
            end_effector_tf = chain.forward_kinematics(q.unsqueeze(0))
            end_effector_tf = start_tf.compose(end_effector_tf)

            # Compute errors

        

            pos_error, rot_error = pk.compute_error(all_tfs[step], end_effector_tf)

            print(f"Step {step+1}/{len(all_tfs)} | IK solution attempt {ii+1}/{num_retries}")
            try:
                print(f"Position Error: {pos_error.item():.2f} meters")
                print(f"Rotation Error: {rot_error.item():.2f} radians")
            except:
                print(f"Position Error: {pos_error[0].item():.2f} meters")
                print(f"Rotation Error: {rot_error[0].item():.2f} radians")
            # write_to_csv(pos_error.item(), rot_error.item(), n, max_iterations, "interpolation.csv")
            # write_to_csv_with_step(pos_error.item(), rot_error.item(), n, max_iterations, step + 1, "steps.csv")
            # time.sleep(0.02)
            if skip:
                input("Press Enter to continue to the next step")
        if skip:
            input("Press Enter to continue to the next solution")
    if not skip:
        p.disconnect()    
    else:
        while True:
            p.stepSimulation()




if __name__ == "__main__":
    print("Testing coalescing functions")
    test_coalesce_and_reshape()
    print("_____________________________________________________")
    print("Test in place no errors")
    test_ik_in_place_no_err(robot="widowx")
    print("_____________________________________________________")
    test_ik_in_place_no_err(robot="kuka_iiwa")
    # print("Testing kuka_iiwa IK")
    # print("_____________________________________________________")
    # test_multiple_robot_ik_jacobian_follower(robot="kuka_iiwa")
    # print("_____________________________________________________")
    # test_multiple_robot_ik_jacobian_follower_iterative_interpolation(robot="kuka_iiwa", n=10, seed=3)
    # print("_____________________________________________________")
    # test_multiple_robot_ik_jacobian_follower_parallel_interpolation(robot="kuka_iiwa", n=10, seed=3)
    # print("_____________________________________________________")
    # print("Testing widowx IK")
    # print("_____________________________________________________")
    # test_multiple_robot_ik_jacobian_follower(robot="widowx")
    # print("_____________________________________________________")
    # test_multiple_robot_ik_jacobian_follower_iterative_interpolation(robot="widowx", n=10, seed=3,delay=True)
    # print("_____________________________________________________")
    # test_multiple_robot_ik_jacobian_follower_parallel_interpolation(robot="widowx", n=10, seed=3,delay=True)
    # print("_____________________________________________________")
    print("Testing fp3 Franka Hand")
    # print("_____________________________________________________")
    # test_multiple_robot_ik_jacobian_follower(robot="fp3_franka_hand")
    # print("_____________________________________________________")
    # test_multiple_robot_ik_jacobian_follower_iterative_interpolation(robot="fp3_franka_hand", n=10, seed=3,delay=True)
    # print("_____________________________________________________")
    # test_multiple_robot_ik_jacobian_follower_parallel_interpolation(robot="fp3_franka_hand", n=10, seed=3,delay=True)
    # print("_____________________________________________________")
    mi=1000
    # test_single_robot_ik_jacobian_follower(robot="widowx", num_retries=10, max_iterations=mi)
    # print("_____________________________________________________")
    # test_single_robot_jacobian_follower_ik_iterative_interpolation(robot="widowx", num_retries=10, max_iterations=mi,delay=True)
    # print("_____________________________________________________")
    # test_single_robot_jacobian_follower_ik_parallel_interpolation(robot="widowx", num_retries=10, max_iterations=mi,skip=True)
    # print("_____________________________________________________")
    # test_single_robot_jacobian_follower_ik_parallel_interpolation(robot="fp3_franka_hand", num_retries=1, max_iterations=mi,skip=True)
    # print("_____________________________________________________")