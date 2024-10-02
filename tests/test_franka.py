


# serial_chain = pk.SerialChain(chain, "fp3_hand_tcp", "base")
# # urdf = "kuka_iiwa/model.urdf"
# # search_path = "CoMMA_pytorch_kinematics/"
# # full_urdf = os.path.join(search_path, urdf)
# # serial_chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")
# serial_chain = serial_chain.to(device=device)
# print(serial_chain)
# # Define Starting Position
# pos = torch.tensor([0.0, 0.0, 0.0], device=device)
# rot = torch.tensor([0.0, 0.0, 0.0], device=device)
# start_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

# seed=3
# pytorch_seed.seed(seed)
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# search_path = pybullet_data.getDataPath()
# max_iterations = 100
# num_retries = 10
# n = 10
# skip=False
# delay = None

# # robot frame
# pos = torch.tensor([0.0, 0.0, 0.0], device=device)
# rot = torch.tensor([0.0, 0.0, 0.0], device=device)
# start_tf = pk.Transform3d(pos=pos, rot=rot, device=device)


# # world frame goal
# M = 1  # Only testing with a single goal
# lim = torch.tensor(chain.get_joint_limits(), device=device)
# goal_q = torch.rand(M, lim.shape[1], device=device) * (lim[1] - lim[0]) + lim[0]

# # get ee pose (in robot frame)
# end_tf = chain.forward_kinematics(goal_q)["fp3_hand_tcp"]
# print("End TF:",end_tf)

# print("Start TF:",start_tf)
# interpolated_tfs = pk.interpolate_poses(start_tf, end_tf, n)
# print("Interpolated TFs:")
# for tf in interpolated_tfs:
#     print("\t",tf)
# print("End TF:",end_tf)

# all_tfs = interpolated_tfs + [end_tf]

# # Transform to world frame for visualization
# goal_tf = start_tf.compose(end_tf)
# goal = goal_tf.get_matrix()
# goal_pos = goal[..., :3, 3]
# goal_rot = pk.matrix_to_euler_angles(goal[..., :3, :3], "XYZ")


# # Initialize IK solver
# ik = pk.PseudoInverseIK(chain, max_iterations=max_iterations, num_retries=num_retries,
#                         joint_limits=lim.T,
#                         early_stopping_any_converged=True,
#                         early_stopping_no_improvement="all",
#                         debug=False,
#                         lr=0.2)

# # Solve IK for all interpolated transforms (batch solution)
# sol = ik.parallel_solve(start_tf, end_tf, n)


# # print("IK converged number: %d / %d" % (sol.converged.sum(), sol.converged.numel()))
# # print("IK took %d iterations" % sol.iterations)

# # Visualization setup
# p.connect(p.GUI)
# p.setRealTimeSimulation(False)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# p.setAdditionalSearchPath(search_path)

# # Setup camera view
# yaw = 90
# pitch = -35
# dist = 1
# target = np.array([0, 0, 0])
# p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

# # Load plane
# plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
# p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])

# # Load the robot in PyBullet
# m = start_tf.get_matrix()
# pos = m[0, :3, 3]
# rot = m[0, :3, :3]
# quat = pk.matrix_to_quaternion(rot)
# pos = pos.cpu().numpy()
# rot = pk.wxyz_to_xyzw(quat).cpu().numpy()

# armId = p.loadURDF(urdf, basePosition=pos, baseOrientation=rot, useFixedBase=True)

# # Visualize goal pose (green cone)
# visId = p.createVisualShape(p.GEOM_MESH, fileName="meshes/cone.obj", meshScale=1.0,
#                             rgbaColor=[0., 1., 0., 0.5])
# goal_marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visId)

# # Set goal marker position and orientation
# xyzw = pk.wxyz_to_xyzw(pk.matrix_to_quaternion(pk.euler_angles_to_matrix(goal_rot, "XYZ")))
# p.resetBasePositionAndOrientation(goal_marker, goal_pos[0].cpu().numpy(), xyzw[0].cpu().numpy())

# # Original joint states
# original_joint_states = []
# for dof in range(p.getNumJoints(armId)):
#     original_joint_states.append(p.getJointState(armId, dof)[0])  # Save the joint position
# # print("Length: ",len(sol))
# # Apply IK solutions to the robot's joints and visualize each
# for ii in range(num_retries):

#     # Reset to the original joint states
#     for dof in range(len(original_joint_states)):
#             p.resetJointState(armId, dof, original_joint_states[dof])
#     for step in range(len(sol)):
#         if delay:
#             time.sleep(0.05)
#         # Get IK solution for this transform
#         # print(sol[step].solutions.shape)
#         solutions = sol[step].solutions[0,:,:]           
        

#         # Apply the IK solution
#         q = solutions[ii, :]
#         for dof in range(q.shape[0]):
#             p.resetJointState(armId, dof, q[dof])

#         # Compute the end-effector pose
#         end_effector_tf = chain.forward_kinematics(q.unsqueeze(0))
#         end_effector_tf = start_tf.compose(end_effector_tf)

#         # Compute errors

#         pos_error, rot_error = pk.compute_error(all_tfs[step], end_effector_tf)

#         print(f"Step {step+1}/{len(all_tfs)} | IK solution attempt {ii+1}/{num_retries}")
#         try:
#             print(f"Position Error: {pos_error.item():.2f} meters")
#             print(f"Rotation Error: {rot_error.item():.2f} radians")
#         except:
#             print(f"Position Error: {pos_error[0].item():.2f} meters")
#             print(f"Rotation Error: {rot_error[0].item():.2f} radians")
#         # write_to_csv(pos_error.item(), rot_error.item(), n, max_iterations, "interpolation.csv")
#         # write_to_csv_with_step(pos_error.item(), rot_error.item(), n, max_iterations, step + 1, "steps.csv")
#         # time.sleep(0.02)
#         if skip:
#             input("Press Enter to continue to the next step")
#     if skip:
#         input("Press Enter to continue to the next solution")
# if not skip:
#     p.disconnect()    
# else:
#     while True:
#         p.stepSimulation()