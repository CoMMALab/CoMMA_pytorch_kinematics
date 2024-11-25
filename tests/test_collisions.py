import pybullet as p
import pybullet_data
import time
import numpy as np

def visualize_robot_poses(data, urdf, n=10, visualize=True, delay=True):
    """
    Visualize pre-computed joint configurations in PyBullet with collision detection.
    
    Args:
        data: List of dictionaries containing pre-computed joint angles for robot poses.
        urdf: Path to the robot URDF file.
        n: Number of poses to visualize.
        visualize: Whether to visualize the robot in PyBullet.
        delay: Whether to add a delay between visualizing poses.
    """
    if visualize:
        # Initialize PyBullet in GUI mode
        p.connect(p.GUI)
        p.setRealTimeSimulation(False)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set up camera for better visualization
        yaw = 90
        pitch = -35
        dist = 2
        target = np.array([0, 0, 0])
        p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

        # Load ground plane
        plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])

        # Load robot model
        robot_id = p.loadURDF(urdf, basePosition=[0, 0, 0], useFixedBase=True)

        # Add a sphere nearby
        sphere_radius = 0.2
        sphere_position = [0.5, 0.5, sphere_radius]  # Position of the sphere
        sphere_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)
        sphere_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=[1, 0, 0, 1])
        sphere_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=sphere_collision_shape,
            baseVisualShapeIndex=sphere_visual_shape,
            basePosition=sphere_position
        )

      

        # Keep the visualization running until manually exited
        while True:
            contact_points_with_sphere = p.getContactPoints(bodyA=robot_id, bodyB=sphere_id)
            if contact_points_with_sphere:
                print(f"Collision detected with the sphere at pose!")
                for contact in contact_points_with_sphere:
                    print(f"Contact: Robot link {contact[3]} with sphere at position {contact[5]}")
            p.stepSimulation()

if __name__ == "__main__":
    search_path = pybullet_data.getDataPath()

    # Define URDF and dataset
    urdf = "franka/fp3_franka_hand.urdf"
    data = [
        {'joint_angles': [0.0, -0.5, 0.5, -1.0, 0.0, 0.8, 0.0]},  # Example pose 1
        {'joint_angles': [0.2, -0.3, 0.7, -0.9, 0.1, 0.5, 0.0]},  # Example pose 2
        # Add more poses here...
    ]

    # Visualize pre-computed joint configurations
    visualize_robot_poses(data=data, urdf=urdf, n=10)
