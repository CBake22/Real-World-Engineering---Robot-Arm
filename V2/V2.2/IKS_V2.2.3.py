# Inverse Kinematic Simulation Program, V2.2.3
# Given robot parameters and a target end effector position, find angles for each joint
# Author(s): Carson Baker
# Last Updated: 4/1/2025

import numpy as np
import matplotlib.pyplot as plt

# Forward kinematics function
def forward_kinematics(joints, DH_params, DOF, T0):
    joint_positions = np.zeros((DOF + 1, 3))
    T = T0.copy()

    for i in range(DOF):
        d, a, alpha = DH_params[i]
        theta = joints[i]

        # Transformation matrix for each joint
        T_i = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

        # Update the total transformation
        T = T @ T_i

        # Store the position of each joint
        joint_positions[i + 1] = T[:3, 3]

    return joint_positions

# Numerical Jacobian calculation
def compute_jacobian(joints, DH_params, DOF, T0, delta=1e-4):
    J = np.zeros((3, DOF))
    initial_positions = forward_kinematics(joints, DH_params, DOF, T0)

    for i in range(DOF):
        joints_delta = joints.copy()
        joints_delta[i] += delta

        # Perturb joint and get new positions
        new_positions = forward_kinematics(joints_delta, DH_params, DOF, T0)

        # Compute the position difference
        J[:, i] = (new_positions[-1] - initial_positions[-1]) / delta

    return J

# Damped Least Squares Inverse Kinematics
def inverse_kinematics(target, joints_init, DH_params, DOF, T0, max_iterations=1000, tolerance=1e-6, damping=0.01):
    joints = joints_init.copy()
    joint_positions = forward_kinematics(joints, DH_params, DOF, T0)

    # Define joint limits
    joint_limits = [
        [-np.pi, np.pi],        # Base rotation
        [0, np.pi],        # Shoulder
        [-np.pi, np.pi],        # Elbow
        [-np.pi, np.pi],        # Wrist pitch
        [-np.pi/2, np.pi/2]         # Wrist roll
    ]

    os = 0  # Oscillation counter
    prev_error = float('inf')

    for iteration in range(max_iterations):
        # Compute the error vector (position only)
        error = target[:3, 3] - joint_positions[-1]
        error_norm = np.linalg.norm(error)

        # Check for convergence
        if error_norm < tolerance:
            print(f"Converged in {iteration} iterations. Final error: {error_norm:.6f}")
            return joints, joint_positions, error_norm, True

        # Compute the Jacobian
        J = compute_jacobian(joints, DH_params, DOF, T0)

        # Damped Least Squares solution
        J_T = J.T
        delta_joints = J_T @ np.linalg.inv(J @ J_T + damping**2 * np.eye(3)) @ error

        # Update joint values and enforce joint limits
        joints += delta_joints
        for i in range(DOF):
            joints[i] = np.clip(joints[i], joint_limits[i][0], joint_limits[i][1])

        # Update joint positions
        joint_positions = forward_kinematics(joints, DH_params, DOF, T0)

        # Oscillation detection
        if error_norm >= prev_error - 1e-3:
            os += 1
        else:
            os = 0

        if os >= 5:
            print("Warning: Potential divergence or oscillation detected.")
            break

        prev_error = error_norm

    print(f"IK failed to converge after {max_iterations} iterations. Final error: {error_norm:.6f}")
    return joints, joint_positions, error_norm, False

# Utility function to ensure equal axis scaling in 3D plot
def set_axes_equal(ax):
    ranges = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
    midpoints = np.mean(ranges, axis=1)
    max_range = max(abs(r[1] - r[0]) for r in ranges)
    for center, axis in zip(midpoints, [ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        axis(center - max_range / 2, center + max_range / 2)

# Visualization function
def plot_robot(joint_positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot links
    ax.plot(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], 'bo-', linewidth=2)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Ensure equal axis scaling
    set_axes_equal(ax)

    plt.show()

# Example usage
if __name__ == "__main__":
    DOF = 5
    T0 = np.eye(4)

    # Order: d, a, alpha
    d1 = 12
    d2 = 0
    d3 = 10.5
    d4 = 18
    d5 = 0

    a1 = 0
    a2 = 14.5
    a3 = 0
    a4 = 0
    a5 = 2.5

    alpha1 = np.radians( 90 )
    alpha2 = np.radians( 0 )
    alpha3 = np.radians( 90 )
    alpha4 = np.radians( -90 )
    alpha5 = np.radians( 0 )

    DH_params = [
        [d1, a1, alpha1],
        [d2, a2, alpha2],
        [d3, a3, alpha3],
        [d4, a4, alpha4],
        [d5, a5, alpha5]
    ]
    # Initial joint values
    joints_init = np.radians(np.array([0.0, 90.0, -90.0, 0.0, 0.0]))

    # Target position
    target = np.eye(4)
    target[:3, 3] = [8.266, -8.038, 43.343]

    # Run inverse kinematics
    joints_rad, joint_positions, error, converged = inverse_kinematics(target, joints_init, DH_params, DOF, T0)
    joints_deg = np.degrees(joints_rad)

    print("Joint angles [deg]:", joints_deg)

    # Plot the robot configuration
    plot_robot(joint_positions)
