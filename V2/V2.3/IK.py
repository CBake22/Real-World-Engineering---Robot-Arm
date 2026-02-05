from serial_sender import send_angles, arduino
import numpy as np

def forward_kinematics(joints, DH_params, DOF, T0):
    joint_positions = np.zeros((DOF + 1, 3))
    T = T0.copy()
    for i in range(DOF):
        d, a, alpha = DH_params[i]
        theta = joints[i]
        T_i = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        T = T @ T_i
        joint_positions[i + 1] = T[:3, 3]
    return joint_positions

def compute_jacobian(joints, DH_params, DOF, T0, delta=1e-4):
    J = np.zeros((3, DOF))
    initial_positions = forward_kinematics(joints, DH_params, DOF, T0)
    for i in range(DOF):
        joints_delta = joints.copy()
        joints_delta[i] += delta
        new_positions = forward_kinematics(joints_delta, DH_params, DOF, T0)
        J[:, i] = (new_positions[-1] - initial_positions[-1]) / delta
    return J

def inverse_kinematics(target, joints_init, DH_params, DOF, T0,
                       max_iterations=1000, tolerance=1e-6, damping=0.01):
    joints = joints_init.copy()
    joint_positions = forward_kinematics(joints, DH_params, DOF, T0)
    joint_limits = [
        [-np.pi, np.pi],  # Base
        [0, np.pi],       # Shoulder
        [-np.pi, np.pi],  # Elbow
        [-np.pi, np.pi],  # Wrist pitch
        [-np.pi/2, np.pi/2]  # Wrist roll
    ]
    os = 0
    prev_error = float('inf')
    for iteration in range(max_iterations):
        error = target[:3, 3] - joint_positions[-1]
        error_norm = np.linalg.norm(error)
        if error_norm < tolerance:
            return joints, joint_positions, error_norm, True
        J = compute_jacobian(joints, DH_params, DOF, T0)
        J_T = J.T
        delta_joints = J_T @ np.linalg.inv(J @ J_T + damping**2 * np.eye(3)) @ error
        joints += delta_joints
        for i in range(DOF):
            joints[i] = np.clip(joints[i], joint_limits[i][0], joint_limits[i][1])
        joint_positions = forward_kinematics(joints, DH_params, DOF, T0)
        if error_norm >= prev_error - 1e-3:
            os += 1
        else:
            os = 0
        if os >= 5:
            break
        prev_error = error_norm
    return joints, joint_positions, error_norm, False

if __name__ == "__main__":
    DOF = 5
    T0 = np.eye(4)

    DH_params = [
        [12, 0, np.radians(90)],
        [0, 14.5, np.radians(0)],
        [10.5, 0, np.radians(90)],
        [18, 0, np.radians(-90)],
        [0, 2.5, np.radians(0)]
    ]
    joints_init = np.radians([0.0, 90.0, -90.0, 0.0, 0.0])

    target = np.eye(4)
    target[:3, 3] = [3, 2, 0]

    joints_rad, joint_positions, error, converged = inverse_kinematics(target, joints_init, DH_params, DOF, T0)
    joints_deg = np.degrees(joints_rad)
    print("Joint angles [deg]:", joints_deg)

    # Only send the first joint angle (Motor 1) to Arduino
    send_angles([joints_deg[0]])

    if arduino is not None:
        arduino.close()
        print("Serial connection closed.")
