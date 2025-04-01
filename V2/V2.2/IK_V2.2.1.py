# Author: Carson Baker
# Date Last Modified: 3/25/2025
# Description: Calculates joint angles for built-in target position (single) and communicates with Arduino for motor control
# Input: DH parameters, target position/rotation, joint bounds position/rotation
# Output: Joint angles to reach target position if converged (to Arduino via serial)
# Version changes: Updated IK program

# Imports and such
import serial
import time
from numpy import *

# Open serial communication with Arduino on COM5
try:
    arduino = serial.Serial('COM5', 9600, timeout=1)
    print("Serial connection established.")
    time.sleep(2)  # Give time for Arduino to initialize
except serial.SerialException as e:
    print(f"Error: {e}")
    exit()

# Function to generate DH transformation matrix using NumPy
def DH_trans_matrix(d, theta, a, alpha):
    return array([
        [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
        [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
        [0, sin(alpha), cos(alpha), d],
        [0, 0, 0, 1]
    ])

# Forward Kinematics
def forward_kinematics(joints, DH_params, DOF, T0):
    positions = [array([0, 0, 0])]
    trans = T0
    for i in range(DOF):
        d, a, alpha = DH_params[i]
        theta = joints[i]
        trans = trans @ DH_trans_matrix(d, theta, a, alpha)
        positions.append(trans[:3, 3])
    return array(positions)

# Inverse Kinematics
def inverse_kinematics(joints_init, target, DH_params, joint_limits, joint_position_limits, DOF, T0, iterations=100, tolerance=0.01):
    joints = joints_init.copy()
    target_pos = target[:3, 3]

    converged = False
    prev_error = float('inf')

    os = 0
    for i in range(iterations):
        current_pos = forward_kinematics(joints, DH_params, DOF, T0)[-1]
        position_error = target_pos - current_pos
        error_norm = linalg.norm(position_error)

        if error_norm < tolerance:
            converged = True
            print(f"Converged in {i + 1} iterations with error: {error_norm:.6f}")
            break

        if error_norm >= prev_error:
            os += 1
            if os > 3:
                print("Warning: Potential divergence or oscillation detected.")
                break
        prev_error = error_norm

        jacobian = zeros((3, DOF))
        delta = 1e-6
        for j in range(DOF):
            joints_delta = joints.copy()
            joints_delta[j] += delta
            pos_delta = forward_kinematics(joints_delta, DH_params, DOF, T0)[-1]
            jacobian[:, j] = (pos_delta - current_pos) / delta

        joint_update = linalg.pinv(jacobian).dot(position_error)
        joints += joint_update

        for j in range(DOF):
            joints[j] = clip(joints[j], joint_limits[j][0], joint_limits[j][1])

        joint_positions = forward_kinematics(joints, DH_params, DOF, T0)
        for j in range(DOF):
            joint_positions[j, 2] = max(joint_positions[j, 2], joint_position_limits[j][2][0])

    final_position = forward_kinematics(joints, DH_params, DOF, T0)[-1]
    final_error = linalg.norm(target_pos - final_position)

    if not converged:
        print(f"Failed to converge within {iterations} iterations. Final error: {final_error:.6f}")
        raise ValueError("Inverse kinematics did not converge.")

    return joints, forward_kinematics(joints, DH_params, DOF, T0), final_error, converged

# Convert Euler angles to rotational matrix
def euler_to_rotation_matrix(pitch, yaw, roll):
    Rx = array([
        [1, 0, 0],
        [0, cos(pitch), -sin(pitch)],
        [0, sin(pitch), cos(pitch)]
    ])
    Ry = array([
        [cos(yaw), 0, sin(yaw)],
        [0, 1, 0],
        [-sin(yaw), 0, cos(yaw)]
    ])
    Rz = array([
        [cos(roll), -sin(roll), 0],
        [sin(roll), cos(roll), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

# Function to send joint angles to Arduino
def send_floats_to_arduino(joint_angles):
    try:
        if not joint_angles:
            print("Error: The list of joint angles is empty")
            return

        data_to_send = ",".join(map(str, joint_angles)) + "\n"
        arduino.write(data_to_send.encode())
        print(f"Sent to Arduino: {data_to_send.strip()}")

        time.sleep(2)
        response = arduino.readline().decode().strip()

        if response:
            print(f"Response from Arduino: {response}")
        else:
            print("No response received from Arduino.")
    except Exception as e:
        print(f"Error during communication: {e}")

# Main function
def main():
    # Define DH parameters
    d1, d2, d3, d4, d5 = 12, 0, 10.5, 18, 0
    a1, a2, a3, a4, a5 = 0, 14.5, 0, 0, 2.5
    alpha1, alpha2, alpha3, alpha4, alpha5 = radians(0), radians(90), radians(0), radians(90), radians(-90)

    DH_params = [[d1, a1, alpha1], [d2, a2, alpha2], [d3, a3, alpha3], [d4, a4, alpha4], [d5, a5, alpha5]]

    joints_init = array([0, pi/2, -pi/2, 0, -pi/2])

    x_tar, y_tar, z_tar = 5.25, -9.0932, 6.0

    rotation_tar = euler_to_rotation_matrix(0, 0, 0)

    target = identity(4)
    target[:3, :3] = rotation_tar
    target[:3, 3] = [x_tar, y_tar, z_tar]

    try:
        joints_solution, _, _, _ = inverse_kinematics(joints_init, target, DH_params, [(-pi, pi)] * 5, [(-inf, inf)] * 5, 5, identity(4))
        send_floats_to_arduino(joints_solution * 180 / pi)

    except ValueError as e:
        print(f"Error in IK solution: {e}")

    arduino.close()
    print("Serial connection closed.")

if __name__ == "__main__":
    main()
