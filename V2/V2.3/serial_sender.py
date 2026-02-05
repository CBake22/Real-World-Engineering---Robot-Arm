import serial
import time

arduino = serial.Serial('/dev/cu.usbmodem112201',9600, timeout=2)
print("Serial connection established.")
time.sleep(2)  # wait for the serial connection to initialize

def send_angles(joint_angles):
    if not joint_angles:
        print("Error: Empty joint angle list")
        return

    try:
        data_to_send = ",".join([f"{a:.3f}" for a in joint_angles]) + "\n"
        arduino.write(data_to_send.encode('utf-8'))
        print(f"Sent: {data_to_send.strip()}")
    except Exception as e:
        print(f"Communication error: {e}")