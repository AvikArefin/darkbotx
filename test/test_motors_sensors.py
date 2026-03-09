import time
import board
import busio

from code.sensor import Sensor
from code.motor import RobotArm

# 1. Shared I2C Bus (The "Highway")
i2c = busio.I2C(board.SCL, board.SDA)

# 2. Initialize our modules
sensor = Sensor(i2c, v_min=1.5, v_max=2.5)
arm    = RobotArm(i2c)

print("System Initialized. Starting loop...")

while True:
    # Get data from sensor module
    raw_volt = sensor.get_voltage(0)
    
    # Logic: Use the sensor module's mapping function

    # --- USE DATA IN OTHER PLACES ---
    # Example: Only move if voltage is high enough, otherwise log an error
    if raw_volt > 0.1:
        sensor1 = int(sensor.get_mapped_value(0, 0, 270))
        feedback = sensor.get_voltage(3)
        # arm.move_servo(0, sensor1)
    else:
        print("Sensor disconnected or under-voltage!")

    print(f"Voltage: {raw_volt:.2f}V | Target: {sensor1:.1f}° | Feedback: {feedback}", end="\r")
    
    time.sleep(0.05)