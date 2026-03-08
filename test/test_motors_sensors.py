import time
import board
import busio

from code.sensor import Sensor
from code.motor import RobotArm

# 1. Shared I2C Bus (The "Highway")
i2c = busio.I2C(board.SCL, board.SDA)

# 2. Initialize our modules
sensor = Sensor(i2c)
arm    = RobotArm(i2c)

print("System Initialized. Starting loop...")

while True:
    # Get data from sensor module
    raw_volt = sensor.get_voltage()
    
    # Logic: Use the sensor module's mapping function

    # --- USE DATA IN OTHER PLACES ---
    # Example: Only move if voltage is high enough, otherwise log an error
    if raw_volt > 0.1:
        target_angle = int(sensor.get_mapped_value(0, 270))
        arm.move_servo(0, target_angle)
    else:
        print("Sensor disconnected or under-voltage!")

    print(f"Voltage: {raw_volt:.2f}V | Target: {target_angle:.1f}°", end="\r")
    
    time.sleep(0.05)