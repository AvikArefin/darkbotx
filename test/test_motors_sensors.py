import time
import board
import busio

from code.sensor import Sensor
from code.motor import RobotArm

i2c = busio.I2C(board.SCL, board.SDA)

sensor = Sensor(i2c, v_min=1.5, v_max=2.5)
arm    = RobotArm(i2c)

EMA_ALPHA = 0.1
# EMA_ALPHA = 0.1 at your 50ms loop = roughly a 500ms smoothing window. Tune it:

# 0.05 → very smooth, ~1s lag (good for slow deliberate motion)
# 0.15 → moderate, ~300ms lag
# 0.3 → light smoothing, reacts quickly

ema_angle = None  # Smooth the mapped angle, not the voltage

sensor1 = 0

print("System Initialized. Starting loop...")

while True:
    raw_volt = sensor.get_voltage(0)

    if raw_volt > 0.1:
        raw_angle = sensor.get_mapped_value(0, 0, 270)

        # Apply EMA to the angle output
        if ema_angle is None:
            ema_angle = raw_angle
        else:
            ema_angle = EMA_ALPHA * raw_angle + (1 - EMA_ALPHA) * ema_angle

        sensor1  = int(ema_angle)
        feedback = sensor.get_voltage(3)
        arm.move_servo(0, sensor1)
    else:
        ema_angle = None
        print("Sensor disconnected or under-voltage!")

    print(f"Voltage: {raw_volt:.2f}V | Target: {sensor1}° | Feedback: {feedback}", end="\r")

    time.sleep(0.05)


