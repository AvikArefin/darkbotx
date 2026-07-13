import time

from robot import RobotArm

arm = RobotArm()

def get_mapped_value(voltage: float, v_min: float, v_max: float, out_min: float, out_max: float) -> float:
    if voltage <= v_min:
        return out_min
    if voltage >= v_max:
        return out_max
    return out_min + (voltage - v_min) * (out_max - out_min) / (v_max - v_min)

EMA_ALPHA = 0.1
# EMA_ALPHA = 0.1 at your 50ms loop = roughly a 500ms smoothing window. Tune it:

# 0.05 → very smooth, ~1s lag (good for slow deliberate motion)
# 0.15 → moderate, ~300ms lag
# 0.3 → light smoothing, reacts quickly

ema_angle: float | None = None
sensor1 = 0
feedback = 0.0

print("System Initialized. Starting loop...")

while True:
    raw_volt = arm.sensor.get_voltage(0)

    if raw_volt > 0.1:
        raw_angle = get_mapped_value(raw_volt, 1.5, 2.5, 0.0, 270.0)

        # Apply EMA to the angle output
        if ema_angle is None:
            ema_angle = raw_angle
        else:
            ema_angle = EMA_ALPHA * raw_angle + (1 - EMA_ALPHA) * ema_angle

        sensor1  = int(ema_angle)
        feedback = arm.sensor.get_voltage(3)
        arm.move_smooth(0, sensor1)
    else:
        ema_angle = None
        print("Sensor disconnected or under-voltage!")

    print(f"Voltage: {raw_volt:.2f}V | Target: {sensor1}° | Feedback: {feedback}", end="\r")

    time.sleep(0.05)
