import sys
import os
import time
import random
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from robot import RobotArm

def adapter_angle_to_width(angle: float, max_angle: float = 250.0, min_angle: float = 0.0, max_width: float = 11.20, min_width: float = 6.2) -> float:
    """
    Converts the FINGER servo angle to a physical width (mm).
    MAX (250°) -> 11.2cm
    MIN (0°)   -> 6.2cm
    """
    angle = max(min_angle, min(max_angle, angle))
    width = min_width + (angle - min_angle) * ((max_width - min_width) / (max_angle - min_angle))
    return width

def read_simulated_sensors(current_angle: float) -> tuple[float, float]:
    """
    Simulates two flex sensors.
    Baseline starts safely at 2.6V to prevent premature lower-bound triggers.
    Sensor 1 crosses 3.4V around 125°.
    Sensor 2 crosses 3.4V around 115°.
    """
    if current_angle > 135.0:
        s1_base = 2.6
    else:
        s1_base = 2.6 + (135.0 - current_angle) * 0.1

    if current_angle > 125.0:
        s2_base = 2.6
    else:
        s2_base = 2.6 + (125.0 - current_angle) * 0.1

    s1 = s1_base + random.uniform(0.01, 0.05)
    s2 = s2_base + random.uniform(0.01, 0.05)

    return s1, s2

GRIPPER_F = 15
GRIPPER_R = 1

def scan_sequence(arm: RobotArm, slice: int) -> list[tuple[float, float, str]]:
    home_angle = arm.HOME_POSITION[GRIPPER_F]
    closed_angle = 0.0

    scan_results: list[tuple[float, float, str]] = []

    print("Initializing gripper FINGERS to home position...")
    arm.move_smooth(GRIPPER_F, home_angle)
    time.sleep(0.5)

    for i in range(slice):
        gripper_r_target_angle = (i / slice) * 180.0

        print(f"\n--- Sequence {i+1}/{slice} ---")
        print(f"Rotating Gripper ROTOR to {gripper_r_target_angle:.1f}°")
        arm.move_smooth(GRIPPER_R, gripper_r_target_angle)
        time.sleep(0.5)

        print("Closing Gripper FINGER (Channel 0 or 15) and monitoring sensors...")
        current_gripper_f_angle = home_angle

        normal_speed_step = 2.0
        slow_speed_step = 0.5
        current_step = normal_speed_step

        s1_trigger_angle = None
        s2_trigger_angle = None

        while current_gripper_f_angle > closed_angle:
            v1, v2 = read_simulated_sensors(current_gripper_f_angle)

            s1_out_of_bounds = not (2.5 <= v1 <= 3.4)
            s2_out_of_bounds = not (2.5 <= v2 <= 3.4)

            if s1_out_of_bounds and s1_trigger_angle is None:
                s1_trigger_angle = current_gripper_f_angle
                print(f"--> Sensor 1 Triggered at {current_gripper_f_angle:.1f}° (Value: {v1:.2f}V)")

            if s2_out_of_bounds and s2_trigger_angle is None:
                s2_trigger_angle = current_gripper_f_angle
                print(f"--> Sensor 2 Triggered at {current_gripper_f_angle:.1f}° (Value: {v2:.2f}V)")

            if s1_trigger_angle is not None and s2_trigger_angle is not None:
                print("--> STOP: Both sensors triggered. Stopping motor.")
                break

            elif s1_trigger_angle is not None or s2_trigger_angle is not None:
                if current_step != slow_speed_step:
                    print("--> SLOWING DOWN: Single contact made, proceeding cautiously.")
                    current_step = slow_speed_step
            else:
                current_step = normal_speed_step

            current_gripper_f_angle -= current_step

            arm.kit.servo[GRIPPER_F].angle = current_gripper_f_angle # type: ignore
            arm.current_angles[GRIPPER_F] = current_gripper_f_angle
            time.sleep(0.02)

        print(f"Final Stopped Angle of FINGERS: {current_gripper_f_angle:.1f}°")

        ch1_rounded = round(gripper_r_target_angle, 2)

        if (s1_trigger_angle is not None and
            s2_trigger_angle is not None and
            s1_trigger_angle == s2_trigger_angle):
            width = adapter_angle_to_width(s1_trigger_angle)
            scan_results.append((ch1_rounded, round(width, 2), "both"))
        else:
            if s1_trigger_angle is not None:
                width1 = adapter_angle_to_width(s1_trigger_angle)
                scan_results.append((ch1_rounded, round(width1, 2), "left"))

            if s2_trigger_angle is not None:
                width2 = adapter_angle_to_width(s2_trigger_angle)
                scan_results.append((ch1_rounded, round(width2, 2), "right"))

        print("Opening FINGERS to home position...")
        arm.move_smooth(GRIPPER_F, home_angle)
        time.sleep(0.5)

    print("\nScanning sequence completed.")
    return scan_results

if __name__ == "__main__":
    arm = RobotArm()
    final_data = scan_sequence(arm, slice=3)

    print("\n--- Final Gathered Data (Ready for DarkBot) ---")
    print("measurements = [")
    for angle, width, side in final_data:
        print(f"    ({angle:5.1f}, {width:5.2f}, \"{side}\"),")
    print("]")