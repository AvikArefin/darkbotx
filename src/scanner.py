import time
import random

from robot import RobotArm

def adapter_angle_to_width(angle: float, max_angle: float = 250.0, min_angle: float = 0.0, max_width: float = 11.20, min_width: float = 6.2) -> float:
    """
    Converts the FINGER servo angle to a physical width (mm).
    MAX (250°) -> 11.2cm
    MIN (0°)   -> 6.2cm
    """
    # Ensure angle is within bounds
    angle = max(min_angle, min(max_angle, angle))
    
    # Linear interpolation
    width = min_width + (angle - min_angle) * ((max_width - min_width) / (max_angle - min_angle))
    return width

def read_simulated_sensors(current_angle: float) -> tuple[float, float]:
    """
    Simulates two flex sensors.
    Baseline starts safely at 2.6V to prevent premature lower-bound triggers.
    Sensor 1 crosses 3.4V around 125°.
    Sensor 2 crosses 3.4V around 115°.
    """
    # Sensor 1 logic
    if current_angle > 135.0:
        s1_base = 2.6
    else:
        # Voltage increases as angle drops below 135°
        s1_base = 2.6 + (135.0 - current_angle) * 0.1 

    # Sensor 2 logic (delayed)
    if current_angle > 125.0:
        s2_base = 2.6
    else:
        # Voltage increases as angle drops below 125°
        s2_base = 2.6 + (125.0 - current_angle) * 0.1 
        
    # Noise is kept positive so it never subtracts from the 2.6V baseline
    s1 = s1_base + random.uniform(0.01, 0.05)
    s2 = s2_base + random.uniform(0.01, 0.05)
    
    return s1, s2

import math
import random

def read_sensors(current_finger_angle: float, current_rotor_angle: float, a: float = 7.0) -> tuple[float, float]:
    """
    Simulates two flex sensors scanning a square of side length 'a' (cm).
    Voltage stays at baseline 2.6V, then spikes above 3.4V threshold 
    when the finger angle matches the physical width of the square at the current rotation.
    """
    # Calculate the bounding width of a square rotated by current_rotor_angle
    # W(theta) = a * (|cos(theta)| + |sin(theta)|)
    rad_angle = math.radians(current_rotor_angle)
    target_width = a * (abs(math.cos(rad_angle)) + abs(math.sin(rad_angle)))
    
    # Calculate what finger angle corresponds to this target physical width
    # Inverting the logic from adapter_angle_to_width()
    max_angle, min_angle = 250.0, 0.0
    max_width, min_width = 11.20, 6.2
    
    # Check if width is within measurable limits, cap if necessary
    clamped_width = max(min_width, min(max_width, target_width))
    target_finger_angle = min_angle + (clamped_width - min_width) * ((max_angle - min_angle) / (max_width - min_width))

    # Sensor logic: trigger (jump > 3.4V) when current angle squeezes down to the target angle
    if current_finger_angle > target_finger_angle:
        s1_base = 2.6
        s2_base = 2.6
    else:
        # Voltage rapidly increases to simulate direct rigid contact
        s1_base = 3.5 + (target_finger_angle - current_finger_angle) * 0.1
        s2_base = 3.5 + (target_finger_angle - current_finger_angle) * 0.1
        
    # Noise is kept positive so it never subtracts from the baseline
    s1 = s1_base + random.uniform(0.01, 0.05)
    s2 = s2_base + random.uniform(0.01, 0.05)
    
    return s1, s2

    
GRIPPER = 15  # FOR ACTUAL USE 0, FOR TESTING USE 15
WRIST_ROT = 1
    
def scan_sequence(arm : RobotArm, slice: int) -> list[tuple[float, float, str]]:
    # DO NOT CHANGE HOME_POSITION TO SERVO_CONFIG
    home_angle = arm.HOME_POSITION[GRIPPER]
    closed_angle = 0.0
    
    # List to hold the formatted (angle, width, side) tuples
    scan_results = []
    
    print("Initializing gripper FINGERS to home position...")
    arm.move_smooth(GRIPPER, home_angle)
    time.sleep(0.5)

    for i in range(slice):
        gripper_r_target_angle = (i / slice) * 180.0
        
        print(f"\n--- Sequence {i+1}/{slice} ---")
        print(f"Rotating Gripper ROTOR to {gripper_r_target_angle:.1f}°")
        arm.move_smooth(1, gripper_r_target_angle)
        time.sleep(0.5)
        
        print("Closing Gripper FINGER (Channel 0 or 15) and monitoring sensors...")
        current_gripper_f_angle = home_angle
        
        normal_speed_step = 2.0
        slow_speed_step = 0.5
        current_step = normal_speed_step
        
        s1_trigger_angle = None
        s2_trigger_angle = None
        
        while current_gripper_f_angle > closed_angle:
            v1, v2 = read_sensors(current_gripper_f_angle, gripper_r_target_angle)
            
            # Bounds check: Must stay between 2.5 and 3.4
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
            
            arm.kit.servo[GRIPPER].angle = current_gripper_f_angle # type: ignore
            arm.current_angles[GRIPPER] = current_gripper_f_angle 
            time.sleep(0.02) 

        print(f"Final Stopped Angle of FINGERS: {current_gripper_f_angle:.1f}°")
        
        # Package the data points for this angle
        ch1_rounded = round(gripper_r_target_angle, 2)
        
        # If both trigger at the exact same time
        if (s1_trigger_angle is not None and 
            s2_trigger_angle is not None and 
            s1_trigger_angle == s2_trigger_angle):
            width = adapter_angle_to_width(s1_trigger_angle)
            scan_results.append((ch1_rounded, round(width, 2), "both"))
        else:
            # If they triggered at different times, or if only one triggered
            if s1_trigger_angle is not None:
                width1 = adapter_angle_to_width(s1_trigger_angle)
                scan_results.append((ch1_rounded, round(width1, 2), "left"))
                
            if s2_trigger_angle is not None:
                width2 = adapter_angle_to_width(s2_trigger_angle)
                scan_results.append((ch1_rounded, round(width2, 2), "right"))
        
        print("Opening FINGERS to home position...")
        arm.move_smooth(GRIPPER, home_angle)
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