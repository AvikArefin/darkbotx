import time
import random

from motor import RobotArm

def adapter_angle_to_width(angle: float, max_angle: float = 270.0, min_angle: float = 0.0, max_width: float = 72.0, min_width: float = 32.0) -> float:
    """
    Converts the Channel 0 servo angle to a physical width (mm).
    MAX (270°) -> 72mm
    MIN (0°) -> 32mm
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

def execute_scanning_sequence(arm : RobotArm, n: int) -> list[tuple[float, float, str]]:
    widest_angle = 270.0
    closed_angle = 0.0
    
    # List to hold the formatted (angle, width, side) tuples
    scan_results = []
    
    print("Initializing gripper to widest position...")
    arm.move_smooth(0, widest_angle)
    time.sleep(0.5)

    for i in range(n):
        ch1_target_angle = (i / n) * 180.0
        
        print(f"\n--- Sequence {i+1}/{n} ---")
        print(f"Rotating Channel 1 to {ch1_target_angle:.1f}°")
        arm.move_smooth(1, ch1_target_angle)
        time.sleep(0.5)
        
        print("Closing gripper (Channel 0) and monitoring sensors...")
        current_ch0_angle = widest_angle
        
        normal_speed_step = 2.0
        slow_speed_step = 0.5
        current_step = normal_speed_step
        
        s1_trigger_angle = None
        s2_trigger_angle = None
        
        while current_ch0_angle > closed_angle:
            v1, v2 = read_simulated_sensors(current_ch0_angle)
            
            # Bounds check: Must stay between 2.5 and 3.4
            s1_out_of_bounds = not (2.5 <= v1 <= 3.4)
            s2_out_of_bounds = not (2.5 <= v2 <= 3.4)
            
            if s1_out_of_bounds and s1_trigger_angle is None:
                s1_trigger_angle = current_ch0_angle
                print(f"--> Sensor 1 Triggered at {current_ch0_angle:.1f}° (Value: {v1:.2f}V)")
                
            if s2_out_of_bounds and s2_trigger_angle is None:
                s2_trigger_angle = current_ch0_angle
                print(f"--> Sensor 2 Triggered at {current_ch0_angle:.1f}° (Value: {v2:.2f}V)")
            
            if s1_trigger_angle is not None and s2_trigger_angle is not None:
                print("--> STOP: Both sensors triggered. Stopping motor.")
                break 
                
            elif s1_trigger_angle is not None or s2_trigger_angle is not None:
                if current_step != slow_speed_step:
                    print("--> SLOWING DOWN: Single contact made, proceeding cautiously.")
                    current_step = slow_speed_step
            else:
                current_step = normal_speed_step
                    
            current_ch0_angle -= current_step
            
            arm.kit.servo[0].angle = current_ch0_angle
            arm.current_angles[0] = current_ch0_angle 
            time.sleep(0.02) 

        print(f"Final Stopped Angle of Channel 0: {current_ch0_angle:.1f}°")
        
        # Package the data points for this angle
        ch1_rounded = round(ch1_target_angle, 2)
        
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
        
        print("Opening Channel 0 to widest position...")
        arm.move_smooth(0, widest_angle)
        time.sleep(0.5) 

    print("\nScanning sequence completed.")
    return scan_results

if __name__ == "__main__":
    arm = RobotArm()
    final_data = execute_scanning_sequence(arm, n=3)
    
    print("\n--- Final Gathered Data (Ready for DarkBot) ---")
    print("measurements = [")
    for angle, width, side in final_data:
        print(f"    ({angle:5.1f}, {width:5.2f}, \"{side}\"),")
    print("]")