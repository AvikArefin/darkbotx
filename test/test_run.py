from motor import RobotArm
import time
import random

def read_simulated_sensors(current_angle: float, start_angle: float, close_angle: float) -> tuple[float, float]:
    """
    Simulates two flex sensors. Values start around 2.5V and gradually increase.
    It will eventually cross the 3.4V boundary as the gripper closes to test the stop logic.
    """
    total_travel = start_angle - close_angle
    traveled = start_angle - current_angle
    progress = traveled / total_travel if total_travel != 0 else 1
    
    # Base voltage scales up from 2.5V to ~3.6V
    base_voltage = 2.5 + (progress * 1.1) 
    
    # Add random noise
    s1 = base_voltage + random.uniform(-0.05, 0.1)
    s2 = base_voltage + random.uniform(-0.1, 0.05)
    
    return s1, s2

def execute_scanning_sequence(arm, n: int) -> dict[float, float]:
    """
    Executes a radial scanning sequence.
    Returns a dictionary mapping Channel 1 target angles to Channel 0 stopping angles.
    """
    widest_angle = 270.0
    closed_angle = 0.0
    
    # Initialize the dictionary to store the results
    scan_results = {}
    
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
        
        while current_ch0_angle > closed_angle:
            v1, v2 = read_simulated_sensors(current_ch0_angle, widest_angle, closed_angle)
            
            # Check if sensors are OUTSIDE the acceptable 2.5 to 3.4 range
            s1_out_of_bounds = not (2.5 <= v1 <= 3.4)
            s2_out_of_bounds = not (2.5 <= v2 <= 3.4)
            
            if s1_out_of_bounds and s2_out_of_bounds:
                print(f"--> STOP: Both sensors out of bounds (S1: {v1:.2f}V, S2: {v2:.2f}V)")
                break 
                
            elif s1_out_of_bounds or s2_out_of_bounds:
                if current_step != slow_speed_step:
                    print(f"--> SLOWING DOWN: One sensor out of bounds (S1: {v1:.2f}V, S2: {v2:.2f}V)")
                    current_step = slow_speed_step
            else:
                # Both sensors are strictly between 2.5 and 3.4. Maintain or return to normal speed.
                current_step = normal_speed_step
                    
            current_ch0_angle -= current_step
            
            arm.kit.servo[0].angle = current_ch0_angle
            arm.current_angles[0] = current_ch0_angle 
            time.sleep(0.02) 

        print(f"Final Angle of Channel 0: {current_ch0_angle:.1f}°")
        
        # Store the data in the dictionary
        scan_results[round(ch1_target_angle, 2)] = round(current_ch0_angle, 2)
        
        print("Opening Channel 0 to widest position...")
        arm.move_smooth(0, widest_angle)
        time.sleep(0.5) 

    print("\nScanning sequence completed.")
    
    # Return the populated dictionary after the entire operation
    return scan_results

if __name__ == "__main__":
    # Assuming RobotArm is properly instantiated
    my_arm = RobotArm()
    
    # Capture the returned dictionary
    final_data = execute_scanning_sequence(my_arm, n=3)
    
    print("\n--- Final Gathered Data ---")
    for ch1_angle, ch0_stop_angle in final_data.items():
        print(f"Channel 1 Angle: {ch1_angle:5.1f}°  |  Channel 0 Stop Angle: {ch0_stop_angle:5.1f}°")