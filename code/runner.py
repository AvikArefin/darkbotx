import os
import atexit
import time

from robot import RobotArm
from scanner import scan_sequence
from pointnet import DarkBot, export_dented_to_stl, generate_urdf

def main():
    print("=== STARTING DARKBOT ORCHESTRATION PIPELINE ===")

    print("\n[1/9] Initializing Robot Arm...")
    arm = RobotArm()
    atexit.register(arm.relax_all)
    
    print("\n[2/9] Smooth Transition to Grab Position")
    arm.go_grab_smooth()

    print("\n[3/9] Executing Hardware Scan")
    # Run the sequence (e.g., n=12 for a scan every 15 degrees)
    live_measurements = scan_sequence(arm, slice=4)
    print(f"\nScan complete. Gathered {len(live_measurements)} data points:")
    for m in live_measurements:
        print(f"  -> {m}")

    print("\n[4/9] Grabbing the object based on scan data...")
    if live_measurements:
        # Calculate a characteristic size (e.g., average radius) from the scan data
        avg_measurement = sum(live_measurements) / len(live_measurements)
        
        # Map the measurement to a suitable grip angle.
        # Assuming channel 0 config: 250 is the maximum safe open boundary.
        # We adjust the clamping angle based on the measured object size.
        # (This heuristic maps larger measurements to wider grip angles)
        scale_factor = 2.5
        calculated_angle = int(avg_measurement * scale_factor)
        
        # Clamp the angle to prevent crushing (e.g., min 90) or over-extending (max 250)
        optimal_grip_angle = max(90, min(250, calculated_angle))
        print(f"Calculated optimal grip angle: {optimal_grip_angle}° from scan average: {avg_measurement:.2f}")
    else:
        print("No scan data received. Defaulting to safe grip angle.")
        optimal_grip_angle = 150
        
    arm.move_smooth(0, optimal_grip_angle)
    time.sleep(0.5)

    print("\n[5/9] Returning arm to HOME position...")
    arm.go_home_smooth()

    print("\n[6/9] Initializing PointNet with Live Data...")
    # Object settings
    height_val = 5.5 #cm
    obj_name = "live_scanned_target"
    # Instantiate the PointNet class using the live data directly
    darkbot = DarkBot(measurements=live_measurements, height=height_val)

    print("\n[7/9] Generating 3D Meshes and URDF...")
    # Setup directory structure
    folder_path = f"assets/{obj_name}"
    os.makedirs(os.path.join(folder_path, "meshes"), exist_ok=True)
    stl_path = os.path.join(folder_path, "meshes", f"{obj_name}.stl")
    urdf_path = os.path.join(folder_path, f"{obj_name}.urdf")
    # Generate files
    export_dented_to_stl(darkbot, stl_path)
    generate_urdf(stl_path, obj_name, scale=0.1, urdf_filepath=urdf_path)

    print("Launching visualization...")
    darkbot.visualize()

    print("\n[8/9] Moving to grab position again...")
    arm.go_grab_smooth()

    print("\n[9/9] Placing the object...")
    # Move the base (Channel 1) to a new angle to relocate the object,
    # while maintaining the rest of the arm in the grab posture.
    place_target = arm.GRAB_POSITION.copy()
    place_target[1] = 180  # Rotate base to 180 degrees for placement
    arm.move_all_smooth(place_target)
    time.sleep(0.5)

    # Open the gripper back to 250 to release the object
    print("Releasing object...")
    arm.move_smooth(0, 250)
    time.sleep(0.5)

    print("\n=== PIPELINE COMPLETE ===")
    
if __name__ == "__main__":
    main()