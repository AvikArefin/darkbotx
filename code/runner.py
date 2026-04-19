import os

from motor import RobotArm
from scanner import scan_sequence
from pointnet import DarkBot, export_dented_to_stl, generate_urdf

def main():
    print("=== STARTING DARKBOT ORCHESTRATION PIPELINE ===")

    print("\n[1/4] Initializing Robot Arm...")
    arm = RobotArm()

    print("\n[2/4] Executing Hardware Scan...")
    # Run the sequence (e.g., n=12 for a scan every 15 degrees)
    live_measurements = scan_sequence(arm, slice=4)
    print(f"\nScan complete. Gathered {len(live_measurements)} data points:")
    for m in live_measurements:
        print(f"  -> {m}")

    print("\n[3/4] Initializing PointNet (DarkBot) with Live Data...")
    # Object settings
    height_val = 30
    obj_name = "live_scanned_target"
    # Instantiate the PointNet class using the live data directly!
    darkbot = DarkBot(measurements=live_measurements, height=height_val)


    print("\n[4/4] Generating 3D Meshes and URDF...")
    # Setup directory structure
    folder_path = f"assets/{obj_name}"
    os.makedirs(os.path.join(folder_path, "meshes"), exist_ok=True)
    stl_path = os.path.join(folder_path, "meshes", f"{obj_name}.stl")
    urdf_path = os.path.join(folder_path, f"{obj_name}.urdf")
    # Generate files
    export_dented_to_stl(darkbot, stl_path)
    generate_urdf(stl_path, obj_name, scale=0.1, urdf_filepath=urdf_path)

    print("\n=== PIPELINE COMPLETE ===")
    
    
    print("Launching visualization...")
    darkbot.visualize()

if __name__ == "__main__":
    main()