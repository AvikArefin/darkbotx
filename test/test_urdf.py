import numpy as np
import genesis as gs
from tabulate import tabulate

def main():
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
        ),
        show_FPS=False,
    )

    plane = scene.add_entity(gs.morphs.Plane())

    # Import the URDF
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="assets/Hiwonder_description/Hiwonder.urdf",
            fixed=True,
        ),
    )

    # CRITICAL: You must build the scene before accessing joint/link data
    scene.build()

    table_data = []
    
# List to store actuated joint parameters for the animation
    actuated_dofs = []

    for joint in robot.joints:
        link_name = joint.link.name
        pos_limit = joint.dofs_limit

        # Initialize display variables
        limit_val = "N/A"
        unit = ""

        if pos_limit[0] is not None:
            
            is_revolute = (joint.type == gs.JOINT_TYPE.REVOLUTE)
            # CRITICAL: Only grab the Master slider, ignore Slider 31 (the slave)
            is_master_slider = (joint.type == gs.JOINT_TYPE.PRISMATIC and joint.name == "Slider 29")

            if joint.type == gs.JOINT_TYPE.PRISMATIC:
                limit_val = [f"{limit:.3f}" for limit in pos_limit[0]]
                unit = " (m)"
                
            elif joint.type == gs.JOINT_TYPE.REVOLUTE:
                limit_deg = [np.rad2deg(limit) for limit in pos_limit[0]]
                limit_val = [f"{limit:.2f}" for limit in limit_deg]
                unit = " (deg)"
            else:
                limit_val = pos_limit[0]
                unit = " (raw)"

            # Add to our animation queue IF it's a servo or the master finger
            if is_revolute or is_master_slider:
                actuated_dofs.append({
                    "name": joint.name,
                    "idx": joint.dofs_idx_local[0],
                    "min": pos_limit[0][0],
                    "max": pos_limit[0][1],
                    "is_rotational": is_revolute
                })

        table_data.append(
            [
                link_name,
                joint.name,
                f"{joint.type}".split(".")[-1],  
                f"{limit_val}{unit}",
            ]
        )

    headers = ["Link Name", "Joint Name", "Type", "Limits"]
    print("\nRobot Hierarchy and Limits:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # --- Animation Sequence ---
    
    target_positions = np.zeros(robot.n_dofs)
    steps_per_movement = 150 

    while True:
        for dof in actuated_dofs:
            # Print friendly units based on joint type
            if dof["is_rotational"]:
                print(f"Moving {dof['name']} from {np.rad2deg(dof['min']):.1f}° to {np.rad2deg(dof['max']):.1f}°")
            else:
                print(f"Moving {dof['name']} from {dof['min']:.3f}m to {dof['max']:.3f}m")
            
            # 1. Sweep from min to max limit
            forward_trajectory = np.linspace(dof['min'], dof['max'], steps_per_movement)
            for val in forward_trajectory:
                target_positions[dof['idx']] = val
                robot.control_dofs_position(target_positions)
                scene.step()
                
            # 2. Sweep back to zero position cleanly
            return_trajectory = np.linspace(dof['max'], 0.0, steps_per_movement)
            for val in return_trajectory:
                target_positions[dof['idx']] = val
                robot.control_dofs_position(target_positions)
                scene.step()

if __name__ == "__main__":
    main()