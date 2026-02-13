import numpy as np
import genesis as gs
from tabulate import tabulate # You may need to pip install tabulate

def main():
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
        ),
        show_FPS = False,
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

    # --- Updated Data Extraction Section ---
    table_data = []
    
    for joint in robot.joints:
        link_name = joint.link.name 
        pos_limit = joint.dofs_limit
        
        # Initialize display variables
        limit_val = "N/A"
        unit = ""

        if pos_limit[0] is not None:
            # Check if it's a prismatic (slider) joint
            # Genesis typically uses 0 for fixed, 1 for revolute, 2 for prismatic
            if joint.type == gs.JOINT_TYPE.PRISMATIC:
                limit_val = [f"{limit:.3f}" for limit in pos_limit[0]]
                unit = " (m)"
            elif joint.type == gs.JOINT_TYPE.REVOLUTE:
                limit_deg = [np.rad2deg(limit) for limit in pos_limit[0]]
                limit_val = [f"{limit:.2f}" for limit in limit_deg]
                unit = " (deg)"
            else:
                # Fallback for continuous or other types
                limit_val = pos_limit[0]
                unit = " (raw)"

        table_data.append([
            link_name, 
            joint.name,
            f"{joint.type}".split('.')[-1], # Displays 'REVOLUTE' or 'PRISMATIC'
            f"{limit_val}{unit}"
        ])

    headers = ["Link Name", "Joint Name", "Type", "Limits"]
    print("\nRobot Hierarchy and Limits:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    # -------------------------------

    while True:
        scene.step()

if __name__ == "__main__":
    main()