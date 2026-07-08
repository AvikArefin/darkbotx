import genesis as gs
import sys
import os

# Add the src directory to path so config can be imported if run from elsewhere
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DJoint

def main():
    # Initialize genesis with CPU backend since we just want to parse the URDF
    gs.init(backend=gs.cpu)
    
    scene = gs.Scene(show_viewer=False)
    
    # Load the robot
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="assets/Hiwonder/Hiwonder.urdf",
            fixed=True,
        ),
    )
    
    # Build the scene to populate the internal properties
    scene.build()
    
    print("=" * 80)
    print(f"{'Logical Name':<15} | {'Joint Name':<12} | {'Type':<10} | {'Lower Limit':<12} | {'Upper Limit'}")
    print("-" * 80)
    
    target_dofs = (DJoint.WRIST_ROLL, DJoint.GRIPPER_LEFT)
    
    for joint in robot.joints:
        if len(joint.dofs_idx_local) > 0:
            dof_idx = joint.dofs_idx_local[0]
            if dof_idx in target_dofs:
                logical_name: str = DJoint(dof_idx).name
                limit = joint.dofs_limit
                if limit[0] is not None:
                    lower = float(limit[0][0])
                    upper = float(limit[0][1])
                    print(f"{logical_name:<15} | {joint.name:<12} | {joint.type.name:<10} | {lower:<12.4f} | {upper:.4f}")
                else:
                    print(f"{logical_name:<15} | {joint.name:<12} | {joint.type.name:<10} | {'None':<12} | None")
            
    print("-" * 80)


if __name__ == "__main__":
    main()
