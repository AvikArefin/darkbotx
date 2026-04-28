import numpy as np
import genesis as gs
from enum import Enum, auto

class Joint(Enum):
    """Logical names for the robot joints."""
    GRIPPER = auto()
    WRIST_ROT = auto()
    WRIST = auto()
    ELBOW = auto()
    SHOULDER = auto()
    BASE = auto()

# Centralized mapping: logical joint -> URDF joint name
URDF_MAPPING: dict[Joint, str] = {
    Joint.GRIPPER:   "Slider 37",
    Joint.WRIST_ROT: "Revolute 19", #
    Joint.WRIST:     "Revolute 35",
    Joint.ELBOW:     "Revolute 34", #
    Joint.SHOULDER:  "Revolute 36",
    Joint.BASE:      "Revolute 13", #
}

STEPS_PER_MOVEMENT = 200

def main():
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -0.5, 0.8),
            camera_lookat=(0.0, 0.2, 0.2),
            camera_fov=30,
        ),
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="assets/Hiwonder_urdf_simplified/Hiwonder.urdf",
            fixed=True,
            decimate=False,
            convexify=False, 
        ),
    )
    scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.007, -0.120, 0.0)))
    scene.build()

    # Joint data is resolved and cached during the initialization phase.
    raw_joint_info = {j.name: j for j in robot.joints if j.dofs_limit[0] is not None}
    
    j_idx = {}
    j_limits = {}
    
    for logical_joint, urdf_name in URDF_MAPPING.items():
        joint_obj = raw_joint_info[urdf_name]
        j_idx[logical_joint] = joint_obj.dofs_idx_local[0]
        j_limits[logical_joint] = {
            "min": joint_obj.dofs_limit[0][0],
            "max": joint_obj.dofs_limit[0][1],
        }

    target_positions = np.zeros(robot.n_dofs)

    def move_joints(movement_dict, steps=STEPS_PER_MOVEMENT):
        """Executes movement using logical Joint Enums."""
        start_pos = target_positions.copy()
        targets = np.zeros(robot.n_dofs)
        
        for joint_enum, deg_value in movement_dict.items():
            targets[j_idx[joint_enum]] = np.deg2rad(deg_value)
            
        for t in np.linspace(0, 1, steps):
            for joint_enum in movement_dict.keys():
                idx = j_idx[joint_enum]
                target_positions[idx] = start_pos[idx] + (targets[idx] - start_pos[idx]) * t
            robot.control_dofs_position(target_positions)
            scene.step()

    def open_gripper():
        target_positions[j_idx[Joint.GRIPPER]] = j_limits[Joint.GRIPPER]["max"]
        for _ in range(50):
            robot.control_dofs_position(target_positions)
            scene.step()

    def close_gripper():
        target_positions[j_idx[Joint.GRIPPER]] = j_limits[Joint.GRIPPER]["min"]
        for _ in range(50):
            robot.control_dofs_position(target_positions)
            scene.step()

    print("[1] Opening gripper...")
    open_gripper()

    print("[2] Moving arm above cube...")
    move_joints({
        Joint.WRIST_ROT: 0.0,
        Joint.WRIST: 0,
        Joint.ELBOW: 37.3,
        Joint.SHOULDER: 115.5,
        Joint.BASE: 90,
    })


    print("[3] Lowering onto cube...")
    move_joints({
        Joint.WRIST_ROT: 0.0,
        Joint.WRIST: 0,
        Joint.ELBOW: 37.3,
        Joint.SHOULDER: 123,
        Joint.BASE: 90,
    }, steps=50)

    print("[4] Closing gripper...")
    close_gripper()

    print("[5] Lifting cube...")
    move_joints({
        Joint.WRIST: 18.0,
        Joint.ELBOW: 57,
        Joint.SHOULDER: 41,
    }, steps=100)

    print("[6] Rotating gripper...")
    move_joints({Joint.WRIST_ROT: 180.0})

    print("[7] Placing cube back down...")
    move_joints({
        Joint.WRIST: 0,
        Joint.ELBOW: 37.3,
        Joint.SHOULDER: 115.5,
        Joint.BASE: 90,
    }, steps=100)
    move_joints({
        Joint.WRIST: 0,
        Joint.ELBOW: 37.3,
        Joint.SHOULDER: 123,
        Joint.BASE: 90,
    }, steps=50)

    print("[8] Releasing...")
    open_gripper()

if __name__ == "__main__":
    main()