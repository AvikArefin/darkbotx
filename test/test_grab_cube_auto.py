import numpy as np
import genesis as gs

GRIPPER_ROTATION_180 = 180
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
            file="assets/Hiwonder_description/Hiwonder.urdf",
            fixed=True,
            # Keep simplification on for the heavy lifting parts
            decimate=True, 
            decimate_aggressiveness=5,
            decimate_face_num=500, 
            convexify=False, 
        ),
    )

    box = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.0, -0.1, 0.0)),
    )

    scene.build()

    joint_limits = {}
    for joint in robot.joints:
        if joint.dofs_limit[0] is not None:
            is_master_slider = (
                joint.type == gs.JOINT_TYPE.PRISMATIC and joint.name == "Slider 29"
            )
            if joint.type == gs.JOINT_TYPE.REVOLUTE or is_master_slider:
                joint_limits[joint.name] = {
                    "idx": joint.dofs_idx_local[0],
                    "min": joint.dofs_limit[0][0],
                    "max": joint.dofs_limit[0][1],
                    "is_revolute": joint.type == gs.JOINT_TYPE.REVOLUTE,
                }

    dof_names = list(joint_limits.keys())
    dof_indices = [joint_limits[name]["idx"] for name in dof_names]
    dof_mins = [joint_limits[name]["min"] for name in dof_names]
    dof_maxs = [joint_limits[name]["max"] for name in dof_names]

    home_pos = [0.0] * robot.n_dofs
    joint1_idx = joint_limits["Revolute 13"]["idx"]
    joint2_idx = joint_limits["Revolute 14"]["idx"]
    joint3_idx = joint_limits["Revolute 15"]["idx"]
    joint4_idx = joint_limits["Revolute 16"]["idx"]
    joint5_idx = joint_limits["Revolute 19"]["idx"]
    slider_idx = joint_limits["Slider 29"]["idx"]

    target_positions = np.zeros(robot.n_dofs)

    def move_joints(joints_dict, steps=STEPS_PER_MOVEMENT):
        start_pos = target_positions.copy()
        targets = np.zeros(robot.n_dofs)
        for name, value in joints_dict.items():
            targets[joint_limits[name]["idx"]] = np.deg2rad(value)
        for t in np.linspace(0, 1, steps):
            for i, name in enumerate(joints_dict.keys()):
                idx = joint_limits[name]["idx"]
                target_positions[idx] = (
                    start_pos[idx] + (targets[idx] - start_pos[idx]) * t
                )
            robot.control_dofs_position(target_positions)
            scene.step()

    def open_gripper():
        target_positions[slider_idx] = joint_limits["Slider 29"]["max"]
        for _ in range(50):
            robot.control_dofs_position(target_positions)
            scene.step()

    def close_gripper():
        target_positions[slider_idx] = joint_limits["Slider 29"]["min"]
        for _ in range(50):
            robot.control_dofs_position(target_positions)
            scene.step()

    print("[1] Moving arm above cube...")
    move_joints(
        {
            "Revolute 13": 0.0,
            "Revolute 14": 104.3,
            "Revolute 15": 65.9,
            "Revolute 16": 90.0,
            "Revolute 19": 0.0,
        }
    )

    print("[2] Opening gripper...")
    open_gripper()

    print("[3] Lowering onto cube...")
    move_joints(
        {
            "Revolute 13": 0.0,
            "Revolute 14": 119.0,
            "Revolute 15": 54.6,
            "Revolute 16": 89.0,
            "Revolute 19": 0.0,
        },
        steps=50,
    )

    print("Closing gripper to pick up cube...")
    close_gripper()

    print("[4] Lifting cube...")
    move_joints(
        {
            "Revolute 14": 68.8,
            "Revolute 15": 57.3,
            "Revolute 16": 34.4,
            "Revolute 19": 0.0,
        },
        steps=100,
    )

    print("[5] Rotating gripper 180 degrees...")
    for t in np.linspace(0, 180, STEPS_PER_MOVEMENT):
        target_positions[joint5_idx] = np.deg2rad(t)
        robot.control_dofs_position(target_positions)
        scene.step()

    print("[6] Lowering cube back to ground...")
    move_joints(
        {
            "Revolute 13": 0.0,
            "Revolute 14": 119.0,
            "Revolute 15": 54.6,
            "Revolute 16": 89.0,
            "Revolute 19": 180.0,
        },
        steps=50,
    )

    print("[7] Opening gripper to release cube...")
    open_gripper()

    print(
        "Task complete! Cube has been picked up, rotated 180 degrees, and placed back on ground."
    )


if __name__ == "__main__":
    main()
