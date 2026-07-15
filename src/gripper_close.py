from robot import RobotArm, RJoint

arm = RobotArm()
arm.move_smooth(RJoint.GRIPPER, target_angle=0)  # Move gripper test to closed position (0 degrees)
print("Gripper closed.")