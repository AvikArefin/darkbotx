from robot import RobotArm, RJoint

arm = RobotArm()
arm.move_smooth(RJoint.GRIPPER, 270)
print("Gripper disengaged.")