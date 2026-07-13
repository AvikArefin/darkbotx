from robot import RobotArm, RJoint

arm = RobotArm()
arm.move_smooth(RJoint.GRIPPER, 250)
print("Gripper engaged.")