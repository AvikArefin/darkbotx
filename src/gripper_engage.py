from robot import RobotArm, RJoint, WIDE_GRIP

arm = RobotArm()
arm.move_smooth(RJoint.GRIPPER, target_angle=WIDE_GRIP, delay=0.1)
print("Gripper engaged.")