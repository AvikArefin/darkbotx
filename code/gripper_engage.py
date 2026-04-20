from motor import RobotArm

arm = RobotArm()
arm.move_smooth(0, 250)
arm.move_smooth(15, 250)