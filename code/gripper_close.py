from robot import RobotArm

arm = RobotArm()

arm.move_smooth(0, 0)  # Move gripper to closed position (0 degrees)
arm.move_smooth(15, 0)  # Move gripper test to closed position (0 degrees)
print("Gripper closed.")