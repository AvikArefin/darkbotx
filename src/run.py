import time

import torch

from robot import RobotArm, RJoint
from pointnet import PointNet


if __name__ == "__main__":
    print("=================== DARKBOTX ===================")
    arm = RobotArm()
    try:
        arm.go_grab_smooth()
        time.sleep(5)

        scan_results: list[tuple[float, float, str]] = arm.scan(slice=3)

        object = PointNet(measurements=scan_results, height=5.5)
        object.export("test", scale=0.006)

        arm.go_home_smooth()


        policy_path = "logs/real/deployed_policy.pt"
        policy = torch.jit.load(policy_path, map_location="cpu")
        policy.eval()  


        # TODO: Properly create the observation tensor
        obs = torch.cat([
            periphery_2d,
            object_yaw,
            object_height,
            current_gripper_state,
        ], dim=-1) 

        with torch.no_grad():
            action = policy(obs)
            
        arm.move_smooth(RJoint.WRIST_ROLL, action[0][0])
        arm.move_smooth(RJoint.WRIST, action[0][1])

        # TODO: lift while holding the object
        # TODO: put down the object

        arm.go_home_smooth()
        

    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        arm.deinit()
