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

        results, emergency= arm.scan(slice=2)

        print(results)

        object = PointNet(measurements=results, height=3)
        object.export("test", scale=0.006)

        arm.go_home_smooth()

        try:
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
            arm.move_smooth(RJoint.GRIPPER, action[0][1])

        except Exception:
            # TODO: Emergency Policy
            arm.move_smooth(RJoint.WRIST_ROLL, emergency[0])
            arm.move_smooth(RJoint.GRIPPER, emergency[1])

        arm.go_lift_smooth()  # DOES NOT CONTROL GRIPPER
        arm.go_put_smooth()   # DOES NOT CONTROL GRIPPER
        arm.go_grab_smooth()

        arm.go_home_smooth()
        

    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        arm.deinit()
