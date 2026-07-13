import os
import sys
import time

import types
from unittest.mock import MagicMock
from typing import Any

import genesis as gs
import numpy as np


# 1. Hardware Mocks - Must happen before any project imports
sys.modules.update({
    "adafruit_servokit": MagicMock(),
    "board": MagicMock(),
    "busio": MagicMock(),
    "digitalio": MagicMock(),
})

# 2. Path Setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# --- Class Definition ---
class MockServo:
    def __init__(self, channel: int, arm: Any) -> None:
        self.channel = channel
        self.arm = arm
        self._angle = float(arm.HOME_POSITION.get(channel, 90.0))
    
    @property
    def angle(self) -> float:
        return self._angle
    
    @angle.setter
    def angle(self, value: float | None) -> None:
        if value is None:
            return
        self._angle = float(value)
        self.arm._update_sim_single(self.channel, self._angle)

    def set_pulse_width_range(self, min_pulse: int, max_pulse: int) -> None:
        pass

class MockKit:
    def __init__(self, arm: Any) -> None:
        self.servo = {ch: MockServo(ch, arm) for ch in [0, 1, 2, 3, 4, 5, 15]}

class SRobotArm:
    """A eplacement for RobotArm that uses Genesis."""
    SERVO_CONFIG: dict[int, str] = {
        0:  "Gripper",
        1:  "Wrist Rot",
        2:  "Wrist",
        3:  "Elbow",
        4:  "Shoulder",
        5:  "Base",
        15: "Gripper",
    }
    
    HOME_POSITION: dict[int, int] = {
        0: 250, 1: 0, 2: 90, 3: 90, 4: 90, 5: 105, 15: 250,
    }

    GRAB_POSITION: dict[int, int] = {
        0: 250, 1: 0, 2: 0, 3: 37, 4: 130, 5: 105, 15: 250,
    }
    
    # Updated to match the new URDF mapping
    CHANNEL_TO_JOINT = {
        0: "Slider 37",   # Gripper
        1: "Revolute 19", # Wrist Rot
        2: "Revolute 35", # Wrist
        3: "Revolute 34", # Elbow
        4: "Revolute 36", # Shoulder
        5: "Revolute 13", # Base
        15: "Slider 37",  # Gripper
    }

    def __init__(self, i2c_bus: Any = None) -> None:

        print("Initializing Genesis Simulation...")
        gs.init(backend=gs._gs_backend.gpu)
        self.scene = gs.Scene(
            show_viewer=False,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.6, -0.6, 0.6),
                camera_lookat=(0.0, 0.0, 0.2),
                camera_fov=30,
            ),
            show_FPS=False,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="assets/Hiwonder/Hiwonder.urdf",
                fixed=True,
                decimate=False,
                convexify=False, 
            ),
            # vis_mode='collision',
        )
        # Add a cube to scan
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(-0.011, -0.116, 0.0)),
        )
        self.scene.build()
        
        self.j_idx = {}
        for ch, name in self.CHANNEL_TO_JOINT.items():
            for joint in self.robot.joints:
                if joint.name == name:
                    self.j_idx[ch] = joint.dofs_idx_local[0]
                    break
        
        self.target_positions = np.zeros(self.robot.n_dofs)
        self.current_angles = {ch: float(ang) for ch, ang in self.HOME_POSITION.items()}
        self.kit = MockKit(self)
        self._initialized = True
        
        # Initial sync
        for ch, ang in self.current_angles.items():
            self._update_target_pos(ch, ang)
        self.robot.control_dofs_position(self.target_positions)
        self.scene.step()
        print("Genesis Simulation Ready.")

    def _update_target_pos(self, channel: int, angle: float) -> None:
        if channel not in self.j_idx: 
            return
        idx = self.j_idx[channel]
        
        if channel in [0, 15]: # Gripper
            # Slider 37 limits are [-0.01, 0.02] m (Total range: 0.03)
            val = (angle / 250.0) * 0.03 - 0.01
        elif channel == 1: # Wrist Rot
            val = np.deg2rad(angle)
        elif channel == 2: # Wrist
            val = np.deg2rad(angle)
        elif channel == 3: # Elbow
            val = np.deg2rad(angle)
        elif channel == 4: # Shoulder
            val = np.deg2rad(angle)
        elif channel == 5: # Base
            val = np.deg2rad(angle - 15) # Base has a 15 degree offset in Real Bot
        else:
            val = np.deg2rad(angle)
            
        self.target_positions[idx] = val

    def _update_sim_single(self, channel: int, angle: float) -> None:
        self._update_target_pos(channel, angle)
        self.robot.control_dofs_position(self.target_positions)
        self.scene.step()

    def move_smooth(self, channel: int, target_angle: float, delay: float = 0.01, step_size: float = 2.0) -> bool:
        current = self.current_angles[channel]
        direction = 1 if target_angle > current else -1
        while abs(target_angle - current) > step_size:
            current += step_size * direction
            self.current_angles[channel] = current
            self._update_sim_single(channel, current)
            time.sleep(0.001)
        
        self.current_angles[channel] = target_angle
        self._update_sim_single(channel, target_angle)
        return True

    def move_all_smooth(self, target_positions: dict[int, Any], delay: float = 0.01, max_step: float = 2.0) -> None:
        steps = 30
        start_angles = self.current_angles.copy()
        for i in range(1, steps + 1):
            t = i / steps
            for ch, target in target_positions.items():
                if ch in self.current_angles:
                    curr = start_angles[ch] + (target - start_angles[ch]) * t
                    self.current_angles[ch] = curr
                    self._update_target_pos(ch, curr)
            self.robot.control_dofs_position(self.target_positions)
            self.scene.step()
            time.sleep(0.005)

    def go_home_smooth(self, delay: float = 0.01, max_step: float = 1.0) -> None:
        print("Returning to HOME...")
        self.move_all_smooth(self.HOME_POSITION)

    def go_grab_smooth(self, delay: float = 0.01, max_step: float = 1.0) -> None:
        print("Moving to GRAB position...")
        self.move_all_smooth(self.GRAB_POSITION)

# 3. Inject SRobotArm into sys.modules BEFORE importing scanner/pointnet
# This satisfies the requirement that scanner imports from 'robot'
robot_mock = types.ModuleType("robot")
robot_mock.RobotArm = SRobotArm  # type: ignore
sys.modules["robot"] = robot_mock

# 4. Project Imports - Now safe to import because 'robot' is mocked
from scanner import scan_sequence
from pointnet import PointNet

def main():
    print("=== STARTING DARKBOT PIPELINE ===")

    print("\n[1/9] Initializing Robot...")
    arm = SRobotArm()
    
    print("\n[2/9] Smooth Transition to Grab Position")
    arm.go_grab_smooth()

    print("\n[3/9] Executing Scan")
    live_measurements = scan_sequence(arm, slice=4) # type: ignore
    print(f"\nScan complete. Gathered {len(live_measurements)} data points:")
    for m in live_measurements:
        print(f"  -> {m}")

    print("\n[4/9] Grabbing the object based on scan data...")
    if live_measurements:
        widths = [m[1] for m in live_measurements]
        avg_measurement = sum(widths) / len(widths)
        
        scale_factor = 25.0 
        calculated_angle = int(avg_measurement * scale_factor)
        
        optimal_grip_angle = max(90, min(250, calculated_angle))
        print(f"Calculated optimal grip angle: {optimal_grip_angle} from scan average: {avg_measurement:.2f}")
    else:
        print("No scan data received. Defaulting to safe grip angle.")
        optimal_grip_angle = 150
        
    arm.move_smooth(0, optimal_grip_angle)
    time.sleep(0.5)

    print("\n[5/9] Returning arm to HOME position...")
    arm.go_home_smooth()

    print("\n[6/9] Initializing PointNet with Live Data...")
    height_val = 5.5 #cm
    obj_name = "starget"
    darkbot = PointNet(measurements=live_measurements, height=height_val)

    print("\n[7/9] Generating 3D Meshes and URDF...")
    stl_path, urdf_path = darkbot.export(obj_name, scale=0.1)

    print("\n[8/9] Moving to grab position again...")
    arm.go_grab_smooth()

    print("\n[9/9] Placing the object...")
    place_target = arm.GRAB_POSITION.copy()
    place_target[1] = 180 
    arm.move_all_smooth(place_target)
    time.sleep(0.5)

    print("Releasing object...")
    arm.move_smooth(0, 250)
    time.sleep(1.0)

    print("\n=== PIPELINE COMPLETE ===")
    time.sleep(5)

if __name__ == "__main__":
    main()