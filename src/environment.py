import sys
import math
import os
import xml.etree.ElementTree as ET

import trimesh

import numpy as np

import torch
from torch import Tensor

import genesis as gs
from genesis import Scene
from genesis.utils.geom import trans_quat_to_T, transform_quat_by_quat, quat_to_R

from rsl_rl.env import VecEnv

from tensordict import TensorDict

from typing import Any, cast, override

from config import DJoint

def get_urdf_meshes(urdf_path: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    base_dir = os.path.dirname(urdf_path)
    mesh_paths = []
    
    for mesh in root.findall(".//collision/geometry/mesh") + root.findall(".//visual/geometry/mesh"):
        filename = mesh.get('filename')
        if filename:
            if filename.startswith("package://"):
                filename = filename.replace("package://", "")
            full_path = os.path.join(base_dir, filename)
            scale_str = mesh.get('scale', '1.0 1.0 1.0')
            scale = np.array([float(s) for s in scale_str.split()])
            mesh_paths.append((full_path, scale))
            
    return list({(p, tuple(s)) for p, s in mesh_paths})

class GraspEnv(VecEnv):
    def __init__(self, CONFIG: dict[str, Any]) -> None :
        # self.num_envs: int = CONFIG.get("num_envs", 1)
        self.cfg: dict[str, Any] | object = CONFIG
        self.num_envs: int = self.cfg.get("num_envs", 1)
        self.num_actions: int = self.cfg.get("num_actions", 2)
        self.max_episode_length: int | Tensor = cast(int, self.cfg.get("max_episode_length", 300))
        self.episode_length_buf: torch.Tensor = torch.zeros(self.num_envs, dtype=torch.long, device=gs.device)
        # pyrefly: ignore [bad-assignment]
        self.device = gs.device
        
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res = self.cfg.get("image_resolution", (1280, 960)),
                camera_pos=(.3, -0.1, 0.3),
                camera_lookat=(0.1, -.1, 0.1),
                camera_fov=55,
            ),
            rigid_options=gs.options.RigidOptions(
                noslip_iterations=5,
            ),
            show_viewer=self.cfg.get("show_viewer", False),
            profiling_options=gs.options.ProfilingOptions(
                show_FPS=False,
            ),
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = Manipulator(self.scene, self.cfg)
        
        obj_type = self.cfg["object_type"]
        obj_kwargs = cast(dict[str, Any], self.cfg.get("object_configs", {}).get(obj_type, {}))
        spawn_pos = self.cfg.get("object_spawn_pos", (-0.0061, -0.0617, 0.015))
        # moved it to a different location for testing purpose only to test the final left finger position without box
        # spawn_pos = (-0.0061, -0.0617, 0.015)
        self.spawn_pos = torch.tensor(spawn_pos, device=gs.device)
        if obj_type == "urdf":
            morph = gs.morphs.URDF(pos=spawn_pos, **obj_kwargs)
        else:
            morph = gs.morphs.Box(pos=spawn_pos, **obj_kwargs)

        self.object = self.scene.add_entity(
            morph,
            material=gs.materials.Rigid(friction=2.0),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.0, 0.0),
                ),
            ),
            vis_mode=self.cfg.get("vis_mode", None),
        )

        self._debug_draw: bool = self.cfg.get("debug_draw", False)
        self._debug_dashboard: bool = self.cfg.get("debug_dashboard", False)
        self.debugline: bool = self.cfg.get("debugline", False)

        self.scene.build(n_envs=self.num_envs, env_spacing=(1, 1))
        
        print(f"The mass of the object is: {self.object.get_mass()} kg")
        
        # Make the gripper joints much stiffer so they can apply enough force to hold the object
        _kp = self.robot._robot_entity.get_dofs_kp()
        _kp[:] = 500.0  # stiffer arm
        _kp[DJoint.GRIPPER_LEFT] = 5000.0
        _kp[DJoint.GRIPPER_RIGHT] = 5000.0
        self.robot._robot_entity.set_dofs_kp(_kp)
        
        _kv = self.robot._robot_entity.get_dofs_kv()
        _kv[:] = 20.0
        _kv[DJoint.GRIPPER_LEFT] = 100.0
        _kv[DJoint.GRIPPER_RIGHT] = 100.0
        self.robot._robot_entity.set_dofs_kv(_kv)
        
        # Cache local periphery points
        if obj_type == "urdf":
            urdf_path = cast(str, obj_kwargs.get("file"))
            if not os.path.isabs(urdf_path):
                urdf_path = os.path.join(os.path.dirname(__file__), "..", urdf_path)
                
            meshes_info = get_urdf_meshes(urdf_path)
            if meshes_info:
                mesh_file, scale = meshes_info[0]
                t_mesh = cast(trimesh.Trimesh, trimesh.load(mesh_file))
                t_mesh.apply_scale(scale)
                path2d = t_mesh.projected(normal=[0, 0, 1])
                boundary_points = path2d.discrete[0][:-1]  # Exclude the last point since it matches the first
                
                pts_3d = np.zeros((len(boundary_points), 3))
                pts_3d[:, :2] = boundary_points
                self._local_periphery = torch.tensor(pts_3d, dtype=torch.float32, device=gs.device)
            else:
                self._local_periphery = torch.zeros((4, 3), device=gs.device)
        else:
            # Fallback for box shape
            size: list[float] = cast(list[float], obj_kwargs.get("size", [0.03, 0.03, 0.03]))
            lx, ly = size[0], size[1]
            self._local_periphery = torch.tensor([
                [lx/2, ly/2, 0],
                [-lx/2, ly/2, 0],
                [-lx/2, -ly/2, 0],
                [lx/2, -ly/2, 0]
            ], dtype=torch.float32, device=gs.device)
    
    @override
    def step(self, actions: Tensor) -> tuple[TensorDict, Any, Tensor, dict[str, Any]]:
        """
        actions: [num_envs, 2] tensor containing [angle, squeeze_width]
        """
        self._last_action = actions
        
        # 1. Rotation [RL]
        self.robot.apply_rotation_action(actions[:, 0])
        for _ in range(120):
            self.scene.step()

        # 2. Move to grasp position
        self.robot.move_to_grasp_position()
        for _ in range(120):
            self.scene.step()


        # actions[:, 1] = torch.full_like(actions[:, 1], -1.0)

        # 3. Squeeze [RL]
        self.robot.apply_squeeze_action(actions[:, 1])
        for _ in range(120):
            self.scene.step()


        # 4. Lift up [Manual] — multi-stage to prevent flinging the object
        grasp_shoulder = 1.7453
        final_shoulder = 0.34
        lift_steps = 4
        for i in range(1, lift_steps + 1):
            t = i / lift_steps
            shoulder = grasp_shoulder + t * (final_shoulder - grasp_shoulder)
            self.robot.move_to_lift_position(shoulder_target=shoulder, gripper_action=actions[:, 1])
            for _ in range(50):
                self.scene.step()

        # 3. Evaluate Reward
        pos = self.object.get_pos()
        # Example condition: Z is above 0.1 
        # [on the ground it says ~0.015, and ~0.2 is kind of at the max]
        reward = (pos[:, 2] > 0.1).float() 
        self._last_reward = reward
        
        # update time
        self.episode_length_buf += 1
        
        if self.debugline:
            # Print the last value of the left gripper finger
            qpos = self.robot._robot_entity.get_qpos()
            left_gripper_val = qpos[:, DJoint.GRIPPER_LEFT]
            print(f"Last left gripper finger value: {left_gripper_val.tolist()}")
        
        # 4. End Episode
        # For a single-step approach, every step is a terminal evaluation state.
        done = torch.ones(self.num_envs, dtype=torch.bool, device=gs.device)

        info: dict[str, Any] = {
            # 0.0 because it's a true terminal state (success/failure), not a truncation timeout
            "time_outs": torch.zeros(self.num_envs, dtype=torch.float32, device=gs.device)
        }
        
        # RL frameworks expect the environment to auto-reset when 'done' is True.
        if done.any():
            self._reset_idx(done)
            self.episode_length_buf[done] = 0
        
        # Return same initial observation, reward, done, and info
        return self.get_observations(), reward, done, info

    def reset(self)-> TensorDict:
        self._reset_idx()
        return self.get_observations()

    def _reset_idx(self, envs_idx: None | Tensor = None)-> None:
        """Reset specified environments.

        Parameters
        ----------
        envs_idx : torch.Tensor or None
            Boolean mask of shape (num_envs,) or integer indices for selective reset, or None for full reset.
        """
        # Safely convert boolean masks to integer indices for Genesis API compatibility
        if envs_idx is not None and envs_idx.dtype == torch.bool:
            envs_idx = envs_idx.nonzero(as_tuple=False).flatten()
            if len(envs_idx) == 0:
                return

        self.robot.reset(envs_idx)


        # >>>>>>>>>>>>>>>>> RESET OBJECT >>>>>>>>>>>>>>>>>
        pos = self.spawn_pos.clone().detach().to(gs.device).expand(self.num_envs, -1)

        # Get random value of -45 to +45 (in radians) | pi = 180
        random_yaw = (
            torch.rand(self.num_envs) * 2 * math.pi - math.pi
        ) * 0.25

        q_yaw = torch.stack(
            [
                torch.cos(random_yaw / 2),
                torch.zeros(self.num_envs),
                torch.zeros(self.num_envs),
                torch.sin(random_yaw / 2),
            ],
            dim=-1,
        )

        q_downward = torch.tensor([0.0, 1.0, 0.0, 0.0]).expand(self.num_envs, -1)
        goal_yaw = transform_quat_by_quat(q_yaw, q_downward)
        
        if envs_idx is None:
            self.object.set_pos(pos, skip_forward=True)
            self.object.set_quat(goal_yaw, skip_forward=False)
        else:
            self.object.set_pos(pos, envs_idx=envs_idx, skip_forward=True)
            self.object.set_quat(goal_yaw, envs_idx=envs_idx, skip_forward=False)
        
        # <<<<<<<<<<<<<<<<<<<<< RESET OBJECT <<<<<<<<<<<<<<<<<<<<
        



    @override
    def get_observations(self)-> TensorDict:
        pos = self.object.get_pos() # [num_envs, 3]
        quat = self.object.get_quat() # [num_envs, 4]
        
        # 1. Height
        height = pos[:, 2:3]
        
        # 2. 2D Periphery relative to object center
        R = quat_to_R(quat) # [num_envs, 3, 3]
        
        # Rotate corners
        rotated_periphery = torch.einsum('nij,kj->nki', R, self._local_periphery)
        
        # Extract x,y and flatten
        num_points = self._local_periphery.shape[0]
        periphery_2d = rotated_periphery[:, :, :2].reshape(self.num_envs, num_points * 2)
        
        # 3. Gripper State (Wrist rotation and gripper left)
        qpos = self.robot._robot_entity.get_qpos()
        angle = qpos[:, DJoint.WRIST_ROLL:DJoint.WRIST_ROLL+1] 
        width = qpos[:, DJoint.GRIPPER_LEFT:DJoint.GRIPPER_LEFT+1] 
        gripper_state = torch.cat([angle, width], dim=-1)

        # Draw the periphery for env_idx 0 in the 3D viewer
        if self._debug_draw:
            world_periphery_0 = rotated_periphery[0] + pos[0]
            self.draw_debug_periphery(world_periphery_0)

        # --- DASHBOARD ---
        if self._debug_dashboard:
            sys.stdout.write("\033[H\033[J") # Move to home, clear screen
            print("=== DARKBOTX ENVIRONMENT DASHBOARD ===")
            if hasattr(self, '_last_action') and self._last_action is not None:
                print(f"Action:          {self._last_action[0].tolist()}")
            if hasattr(self, '_last_reward') and self._last_reward is not None:
                print(f"Reward:          {self._last_reward[0].item():.4f}")
            print(f"pos[0]:          {pos[0].tolist()}")
            print(f"quat[0]:         {quat[0].tolist()}")
            print(f"height[0]:       {height[0].item():.4f}")
            print(f"periphery_2d:    {periphery_2d[0].tolist()}")
            print(f"gripper_state:   {gripper_state[0].tolist()}")
            print("======================================")
            sys.stdout.flush()
        
        return TensorDict({
            "object_2d_profile": periphery_2d,
            "object_height": height,
            "current_gripper_state": gripper_state
        }, batch_size=[self.num_envs])


    def draw_debug_periphery(self, world_periphery: torch.Tensor, radius: float = 0.005, color: tuple[float, float, float, float]=(0.0, 1.0, 0.0, 1.0)):
        """Draws the 2D periphery points in the Genesis viewer for debugging."""
        if hasattr(self.scene, 'clear_debug_objects'):
            self.scene.clear_debug_objects()
            
        if hasattr(self.scene, 'draw_debug_spheres'):
            self.scene.draw_debug_spheres(poss=world_periphery.cpu().numpy(), radius=radius, color=color)

    def draw_debug_frame(self, pos: torch.Tensor, quat: torch.Tensor, axis_len: float = 0.05, radius: float = 0.002, env_idx: int = 0):
        """Draws a coordinate frame using Genesis's built-in draw_debug_frame."""
        # Convert position and quaternion to a 4x4 transformation matrix T
        pos_slice = pos[env_idx:env_idx+1]
        quat_slice = quat[env_idx:env_idx+1]
        T = trans_quat_to_T(pos_slice, quat_slice)[0]
        
        # Genesis's internal method requires CPU numpy arrays for drawing
        self.scene.draw_debug_frame(T.cpu().numpy(), axis_length=axis_len, axis_radius=radius)


class Manipulator:
    def __init__(self, scene: Scene, cfg: dict[str, Any]) -> None:
        morph = gs.morphs.URDF(file=cfg["path"], fixed=True)
        material = gs.materials.Rigid(friction=2.0)
        self._robot_entity = scene.add_entity(material=material, morph=morph)
        
        _init_qpos_deg = torch.tensor(cfg["rl_start_angles"], dtype=torch.float32, device=gs.device)
        self._init_qpos = torch.deg2rad(_init_qpos_deg)
        self._limits_cache: dict[int, tuple[float, float]] = {}

    def _get_limit(self, dof_idx: int) -> tuple[float, float]:
        if not self._limits_cache:
            for joint in self._robot_entity.joints:
                if len(joint.dofs_idx_local) > 0:
                    idx = joint.dofs_idx_local[0]
                    limit = joint.dofs_limit
                    if limit[0] is not None:
                        self._limits_cache[idx] = (float(limit[0][0]), float(limit[0][1]))
        return self._limits_cache[dof_idx]

    
    def reset(self, envs_idx: None | Tensor = None, skip_forward: bool =True) -> None:
        self._robot_entity.set_qpos(
            self._init_qpos,
            envs_idx=envs_idx,
            zero_velocity=True,
            skip_forward=skip_forward,
        )
        # Establish PD controller targets so all joints are actively held in place
        # Exclude GRIPPER_RIGHT (mimic joint) — its motion is driven by the mimic
        # constraint on GRIPPER_LEFT. Setting a controller target on it would fight
        # the constraint and prevent GRIPPER_LEFT from reaching its target.
        controlled_dofs = [j.value for j in DJoint if j != DJoint.GRIPPER_RIGHT]
        self._robot_entity.control_dofs_position(
            position=self._init_qpos[controlled_dofs].expand(self._robot_entity.get_qpos().shape[0], -1),
            dofs_idx_local=controlled_dofs,
            envs_idx=envs_idx,
        )

    def rotation_scale(self, action: Tensor, motor_idx: int) -> Tensor:
        action_clipped = torch.clamp(action, min=-1.0, max=1.0)
        
        lower_rad, upper_rad = self._get_limit(motor_idx)
        lower_rad = lower_rad / 2.0
        upper_rad = upper_rad / 2.0
        return lower_rad + ((action_clipped + 1.0) / 2.0) * (upper_rad - lower_rad)
    
    def prismatic_scale(self, action: Tensor, motor_idx: int) -> Tensor:
        action_clipped = torch.clamp(action, min=-1.0, max=1.0)
        
        lower_m, upper_m = self._get_limit(motor_idx)
        return lower_m + ((action_clipped + 1.0) / 2.0) * (upper_m - lower_m)

    def apply_rotation_action(self, rotation: Tensor)-> None:
        """
        Apply the rotation action from the RL agent.
        """
        targets: dict[DJoint, int | float | Tensor]  = {
            DJoint.WRIST_ROLL: self.rotation_scale(rotation, DJoint.WRIST_ROLL),
        }
        self.control_motors(targets)

    def apply_squeeze_action(self, squeeze: Tensor)-> None:
        """
        Apply the squeeze action from the RL agent.
        """
        targets: dict[DJoint, int | float | Tensor]  = {
            DJoint.GRIPPER_LEFT: self.prismatic_scale(squeeze, DJoint.GRIPPER_LEFT),
        }
        self.control_motors(targets)

    def move_to_grasp_position(self):
        targets: dict[DJoint, int | float | Tensor] = {
            # HiwonderJoint.BASE: 0.0,
            DJoint.SHOULDER: 1.7453, # 100 degrees
            DJoint.ELBOW: 0.1745, # 10 degrees
            # HiwonderJoint.WRIST: 0.1745,
            # HiwonderJoint.WRIST_ROLL: 0.1745,
            DJoint.GRIPPER_LEFT: 0.02, # fully open (meters)
        }
        self.control_motors(targets)

    def move_to_lift_position(self, shoulder_target: float = 0.34, gripper_action: Tensor | None = None):
        targets: dict[DJoint, int | float | Tensor] = {
            DJoint.SHOULDER: shoulder_target,
        }
        if gripper_action is not None:
            targets[DJoint.GRIPPER_LEFT] = self.prismatic_scale(gripper_action, DJoint.GRIPPER_LEFT)
            
        self.control_motors(targets)


    def control_motors(self, targets: dict[DJoint, int | float | Tensor]) -> None:
        """
        Controls multiple motors at once without changing the target state of other motors.
        
        Args:
            targets: A dictionary mapping motor indices to target values (radians for revolute, meters for prismatic).
        """
        dof_indices = list(targets.keys())
        
        # Build a [num_envs, len(targets)] position tensor
        vals: list[Tensor] = []
        for idx in dof_indices:
            v: Tensor | float | int = targets[idx]
            if isinstance(v, Tensor):
                vals.append(v.unsqueeze(-1) if v.dim() == 1 else v)
            else:
                vals.append(torch.full((self._robot_entity.get_qpos().shape[0], 1), float(v), device=gs.device))
        
        position = torch.cat(vals, dim=-1)
        
        self._robot_entity.control_dofs_position(
            position=position,
            dofs_idx_local=dof_indices,
        )


