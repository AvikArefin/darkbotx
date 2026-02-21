import math
import torch
import genesis as gs
from rsl_rl.env import VecEnv
from tensordict import TensorDict

# Following the RSL RL API
class FastFrankaEnv(VecEnv):
    def __init__(self, env_cfg: dict, reward_cfg: dict, robot_cfg: dict, show_viewer: bool = False):
        # RSL-RL required attributes
        self.show_viewer = show_viewer
        self.is_monitor : bool = env_cfg["is_monitor"]
        self.num_envs : int = env_cfg["num_envs"] 
        self.num_actions : int = env_cfg["num_actions"]
        self.num_obs : int = env_cfg["num_obs"]
        self.num_privileged_obs = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.ctrl_dt : float = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        # configs
        self.cfg : dict = env_cfg
        self.reward_scales : dict = reward_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)

        # Tracking buffers for RSL-RL logic
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        gs.init(
            seed=None,
            precision="32",
            logging_level=env_cfg["logging_level"],
            debug=env_cfg["is_debug"],
            performance_mode=env_cfg["performance_mode"],
            backend=gs.gpu if self.device.type != "cpu" else gs.cpu,
        )

        # === SETUP SCENE ===
        if self.device.type == "cuda":
            scene_renderer = gs.options.renderers.BatchRenderer(
                use_rasterizer=env_cfg["use_rasterizer"],
            )
        else:
            scene_renderer = gs.renderers.Rasterizer()

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.ctrl_dt,
                substeps=10,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(self.num_envs, 10)))),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(512, 512),
            ),
            profiling_options=gs.options.ProfilingOptions(
                show_FPS=env_cfg["show_FPS"]
            ),
            renderer=scene_renderer,
            show_viewer=show_viewer,
        )
        
        # === ADD ELEMENTS TO THE SCENE ===
        self.scene.add_entity(gs.morphs.Plane())

        # == robot ==
        self.init_robot_dof_pos = torch.tensor(robot_cfg["home_pos"], device=self.device)
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file='xml/franka_emika_panda/panda.xml', 
                pos = (0.0, 0.0, 0.0),
            )
        )

        # == target ==
        self.init_target_pos = torch.tensor([0.5, 0.5, 0.0], device=self.device)
        self.target = self.scene.add_entity(
            gs.morphs.URDF(
                file='assets/DarkCube/DarkCube.urdf', 
            )
        )

        # build scene
        self.env_spacing = 1.5
        self.scene.build(
            n_envs=self.num_envs, 
            env_spacing=(self.env_spacing, self.env_spacing)
        )

        # == robot ==
        # pos range
        lower, upper = self.robot.get_dofs_limit()
        self.dof_lower = lower.to(self.device)
        self.dof_upper = upper.to(self.device)

        # force range
        force_lower, force_upper = self.robot.get_dofs_force_range()
        self.dof_force_upper = force_upper.to(self.device)
        self.dof_force_lower = force_lower.to(self.device)

        if True:
            self._set_pd_gains()
            self.analyze_robot()

        # == target ==
        lower_bound, upper_bound = self.target.get_AABB(envs_idx=0)
        self.init_target_pos[2] = (upper_bound[2] - lower_bound[2]) / 2.0
        self.target.set_pos(self.init_target_pos)

    def _set_pd_gains(self):
        # Set up PD gains for all joints
        # kp: how hard it pulls toward the target (stiffness)
        # kv: how much it resists motion (damping/viscosity)

        self.robot.set_dofs_kp(
            torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100], device=self.device),
        )

        self.robot.set_dofs_kv(
            torch.tensor([450, 450, 350, 350, 200, 200, 200, 10, 10], device=self.device),
        )

        self.robot.set_dofs_force_range(
            self.dof_force_lower,
            self.dof_force_upper,
        )

    def analyze_robot(self):
        """
        Analyzes the robot's properties.
        """
        self.robot_mass = sum(link.inertial_mass for link in self.robot.links)
        print(f"Calculated total robot mass: {self.robot_mass:.2f} kg")

        print("\n--- Link Inertial Properties ---")
        for link in self.robot.links:
            print(f"\nLink: {link.name}")
            print(f"  Mass: {link.inertial_mass}")
            print(f"  Inertia tensor:\n{link.inertial_i}")
            print(f"  COM position: {link.inertial_pos}")
            print(f"  COM orientation (quat): {link.inertial_quat}")
        print("---------------------------------\n")

    def reset_at(self, env_ids):
        print(f"RESET AT: {env_ids}")
        # robot
        self.robot.set_dofs_position(self.init_robot_dof_pos, envs_idx=env_ids)
        self.robot.set_dofs_velocity(torch.zeros_like(self.init_robot_dof_pos), envs_idx=env_ids)

        # target
        self.target.set_pos(self.init_target_pos, envs_idx=env_ids)

    def get_observations(self) -> TensorDict:
        """Fetches the current state of the robot and target."""
        dofs_pos = self.robot.get_dofs_position()
        target_pos = self.target.get_pos()
        
        # Combine robot state + target state
        obs = torch.cat([dofs_pos, target_pos], dim=-1)
        
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device) 

    def _compute_reward(self):
        """Calculates rewards and termination conditions."""
        rewards = 0
        ee_pos = (self.robot.get_link('left_finger').get_pos() + self.robot.get_link('right_finger').get_pos()) / 2
        target_pos = self.target.get_pos()
        
        # time penalty
        rewards += -0.1

        # Distance from robot gripper to target
        dist = torch.norm(ee_pos - target_pos, dim=-1)
        rewards += -dist  

        # Horizontal (XY) displacement of target
        target_xy_dist = torch.norm(target_pos[:, :2] - self.init_target_pos[:2], dim=-1)
        
        # When sliding, in most cases, it does not stay on the ground
        # is_on_ground = target_pos[:, 2] < 0.03
        did_slide = (target_xy_dist > 0.02)

        # print(f"------------{(target_xy_dist > 0.02)} - {is_on_ground} {is_sliding}")
        rewards += -did_slide.float() * 10.0
        
        # is_success
        # Calculate Task Terminations
        termination_dones = (dist < 0.05) | did_slide
        
        return rewards, termination_dones 

    def _debug_vis(self) -> None:
        ee_pos = (self.robot.get_link('left_finger').get_pos() + self.robot.get_link('right_finger').get_pos()) / 2
        target_pos = self.target.get_pos()
        
        self.scene.clear_debug_objects()
        num_rendered = min(self.num_envs, 10) 
        
        # --- Simplified Grid Math ---
        # Assuming self.env_spacing is a tuple like (5.0, 5.0), we just take the first value
        spacing = self.env_spacing[0] if isinstance(self.env_spacing, tuple) else self.env_spacing
        
        n_cols = math.ceil(math.sqrt(self.num_envs))
        n_rows = math.ceil(self.num_envs / n_cols)
        
        # Calculate the "center index" (how many rows/cols to shift back)
        center_x_idx = (n_rows - 1) / 2.0
        center_y_idx = (n_cols - 1) / 2.0

        for i in range(num_rendered):
            row = i // n_cols
            col = i % n_cols
            
            # Clean, single-multiplier offset calculation
            visual_offset = torch.tensor([
                (row - center_x_idx) * spacing, 
                (col - center_y_idx) * spacing, 
                0.0
            ], device=self.device)
            
            global_start = target_pos[i] + visual_offset
            global_end = ee_pos[i] + visual_offset

            self.scene.draw_debug_line(
                start=global_start, 
                end=global_end,
                color=(0, 1, 0),
            )       

    def step(self, actions):
        # 1. Apply actions (Delta Control)
        current_dofs_pos = self.robot.get_dofs_position()
        target_dofs_pos = self.dof_lower + actions * (self.dof_upper - self.dof_lower)
        target_dofs_pos = torch.clamp(target_dofs_pos, self.dof_lower, self.dof_upper)

        self.robot.control_dofs_position(target_dofs_pos)
        self.scene.step()

        # 2. Debug Visualization (Optional)
        if self.show_viewer:
            self._debug_vis()

        # Compute Rewards and Dones
        rewards, termination_dones = self._compute_reward()
        
        # Handle Timeouts & Total Dones
        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length
        total_dones = termination_dones | time_outs
        
        # Automatic Resets
        if total_dones.any():
            env_ids = total_dones.nonzero(as_tuple=False).flatten()
            self.reset_at(env_ids)
            self.episode_length_buf[env_ids] = 0
        
        # Fetch Final Observations
        obs = self.get_observations()

        if self.is_monitor:
            dofs_pos = self.robot.get_dofs_position()
            target_pos = self.target.get_pos()
            ee_pos = (self.robot.get_link('left_finger').get_pos() + self.robot.get_link('right_finger').get_pos()) / 2
            dist = torch.norm(ee_pos - target_pos, dim=-1)

            infos = {
                "time_outs": time_outs,
                "dofs_pos": dofs_pos,
                "target_pos": target_pos,
                "dist": dist,
            }
        else:
            infos = {
                "time_outs": time_outs
            }
        
        return obs, rewards, total_dones, infos 

    def reset(self):
        # Reset all environments
        self.reset_at(torch.arange(self.num_envs, device=self.device))
        self.episode_length_buf.zero_()
        
        # Return initial observations
        return self.get_observations(), {} 
