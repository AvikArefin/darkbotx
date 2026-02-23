import math
import torch
import genesis as gs
from rsl_rl.env import VecEnv
from tensordict import TensorDict

# custom import
from code.logger_setup import setup_logger

# logger setup
logger = setup_logger(__name__)

def _check_nan(tensor: torch.Tensor, name: str, context: str = "") -> bool:
    """Returns True if NaN detected, logs details."""
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        nan_indices = torch.isnan(tensor).nonzero(as_tuple=False)
        logger.error(
            f"[NaN DETECTED] {name} | context={context} | "
            f"count={nan_count}/{tensor.numel()} | "
            f"first indices={nan_indices[:5].tolist()} | "
            f"shape={tuple(tensor.shape)} | "
            f"min={tensor[~torch.isnan(tensor)].min().item() if (~torch.isnan(tensor)).any() else 'all nan'} | "
            f"max={tensor[~torch.isnan(tensor)].max().item() if (~torch.isnan(tensor)).any() else 'all nan'}"
        )
        return True

    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        logger.error(
            f"[INF DETECTED] {name} | context={context} | "
            f"count={inf_count}/{tensor.numel()} | "
            f"shape={tuple(tensor.shape)}"
        )
        return True
    return False

# Following the RSL RL API
class FastFrankaEnv(VecEnv):
    def __init__(self, env_cfg: dict, reward_cfg: dict, robot_cfg: dict, show_viewer: bool = False):
        # INFO: RSL-RL required attributes
        self.show_viewer : bool = show_viewer
        self.is_monitor : bool = env_cfg["is_monitor"]
        self.num_envs : int = env_cfg["num_envs"] 
        self.num_actions : int = env_cfg["num_actions"]
        self.num_obs : int = env_cfg["num_obs"]
        self.num_privileged_obs = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.ctrl_dt : float = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        self.success_range = 0.15 # WARN: This property will be deprecated in future.  

        # INFO: configs
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

        # INFO: SETUP SCENE
        if self.device.type == "cuda":
            scene_renderer = gs.options.renderers.BatchRenderer(
                use_rasterizer=env_cfg["use_rasterizer"],
            )
        else:
            scene_renderer = gs.renderers.Rasterizer()

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.ctrl_dt,
                substeps=env_cfg["substeps"],
            ),
            rigid_options=gs.options.RigidOptions(
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
        
        # INFO: ADD ELEMENTS TO THE SCENE
        self.scene.add_entity(gs.morphs.Plane())

        # NOTE: robot
        self.init_robot_dof_pos = torch.tensor(robot_cfg["home_pos"], device=self.device)
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file='xml/franka_emika_panda/panda.xml', 
                pos = (0.0, 0.0, 0.0),
            )
        )

        # NOTE: target 
        self.init_target_pos = torch.tensor([0.5, 0.5, 0.0], device=self.device)
        self.r_min, self.r_max = 0.3, 0.8
        self.target = self.scene.add_entity(
            gs.morphs.URDF(
                file='assets/DarkCube/DarkCube.urdf', 
            )
        )

        # NOTE: build scene
        self.env_spacing = 1.5
        self.scene.build(
            n_envs=self.num_envs, 
            env_spacing=(self.env_spacing, self.env_spacing)
        )

        # NOTE: robot 
        # pos range
        lower, upper = self.robot.get_dofs_limit()
        self.dof_lower = lower.to(self.device)
        self.dof_upper = upper.to(self.device)

        _check_nan(self.dof_lower, "dof_lower", "__init__")
        _check_nan(self.dof_upper, "dof_upper", "__init__")

        # force range
        force_lower, force_upper = self.robot.get_dofs_force_range()
        self.dof_force_upper = force_upper.to(self.device)
        self.dof_force_lower = force_lower.to(self.device)

        self._set_pd_gains()
        # self.analyze_robot()

        # NOTE: target
        lower_bound, upper_bound = self.target.get_AABB(envs_idx=0)
        self.init_target_pos[2] = (upper_bound[2] - lower_bound[2]) / 2.0
        self.target.set_pos(self.init_target_pos)

        if show_viewer:
            self._init_vis_offsets()

        self.nan_counter = 0

    # INFO: INIT HELPERS
    def _set_pd_gains(self):
        # Set up PD gains for all joints
        # kp: how hard it pulls toward the target (stiffness)
        # kv: how much it resists motion (damping/viscosity)

        try:
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
        except Exception as e:
            logger.exception(f"[_set_pd_gains ERROR] {e}")

    def analyze_robot(self):
        """
        Analyzes the robot's properties.
        """
        try:
            self.robot_mass = sum(link.inertial_mass for link in self.robot.links)
            logger.debug(f"Calculated total robot mass: {self.robot_mass:.2f} kg")

            logger.debug("\n--- Link Inertial Properties ---")
            for link in self.robot.links:
                logger.debug(f"\nLink: {link.name}")
                logger.debug(f"  Mass: {link.inertial_mass}")
                logger.debug(f"  Inertia tensor:\n{link.inertial_i}")
                logger.debug(f"  COM position: {link.inertial_pos}")
                logger.debug(f"  COM orientation (quat): {link.inertial_quat}")
            logger.debug("---------------------------------\n")
        except Exception as e:
            logger.exception(f"[DEBUG_VIS ERROR] {e}")

    def _init_vis_offsets(self):
        """Precomputes the (num_rendered, 3) grid-offset tensor once on GPU."""
        try:
            num_rendered = min(self.num_envs, 10)
            spacing      = self.env_spacing[0] if isinstance(self.env_spacing, tuple) else self.env_spacing
            n_cols       = math.ceil(math.sqrt(self.num_envs))
            n_rows       = math.ceil(self.num_envs / n_cols)
            cx, cy       = (n_rows - 1) / 2.0, (n_cols - 1) / 2.0

            offsets = [
                [(i // n_cols - cx) * spacing, (i % n_cols - cy) * spacing, 0.0]
                for i in range(num_rendered)
            ]
            self.vis_offsets = torch.tensor(offsets, dtype=torch.float32, device=self.device)
        except Exception as e:
            logger.exception(f"[_init_vis_offsets ERROR] {e}")

    def _debug_vis(self):
        try:
            ee_pos = (
                self.robot.get_link("left_finger").get_pos()
                + self.robot.get_link("right_finger").get_pos()
            ) / 2.0

            target_pos   = self.target.get_pos()
            num_rendered = self.vis_offsets.shape[0]

            self.scene.clear_debug_objects()
            for i in range(num_rendered):
                offset = self.vis_offsets[i]       # precomputed (3,) slice — no allocation
                self.scene.draw_debug_line(
                    start=target_pos[i] + offset,
                    end=ee_pos[i]       + offset,
                    color=(0, 1, 0),
                )       
        except Exception as e:
            logger.exception(f"[DEBUG_VIS ERROR] {e}")

    # INFO: CORE API
    def get_observations(self):
        """Fetches the current state of the robot and target."""
        try:
            dofs_pos = self.robot.get_dofs_position()
            dofs_vel = self.robot.get_dofs_velocity()
            ee_pos = (
                self.robot.get_link('left_finger').get_pos() 
                + self.robot.get_link('right_finger').get_pos()
            ) / 2.0
            ee_quat = self.robot.get_link("hand").get_quat()
            ee_lin_vel = self.robot.get_link("hand").get_vel()
            ee_ang_vel = self.robot.get_link("hand").get_ang()
            target_pos = self.target.get_pos()
            ee_to_target_vector = target_pos - ee_pos
            dist = torch.norm(ee_to_target_vector, dim=-1, keepdim=True)

            # Combine robot state + target state
            obs = torch.cat([
            	dofs_pos,               # (n_envs, 9)
            	dofs_vel,               # (n_envs, 9)
            	ee_pos,                 # (n_envs, 3)
            	ee_quat,                # (n_envs, 4)
            	ee_lin_vel,             # (n_envs, 3)
            	ee_ang_vel,             # (n_envs, 3)
            	target_pos,             # (n_envs, 3)
            	ee_to_target_vector,    # (n_envs, 3)
            	dist,                   # (n_envs, 1)
            ], dim=-1)                  # (n_envs, 38)

            _check_nan(obs, "obs", "get_observations")

            return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device) 
        except Exception as e:
            logger.exception(f"[GET_OBS ERROR] {e}")

    def _compute_reward(self, obs_tensor):
        """Calculates rewards and termination conditions purely from observations."""
        try:
            ee_lin_vel = obs_tensor[:, 25:28]
            ee_to_target_vector = obs_tensor[:, 34:37]
            dist = obs_tensor[:, 37]

            # NOTE: Distance Reward (Linear + Exponential)
            distance_reward = -dist + torch.exp(-10 * dist)

            # NOTE: Approach bonus: reward moving toward the target
            # Unsqueeze dist to (n_envs, 1) to match ee_lin_vel's shape for division
            directional_unit_vector = ee_to_target_vector / (dist.unsqueeze(-1) + 1e-6)
            approach_reward = (ee_lin_vel * directional_unit_vector).sum(dim=-1).clamp(min=0.0)

            # NOTE: Success reward
            is_success = dist < self.success_range
            success_reward = is_success.float() * 100.0

            # NOTE: time penalty: encouraging to find optimal path
            time_penalty = -0.5

            rewards = (
                2.0 * distance_reward
                + 0.5 * approach_reward
                + success_reward
                + time_penalty
            )
            
            termination_dones = is_success
            _check_nan(rewards, "rewards", "_compute_reward")

            return rewards, termination_dones 
            
        except Exception as e:
            logger.exception(f"[COMPUTE_REWARD ERROR] {e}")
            raise 

    def step(self, actions):
        try:
            _check_nan(actions, "actions", "step")
            
            # NOTE: Apply actions (Delta Control)
            dof_center = (self.dof_upper + self.dof_lower) / 2.0
            dof_span = (self.dof_upper - self.dof_lower) / 2.0
            target_dofs_pos = dof_center + actions * dof_span 
            _check_nan(target_dofs_pos, "target_dofs_pos (pre-clamp)", "step")

            target_dofs_pos = torch.clamp(target_dofs_pos, self.dof_lower, self.dof_upper)
            _check_nan(target_dofs_pos, "target_dofs_pos (clamped)", "step")

            self.robot.control_dofs_position(target_dofs_pos)

            # WARN: Physics step NaN recovery. Need to be improved later
            try:
                self.scene.step()
            except gs.GenesisException as physics_err:
                logger.exception(
                    f"[PHYSICS NaN] Solver diverged: {physics_err}. "
                    f"Resetting all {self.num_envs} envs and continuing."
                )
                all_ids = torch.arange(self.num_envs, device=self.device)
                self.reset_at(all_ids)
                self.episode_length_buf.zero_()
                obs = self.get_observations()
                rewards = torch.zeros(self.num_envs, device=self.device)
                dones   = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
                infos   = {"time_outs": dones}

                self.nan_counter += 1
                return obs, rewards, dones, infos 

            # Debug Visualization (Optional)
            if self.show_viewer:
                self._debug_vis()

            # NOTE: OBSERVATION & REWARD
            
            # Fetch observations once
            obs = self.get_observations()
            obs_tensor = obs["policy"]

            # Compute Rewards and Dones using the pre-fetched tensor
            rewards, termination_dones = self._compute_reward(obs_tensor)
            _check_nan(rewards, "rewards", "step")
            
            # Handle Timeouts & Total Dones
            self.episode_length_buf += 1
            time_outs = self.episode_length_buf >= self.max_episode_length
            total_dones = termination_dones | time_outs
            
            # NOTE: BUILD INFOS USING EXISTING TENSOR
            dist = obs_tensor[:, 37]
            
            infos = {
                "time_outs": time_outs,
                "episode": {
                    "nan_counter": self.nan_counter,
                    "mean_distance": dist.mean().item(),
                    "success_rate": (dist < self.success_range).float().mean().item(),
                }
            }

            if self.is_monitor:
                infos["dofs_pos"] = obs_tensor[:, 0:9]
                infos["target_pos"] = obs_tensor[:, 31:34]
                infos["dist"] = dist

            # NOTE: HANDLE RESETS (Only update obs if necessary)
            if total_dones.any():
                env_ids = total_dones.nonzero(as_tuple=False).flatten()
                self.reset_at(env_ids)
                self.episode_length_buf[env_ids] = 0
                
                # ONLY fetch observations a second time if environments were teleported
                # RSL-RL requires the returned 'obs' to reflect the post-reset state
                obs = self.get_observations()
            
            return obs, rewards, total_dones, infos 
            
        except Exception as e:
            logger.exception(f"[STEP ERROR] {e}")
            raise 

    def reset_at(self, env_ids: torch.Tensor):
        try:
            n = len(env_ids)
            # NOTE: ROBOT 
            # Expand 1-D init tensors to (n, dofs) / (n, 3) — all on GPU
            self.robot.set_dofs_position(
                self.init_robot_dof_pos.unsqueeze(0).expand(n, -1),
                envs_idx=env_ids,
            )
            self.robot.set_dofs_velocity(
                torch.zeros(n, self.num_actions, device=self.device),
                envs_idx=env_ids,
            )

            # NOTE: TARGET
            r = self.r_min + (self.r_max - self.r_min) * torch.rand(n, device=self.device)
            theta = 2 * math.pi * torch.rand(n, device=self.device)

            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            z = torch.full((n,), self.init_target_pos[2].item(), device=self.device)

            random_target_pos = torch.stack((x, y, z), dim=1)
            self.target.set_pos(
                random_target_pos,
                envs_idx=env_ids,
            )
        except Exception as e:
            logger.exception(f"[RESET_AT ERROR] env_ids={env_ids.tolist()}: {e}")
            raise

    def reset(self):
        # Reset all environments
        try:
            self.reset_at(torch.arange(self.num_envs, device=self.device))
            self.episode_length_buf.zero_()
            obs = self.get_observations()
            # Return initial observations
            return obs, {} 
        except Exception as e:
            logger.exception(f"[RESET ERROR] {e}")
            raise
