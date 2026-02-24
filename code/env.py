import math
import torch
import genesis as gs
from rsl_rl.env import VecEnv
from tensordict import TensorDict

# custom import
from code.logger_setup import setup_logger

# logger setup
logger = setup_logger(__name__)

# dummy class for wandb
class ConfigDict(dict):
    """A dictionary that can pass rsl_rl's .to_dict() check for Weights & Biases."""
    def to_dict(self):
        return dict(self)


@torch.inference_mode()
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
    @torch.no_grad()
    def __init__(self, env_cfg: dict, reward_cfg: dict, robot_cfg: dict, show_viewer: bool = False):
        # INFO: __INIT__ RSL-RL 
        self.show_viewer : bool = show_viewer
        self.is_monitor : bool = env_cfg["is_monitor"]
        self.num_envs : int = env_cfg["num_envs"] 
        self.num_actions : int = env_cfg["num_actions"]
        self.num_obs : int = env_cfg["num_obs"]
        self.num_privileged_obs = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.ctrl_dt : float = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        self.success_range : torch.Tensor = torch.tensor( 0.087, device=self.device) # WARN: This property will be deprecated in future.  

        self.env_cfg : dict = env_cfg
        self.reward_scales : dict = reward_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)

        self.cfg = ConfigDict({
            "env": env_cfg,
            "robot": robot_cfg,
            "rewards": reward_cfg
        })

        # Tracking buffers for RSL-RL logic
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)

        # INFO: __INIT__ GENESIS ENGINE
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
        # add plane to the scene
        self.scene.add_entity(gs.morphs.Plane())

        # add robot to the scene
        self.init_robot_dof_pos = torch.tensor(robot_cfg["home_pos"], device=self.device)
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file='xml/franka_emika_panda/panda.xml', 
                pos = (0.0, 0.0, 0.0),
            )
        )
        self.left_finger = self.robot.get_link("left_finger")
        self.right_finger = self.robot.get_link("right_finger")
        self.hand_link = self.robot.get_link("hand")

        # add target to the scene
        self.init_target_pos = torch.tensor([0.5, 0.5, 0.0], device=self.device)
        self.r_min, self.r_max = 0.3, 0.8
        self.target = self.scene.add_entity(
            gs.morphs.URDF(
                file='assets/DarkCube/DarkCube.urdf', 
            )
        )

        # NOTE: build the scene
        self.env_spacing = 1.5
        self.scene.build(
            n_envs=self.num_envs, 
            env_spacing=(self.env_spacing, self.env_spacing)
        )

        # NOTE: init robot params
        # pos range
        lower, upper = self.robot.get_dofs_limit()
        self.dof_lower = lower.to(self.device)
        self.dof_upper = upper.to(self.device)
        self.dof_center = ((self.dof_upper + self.dof_lower) / 2.0).to(self.device)
        self.dof_span = ((self.dof_upper - self.dof_lower) / 2.0).to(self.device)

        # force range
        force_lower, force_upper = self.robot.get_dofs_force_range()
        self.dof_force_upper = force_upper.to(self.device)
        self.dof_force_lower = force_lower.to(self.device)

        self._set_pd_gains()

        # NOTE: init target params
        lower_bound, upper_bound = self.target.get_AABB(envs_idx=0)
        self.init_target_pos[2] = (upper_bound[2] - lower_bound[2]) / 2.0
        self.target_pos = tuple(self.init_target_pos)
        self.target.set_pos(self.init_target_pos)

        # NOTE: init debug visualization
        if show_viewer:
            # self.analyze_robot()
            self._init_vis_debug()
            self._init_spawn_vis()
            self._init_success_sphere_vis()

        self.nan_counter = 0

        self.obs_buf = torch.zeros(self.num_envs, 39, device=self.device, dtype=torch.float32)

    # INFO: INIT HELPERS
    @torch.inference_mode()
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

    @torch.no_grad()
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

    @torch.no_grad()
    def _init_vis_debug(self):
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

    @torch.no_grad()
    def _init_spawn_vis(self, n_segments: int = 32):
        """Precomputes r_min and r_max circle line segments for the floor spawn area."""
        try:
            angles = torch.linspace(0, 2 * math.pi, n_segments + 1, device=self.device)
            cos_a  = torch.cos(angles)
            sin_a  = torch.sin(angles)
            z      = torch.zeros(n_segments, device=self.device)

            def make_segments(r):
                starts = torch.stack([r * cos_a[:-1], r * sin_a[:-1], z], dim=1)  # (n_seg, 3)
                ends   = torch.stack([r * cos_a[1:],  r * sin_a[1:],  z], dim=1)  # (n_seg, 3)
                return starts, ends

            self.spawn_inner_starts, self.spawn_inner_ends = make_segments(self.r_min)
            self.spawn_outer_starts, self.spawn_outer_ends = make_segments(self.r_max)

            logger.debug(f"Spawn vis initialized | r_min={self.r_min} r_max={self.r_max} segments={n_segments}")
        except Exception as e:
            logger.exception(f"[_init_spawn_vis ERROR] {e}")

    @torch.no_grad()
    def _init_success_sphere_vis(self, n_lat: int = 4, n_lon: int = 8):
        """Precomputes unit-sphere wireframe segments, scaled by success_range."""
        try:
            r = self.success_range
            segments_s, segments_e = [], []

            # Latitude rings (horizontal circles at different heights)
            for i in range(1, n_lat):
                phi   = math.pi * i / n_lat          # 0 (top) → pi (bottom)
                ring_r = r * math.sin(phi)
                z_val  = r * math.cos(phi)
                angles = torch.linspace(0, 2 * math.pi, n_lon + 1, device=self.device)
                xs = ring_r * torch.cos(angles)
                ys = ring_r * torch.sin(angles)
                zs = torch.full_like(xs, z_val)
                pts = torch.stack([xs, ys, zs], dim=1)  # (n_lon+1, 3)
                segments_s.append(pts[:-1])
                segments_e.append(pts[1:])

            # Longitude lines (vertical arcs)
            for i in range(n_lon):
                phi_vals = torch.linspace(0, math.pi, n_lat * 4, device=self.device)
                lon_angle = 2 * math.pi * i / n_lon
                xs = r * torch.sin(phi_vals) * math.cos(lon_angle)
                ys = r * torch.sin(phi_vals) * math.sin(lon_angle)
                zs = r * torch.cos(phi_vals)
                pts = torch.stack([xs, ys, zs], dim=1)
                segments_s.append(pts[:-1])
                segments_e.append(pts[1:])

            self.sphere_starts = torch.cat(segments_s, dim=0)  # (total_segs, 3)
            self.sphere_ends   = torch.cat(segments_e, dim=0)

            logger.debug(f"Success sphere vis | r={r} | total segments={self.sphere_starts.shape[0]}")
        except Exception as e:
            logger.exception(f"[_init_success_sphere_vis ERROR] {e}")

    @torch.no_grad()
    def _debug_vis(self):
        try:
            ee_pos = (self.robot.get_link("left_finger").get_pos() + self.robot.get_link("right_finger").get_pos()) / 2.0
            target_pos = self.target.get_pos()
            num_rendered = self.vis_offsets.shape[0]

            self.scene.clear_debug_objects()

            for i in range(num_rendered):
                offset = self.vis_offsets[i]  # (3,)
                offset_xy = offset.clone()
                offset_xy[2] = 0.001  # just above floor to avoid z-fighting
                t_pos = target_pos[i] + offset

                # INFO: EE → target line
                self.scene.draw_debug_line(
                    start=target_pos[i] + offset,
                    end=ee_pos[i]       + offset,
                    color=(0, 1, 0),
                )

                # INFO: Spawn area circles (precomputed, no alloc)
                for s, e in zip(self.spawn_inner_starts, self.spawn_inner_ends):
                    self.scene.draw_debug_line(start=s + offset_xy, end=e + offset_xy, color=(0.3, 0.3, 1.0))

                for s, e in zip(self.spawn_outer_starts, self.spawn_outer_ends):
                    self.scene.draw_debug_line(start=s + offset_xy, end=e + offset_xy, color=(0.0, 0.5, 1.0))

                # INFO: SUCCESS SPHERE AROUND TARGET
                # Color shifts green→red based on current distance
                for s, e in zip(self.sphere_starts, self.sphere_ends):
                    self.scene.draw_debug_line(
                        start=s + t_pos,
                        end=e   + t_pos,
                        color=(1.0, 0.0, 0.0),
                    )
        except Exception as e:
            logger.exception(f"[DEBUG VIS ERROR] {e}") 

    # INFO: CORE API
    @torch.no_grad()
    def get_observations(self):
        """Fetches the current state of the robot and target."""
        try:
            dofs_pos = self.robot.get_dofs_position()
            dofs_vel = self.robot.get_dofs_velocity()
            ee_pos = (
                self.left_finger.get_pos() 
                + self.right_finger.get_pos()
            ) / 2.0
            ee_quat = self.hand_link.get_quat()
            ee_lin_vel = self.hand_link.get_vel()
            ee_ang_vel = self.hand_link.get_ang()
            target_pos = self.target.get_pos()
            ee_to_target_vector = target_pos - ee_pos
            dist = torch.norm(ee_to_target_vector, dim=-1, keepdim=True)
            gripper_width = (dofs_pos[:, 7] + dofs_pos[:, 8]).unsqueeze(-1)

        # Combine robot state + target state
        torch.cat([
            dofs_pos,               # (n_envs, 9)
            dofs_vel,               # (n_envs, 9)
            ee_pos,                 # (n_envs, 3)
            ee_quat,                # (n_envs, 4)
            ee_lin_vel,             # (n_envs, 3)
            ee_ang_vel,             # (n_envs, 3)
            target_pos,             # (n_envs, 3)
            ee_to_target_vector,    # (n_envs, 3)
            dist,                   # (n_envs, 1)
            gripper_width,          # (n_envs, 1)
        ], dim=-1, out=self.obs_buf)                  # (n_envs, 39)

        return TensorDict({"policy": self.obs_buf.clone()}, batch_size=[self.num_envs], device=self.device) 

    @torch.compile(fullgraph=True, dynamic=False)
    def _compute_reward(self, obs_tensor: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates rewards and termination conditions purely from observations."""
        ee_lin_vel = obs_tensor[:, 25:28]
        ee_to_target_vector = obs_tensor[:, 34:37]
        dist = obs_tensor[:, 37]

        # NOTE: Distance Reward (Linear + Exponential)
        distance_reward = -5 * dist + torch.exp(-10 * dist)

        # NOTE: Approach bonus: reward moving toward the target
        # Unsqueeze dist to (n_envs, 1) to match ee_lin_vel's shape for division
        directional_unit_vector = ee_to_target_vector / (dist.unsqueeze(-1) + 1e-6)
        approach_reward = (
            (ee_lin_vel * directional_unit_vector).sum(dim=-1).clamp(min=0.0)
        )

        # TODO: Action smoothing
        # action_delta = actions - self.last_actions
        # smoothness_reward = - torch.norm(action_delta, dim=-1) * 0.05

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

        return rewards, termination_dones

    def step(self, actions: torch.Tensor):
        try:
            # NOTE: Apply actions (Delta Control)
            target_dofs_pos = torch.clamp(
                self.dof_center + actions * self.dof_span,            # Or, input = torch.addcmul(self.dof_center, actions, self.dof_span) # same thing.
                self.dof_lower,
                self.dof_upper,
            )

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
                dones = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
                infos = {"time_outs": dones}

                self.nan_counter += 1
                return obs, rewards, dones, infos

            # Debug Visualization (Optional)
            if self.show_viewer:
                self._debug_vis()
                pass

            # NOTE: OBSERVATION & REWARD
            # Fetch observations once
            obs = self.get_observations()
            obs_tensor = obs["policy"]

            # Compute Rewards and Dones using the pre-fetched tensor
            rewards, termination_dones = self._compute_reward(obs_tensor, actions)
            
            # Handle Timeouts & Total Dones
            self.episode_length_buf += 1
            time_outs = self.episode_length_buf >= self.max_episode_length
            total_dones = termination_dones | time_outs
            
            # NOTE: BUILD INFOS USING EXISTING TENSOR
            dist = obs_tensor[:, 37]
            
            infos = {
                "time_outs": time_outs,
                "episode" : {
                    "nan_counter": self.nan_counter,
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

                terminal_dist = dist[total_dones]  # only the envs that just ended

                infos["episode"] = {

                    # Average final distance across envs that just terminated
                    "terminal_mean_distance": terminal_dist.mean().item(),

                    # Breakdown: how many ended from success vs timeout
                    "success_rate": (terminal_dist < self.success_range).float().mean().item(),
                }
                
                # ONLY fetch observations a second time if environments were teleported
                # RSL-RL requires the returned 'obs' to reflect the post-reset state
                obs = self.get_observations()
            else:
                # NOTE: if termination or timout happens, reset_at handles resetting last_actions. 
                # else clone the actions to last_actions so that in next step these actions can be accessible.
                self.last_actions = actions.clone()
            
            return obs, rewards, total_dones, infos 
            
        except Exception as e:
            logger.exception(f"[STEP ERROR] {e}")
            raise 

    @torch.no_grad()
    def reset_at(self, env_ids: torch.Tensor):
        try:
            n = len(env_ids)
            # NOTE: RESET ROBOT 
            # Expand 1-D init tensors to (n, dofs) / (n, 3) — all on GPU
            self.robot.set_dofs_position(
                self.init_robot_dof_pos.unsqueeze(0).expand(n, -1),
                envs_idx=env_ids,
            )
            self.robot.set_dofs_velocity(
                torch.zeros(n, self.num_actions, device=self.device),
                envs_idx=env_ids,
            )

            # NOTE: RESET TARGET
            r = self.r_min + (self.r_max - self.r_min) * torch.rand(n, device=self.device)
            theta = 2 * math.pi * torch.rand(n, device=self.device)

            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            z = torch.full((n,), self.init_target_pos[2].item(), device=self.device)

            self.target_pos = (x, y, z)
            random_target_pos = torch.stack(self.target_pos, dim=1)
            self.target.set_pos(
                random_target_pos,
                envs_idx=env_ids,
            )

            # NOTE: clear last_actions
            self.last_actions[env_ids] = 0.0
            self.obs_buf[env_ids] = 0.0 
        except Exception as e:
            logger.exception(f"[RESET_AT ERROR] env_ids={env_ids.tolist()}: {e}")
            raise

    @torch.no_grad()
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
