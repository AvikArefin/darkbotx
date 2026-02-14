import time
import torch
import genesis as gs
from rsl_rl.env import VecEnv
from tensordict import TensorDict

# Following the RSL RL API
class FastFrankaEnv(VecEnv):
    def __init__(self, n_envs=1, show_viewer=False):
        self.n_envs = n_envs
        # RSL-RL required attributes
        self.num_envs = n_envs
        self.num_actions = 9
        self.max_episode_length = 500
        self.cfg = {} 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        gs.init(backend=gs.gpu if self.device.type != "cpu" else gs.cpu, performance_mode= not show_viewer)
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,       # Simulation timestep (e.g., 100Hz)
                substeps=2,    # Physics substeps per dt (increases stability)
            ),
            show_viewer=show_viewer,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, -3.0, 2.5), camera_lookat=(0, 0, 0.5), res=(512, 512)
            )
        )
        
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
        self.target = self.scene.add_entity(gs.morphs.URDF(file='assets/star/star.urdf', pos=(0.6, 0, 0.45)))

        self.scene.build(n_envs=self.n_envs, env_spacing=(1.5, 1.5))
        
        self.init_robot_pos = torch.zeros(self.robot.n_dofs, device=self.device)
        self.init_target_pos = torch.tensor([0.6, 0.45, 0], device=self.device)
        
        # Tracking buffers for RSL-RL logic
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
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
            torch.tensor([-87, -87, -87, -87, -12, -12, -12, -100, -100], device=self.device),
            torch.tensor([87, 87, 87, 87, 12, 12, 12, 100, 100], device=self.device),
        )
        
        self.action_scales = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], device=self.device)
        
        
    def get_observations(self) -> TensorDict:
        obs = self.robot.get_dofs_position()
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)

    def step(self, actions):
        # 1. Apply actions and simulate
        self.robot.control_dofs_position(actions * self.action_scales)
        self.scene.step()
        
        obs = self.robot.get_dofs_position()
        
        ee_pos = (self.robot.get_link('left_finger').get_pos() + self.robot.get_link('right_finger').get_pos()) / 2 # OK       
        target_pos = self.target.get_pos()                                                                          # OK
        
        # 3. Draw Debug Line (Target to End Effector)
        # We clear previous lines to avoid a "trail" and draw a new one
        # self.scene.clear_debug_objects()
        
        # for i in range(self.num_envs):
        #     self.scene.draw_debug_line(
        #         start=target_pos[i], 
        #         # start=torch.tensor([0, 0, 0]),
        #         end=ee_pos[i],
        #         radius=0.005,
        #         color=(0, 1, 0),
        #     )
        
        # Distance from robot to target
        dist = torch.norm(ee_pos - target_pos, dim=-1)
        
        # Distance from target to its initial starting point
        target_displacement = torch.norm(target_pos - self.init_target_pos, dim=-1)
        
        # If target moves more than 10cm, consider it "thrown away"
        is_thrown = target_displacement > 0.10 
        
        # 3. Rewards & Dones
        # Apply a heavy penalty if the target is thrown
        rewards = -dist - (is_thrown.float() * 10.0) 
        
        termination_dones = (dist < 0.05) | is_thrown
        
        # 3. Handle Timeouts (Truncation)
        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length
        
        # 4. Combined Dones for the solver
        total_dones = termination_dones | time_outs
        
        # 5. Automatic Resets (Crucial for vectorized training)
        if total_dones.any():
            env_ids = total_dones.nonzero(as_tuple=False).flatten()
            self.reset_at(env_ids)
            self.episode_length_buf[env_ids] = 0
            # Refresh observations after resets
            obs = self.robot.get_dofs_position()
        
        # 6. Package for RSL-RL
        obs_td = TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)
        infos = {"time_outs": time_outs}
        
        return obs_td, rewards * 0.01, total_dones, infos

    def reset_at(self, env_ids):
        self.robot.set_dofs_position(self.init_robot_pos, envs_idx=env_ids)
        self.robot.set_dofs_velocity(torch.zeros_like(self.init_robot_pos), envs_idx=env_ids)
        self.target.set_pos(self.init_target_pos, envs_idx=env_ids)

    def reset(self):
        self.scene.reset()
        self.episode_length_buf.zero_()
        obs = self.robot.get_dofs_position()
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device), {}