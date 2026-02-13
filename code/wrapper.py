from rsl_rl.env import VecEnv
from tensordict import TensorDict
import torch # The abstract class you shared earlier

class FrankaRSLWrapper(VecEnv):
    def __init__(self, env, device):
        self.env = env
        self.device = device
        
        # Required by RSL-RL VecEnv interface
        self.num_envs = env.n_envs
        self.num_actions = 9 
        self.max_episode_length = 500
        self.cfg = {} 
        
        # Tracking buffers
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def get_observations(self) -> TensorDict:
        obs = self.env.franka.get_dofs_position()
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)

    def step(self, actions: torch.Tensor):
        # 1. Call the underlying physics env
        obs, rewards, dones, infos = self.env.step(actions)
        
        # 2. Logic for timeouts (Truncation)
        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length
        
        # 3. Combine Termination (dones) and Truncation (time_outs)
        # Ensure both are bool before bitwise OR
        dones = dones.to(torch.bool) | time_outs.to(torch.bool)
        
        # 4. Handle Resets
        if dones.any():
            env_ids = dones.nonzero(as_tuple=False).flatten()
            self.env.reset_at(env_ids)
            self.episode_length_buf[env_ids] = 0
            # Re-fetch observations for the reset environments
            obs = self.env.franka.get_dofs_position()
            
        # 5. Package for RSL-RL
        obs_td = TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)
        infos["time_outs"] = time_outs
        
        return obs_td, rewards, dones, infos

    def reset(self):
        obs = self.env.reset()
        self.episode_length_buf.zero_()
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device), {}