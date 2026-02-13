import torch

class FrankaRSLWrapper:
    def __init__(self, env, device):
        self.env = env
        # Set these based on your panda.xml and star.urdf
        self.num_envs = env.n_envs
        self.num_obs = 9    # 7 joints + 2 finger positions (or your obs size)
        self.num_privileged_obs = None # Optional for teacher-student training
        self.num_actions = 9 # Number of joints to control
        
        # Fixed device - everything must be on GPU
        self.device = device

    def get_observations(self):
        # Genesis provides these directly as torch tensors
        obs = self.env.franka.get_dofs_position()
        return obs

    def reset(self):
        self.env.reset()
        return self.get_observations(), None # (obs, privileged_obs)

    def step(self, actions):
        # 1. Take action and step physics
        # clip actions if necessary: actions = torch.clip(actions, -1.0, 1.0)
        obs, star_pos = self.env.step(actions)
        
        # 2. Vectorized Reward (Stay on GPU!)
        # Distance between Franka and the star
        dist = torch.norm(obs[:, :3] - star_pos, dim=-1)
        rewards = -dist
        
        # 3. Vectorized Dones
        # Example: Reset if within 5cm or after 500 steps
        dones = (dist < 0.05).to(torch.float32)
        
        # 4. Handle auto-resets (Crucial for speed!)
        if dones.any():
            env_ids = dones.nonzero(as_tuple=False).flatten()
            self.env.reset_at(env_ids)
            
        return obs, None, rewards, dones, {} # (obs, priv_obs, rewards, dones, infos)