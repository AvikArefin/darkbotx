import torch
import genesis as gs

class FastFrankaEnv:
    def __init__(self, n_envs=2, show_viewer=False):
        self.n_envs = n_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        gs.init(backend=gs.gpu if self.device.type != "cpu" else gs.cpu)
        
        self.scene = gs.Scene(
            show_viewer=show_viewer,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, -3.0, 2.5), camera_lookat=(0, 0, 0.5), res=(128, 128)
            )
        )
        
        self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
        self.star = self.scene.add_entity(gs.morphs.URDF(file='assets/star/star.urdf', pos=(0.6, 0, 0.45)))

        self.scene.build(n_envs=self.n_envs, env_spacing=(1.5, 1.5))
        
        self.init_franka_pos = torch.zeros(self.franka.n_dofs, device=self.device)
        self.init_star_pos = torch.tensor([0.6, 0, 0.45], device=self.device)

    def step(self, actions):
        self.franka.set_dofs_position(actions)
        self.scene.step()
        
        obs = self.franka.get_dofs_position()
        target_pos = self.star.get_pos()
        
        # Calculate rewards and dones
        dist = torch.norm(obs[:, :3] - target_pos, dim=-1)
        rewards = -dist
        dones = (dist < 0.05) # This is a Bool tensor
        
        return obs, rewards, dones, {}

    def reset_at(self, env_ids):
        # Reset Franka joint positions and velocities
        self.franka.set_dofs_position(self.init_franka_pos, envs_idx=env_ids)
        self.franka.set_dofs_velocity(torch.zeros_like(self.init_franka_pos), envs_idx=env_ids)
    
        # Reset Star object position
        self.star.set_pos(self.init_star_pos, envs_idx=env_ids)

    def reset(self):
        self.scene.reset()
        return self.franka.get_dofs_position()