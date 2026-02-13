import genesis as gs
import torch

class FastFrankaEnv:
    def __init__(self, n_envs=2, show_viewer=False):
        self.n_envs = n_envs
        gs.init(backend=gs.gpu if torch.cuda.is_available() or torch.backends.mps.is_available() else gs.cpu)
        
        # 1. Setup scene with n_envs
        self.scene = gs.Scene(
            show_viewer=show_viewer,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, -3.0, 2.5),
                camera_lookat=(0, 0, 0.5),
                res=(128, 128) # Lower res if viewer is on; zero overhead if off
            )
        )
        
        # 2. Add Entities (Genesis handles the cloning for you)
        self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
        )
        self.star = self.scene.add_entity(
            gs.morphs.URDF(file='assets/star/star.urdf', pos=(0.6, 0, 0.45))
        )

        # 3. Build with n_envs and spacing so they don't overlap in the viewer
        self.scene.build(n_envs=self.n_envs, env_spacing=(1.5, 1.5))

    def step(self, actions):
        # actions: (n_envs, 9)
        self.franka.set_dofs_position(actions)
        self.scene.step()
        
        # observation: (n_envs, 3)
        obs = self.franka.get_dofs_position()
        target_pos = self.star.get_pos()
        
        # 3. Compute Reward (Vectorized - No loops!)
        # Assuming obs[:, :3] is your end-effector or relevant joint
        dist = torch.norm(obs[:, :3] - target_pos, dim=-1)
        rewards = -dist
        
        # 4. Compute Dones (Speed Tip: use a mask)
        # Example: done if distance < 0.05 or step_count > max_steps
        dones = (dist < 0.05).to(torch.float32) 
        
        return obs, rewards, dones, {}

    def reset_at(self, env_ids):
        # Use Genesis's selective reset to only fix 'finished' envs.
        self.scene.reset(env_ids=env_ids)

    def reset(self):
        self.scene.reset()
