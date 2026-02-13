import genesis as gs
import torch

class FastFrankaEnv:
    def __init__(self, n_envs=2, show_viewer=True):
        self.n_envs = n_envs
        gs.init(backend=gs.gpu)
        
        # 1. Setup scene with n_envs
        self.scene = gs.Scene(
            show_viewer=show_viewer,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, -3.0, 2.5),
                camera_lookat=(0, 0, 0.5),
                res=(640, 480)
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
        # actions shape: (n_envs, 9)
        self.franka.set_dofs_position(actions)
        self.scene.step()
        
        # Get all positions at once (n_envs, 3)
        # No .cpu() or .numpy() here for speed!
        obs = self.franka.get_dofs_position()
        star_pos = self.star.get_pos()
        
        return obs, star_pos

    def reset(self):
        self.scene.reset()