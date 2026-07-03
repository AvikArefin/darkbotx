import math
import torch
import genesis as gs
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner


class HiwonderGraspEnv:
    def __init__(self, num_envs: int = 64, show_viewer: bool = True) -> None:
            self.num_envs = num_envs
            self.device = gs.device

            self.ctrl_dt = 0.02
            self.max_episode_length = math.ceil(5.0 / self.ctrl_dt)

            # 1. Initialize Scene
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
                rigid_options=gs.options.RigidOptions(
                    dt=self.ctrl_dt,
                    constraint_solver=gs.constraint_solver.Newton,
                    enable_collision=True,
                    enable_joint_limit=True,
                ),
                vis_options=gs.options.VisOptions(
                    # Render all 64 environments instead of just 10
                    rendered_envs_idx=list(range(self.num_envs)), 
                    env_separate_rigid=True,
                ),
                viewer_options=gs.options.ViewerOptions(
                    res=(1280, 960),
                    # Pull the camera WAY back and up to see the whole 8x8 grid
                    camera_pos=(8.0, -4.0, 6.0),
                    camera_lookat=(2.0, 2.0, 0.0),
                    camera_fov=40,
                    max_FPS=int(0.5 / self.ctrl_dt),
                ),
                show_viewer=show_viewer,
            )

            # 2. Add Entities
            self.scene.add_entity(gs.morphs.Plane())

            self.robot = self.scene.add_entity(
                gs.morphs.URDF(
                    file="assets/Hiwonder_urdf_simplified/Hiwonder.urdf",
                    fixed=True,
                    decimate=False,
                    convexify=False,
                )
            )

            self.box = self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.04, 0.04, 0.04),
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.0, 0.0),
                    ),
                ),
            )

            # 3. Build Scene
            self.scene.build(n_envs=self.num_envs, env_spacing=(0.5, 0.5))

            # 4. Now that the robot exists, set dimensions and RSL_RL requirements
            self.num_actions = self.robot.n_dofs
            self.action_scales = torch.ones(self.num_actions, device=self.device) * 3.14

            # --- Freeze the fingers ---
            # The first 5 joints are the arm. Zero out the action scales for anything after that.
            self.action_scales[5:] = 0.0 
            
            # 7 DoF pos + 7 DoF vel + 3 Box pos = 17 features
            self.num_obs = (self.robot.n_dofs * 2) + 3 
            self.num_privileged_obs = None
            
            self.cfg = {
                "num_envs": self.num_envs, 
                "num_actions": self.num_actions, 
                "episode_length_s": 5.0
            }

            # 5. Set Control Parameters
            self.robot.set_dofs_kp(
                torch.ones(self.robot.n_dofs, device=self.device) * 3000.0
            )
            self.robot.set_dofs_kv(
                torch.ones(self.robot.n_dofs, device=self.device) * 300.0
            )
            self.robot.set_dofs_force_range(
                torch.ones(self.robot.n_dofs, device=self.device) * -1000.0,
                torch.ones(self.robot.n_dofs, device=self.device) * 1000.0,
            )

            self._init_buffers()
            self.reset()

        

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.reset_buf = torch.ones(self.num_envs, dtype=gs.tc_bool, device=self.device)
        self.extras = dict()

    def reset(self) -> TensorDict:
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(env_ids)
        return self.get_observations()

    def _reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return

        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = False

        dof_pos = torch.zeros((len(envs_idx), self.robot.n_dofs), device=self.device)
        self.robot.set_dofs_position(dof_pos, envs_idx=envs_idx)

        box_pos = torch.zeros((len(envs_idx), 3), device=self.device)
        box_pos[:, 0] = 0.0
        box_pos[:, 1] = -0.1
        box_pos[:, 2] = 0.02
        self.box.set_pos(box_pos, envs_idx=envs_idx)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        target_dofs = actions * self.action_scales
        self.robot.control_dofs_position(target_dofs)
        self.scene.step()

        self.episode_length_buf += 1

        box_pos = self.box.get_pos()
        gripper_pos = self.robot.get_links_pos()[:, -1, :]
        dist = torch.norm(box_pos - gripper_pos, dim=-1)

        reward = torch.exp(-dist)

        self.reset_buf = self.episode_length_buf > self.max_episode_length
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._reset_idx(env_ids)

        return self.get_observations(), reward, self.reset_buf, self.extras

    def get_observations(self) -> TensorDict:
        dof_pos = self.robot.get_dofs_position()
        dof_vel = self.robot.get_dofs_velocity()
        box_pos = self.box.get_pos()

        obs_buf = torch.cat([dof_pos, dof_vel, box_pos], dim=-1)
        return TensorDict({"policy": obs_buf}, batch_size=[self.num_envs])

    # ... underneath your step() and get_observations() methods

    def get_privileged_observations(self):
        return None


def get_train_cfg() -> dict:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.0,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "relu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "relu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": "hiwonder_grasp_demo",
        "logger": "tensorboard",
    }


def main() -> None:
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    env = HiwonderGraspEnv(num_envs=64, show_viewer=True)
    rl_train_cfg = get_train_cfg()

    runner = OnPolicyRunner(env, rl_train_cfg, "logs", device=gs.device)
    runner.learn(num_learning_iterations=300, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
