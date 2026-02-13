import torch
import genesis as gs
from code.env import FastFrankaEnv
import time

from code.wrapper import FrankaRSLWrapper
from rsl_rl.runners import OnPolicyRunner

def test():
    # Let's run 64 environments at once
    n_envs = 24
    env = FastFrankaEnv(n_envs=n_envs, show_viewer=True)
    
    print(f"Running {n_envs} parallel environments...")

    for i in range(5000):
        # Generate random target positions for all 64 robots at once
        # Shape: (64, 9)
        random_actions = torch.randn((n_envs, 9), device=gs.device)
        
        # Step all environments
        obs = env.step(random_actions)
        time.sleep(0.01)
        
        if i % 100 == 0:
            print(f"Step {i}: Simulated {n_envs * (i+1)} total frames")



train_cfg = {
    "runner": {
        "class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "max_iterations": 1500,
        "save_interval": 50,
        "experiment_name": "franka_fast_reach",
        "run_name": "genesis_test",
    },
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
    },
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "learning_rate": 1e-3,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "gamma": 0.99,
        "lam": 0.95,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
    "obs_groups": {
        "actor": ["policy"], 
        "critic": ["policy"]
    }
}

def main():
    # 1. Initialize your wrapped env
    device = "cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu:0"
    env_wrapped = FrankaRSLWrapper(FastFrankaEnv(n_envs=2048, show_viewer=False), device=device)

    # 2. Setup the Runner
    runner = OnPolicyRunner(env_wrapped, train_cfg, log_dir="logs/", device="mps")

    # 3. Start Training
    runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"], init_at_random_step=True)

if __name__ == "__main__":
    main()
