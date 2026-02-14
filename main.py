import torch
from code.env import FastFrankaEnv

from rsl_rl.runners import OnPolicyRunner

train_cfg = {
    "class_name": "OnPolicyRunner",
    # General
    "num_steps_per_env": 24,
    "max_iterations": 1500,
    "seed": 1,
    
    # Observations
    "obs_groups": {
        "actor": ["policy"], 
        "critic": ["policy"]
    },
    
    # Logging parameters
    "save_interval": 50,
    "experiment_name": "franka_fast_reach",
    "run_name": "genesis_test",
    
    # Logging writer
    "logger": "tensorboard", # tensorboard, neptune, wandb
    
    # Actor
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
        "obs_normalization": True,
        "stochastic": True,
        "init_noise_std": 1.0,
        "noise_std_type": "scalar", # "scalar" or "log"
        "state_dependent_std": False
    },
    
    # Critic
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
        "obs_normalization": True,
        "stochastic": False
    },
    
    # Algorithm
    "algorithm": {
        "class_name": "PPO",
        
        # Training
        "optimizer": "adam",
        "learning_rate": 0.001,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "schedule": "adaptive",
        
        # Value function
        "value_loss_coef": 1.0,
        "clip_param": 0.2,
        "use_clipped_value_loss": True,
        
        # Surrogate loss    
        "desired_kl": 0.01,
        "entropy_coef": 0.01,
        "gamma": 0.99,
        "lam": 0.95,
        "max_grad_norm": 1.0,
        
        # Miscellaneous
        "normalize_advantage_per_mini_batch": False
    },
}

def main():
    device = "cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu:0"
    
    env = FastFrankaEnv(n_envs=2048, show_viewer=False)

    runner = OnPolicyRunner(env, train_cfg, log_dir="logs/", device=device)

    # Start Training
    runner.learn(num_learning_iterations=train_cfg["max_iterations"])

if __name__ == "__main__":
    main()
