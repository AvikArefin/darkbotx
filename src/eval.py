import os

from pathlib import Path

import torch

import genesis as gs
from genesis import _gs_backend

from rsl_rl.runners import OnPolicyRunner

from config import RL_POLICY_CFG, EVAL_ENV_CFG
from environment import GraspEnv

def evaluate(model_path: str, num_steps: int):
    gs.init(backend=_gs_backend.cpu, precision="32", logging_level="warning")
    
    env = GraspEnv(EVAL_ENV_CFG)
    
    print(f"Evaluating model from: {model_path}")
    print(f"Number of parallel environments: {env.num_envs}")
    
    log_dir = "logs/darkbotx"
    runner = OnPolicyRunner(env, RL_POLICY_CFG, log_dir, device=str(gs.device))
    runner.load(model_path)
    policy = runner.get_inference_policy(device=str(gs.device))
    
    total_rewards = 0.0
    total_episodes = 0
    
    for step in range(num_steps):
        print(f"\n--- Evaluation Step {step + 1}/{num_steps} ---")
        
        # Get initial observation
        obs = env.reset()
        
        # Get action from the loaded model based on obs
        with torch.no_grad():
            action = policy(obs)
        
        # Execute the single-step episode
        next_obs, reward, done, info = env.step(action)
        
        # Reward is 1.0 for success and 0.0 for failure, so sum is the number of successes
        step_reward = reward.sum().item()
        total_rewards += step_reward
        total_episodes += env.num_envs
        
        print(f"Step Successes: {step_reward} / {env.num_envs}")
        
    success_rate = total_rewards / total_episodes if total_episodes > 0 else 0
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Total Episodes Evaluated: {total_episodes}")
    print(f"Total Successful Grasps: {total_rewards}")
    print(f"Success Rate: {success_rate * 100:.2f}%")

if __name__ == "__main__":
    log_dir = Path("logs") / "darkbotx"
    model_files = list(log_dir.glob("model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model checkpoints found in {log_dir}. Please train a model first.")
        
    model_path = str(max(model_files, key=os.path.getmtime))
    print(f"Automatically selected the latest model: {model_path}")
    
    evaluate(model_path, 10)

