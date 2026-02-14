import torch
import genesis as gs
from code.env import FastFrankaEnv
from code.wrapper import FrankaRSLWrapper
from rsl_rl.runners import OnPolicyRunner
from main import train_cfg

def test_model(checkpoint_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")
 
    # 1. Initialize Environment for Visualization
    env_wrapped = FrankaRSLWrapper(FastFrankaEnv(device=device, n_envs=1, show_viewer=True))
    
    runner = OnPolicyRunner(env_wrapped, train_cfg, log_dir="logs/", device=device)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    runner.load(checkpoint_path)
    
    # Switch to evaluation mode
    policy = runner.get_inference_policy(device=device)

    # 3. Inference Loop
    obs, _ = env_wrapped.reset()
    
    print("Starting inference... Press Ctrl+C to stop.")
    for i in range(2000):
        # The policy expects a tensor of observations
        actions = policy(obs)
        
        # Step the environment
        obs, rewards, dones, infos = env_wrapped.step(actions)
        
        # Optional: slow down the loop so you can actually see the movement
        gs.tools.sleep(0.01) 

if __name__ == "__main__":
    # Point this to your latest model file, e.g., 'logs/franka_fast_reach/genesis_test/model_1500.pt'
    CHECKPOINT = "logs/model_0.pt"
    test_model(CHECKPOINT)