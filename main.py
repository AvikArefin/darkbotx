import torch
import genesis as gs
from code.env import FastFrankaEnv

def main():
    # Let's run 64 environments at once
    n_envs = 64*2
    env = FastFrankaEnv(n_envs=n_envs, show_viewer=True)
    
    print(f"Running {n_envs} parallel environments...")

    for i in range(5000):
        # Generate random target positions for all 64 robots at once
        # Shape: (64, 9)
        random_actions = torch.randn((n_envs, 9), device=gs.device)
        
        # Step all environments
        obs, star_pos = env.step(random_actions)
        
        if i % 100 == 0:
            print(f"Step {i}: Simulated {n_envs * (i+1)} total frames")

if __name__ == "__main__":
    main()