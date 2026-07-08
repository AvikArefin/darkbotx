import argparse
import time
import os
import importlib
import genesis as gs
import torch

# Import the modules themselves so we can reload them
import environment
import config

def main():
    # Initialize Genesis
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")

    # Create the environment with the viewer enabled
    env = environment.GraspEnv()
    
    # Initialize the environment for the first time
    obs = env.reset()

    print("="*60)
    print("🚀 Running environment in LIVE RELOAD mode.")
    print("Modify 'environment.py' or 'config.py' and save. The code will update instantly!")
    print("Press Ctrl+C in the terminal to stop.")
    print("="*60)
    
    # Watch environment.py and config.py for file modifications
    environment_path = environment.__file__
    config_path = config.__file__
    
    last_mtime_env = os.path.getmtime(environment_path)
    last_mtime_config = os.path.getmtime(config_path)

    try:
        while True:
            # 1. Live reload check
            current_mtime_env = os.path.getmtime(environment_path)
            current_mtime_config = os.path.getmtime(config_path)
            
            reloaded = False
            
            if current_mtime_config > last_mtime_config:
                print("\n[Live Reload] Detected changes in config.py! Reloading module...")
                try:
                    importlib.reload(config)
                    print("[Live Reload] Successfully reloaded config.py! ✨")
                    last_mtime_config = current_mtime_config
                    reloaded = True
                except Exception as e:
                    print(f"[Live Reload] ❌ Error reloading config (fix your code and save again): {e}")

            if current_mtime_env > last_mtime_env or reloaded:
                print("\n[Live Reload] Reloading environment...")
                try:
                    # Reload the module
                    importlib.reload(environment)
                    # Magically update our existing instance to use the new class definition!
                    # Note: Changes in __init__ (like box_size) won't take effect unless env is re-instantiated, 
                    # but step() logic and updated config values will be available.
                    env.__class__ = environment.GraspEnv
                    if hasattr(env, 'robot'):
                        env.robot.__class__ = environment.Manipulator
                    print("[Live Reload] Successfully reloaded and applied environment changes! ✨")
                    last_mtime_env = current_mtime_env
                except Exception as e:
                    print(f"[Live Reload] ❌ Error reloading environment (fix your code and save again): {e}")

            # 2. Step the environment
            try:
                # Pass a dummy continuous action [angle, squeeze_width]
                action = torch.zeros((env.num_envs, 2), device=gs.device)
                obs, reward, done, info = env.step(action)
                
                # Single-step episode immediately returns done=True
                if done.any():
                    obs = env.reset()
            except Exception as e:
                if "Viewer closed" in str(e):
                    print("\n[Viewer Closed] Exiting script cleanly.")
                    break
                print(f"[Runtime Error] ❌ step() failed: {e}")
                print("Waiting for file changes to recover...")
                # Sleep a bit to prevent spamming errors before the user fixes it
                time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nStopping debug environment.")

if __name__ == "__main__":
    main()
