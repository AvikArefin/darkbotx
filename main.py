import os
import sys
import argparse
import numpy as np

# RL training imports
import torch
from rsl_rl.runners import OnPolicyRunner

# training log
import wandb

# custom imports
from code.env import FastFrankaEnv
from code.logger_setup import setup_logger
from code.monitor import Monitor

# Register the custom environment with Gymnasium
ID: str = "PandaPickAndPlace-v0"                         # <name>-v<version>
ENTRY_POINT: str = "environment:PandaPickAndPlaceEnv"    # <filename/module>:<class_name>

MAX_EPISODE_STEPS: int = int(4e2)
MODEL_PATH = "models/sac_panda_pickandplace.zip"
DEFAULT_TOTAL_TIMESTEPS = 500_000
TOTAL_EPISODES=10
DEFAULT_CONTROL_MODE="ee"

# logger setup
logger = setup_logger(__name__)

# --- ARGS ---
def positive_int(value):
    """Custom argparse type for strictly positive integers."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer.")
        
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Value must be a positive integer, got {value}.")
    return ivalue

def get_args():
    parser = argparse.ArgumentParser(description="Run robot arm simulations.")

    parser.add_argument("-m", "--manual", action="store_true", help="Run manual simulation.")

    parser.add_argument("-t", "--training", nargs="?", const=DEFAULT_TOTAL_TIMESTEPS, type=positive_int, help="RL training. Trains new model if no --load provided. default: %(const)s")
    parser.add_argument("-r","--random", nargs="?", const=TOTAL_EPISODES, type=positive_int, help="Run simulation with random actions. default: %(const)s")
    parser.add_argument("-i", "--inference", nargs="?", const=TOTAL_EPISODES, type=positive_int, help="RL model inference simulation with trained model. Loads default model if no --load arg passed. default: %(const)s")
    
    parser.add_argument("--load", nargs="?", const=MODEL_PATH, type=str, default=None, help="Pass model to -i & -t. default: %(const)s")
    parser.add_argument("--control_mode", nargs="?", const=DEFAULT_CONTROL_MODE, type=str, default=DEFAULT_CONTROL_MODE, help="PyBullet Env Control Mode. default: %(const)s")
    parser.add_argument("--monitor", action="store_true", help="provides monitor for env")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()
args = get_args()

# configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

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

env_cfg = {
    "num_envs": 1,
    "num_obs": 14,
    "num_actions": 9,
    "action_scales": torch.tensor([1.0] * 9, device=device),
    "episode_length_s": 30.0,
    "ctrl_dt": 0.01,
    "use_rasterizer": True,
}

reward_cfg = {
    "keypoints": 1.0,
}

robot_cfg = {
    "ee_link_name": "hand",
    "gripper_link_names": ["left_finger", "right_finger"],
    "home_pos": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
    "ik_method": "dls_ik",
}

# --- TRAINING: functions and classes ---
def run_random_simulation():
    """Runs the simulation with the given model."""
    show_viewer = True

    env = FastFrankaEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=show_viewer
    )

    env.reset()
    try:
        while True:
            random_actions = 2.0 * torch.rand((env_cfg["num_envs"], env_cfg["num_actions"]), device=env.device) - 1.0
            obs, rewards, dones, infos = env.step(random_actions)

            logger.debug(f"infos = {infos}")
            logger.debug(f"rewards = {rewards}")

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        pass

def run_simulation(model, id: str, episodes: int = 3):
    """Runs the simulation with the given model."""

    monitor : Monitor = Monitor(sys.argv)

    # Create a single environment with rendering enabled.
    env = gym.make(id, render_mode="human", control_mode=args.control_mode)
    
    if model is None:
        logger.info("No model provided, using random actions.")
        model = RandomPolicy(env.action_space)
    
    try:
        for episode in range(episodes):
            logger.info(f"--- Starting Episode {episode + 1} ---")

            # GUI: update episode
            monitor.set_episode(episode + 1)
            monitor.reset_plot()

            obs, _ = env.reset()
            terminated = False
            truncated = False
            
            step_counter = 0
            total_reward = 0
            while not terminated and not truncated:
                step_counter += 1

                # logger.info(f"{10*'-'} STEP_{step_counter} {10*'-'}")
                action, _states = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += float(reward)

                # GUI: update debug monitor
                monitor.set_cube_label(info["cube_info"])

                monitor.update_plot(total_reward)
                monitor.update_joints(info)
                monitor.update_gui()

        exit_code = monitor.exec()
        logger.info(f"Monitor close with exit code: {exit_code}")
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
    finally:
        env.close()

def run_manual_simulation():
    """
    Runs the simulation in manual control mode using the Custom Monitor GUI.
    """
    # 1. Setup Environment
    # Force 1 environment for manual control
    manual_env_cfg = env_cfg.copy()
    manual_env_cfg["num_envs"] = 1
    manual_env_cfg["episode_length_s"] = 9999999.0
    
    # Initialize Genesis Environment
    env = FastFrankaEnv(
        env_cfg=manual_env_cfg, 
        reward_cfg=reward_cfg, 
        robot_cfg=robot_cfg, 
        show_viewer=True
    )
    
    # 2. Setup GUI Monitor
    monitor : Monitor = Monitor(sys.argv)
    env.reset()
    
    total_reward: float = 0.
    step_count: int = 0

    def physics_loop():       
        nonlocal step_count, total_reward
        
        # A. Read Sliders -> Action Vector
        # Shape: (1, 9) -> [Batch, Actions]
        # We use [0, 1] range because your env.py uses absolute control:
        # target = lower + action * (upper - lower)
        action = torch.zeros((1, 9), device=env.device)
        
        # 1. Map Arm Sliders (indices 0-6)
        for i in range(7):
            if i < len(monitor.joints_var):
                slider = monitor.joints_var[i]
                val = slider.slider.value() # 0 to 1000
                
                # Map 0..1000 -> 0.0..1.0
                normalized_val = val / 1000.0
                action[0, i] = normalized_val
                
                # Update Label (Approximate visual feedback)
                slider.label.setText(f"{slider.name}: {normalized_val:.2f}")

        # 2. Map Gripper (Index 7 Control -> Action 7 & 8)
        if len(monitor.joints_var) > 7:
            gripper_slider = monitor.joints_var[7]
            g_val = gripper_slider.slider.value() # 0 to 1000
            
            # Map 0..1000 -> 0.0..1.0
            g_norm = g_val / 1000.0
            
            action[0, 7] = g_norm # Left Finger
            action[0, 8] = g_norm # Right Finger
            
            gripper_slider.label.setText(f"{gripper_slider.name}: {g_norm:.2f}")
            
            # Update the 2nd gripper slider visually if it exists
            if len(monitor.joints_var) > 8:
                monitor.joints_var[8].slider.setValue(g_val)
                monitor.joints_var[8].label.setText(f"{monitor.joints_var[8].name}: {g_norm:.2f}")

        # B. Step Physics
        # FastFrankaEnv returns: (obs, rewards, dones, infos)
        obs, reward, done, info = env.step(action)
        
        step_count += 1
        
        # Take the scalar reward from the (1,) tensor
        current_reward = reward.item()
        total_reward += current_reward

        # C. Update Telemetry
        monitor.update_plot(total_reward)
        
        # Optional: Update Cube Label if info is available
        if "cube_info" in info:
            monitor.set_cube_label(info["cube_info"])

        # D. Auto-Reset Handling
        # Genesis auto-resets in step(), so we just handle the GUI side here
        if done.any():
            # Optional: Reset plot on new episode
            # monitor.reset_plot()
            pass

    # 3. Start the GUI Event Loop
    try:
        logger.info("--- GUI: MANUAL SIMULATION STARTED ---")
        monitor.start_manual_loop(physics_loop)
        exit_code = monitor.exec()
        logger.info(f"Monitor closed with exit code: {exit_code}")
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
    except Exception as e:
        logger.error(f"Error in manual simulation: {e}")
    finally:
        logger.info("Manual Simulation Ended.")

def rl_training():
    # For training, episodes are not explicitly defined since we train for a total number of timesteps.
    show_viewer = True

    env = FastFrankaEnv(env_cfg=env_cfg, reward_cfg=reward_cfg, robot_cfg=robot_cfg, show_viewer=show_viewer)
    
def main():
    """Main function to train and evaluate the model."""
    
    # different modes: --manual, --random, --training
    try:
        if args.random:
            logger.info("--- GUI: RANDOM ACTION SIMULATION ---")
            run_random_simulation()
            logger.info("FINISHED. Running simulation with random actions.")
        if args.manual:
            logger.info("--- GUI: MANUAL SIMULATION ---")
            run_manual_simulation()
        if args.training:
            logger.info("--- RL TRAINING ---")
            rl_training()
        if args.inference:
            logger.info(f"--- INFERENCE RL MODEL WITH SIMULATION ---")

            trained_model = None 
            if args.load and os.path.exists(args.load):
                logger.info(f"Inferencing '{args.load}' model")
            elif os.path.exists(MODEL_PATH):
                logger.info(f"Inferencing '{MODEL_PATH}' model")
            else:
                logger.error("No trained model Found!")

            if trained_model:
                # run_simulation(model=trained_model, id=ID, episodes=args.inference)
                pass
    except KeyboardInterrupt:
        logger.warning("\nCtrl + C received! Cleaning up resources ...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        # Force pybullet to disconnect and the app to quit
        logger.info("SHUTDOWN COMPLETE")
 
if __name__ == "__main__":
    main()
