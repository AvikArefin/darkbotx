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
    "episode_length_s": 1.0,
    "ctrl_dt": 0.01,
    "use_rasterizer": True,
    "is_debug": True,
    "logging_level": "info",
    "performance_mode": True,
    "show_FPS": False,
}

reward_cfg = {
    "keypoints": 1.0,
}

robot_cfg = {
    "ee_link_name": "hand",
    "gripper_link_names": ["left_finger", "right_finger"],
    "home_pos": torch.tensor([0.741, 0.63, 0.023, -1.95, 0.26, 2.38, 1.21, 0.41, 0.41], device=device),
    "ik_method": "dls_ik",
}

# --- TRAINING: functions and classes ---
def run_random_simulation(debug: bool = False):
    """
    Runs the simulation with random actions.
    If debug is True, visualizes it using the Monitor and extracts telemetry.
    """
    # 1. Setup Environment
    # show_viewer is tied to the debug flag
    env = FastFrankaEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=debug
    )

    # 2. Setup GUI Monitor conditionally
    monitor = Monitor(sys.argv, num_envs=env.num_envs) if debug else None
    
    env.reset()
    
    # Trackers for rewards
    total_rewards = [0.0] * env.num_envs

    try:
        if debug:
            logger.info("--- GUI: RANDOM ACTION SIMULATION STARTED ---")
        else:
            logger.info("--- HEADLESS: RANDOM ACTION SIMULATION STARTED ---")
        
        # 3. Linear Simulation Loop
        while True:
            # If in debug mode, break the loop if the user closes the Monitor window
            if debug and not monitor.window.isVisible():
                break
            
            # A. Generate Random Actions
            # Shape: (num_envs, 9) in range [0.0, 1.0] for absolute control
            random_actions = torch.rand((env.num_envs, env.num_actions), device=env.device)
            
            # B. Step Environment
            obs, rewards, dones, infos = env.step(random_actions)
            
            # C. Monitor Updates (Only if debug is enabled)
            if debug:
                # Extract telemetry directly from infos
                dofs_pos = infos.get("dofs_pos")
                target_pos = infos.get("target_pos")
                dist = infos.get("dist")

                # Update Monitor (Iterate over all environment panels)
                for i in range(env.num_envs):
                    panel = monitor.env_panels[i]

                    # 1. Update Reward Plot
                    total_rewards[i] += rewards[i].item()
                    panel.update_plot(total_rewards[i])

                    # 2. Update Sliders
                    if dofs_pos is not None:
                        # Pass the exact physics joint angles to the sliders
                        panel.update_joints(dofs_pos[i].tolist())

                    # 3. Update Cube Label
                    if target_pos is not None and dist is not None:
                        # Convert to standard python types to avoid tensor format errors
                        target_p = target_pos[i].tolist()
                        d = dist[i].item()

                        cube_text = f"XYZ: [{target_p[0]:.2f}, {target_p[1]:.2f}, {target_p[2]:.2f}]\ndist: {d:.4f}"
                        panel.set_cube_label(cube_text)

                    # 4. Handle Episode Completion (Reset Plot)
                    if dones[i]:
                        total_rewards[i] = 0.0
                        panel.reset_plot()
                
                # D. Process GUI Events (Keep window responsive)
                monitor.update_gui()

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
    except Exception as e:
        logger.error(f"Error in random simulation: {e}")
    finally:
        # Cleanup
        if debug and monitor:
            monitor.close()
        logger.info("Random Simulation Ended.")

def run_manual_simulation():
    """
    Runs the simulation in manual control mode.

    Slider ranges are set to the strict Panda joint limits (panda_lower/panda_upper).
    Sliders are initialised to robot_cfg["home_pos"].

    Action conversion
    -----------------
    env.step() maps actions in [0, 1] to physical DOF positions via:
        target_dofs_pos = env.dof_lower + action * (env.dof_upper - env.dof_lower)

    So to send a desired physical value `phys` we invert that:
        action = (phys - env.dof_lower) / (env.dof_upper - env.dof_lower)

    Both finger joints (7 & 8) are driven together from slider 7.
    """

    # ------------------------------------------------------------------ #
    # 1.  Environment                                                      #
    # ------------------------------------------------------------------ #
    manual_env_cfg = env_cfg.copy()
    manual_env_cfg["episode_length_s"] = 99999.0

    env = FastFrankaEnv(
        env_cfg=manual_env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=True,
    )
    env.reset()

    # ------------------------------------------------------------------ #
    # 2.  Strict Panda joint limits used for sliders                      #
    # ------------------------------------------------------------------ #
    panda_lower = torch.tensor(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0000, 0.0000],
        device=env.device,
    )
    panda_upper = torch.tensor(
        [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973, 0.0400, 0.0400],
        device=env.device,
    )
    panda_range = panda_upper - panda_lower  # element-wise span

    # ------------------------------------------------------------------ #
    # 3.  Monitor + slider initialisation                                 #
    # ------------------------------------------------------------------ #
    monitor = Monitor(sys.argv, num_envs=env.num_envs)
    home_pos = robot_cfg["home_pos"]  # shape: (9,)

    for panel in monitor.env_panels:
        for i in range(len(panel.joints_var)):
            slider = panel.joints_var[i]

            # Override limits stored in the FloatSlider to match panda_lower/upper
            slider.min_val    = panda_lower[i].item()
            slider.max_val    = panda_upper[i].item()
            slider.range_span = panda_range[i].item()

            # Initialise visual position to the home pose
            slider.set(home_pos[i].item())

    # ------------------------------------------------------------------ #
    # 4.  Physics / GUI loop                                              #
    # ------------------------------------------------------------------ #
    total_rewards = [0.0] * env.num_envs

    def physics_loop():
        nonlocal total_rewards

        # --- A.  Build action tensor from slider physical values -------- #
        action = torch.zeros((env.num_envs, env.num_actions), device=env.device)

        for env_idx in range(env.num_envs):
            panel = monitor.env_panels[env_idx]

            # Arm joints 0-6
            for i in range(7):
                slider = panel.joints_var[i]

                # GUI slider integer (0-1000) -> physical value (rad)
                gui_norm = slider.slider.value() / 1000.0
                phys_val = slider.min_val + gui_norm * slider.range_span

                # Physical value -> action in [0, 1] expected by env.step()
                dof_span  = (env.dof_upper[i] - env.dof_lower[i]).item()
                action_val = (phys_val - env.dof_lower[i].item()) / dof_span if dof_span != 0 else 0.0
                action[env_idx, i] = max(0.0, min(1.0, action_val))

                # Refresh label
                slider.label.setText(f"{slider.name}: {phys_val:.4f}")

            # Gripper joints 7 & 8 --- driven by slider 7, mirrored to slider 8
            g_slider  = panel.joints_var[7]
            g_gui_norm = g_slider.slider.value() / 1000.0
            g_phys     = g_slider.min_val + g_gui_norm * g_slider.range_span

            g_dof_span = (env.dof_upper[7] - env.dof_lower[7]).item()
            g_action   = (g_phys - env.dof_lower[7].item()) / g_dof_span if g_dof_span != 0 else 0.0
            g_action   = max(0.0, min(1.0, g_action))

            action[env_idx, 7] = g_action
            action[env_idx, 8] = g_action  # both fingers move together

            # Refresh gripper labels and mirror slider 8 visually
            g_slider.label.setText(f"{g_slider.name}: {g_phys:.4f}")
            f2 = panel.joints_var[8]
            f2.slider.setValue(g_slider.slider.value())
            f2.label.setText(f"{f2.name}: {g_phys:.4f}")

        # --- B.  Step simulation ---------------------------------------- #
        obs, rewards, dones, infos = env.step(action)

        target_pos = infos.get("target_pos")  # tensor (num_envs, 3), present when show_viewer=True
        dist       = infos.get("dist")        # tensor (num_envs,),   present when show_viewer=True

        # --- C.  Update Monitor UI -------------------------------------- #
        for i in range(env.num_envs):
            panel = monitor.env_panels[i]

            # Cumulative reward plot
            total_rewards[i] += rewards[i].item()
            panel.update_plot(total_rewards[i])

            # Cube position + distance label
            if target_pos is not None and dist is not None:
                tp = target_pos[i].tolist()
                d  = dist[i].item()
                panel.set_cube_label(
                    f"XYZ: [{tp[0]:.3f}, {tp[1]:.3f}, {tp[2]:.3f}]   dist: {d:.4f} m"
                )

            # Reset on episode end
            if dones[i]:
                total_rewards[i] = 0.0
                panel.reset_plot()

    # ------------------------------------------------------------------ #
    # 5.  Start event loop                                                #
    # ------------------------------------------------------------------ #
    try:
        logger.info("--- GUI: MANUAL SIMULATION STARTED ---")
        monitor.start_manual_loop(physics_loop)
        exit_code = monitor.exec()
        logger.info(f"Monitor closed with exit code: {exit_code}")
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
    except Exception as e:
        logger.error(f"Error in manual simulation: {e}", exc_info=True)
    finally:
        logger.info("Manual Simulation Ended.")

def rl_training():
    # For training, episodes are not explicitly defined since we train for a total number of timesteps.
    show_viewer = True

    env = FastFrankaEnv(
        env_cfg=env_cfg, 
        reward_cfg=reward_cfg, 
        robot_cfg=robot_cfg, 
        show_viewer=show_viewer
    )
    
def main():
    """Main function to train and evaluate the model."""
    
    # different modes: --manual, --random, --training
    try:
        if args.random:
            logger.info("--- GUI: RANDOM ACTION SIMULATION ---")
            run_random_simulation(debug=args.monitor)
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
