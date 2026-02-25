from datetime import datetime
import os
import sys
import argparse
from datetime import datetime

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
DEFAULT_CONTROL_MODE="ee"
DEFAULT_DEBUG_MODE = "both"
DEFAULT_MAX_ITERATIONS = 1500
DEFAULT_TOTAL_EPISODES=10                               # WARN: DEPRECATED.
MODEL_PATH = "models/model_panda_pickandplace.pt"

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

    parser.add_argument("-t", "--training", nargs="?", const=DEFAULT_MAX_ITERATIONS, type=positive_int, help="RL training. Trains new model if no --load provided. default: %(const)s")
    parser.add_argument("-r","--random", nargs="?", const=DEFAULT_TOTAL_EPISODES, type=positive_int, help="Run simulation with random actions. default: %(const)s")
    parser.add_argument("-i", "--inference", nargs="?", const=DEFAULT_TOTAL_EPISODES, type=positive_int, help="RL model inference simulation with trained model. Loads default model if no --load arg passed. default: %(const)s")

    debug_modes = ["monitor", "sim", "both"]
    parser.add_argument("--debug", nargs="?", const=DEFAULT_DEBUG_MODE, type=str, choices=debug_modes, default=None, help=f"Available debug modes: {debug_modes}. default: %(const)s")
    
    parser.add_argument("--load", nargs="?", const=MODEL_PATH, type=str, default=None, help="Pass model to -i & -t. default: %(const)s")
    parser.add_argument("--control_mode", nargs="?", const=DEFAULT_CONTROL_MODE, type=str, default=DEFAULT_CONTROL_MODE, help="PyBullet Env Control Mode. default: %(const)s")
    parser.add_argument("--crash", type=str, help="reproducing crash from saved debug")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()

args = get_args()

# configuration
train_cfg = {
    "class_name": "OnPolicyRunner",                     # NOTE: "OnPolicyRunner" (only standard option)

    # --- General ---
    "num_steps_per_env": 64,                            # NOTE: 16 | 24 | 48 | 64 | 96 | 128 | 256
    "max_iterations": args.training,                    # NOTE: any int > 0
    "seed": 1,                                          # NOTE: any integer

    # --- Observations ---
    "obs_groups": {
        "actor":  ["policy"],                           # NOTE: list of keys from env obs dict
        "critic": ["policy"],                           # NOTE: can give critic extra privileged keys e.g. ["policy", "privileged"]
    },

    # --- Logging ---
    "save_interval": 40,                                # NOTE: any int > 0 (iterations between .pt checkpoints)
    "experiment_name": "franka_fast_reach",             # NOTE: any string (parent log folder)
    "run_name": "genesis_test",                         # NOTE: any string (sub-folder for this run)

    "logger": "wandb",                                  # NOTE: "tensorboard" | "wandb" | "neptune"
    "wandb_project": "franka-genesis-rl",               # NOTE: Dashboard project name 

    # --- Actor ---
    "actor": {
        "class_name": "MLPModel",                       # NOTE: "MLPModel" | "RNN" (some forks)
        "hidden_dims": [256, 128, 64],                  # NOTE: [64,64] | [256,128,64] | [512,256,128]
        "activation": "elu",                            # NOTE: "elu" | "relu" | "tanh" | "selu"
        "obs_normalization": True,                      # NOTE: True | False
        "stochastic": True,                             # NOTE: True (Actor) | False (Critic)
        "init_noise_std": 1.0,                          # NOTE: 0.5 | 0.8 | 1.0 | 1.5
        "noise_std_type": "scalar",                     # NOTE: "scalar" | "log"
        "state_dependent_std": False,                   # NOTE: True | False
    },

    # --- Critic ---
    "critic": {
        "class_name": "MLPModel",                       # NOTE: "MLPModel" | "RNN" (some forks)
        "hidden_dims": [256, 128, 64],                  # NOTE: [64,64] | [256,128,64] | [512,256,128]
        "activation": "elu",                            # NOTE: "elu" | "relu" | "tanh" | "selu"
        "obs_normalization": True,                      # NOTE: True | False
        "stochastic": False,                            # NOTE: always False for Critic
    },

    # --- Algorithm (PPO) ---
    "algorithm": {
        "class_name": "PPO",                            # NOTE: "PPO" (only standard option)

        # Optimiser
        "optimizer": "adam",                            # NOTE: "adam" | "sgd"
        "learning_rate": 0.001,                         # NOTE: 1e-4 | 5e-4 | 1e-3
        "schedule": "adaptive",                         # NOTE: "adaptive" | "fixed" | "empirical"

        # Data
        "num_learning_epochs": 5,                       # NOTE: 3 | 4 | 5 | 8 | 10
        "num_mini_batches": 4,                          # NOTE: 2 | 4 | 8

        # Clipping & Loss
        "clip_param": 0.2,                              # NOTE: 0.1 | 0.2 | 0.3
        "use_clipped_value_loss": True,                 # NOTE: True | False
        "value_loss_coef": 1.0,                         # NOTE: 0.5 | 1.0
        "max_grad_norm": 1.0,                           # NOTE: 0.5 | 1.0 | 5.0

        # Exploration
        "entropy_coef": 0.01,                           # NOTE: 0.0 | 0.005 | 0.01 | 0.05
        "desired_kl": 0.01,                             # NOTE: 0.005 | 0.01 | 0.02 | 0.05  (used by "adaptive" schedule)

        # Return & Advantage
        "gamma": 0.99,                                  # NOTE: 0.95 | 0.97 | 0.99 | 0.999
        "lam": 0.95,                                    # NOTE: 0.9 | 0.95 | 0.99 | 1.0
        "normalize_advantage_per_mini_batch": False,    # NOTE: True | False
    },

    # --- Optional / Hidden Keys ---
    # "resume": False,                                  # NOTE: True | False  — auto-load last checkpoint
    # "load_run": "genesis_test_6",                     # NOTE: any run_name string  — used with resume=True
    # "checkpoint": -1,                                 # NOTE: -1 (latest) | any iteration int
    # "clip_rewards": 10.0,                             # NOTE: any float  — clips rewards to [-value, value]
}

env_cfg = {
    "num_envs": 1024,
    "num_obs": 39,
    "num_actions": 9,
    "action_scales": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.3, 0.05, 0.05],
    "episode_length_s": 3.0,
    "ctrl_dt": 0.01,
    "substeps": 10,

    "is_debug": False,
    "logging_level": "info",
    "show_FPS": False,
    "is_monitor": False,

    "use_rasterizer": True,
    "performance_mode": True,
}

reward_cfg = {
    "keypoints": 1.0,
}

robot_cfg = {
    "ee_link_name": "hand",
    "gripper_link_names": ["left_finger", "right_finger"],
    # "home_pos": [0.741, 0.63, 0.023, -1.95, 0.26, 2.38, 1.21, 0.41, 0.41],
    "home_pos": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
    "ik_method": "dls_ik",
}

# --- TRAINING: functions and classes ---
def run_random_simulation(debug: str):
    """
    Runs the simulation with random actions.
    If debug is True, visualizes it using the Monitor and extracts telemetry.
    """
    # INFO: configuration modification
    show_sim = False
    show_monitor = False
    if debug == "both":
        show_sim = True
        show_monitor = True
    elif debug == "sim":
        show_sim = True
    elif debug == "monitor":
        show_monitor = True

    env_cfg["num_envs"] = 1
    env_cfg["is_debug"] = True
    env_cfg["is_monitor"] = show_monitor
    env_cfg["episode_length_s"] = 1.0       # NOTE: 1 seconds / episode for random run

    if show_sim or show_monitor:
        env_cfg["logging_level"] = "debug"
        env_cfg["show_FPS"] = True

    # INFO: env setup
    env = FastFrankaEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=show_sim,
    )

    # INFO: Setup GUI Monitor
    monitor = Monitor(sys.argv, num_envs=env.num_envs) if show_monitor else None
    
    env.reset()
    
    # Trackers for rewards
    total_rewards = [0.0] * env.num_envs

    try:
        while True:
            # If in debug mode, break the loop if the user closes the Monitor window
            if monitor and not monitor.window.isVisible():
                break 
            
            # INFO: Genesis-RL loop part
            # Generate Random Actions
            # Shape: (num_envs, 9) in range [0.0, 1.0] for absolute control
            random_actions = torch.rand((env.num_envs, env.num_actions), device=env.device)
            
            # Step Environment
            obs, rewards, dones, infos = env.step(random_actions)
            
            # INFO: Monitor Updates (Only if debug is enabled)
            if monitor:
                # Extract telemetry directly from infos
                dofs_pos = infos.get("dofs_pos")
                target_pos = infos.get("target_pos")
                dist = infos.get("dist")

                # Update Monitor (Iterate over all environment panels)
                for i in range(env.num_envs):
                    panel = monitor.env_panels[i]

                    # Update Reward Plot
                    total_rewards[i] += rewards[i].item()
                    panel.update_plot(total_rewards[i])

                    # Update Sliders
                    if dofs_pos is not None:
                        # Pass the exact physics joint angles to the sliders
                        panel.update_joints(dofs_pos[i].tolist())

                    # Update Cube Label
                    if target_pos is not None and dist is not None:
                        # Convert to standard python types to avoid tensor format errors
                        target_p = target_pos[i].tolist()
                        d = dist[i].item()

                        cube_text = f"XYZ: [{target_p[0]:.2f}, {target_p[1]:.2f}, {target_p[2]:.2f}]\ndist: {d:.4f}"
                        panel.set_cube_label(cube_text)

                    # Handle Episode Completion (Reset Plot)
                    if dones[i]:
                        total_rewards[i] = 0.0
                        panel.reset_plot()
                
                # Process GUI Events (Keep window responsive)
                monitor.update_gui()
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
    except Exception as e:
        logger.exception(f"Error in random simulation: {e}")
    finally:
        # Cleanup
        if monitor:
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

    # INFO: configuration changes
    env_cfg["num_envs"] = 1
    env_cfg["episode_length_s"] = 99999.0
    env_cfg["is_monitor"] = True
    env_cfg["is_debug"] = True
    env_cfg["logging_level"] = "debug"

    # INFO: env setup
    env = FastFrankaEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=True,
    )

    env.reset()

    # Strict Panda joint limits used for sliders
    panda_lower = torch.tensor(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0000, 0.0000],
        device=env.device,
    )
    panda_upper = torch.tensor(
        [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973, 0.0400, 0.0400],
        device=env.device,
    )
    panda_range = panda_upper - panda_lower  # element-wise span

    # INFO: Monitor + slider initialisation
    monitor = Monitor(sys.argv, num_envs=env.num_envs)
    home_pos = torch.tensor(robot_cfg["home_pos"], device=env.device)  # shape: (9,)

    for panel in monitor.env_panels:
        for i in range(len(panel.joints_var)):
            slider = panel.joints_var[i]

            # Override limits stored in the FloatSlider to match panda_lower/upper
            slider.min_val    = panda_lower[i].item()
            slider.max_val    = panda_upper[i].item()
            slider.range_span = panda_range[i].item()

            # Initialise visual position to the home pose
            slider.set(home_pos[i].item())

    # INFO: Physics / GUI loop
    total_rewards = [0.0] * env.num_envs

    def physics_loop():
        nonlocal total_rewards

        # INFO: Build action tensor from slider physical values
        action = torch.zeros((env.num_envs, env.num_actions), device=env.device)

        for env_idx in range(env.num_envs):
            panel = monitor.env_panels[env_idx]

            # Arm joints 0-6
            for i in range(7):
                slider = panel.joints_var[i]

                # GUI slider integer (0-1000) -> physical value (rad)
                gui_norm = slider.slider.value() / 1000.0
                phys_val = slider.min_val + gui_norm * slider.range_span

                # Physical value -> action in [-1, 1] expected by env.step()
                dof_span  = (env.dof_upper[i] - env.dof_lower[i]).item()
                
                # Get 0 to 1 ratio
                ratio = (phys_val - env.dof_lower[i].item()) / dof_span if dof_span != 0 else 0.0
                
                # Convert ratio to [-1, 1]
                action_val = (ratio * 2.0) - 1.0
                action[env_idx, i] = max(-1.0, min(1.0, action_val))

                # Refresh label
                slider.label.setText(f"{slider.name}: {phys_val:.4f}")

            # Gripper joints 7 & 8 --- driven by slider 7, mirrored to slider 8
            g_slider  = panel.joints_var[7]
            g_gui_norm = g_slider.slider.value() / 1000.0
            g_phys     = g_slider.min_val + g_gui_norm * g_slider.range_span

            g_dof_span = (env.dof_upper[7] - env.dof_lower[7]).item()
            g_ratio    = (g_phys - env.dof_lower[7].item()) / g_dof_span if g_dof_span != 0 else 0.0
            g_action   = (g_ratio * 2.0) - 1.0
            g_action   = max(-1.0, min(1.0, g_action))

            action[env_idx, 7] = g_action
            action[env_idx, 8] = g_action  # both fingers move together 

            # Refresh gripper labels and mirror slider 8 visually
            g_slider.label.setText(f"{g_slider.name}: {g_phys:.4f}")
            f2 = panel.joints_var[8]
            f2.slider.setValue(g_slider.slider.value())
            f2.label.setText(f"{f2.name}: {g_phys:.4f}")

        # INFO: simulation
        obs, rewards, dones, infos = env.step(action)

        target_pos = infos.get("target_pos")  # tensor (num_envs, 3), present when show_viewer=True
        dist       = infos.get("dist")        # tensor (num_envs,),   present when show_viewer=True

        # INFO: Update Monitor UI
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
    try:
        logger.info("--- GUI: MANUAL SIMULATION STARTED ---")
        monitor.start_manual_loop(physics_loop)
        exit_code = monitor.exec()
        logger.info(f"Monitor closed with exit code: {exit_code}")
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
    except Exception as e:
        logger.exception(f"Error in manual simulation: {e}", exc_info=True)
    finally:
        logger.info("Manual Simulation Ended.")

def rl_training(checkpoint_model_path):
    # INFO: run_id
    timestamp = datetime.now().strftime("%H-%M_%d-%m-%Y")
    train_cfg["run_name"] = f"{train_cfg['run_name']}_{timestamp}"

    # INFO: env setup
    env = FastFrankaEnv(
        env_cfg=env_cfg, 
        reward_cfg=reward_cfg, 
        robot_cfg=robot_cfg, 
        show_viewer=False
    )

    log_dir = os.path.join("logs", train_cfg["run_name"])
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Format device string safely (e.g., "cuda:0" or "cpu")
    device_str = f"{env.device.type}:0" if env.device.type in ["cuda", "mps"] else "cpu:0"

    # INFO: Initialize the Runner
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device_str)
    if checkpoint_model_path:
        if os.path.exists(checkpoint_model_path):
            logger.info(f"Loading checkpoint to resume training: '{checkpoint_model_path}'")
            runner.load(checkpoint_model_path)
        else:
            logger.error(f"Checkpoint path '{checkpoint_model_path}' not found!")
            exit(1)

    # INFO: start training
    try:
        runner.learn(train_cfg["max_iterations"])
    except KeyboardInterrupt:
        logger.warning("\nCtrl + C received! Cleaning up resources ...")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    finally:
        try:
            runner.save(MODEL_PATH)
        except Exception:
            pass
        logger.info(f"RL TRAINING STOPPED! model saved to: {MODEL_PATH}")

def inference_model(model_path : str, debug : str):
    # INFO: config modification
    show_sim = False
    show_monitor = False
    if debug == "both":
        show_sim = True
        show_monitor = True
    elif debug == "sim":
        show_sim = True
    elif debug == "monitor":
        show_monitor = True

    env_cfg["is_monitor"] = show_monitor
    env_cfg["num_envs"] = 1

    if show_sim or show_monitor:
        env_cfg["logging_level"] = "debug"
        env_cfg["show_FPS"] = True

    # INFO: env setup
    env = FastFrankaEnv(
        env_cfg=env_cfg, 
        reward_cfg=reward_cfg, 
        robot_cfg=robot_cfg, 
        show_viewer=show_sim
    ) 

    log_dir = os.path.join("logs", "inference", train_cfg["experiment_name"], train_cfg["run_name"])
    device_str = f"{env.device.type}:0" if env.device.type in ["cuda", "mps"] else "cpu"

    os.makedirs(log_dir, exist_ok=True)
    
    # INFO: Initialize the Runner and load the model
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device_str)

    logger.info(f"Loading model: {model_path}")
    # Load the specific model zip file
    runner.load(model_path)

    # Get the inference policy
    policy = runner.get_inference_policy(device=device_str)

    # INFO: MONITOR SETUP
    # Only initialize if the --monitor arg was passed
    monitor = Monitor(sys.argv, num_envs=env.num_envs) if show_monitor else None
    total_rewards = [0.0] * env.num_envs

    obs, _ = env.reset() 
    
    try:
        while True:
            # Check if user closed the Monitor window
            if monitor and not monitor.window.isVisible():
                break

            # INFO: INFERENCE LOOP
            # Safely extract the tensor if obs is the dictionary we created
            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            
            # Predict actions using the loaded model
            actions = policy(obs_tensor)

            # Step the environment
            obs, rewards, dones, infos = env.step(actions)

            # INFO: MONITOR UPDATES
            if monitor:
                # Extract telemetry directly from infos
                dofs_pos = infos.get("dofs_pos")
                target_pos = infos.get("target_pos")
                dist = infos.get("dist")

                # Update Monitor (Iterate over all environment panels)
                for i in range(env.num_envs):
                    panel = monitor.env_panels[i]

                    # Update Reward Plot
                    total_rewards[i] += rewards[i].item()
                    panel.update_plot(total_rewards[i])

                    # Update Sliders
                    if dofs_pos is not None:
                        # Pass the exact physics joint angles to the sliders
                        panel.update_joints(dofs_pos[i].tolist())

                    # Update Cube Label
                    if target_pos is not None and dist is not None:
                        # Convert to standard python types to avoid tensor format errors
                        target_p = target_pos[i].tolist()
                        d = dist[i].item()

                        cube_text = f"XYZ: [{target_p[0]:.2f}, {target_p[1]:.2f}, {target_p[2]:.2f}]\ndist: {d:.4f}"
                        panel.set_cube_label(cube_text)

                    # Handle Episode Completion (Reset Plot)
                    if dones[i]:
                        total_rewards[i] = 0.0
                        panel.reset_plot()
                
                # Process GUI Events (Keep window responsive)
                monitor.update_gui() 
    except KeyboardInterrupt:
        logger.warning("\nCtrl + C received! Cleaning up resources ...")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if monitor:
            monitor.close()
        logger.info(f"INFERENCE STOPPED! model={model_path}")

def reproduce_crash(crash_file: str):
    """
    Runs 1 environment and sequentially tests every action from the crashed batch.
    """
    if not os.path.exists(crash_file):
        logger.error(f"Crash file not found: {crash_file}")
        return
        
    logger.info(f"Loading crash data from: {crash_file}")
    crash_data = torch.load(crash_file)
    
    # Extract actions and count them
    crash_actions = crash_data["actions"]
    num_actions = crash_actions.shape[0]
    logger.info(f"Loaded {num_actions} actions to test.")

    # INFO: Force simulation to exactly 1 environment
    show_sim = True
    show_monitor = True

    env_cfg["num_envs"] = 1 
    env_cfg["is_debug"] = True
    env_cfg["is_monitor"] = show_monitor
    env_cfg["episode_length_s"] = 10.0   

    if show_sim or show_monitor:
        env_cfg["logging_level"] = "debug"
        env_cfg["show_FPS"] = True

    # INFO: env setup
    env = FastFrankaEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=show_sim,
    )

    # Move the entire batch of actions to the environment's device
    crash_actions = crash_actions.to(env.device)

    # INFO: Setup GUI Monitor
    monitor = Monitor(sys.argv, num_envs=env.num_envs) if show_monitor else None
    
    env.reset()
    total_rewards = [0.0] * env.num_envs

    logger.warning(">>> TESTING CRASH ACTIONS SEQUENTIALLY <<<")

    action_idx = 0  # Track which action we are currently testing

    try:
        while True:
            # If in debug mode, break the loop if the user closes the window
            if monitor and not monitor.window.isVisible():
                break 
            
            # --- THE LOOP LOGIC ---
            # Grab one single action from the batch and shape it to (1, 9)
            current_action = crash_actions[action_idx].unsqueeze(0)
            
            # Print the index BEFORE stepping so if it crashes, we know exactly who did it
            logger.debug(f"Testing action index: {action_idx}")
            
            # Step the environment
            obs, rewards, dones, infos = env.step(current_action)
            
            # Move to the next action. If we reach the end, loop back to the beginning.
            action_idx = (action_idx + 1) % num_actions
            # ----------------------
            
            # INFO: Monitor Updates
            if monitor:
                dofs_pos = infos.get("dofs_pos")
                target_pos = infos.get("target_pos")
                dist = infos.get("dist")

                # We only have 1 panel now, so i is always 0
                panel = monitor.env_panels[0]
                total_rewards[0] += rewards[0].item()
                panel.update_plot(total_rewards[0])

                if dofs_pos is not None:
                    panel.update_joints(dofs_pos[0].tolist())

                if target_pos is not None and dist is not None:
                    target_p = target_pos[0].tolist()
                    d = dist[0].item()
                    cube_text = f"XYZ: [{target_p[0]:.2f}, {target_p[1]:.2f}, {target_p[2]:.2f}]\ndist: {d:.4f}"
                    panel.set_cube_label(cube_text)

                if dones[0]:
                    total_rewards[0] = 0.0
                    panel.reset_plot()
                
                monitor.update_gui()
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
    except Exception as e:
        logger.exception(f"CRASH REPRODUCED at action index {action_idx}! Error: {e}")
        logger.error(f"The exact action tensor that broke it: {current_action}")
    finally:
        if monitor:
            monitor.close()
        logger.info("Crash Reproduction Ended.")

def main():
    """Main function to train and evaluate the model."""
    
    # INFO: different modes: --manual, --random, --training
    try:
        if args.random:
            logger.info("--- GUI: RANDOM ACTION SIMULATION ---")
            run_random_simulation(debug=args.debug)
            logger.info("FINISHED. Running simulation with random actions.")
        if args.manual:
            logger.info("--- GUI: MANUAL SIMULATION ---")
            run_manual_simulation()
        if args.training:
            logger.info("--- RL TRAINING ---")
            rl_training(checkpoint_model_path=args.load)
        if args.inference:
            logger.info("--- INFERENCE RL MODEL WITH SIMULATION ---")

            if args.load and os.path.exists(args.load):
                logger.info(f"Inferencing '{args.load}' model")
                inference_model(model_path=args.load, debug=args.debug)
            else:
                logger.error("No trained model Found!")
        if args.crash and os.path.exists(args.crash):
            logger.info("--- CRASH REPRODUCE ---")
            reproduce_crash(args.crash)
    except KeyboardInterrupt:
        logger.warning("\nCtrl + C received! Cleaning up resources ...")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    finally:
        # Force pybullet to disconnect and the app to quit
        logger.info("SHUTDOWN COMPLETE")
 
if __name__ == "__main__":
    main()
