from enum import IntEnum


class DJoint(IntEnum):
    """
    Hiwonder Joints

    Don't use GRIPPER_RIGHT that one is already mirrored from gripper left
    """
    BASE = 0           
    SHOULDER = 1     
    ELBOW = 2          
    WRIST = 3     
    WRIST_ROLL = 4        
    GRIPPER_LEFT = 5    
    GRIPPER_RIGHT = 6  

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> RSL_RL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

RL_POLICY_CFG = {
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
        "hidden_dims": [256, 256, 128],
        "activation": "relu",
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 256, 128],
        "activation": "relu",
    },
    "obs_groups": {
        "actor": ["object_2d_profile", "object_height", "current_gripper_state"],
        "critic": ["object_2d_profile", "object_height", "current_gripper_state"],
    },
    "num_steps_per_env": 1,
    "save_interval": 100,
    "num_max_iteration": 300,
}

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< RSL_RL <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ENV_CFG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

BASE_ENV_CFG = {
    "num_actions": 2,

    # Debug / Actual Observation Related
    "image_resolution": (1280, 960),
    "vis_mode": None,
    "debug_draw": False,
    "debug_dashboard": False,
    "debugline": False,

    # ROBOT RELATED
    "path": "assets/Hiwonder/Hiwonder.urdf",
    "rl_start_angles": [0.1, 0.2, 0, 0.0, 0, 0.0, 0.0],

    # Object related
    "object_spawn_pos": (-0.0061, -0.0617, 0.015),
    "object_type": "urdf", # Change this to "box" or "urdf" to swap shapes
    "object_configs": {
        "box": {
            "size": [0.03, 0.03, 0.03],
            "fixed": False,
            "batch_fixed_verts": True,
        },
        "urdf": {
            "file": "assets/cube/cube.urdf", 
            "fixed": False,
        }
    },


}

TRAIN_ENV_CFG = {
    **BASE_ENV_CFG,
    "num_envs": 1000,
    "show_viewer": False,
}

EVAL_ENV_CFG = {
    **BASE_ENV_CFG,
    "num_envs": 1,
    "show_viewer": True,
}

TEST_ENV_CFG = {
    **BASE_ENV_CFG,
    "num_envs": 1,
    "vis_mode": 'collision',
    "show_viewer": True,
    "debug_dashboard": True, 
}

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ENV_CFG <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
