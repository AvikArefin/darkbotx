from enum import IntEnum


class SJoint(IntEnum):
    """
    Hiwonder Sim Joints

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
        "entropy_coef": 0.03,
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
        "actor": [
            "object_2d_profile",
            "object_yaw",
            "object_height",
            "current_gripper_state",
        ],
        "critic": [
            "object_2d_profile",
            "object_yaw",
            "object_height",
            "current_gripper_state",
        ],
    },
    "num_steps_per_env": 1,
    "save_interval": 25,
    "num_max_iteration": 100,
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
    "use_mtrick": False,
    # ROBOT RELATED
    "path": "assets/Hiwonder/Hiwonder.urdf",
    "rl_start_angles": [0.1, 0.2, 0, 0.0, 0, 0.0, 0.0],
    # Object related
    "object_spawn_pos": (-0.0061, -0.0617, 0.015),
    "object_spawn_yaw": 0.0,
    "object_type": "random",  # Can be "random" or a specific type like "cube", "rect", "star", "box"
    "num_periphery_points": 16,
    "object_configs": {
        # "box": {
        #     # "size": [0.03, 0.03, 0.03],
        #     "file": "assets/box/box.urdf",
        #     "fixed": False,
        #     # "batch_fixed_verts": True,
        # },
        "cube": {
            "file": "assets/cube/cube.urdf",
            "fixed": False,
        },
        "rect": {
            "file": "assets/rect/rect.urdf",
            "fixed": False,
        },
        # "star": {
        #     "file": "assets/star/star.urdf",
        #     "fixed": False,
        # },
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
    "object_type": "rect"
}

TEST_ENV_CFG = {
    **BASE_ENV_CFG,
    "num_envs": 3,
    "vis_mode": "collision",
    "show_viewer": True,
    "debug_dashboard": False,
    "debug_draw": True,
    "use_mtrick": True,
}

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ENV_CFG <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
