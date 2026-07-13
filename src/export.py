import os
import argparse
import torch
from pathlib import Path

import genesis as gs
from genesis import _gs_backend

from rsl_rl.runners import OnPolicyRunner

from config import RL_POLICY_CFG, EVAL_ENV_CFG
from eval import GraspEnv

def export_policy(model_path: str, output_path: str):
    # 1. Initialize Genesis in CPU mode
    gs.init(backend=_gs_backend.cpu, precision="32", logging_level="warning")

    # 2. Create the evaluation environment to get observation/action spaces
    env = GraspEnv(EVAL_ENV_CFG)

    print(f"Loading checkpoint from: {model_path}")
    log_dir = "logs/export_temp"
    
    # 3. Initialize runner and load model
    runner = OnPolicyRunner(env, RL_POLICY_CFG, log_dir, device=str(gs.device))
    runner.load(model_path)
    policy = runner.get_inference_policy(device=str(gs.device))
    jit_policy = policy.as_jit()

    # 4. Create dummy input with the environment's observation shape
    # policy expects shape: (num_envs, num_obs)
    dummy_obs = torch.zeros(1, policy.obs_dim, device=str(gs.device))

    # 5. Trace the policy using TorchScript
    print("Tracing policy using TorchScript...")
    with torch.no_grad():
        traced_policy = torch.jit.trace(jit_policy, dummy_obs)

    # 6. Save the traced policy to the output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced_policy.save(output_path)  # type: ignore
    print(f"[SUCCESS] Standalone TorchScript policy exported to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export OnPolicyRunner checkpoint to standalone TorchScript policy")
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        default=None,
        help="Path to the checkpoint file (.pt) to load (default: latest model in logs/darkbotx)"
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        default="logs/real/deployed_policy.pt",
        help="Path where the exported TorchScript policy will be saved (default: logs/real/deployed_policy.pt)"
    )
    
    args = parser.parse_args()
    
    model_path = args.model_path
    if model_path is None:
        log_dir = Path("logs") / "darkbotx"
        model_files = list(log_dir.glob("model_*.pt"))
        if not model_files:
            raise FileNotFoundError(
                f"No model checkpoints found in {log_dir}. Please train a model first."
            )
        model_path = str(max(model_files, key=os.path.getmtime))
        print(f"Automatically selected the latest model: {model_path}")

    export_policy(model_path, args.output_path)
