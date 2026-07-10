import genesis as gs
from genesis import _gs_backend

from rsl_rl.runners import OnPolicyRunner

from environment import GraspEnv
from config import RL_POLICY_CFG, TRAIN_ENV_CFG


def main():
    gs.init(backend = _gs_backend.cpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = "logs/darkbotx"
    env = GraspEnv(TRAIN_ENV_CFG)
    runner = OnPolicyRunner(env, RL_POLICY_CFG, log_dir, device=str(gs.device))
    runner.learn(num_learning_iterations=RL_POLICY_CFG.get("num_max_iteration", 300), init_at_random_ep_len=True)



if __name__ == "__main__":
    main()
