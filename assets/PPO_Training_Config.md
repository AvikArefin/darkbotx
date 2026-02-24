# PPO Training Config

# 🧠 `train_cfg` — PPO Training Configuration Reference

> **Library:** `rsl_rl` · **Algorithm:** PPO · **Runner:** `OnPolicyRunner`
> 

---

## 📦 1. General & Runner

> Controls the overall training loop — how long it runs, how data is collected, and where results are saved.
> 

| Parameter | Your Value | Type | What It Does | Effect on Training | Options |
| --- | --- | --- | --- | --- | --- |
| `class_name` | `"OnPolicyRunner"` | str | Selects the training runner class | On-policy = collects fresh data every iteration. No experience replay. | `"OnPolicyRunner"` only in standard rsl_rl |
| `num_steps_per_env` | `24` | int | Steps each env takes before a PPO update. **Horizon length H.** Total batch = `num_envs × H` | ⬆ larger = more stable gradient, slower updates. ⬇ smaller = noisier gradient, faster updates. `24` with `10 envs` = only **240 samples/update** — very small | `16`, `24`, `48`, `96`, `128`, `256` |
| `max_iterations` | `args.training` | int | Total PPO update iterations before stopping | Total env steps = `max_iterations × num_envs × num_steps_per_env` | Any `int > 0` |
| `seed` | `1` | int | Global random seed for reproducibility | Same seed = same trajectory order, weight init, env resets | Any integer |
| `obs_groups` | `{"actor": ["policy"], "critic": ["policy"]}` | dict | Maps network inputs to env observation keys | Actor and Critic both receive `obs["policy"]`. Can give Critic extra privileged obs. | Keys: `"actor"`, `"critic"`. Values: list of obs dict keys |

### 💡 `num_steps_per_env` — Suggested Values

| Scenario | Value |
| --- | --- |
| Fast iteration / debugging | `24` – `48` |
| Balanced (recommended) | `64` – `96` |
| Stable training / complex tasks | `128` – `256` |

---

## 📁 2. Logging & Checkpoints

> Controls where and how often training state is saved and reported.
> 

| Parameter | Your Value | Type | What It Does | Effect on Training | Options |
| --- | --- | --- | --- | --- | --- |
| `save_interval` | `40` | int | Save a `.pt` checkpoint every N iterations | Lower = more disk usage, safer recovery. Higher = less I/O overhead. | Any `int > 0` |
| `experiment_name` | `"franka_fast_reach"` | str | Parent folder for all logs/models | Organises runs under `logs/franka_fast_reach/` | Any string |
| `run_name` | `"genesis_test_7"` | str | Sub-folder for this specific run | Each unique `run_name` = separate log stream | Any string |
| `logger` | `"tensorboard"` | str | Backend for metric logging | Determines where loss/reward curves appear | `"tensorboard"`, `"wandb"`, `"neptune"` |

---

## 🎭 3. Actor Network

> The **policy** — outputs action distributions. Stochastic during training, deterministic at inference.
> 

| Parameter | Your Value | Type | What It Does | Effect on Training | Options |
| --- | --- | --- | --- | --- | --- |
| `class_name` | `"MLPModel"` | str | Neural network architecture | MLP = fast, simple feedforward. RNN = has memory over time. | `"MLPModel"`, `"RNN"` (some forks) |
| `hidden_dims` | `[256, 128, 64]` | list[int] | Size of each hidden layer | ⬆ larger = more capacity, slower. ⬇ smaller = faster, may underfit. 3-layer taper is common. | `[64,64]`, `[256,128,64]`, `[512,256,128]` |
| `activation` | `"elu"` | str | Nonlinearity between layers | `elu` avoids dead neurons (unlike `relu`), smooth gradients | `"elu"` ✅, `"relu"`, `"tanh"`, `"selu"` |
| `obs_normalization` | `True` | bool | Tracks running mean/variance of obs, normalizes input | **Almost always True.** Prevents large obs values from destabilising training | `True`, `False` |
| `stochastic` | `True` | bool | Output is a distribution (mean + std) rather than a single value | Must be `True` for Actor — needed for PPO's entropy and log-prob calculations | Actor: `True` · Critic: `False` |
| `init_noise_std` | `1.0` | float | Initial std dev of action distribution — controls early exploration width | `1.0` = broad exploration early. Decays as training progresses via adaptive KL. Too low = premature convergence. | `0.5` – `1.5` |
| `noise_std_type` | `"scalar"` | str | How std dev is parameterised internally | `"scalar"` = single shared std. `"log"` = learn log(std), more stable for very small values. | `"scalar"`, `"log"` |
| `state_dependent_std` | `False` | bool | If True, network outputs std from obs. If False, std is a standalone learnable param. | `False` = simpler, more stable. `True` = richer but harder to train. | `True`, `False` |

---

## 🧑‍⚖️ 4. Critic Network

> The **value function** — estimates expected future return. Not used at inference time.
> 

| Parameter | Your Value | Type | What It Does | Effect on Training | Options |
| --- | --- | --- | --- | --- | --- |
| `class_name` | `"MLPModel"` | str | Same as Actor | — | `"MLPModel"` |
| `hidden_dims` | `[256, 128, 64]` | list[int] | Critic capacity | Critic often benefits from same or slightly larger dims than Actor | `[256,128,64]`, `[512,256,128]` |
| `activation` | `"elu"` | str | Same as Actor | — | `"elu"` ✅ |
| `obs_normalization` | `True` | bool | Same as Actor | Especially important for Critic since value targets can have large scale | `True`, `False` |
| `stochastic` | `False` | bool | Critic outputs a single scalar V(s) | Must be `False` — Critic is deterministic | `False` |

---

## ⚙️ 5. PPO Algorithm Hyperparameters

> The core math of PPO. These have the biggest impact on convergence quality and speed.
> 

### 5a. Optimiser & Learning Rate

| Parameter | Your Value | Type | What It Does | Effect on Training | Options |
| --- | --- | --- | --- | --- | --- |
| `optimizer` | `"adam"` | str | Gradient descent algorithm | Adam adapts per-parameter lr, works well out of the box | `"adam"` ✅, `"sgd"` |
| `learning_rate` | `0.001` | float | Step size for weight updates | ⬆ faster learning but unstable. ⬇ slower but more stable. With `"adaptive"` schedule this adjusts automatically. | `1e-4` – `1e-3` |
| `schedule` | `"adaptive"` | str | How lr changes over training | `"adaptive"` watches KL divergence and rescales lr to stay near `desired_kl`. Highly recommended. | `"adaptive"` ✅, `"fixed"`, `"empirical"` |

### 5b. Data & Epochs

| Parameter | Your Value | Type | What It Does | Effect on Training | Options |
| --- | --- | --- | --- | --- | --- |
| `num_learning_epochs` | `5` | int | How many full passes over the collected rollout data per PPO update | ⬆ more gradient steps from same data = sample efficient but risks overfit. `4–8` is standard. | `3` – `10` |
| `num_mini_batches` | `4` | int | Splits the rollout into N chunks for each epoch pass. Mini-batch size = `total_steps / N` | ⬆ more minibatches = smaller batches = noisier but regularising gradient. With only 240 total steps, `4` gives **60 samples/minibatch** — very small. | `2`, `4`, `8` |

> ⚠️ **With `num_envs=10` and `num_steps_per_env=24`:** total = 240 steps → 4 minibatches of 60 each. Increasing `num_steps_per_env` to `64` gives 640 steps → minibatches of 160, much healthier.
> 

### 5c. Clipping & Loss

| Parameter | Your Value | Type | What It Does | Effect on Training | Options |
| --- | --- | --- | --- | --- | --- |
| `clip_param` | `0.2` | float | PPO's ε — limits how much the policy can change per update: ratio ∈ [1-ε, 1+ε] | ⬆ allows larger updates, can destabilise. ⬇ overly conservative. `0.2` is the canonical default. | `0.1` – `0.3` |
| `use_clipped_value_loss` | `True` | bool | Applies same clipping mechanism to the Critic's value loss | Prevents Critic from updating too aggressively. Recommended `True`. | `True`, `False` |
| `value_loss_coef` | `1.0` | float | Weight of Critic loss in the combined loss: `L = L_actor + coef × L_critic` | ⬆ Critic trains faster relative to Actor. `1.0` is balanced. | `0.5` – `1.0` |
| `max_grad_norm` | `1.0` | float | Gradient clipping threshold — prevents exploding gradients | Essential for stability. `1.0` is standard. | `0.5` – `5.0` |

### 5d. Exploration & Reward Shaping

| Parameter | Your Value | Type | What It Does | Effect on Training | Options |
| --- | --- | --- | --- | --- | --- |
| `entropy_coef` | `0.01` | float | Bonus reward for policy randomness: `L -= coef × H(π)` | Prevents premature convergence. ⬆ more exploration but slower convergence. `0.01` is mild. | `0.0` – `0.05` |
| `desired_kl` | `0.01` | float | Target KL divergence between old and new policy per update (used by `"adaptive"` lr schedule) | Higher = larger allowed policy jumps. Lower = conservative updates. `0.01` is standard. | `0.005` – `0.05` |

### 5e. Return & Advantage Estimation

| Parameter | Your Value | Type | What It Does | Effect on Training | Options |
| --- | --- | --- | --- | --- | --- |
| `gamma` | `0.99` | float | Discount factor — how much future rewards are discounted | `0.99` = cares about ~100 steps ahead. ⬇ shorter horizon, myopic. ⬆ longer horizon, harder to train. | `0.95` – `0.999` |
| `lam` | `0.95` | float | GAE λ — trades off bias vs variance in advantage estimates | `lam=1.0` = pure Monte Carlo (high variance). `lam=0.0` = pure TD (high bias). `0.95` is standard. | `0.9` – `0.99` |
| `normalize_advantage_per_mini_batch` | `False` | bool | Normalises advantages within each minibatch vs the full batch | `True` = more consistent gradient scale across minibatches. Useful when batch sizes vary. | `True`, `False` |

---

## ➕ 6. Hidden / Optional Keys (not in your config)

These are valid `rsl_rl` keys you can add when needed:

| Key | Type | What It Does | When to Use |
| --- | --- | --- | --- |
| `resume` | bool | Auto-load last checkpoint from `experiment_name/run_name` | When resuming an interrupted run |
| `load_run` | str | Folder name to load checkpoint from (used with `resume: True`) | Loading from a different run name |
| `checkpoint` | int | Specific iteration to resume from. `-1` = latest | Resuming from a particular point |
| `clip_rewards` | float | Clips env rewards to `[-value, value]` before network sees them | When reward scale is unstable or very large |

---
