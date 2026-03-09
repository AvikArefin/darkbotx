# RL Training Graph Analysis

# 📊 RL Training — TensorBoard Monitoring Guide

> **How to use this guide:** Check these graphs in order during training. Start with Agent Intelligence to know *if* it's learning, then drill into the others only if something looks wrong.
> 

---

## 🧠 1. Agent Intelligence — *Is the robot actually learning?*

| Graph | What It Means | Ideal Trend | Good Range | 🚨 Red Flag |
| --- | --- | --- | --- | --- |
| `Train/mean_reward` | Average score per episode | 📈 UP | Climbing toward `0` or positive | Flatlines indefinitely or collapses downward |
| `Train/mean_episode_length` | How long before a reset | 📉 DOWN | Well below your max (e.g. `< 280` if max is `300`) | Perfectly flat at max — agent is never succeeding |
| `Train/success_rate` *(custom)* | % of envs hitting `dist < success_range` | 📈 UP | Climbing from `0%` toward `100%` | Stuck at `0%` after 500+ iterations |
| `Train/mean_distance` *(custom)* | Average EE-to-target distance | 📉 DOWN | Approaching `success_range` | Flatlines above `0.3m` — agent is stuck far away |

> 💡 **Note:** `mean_episode_length` and `mean_reward` must be read **together**. A rising reward with flat episode length means the agent is getting closer but not succeeding. A falling episode length with flat reward means it's resetting often (possibly from your physics NaN recovery) not from success.
> 

---

## 🎲 2. Exploration Health — *Is the robot exploring or panicking?*

| Graph | What It Means | Ideal Trend | Good Range | 🚨 Red Flag |
| --- | --- | --- | --- | --- |
| `Policy/mean_noise_std` | Joint exploration randomness | 📉 DOWN | `1.0` → `~0.1` over training | Steady increase — agent is hopelessly confused |
| `Loss/entropy` | Mathematical measure of policy randomness | 📉 DOWN | Smooth downward slope | Climbing or bouncing violently — entropy collapse or chaos |

> 💡 **Note:** Entropy dropping **too fast** is also a red flag — it means the policy collapsed to a deterministic behavior before finding a good solution. If entropy hits near-zero before `mean_reward` is good, your `entropy_coef` in PPO config is too low.
> 

---

## 🧮 3. PPO Algorithm Health — *Is the math working?*

| Graph | What It Means | Ideal Trend | Good Range | 🚨 Red Flag |
| --- | --- | --- | --- | --- |
| `Loss/learning_rate` | Step size for weight updates | ➡️ FLAT or ↘️ slow decay | `~0.001` | Drops to `0.0` instantly — training frozen |
| `Loss/value` | Critic's reward prediction error | 📉 DOWN | Low and stable | Violent, sustained spikes — critic can't keep up |
| `Loss/surrogate` | Actor policy update magnitude | 〰️ STABLE | Slight fluctuation below `0` | Extreme vertical swings — policy changing too violently (lower `clip_param`) |
| `Loss/clip_fraction` | % of updates hitting the PPO clip boundary | 〰️ STABLE | `0.1` – `0.3` | Consistently above `0.5` — learning rate or `clip_param` is too high |

> 💡 **Note:** `Loss/value` spiking **at the start of training** is completely normal — the critic has no reference point yet. Only worry if it spikes *after* it was already stable, which signals the reward distribution suddenly changed (e.g. the agent found a new behavior).
> 

---

## ⚙️ 4. Reward Decomposition — *What is the agent actually optimizing?*

*Only relevant if you log reward components separately — strongly recommended.*

| Graph | What It Means | Ideal Trend | 🚨 Red Flag |
| --- | --- | --- | --- |
| `Reward/reaching` | `exp(-k * dist)` component | 📈 UP | Flatlines at low value — agent isn't closing distance |
| `Reward/approach_bonus` | Velocity-toward-target reward | 📈 then 〰️ STABLE | Stays at `0` — agent is stationary or moving sideways |
| `Reward/smoothness_penalty` | Action jitter penalty | 〰️ near `0` | Large negative — agent is oscillating violently |
| `Reward/success_bonus` | Sparse `+200` for dist < threshold | Rare → frequent | Never appears after 1000+ iterations — threshold may be too tight |

> 💡 **Why this matters:** Without decomposed rewards you can't tell *why* mean reward is improving. It could be the agent learning to reach, or it could just be learning to reduce jitter. Logging each term separately removes all ambiguity.
> 

---

## 💻 5. Hardware Performance — *Is the machine keeping up?*

| Graph | What It Means | Ideal Trend | Good Range | 🚨 Red Flag |
| --- | --- | --- | --- | --- |
| `Perf/total_fps` | Steps processed per second | 📈 UP or ➡️ FLAT | `150+` | Sudden permanent drop — memory leak or GPU OOM |
| `Perf/collection_time` | Time in Genesis physics sim | 📉 DOWN | `< 0.5s` | Creeping upward over hours — fragmentation or thermal throttle |
| `Perf/learning_time` | Time doing PyTorch backprop | ➡️ FLAT | `< 0.1s` | Spiking heavily — GPU bottleneck, reduce batch size |

> 💡 **Note:** Short spikes in `collection_time` on reset-heavy iterations are **expected and normal** — your physics NaN recovery resets all envs at once, which is expensive. Only worry about the **trend**, not individual spikes.
> 

---

## 🚑 6. Stability Indicators — *Is the simulation healthy?*

*These are custom metrics from your env — worth logging to TensorBoard.*

| Metric | What It Means | Good Range | 🚨 Red Flag |
| --- | --- | --- | --- |
| `Env/nan_counter` | Total physics solver divergences | As low as possible | Increasing every few hundred steps — gains too high or `dt` too large |
| `Env/timeout_rate` | % of episodes ending from timeout vs success | Decreasing over time | Stuck at `100%` — agent never succeeds, only times out |
| `Env/mean_distance_at_done` | EE distance when episode ends | 📉 DOWN | Large and constant — agent isn't improving before timeout |

---

## 🗺️ Quick Diagnosis Flowchart

```
mean_reward not improving?
├── mean_distance also flat?       → Reward shaping issue. Check exp(-k*dist) scale.
├── mean_distance improving?       → Success threshold too tight. Loosen it.
├── entropy collapsed early?       → Increase entropy_coef in PPO config.
└── nan_counter high?              → Reduce PD gains or increase substeps.

mean_reward improving but slowly?
├── success_rate still 0%?         → Add staged success reward (near bonus).
├── approach_bonus near 0?         → Agent is stationary. Check action scaling.
└── collection_time high?          → Reduce num_envs or substeps.
```

---