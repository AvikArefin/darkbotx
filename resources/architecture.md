# Single-Step Reinforcement Learning Architecture

This document describes the environment architecture for a Single-Step Episode (also known as a Contextual Bandit) setup in Reinforcement Learning, specifically tailored for a Robotic Grasping task.

## The Objective
The goal is to train an RL agent to perform a single, critical decision step that outputs two continuous values:
1. **Grasp Angle:** The rotation of the gripper (`DJoint.WRIST_ROLL`).
2. **Squeeze Width:** How much the fingers should squeeze/close (`DJoint.GRIPPER_LEFT`).

These joint indices are mapped via the `DJoint` IntEnum to improve code readability and maintainability.

The rest of the physical movement (approaching, closing the gripper to the specified width, lifting) is handled manually (deterministically).

## Execution Flow

1. **Environment Reset (`reset()`)**
   - **Action:** The simulation resets to the initial state. The object (which can be of dynamic types such as `box` or `urdf` configured via `ENV_CONFIG`) is spawned with randomized properties (size, rotation), and the robot is reset.
   - **Pre-Action Sequence:** The robot manually moves to the "grab position" (down near the object).
   - **Observation:** The environment calculates the *Initial State* and returns it.

2. **Agent Decision (`model.predict(obs)`)**
   - The RL agent receives the initial observation and outputs a 2D continuous action: `[target_angle, squeeze_width]`.

3. **Environment Step (`step(action)`)**
   - **Action Application:** The environment applies the chosen angle and squeeze width to grab the object. This uses a unified `control_motors` method in the `Manipulator` class to ensure multiple joint targets are applied simultaneously.
   - **Post-Action Sequence:** The environment manually moves the robot to the lift position (performed in multiple interpolated stages to prevent flinging the object).
   - **Reward Calculation:** The environment evaluates the final state of the simulation (e.g., is the object lifted above a threshold?). It calculates a sparse, scalar reward.
   - **Termination:** The episode is immediately marked as finished (`done = True`).
   - **Return:** The environment returns `(next_obs, reward, done, info)`.

## Observation Space Design

In a single-step MDP, the agent learns a mapping from the **Initial State** to the **Optimal Action**. Therefore, the observation returned by `reset()` must contain all the context the agent needs to make its decision.

### Initial Observation (Returned by `reset()`)
This is the critical observation. It must describe the object's geometry and pose so the agent knows what angle to choose.

Based on the requirement for a generalized periphery and height:
- **Generalized 2D Periphery:** A discrete set of 2D boundary points relative to the object's center. For URDF objects, these are dynamically calculated at initialization by projecting the 3D mesh onto a 2D plane using `trimesh`, extracting the precise outer boundary. This gives the agent a generalized top-down spatial understanding of the object's exact geometric profile and rotation. For primitive boxes, it falls back to a 4-corner bounding box.

- **Object Height:** The `z` dimension of the object, dictating how deep the gripper should go or how tall the object is.
- **Gripper State:** The current angle and width of the gripper (if it varies before the action is taken).

*Example TensorDict Structure for `reset()`:*
```python
{
    # A 2D profile / periphery (flattened x,y boundary points relative to center)
    # Shape: [num_envs, num_points * 2]  -> N points * 2 (x,y)
    "object_2d_profile": [x1, y1, x2, y2, ..., xN, yN], 
    
    # The height of the object
    # Shape: [num_envs, 1]
    "object_height": [height],
    
    # Initial state of the gripper
    "current_gripper_state": [angle, width]
}
```

### Final Observation (Returned by `step()`)
Because the episode terminates immediately (`done = True`), the RL algorithm will **not** use the `next_obs` returned by `step()` to make another decision. Furthermore, because `done=True`, the RL algorithm does not use `next_obs` to compute future value estimates (the episode is over, so future value is 0).

Therefore, the exact contents of `next_obs` are practically **ignored by the neural network's training loop**. 

However, standard RL APIs (like Gym/Gymnasium/Isaac) require you to return an observation anyway. You have two standard options:
1. **Return the literal final state** of the simulation (e.g., the new object position high in the air, the closed gripper angle). This can be useful if you use the `next_obs` for debugging or logging success metrics.
2. **Return the exact same observation** as the initial state, just to satisfy the dictionary shape requirement.

*It doesn't matter for training which one you pick.*

## Reward Function
The reward function is evaluated *only once* at the very end of the `step()` method.
- **Success:** `+1.0` if `object.position.z > 0.1` (the lift threshold).
- **Failure:** `0.0` otherwise.

Because the reward is sparse and given strictly at the end, the RL algorithm operates as a Contextual Bandit, using policy gradients (like PPO) to maximize the expected return of its single action given the initial context (observation).

## Debugging & Visualization
The environment includes several built-in toggles configured via `ENV_CONFIG` to aid in debugging:
- **`debug_draw`**: Renders the 2D periphery boundary points as green spheres in the 3D Genesis viewer to verify the mesh projection aligns correctly with the object's physical pose.
- **`debug_dashboard`**: Clears the console and prints a real-time textual dashboard showing the most recent action, reward, object position/quaternion, and gripper state.
- **`debugline`**: Outputs the raw position value of the left gripper joint for low-level mechanical verification.
