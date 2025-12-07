# Experiment V7: Pure Standard (Back to Basics) - SUCCESS REPORT

## 1. Overview
After multiple failed attempts (V1~V6) involving custom reward functions, state space reduction, and discrete action spaces, we hypothesized that the modifications themselves were the root cause of the poor performance. V7 tested this hypothesis by reverting the environment to its **pure standard configuration**.

## 2. Configuration (Pure Vanilla)
*   **Environment**: Standard `BipedalWalker-v3`
    *   **State Space**: 24 dimensions (Full Lidar, Join Velocity, Contact info).
    *   **Action Space**: Continuous Box(4) (-1.0 to 1.0 torque).
*   **Reward Function**: Original BipedalWalker Reward.
    *   Formula: `R = 130 * forward_velocity - 5.0 * torque_cost - 100 * fall_penalty + ...`
*   **Algorithm**: PPO (Stable Baselines 3 Default Hyperparameters)
*   **Success Criterion**: 300 Points (Official "Solved" standard).

## 3. Results (Success!)
The agent successfully solved the environment, reaching the target score of **300+** within **820,000 steps**.

*   **Normal Mode**: **301.04** (Solved)
    *   Result: Perfect bipedal walking. No reward hacking (kneeling/crawling) observed.
*   **Heavy Mode**: **286.10** (Robust)
    *   Even with increased leg density (x2.0), the agent adapted well.
*   **Slippery Mode**: **57.24** (Challenging)
    *   The agent struggled on low friction surfaces but avoided falling immediately.

## 4. Key Learnings (Root Cause Analysis of Previous Failures)
1.  **State Reduction was Harmful**: Reducing the state space from 24 to 14 (especially reducing Lidar to 2) blinded the agent, making it impossible to anticipate terrain.
2.  **Discrete Actions Limiting**: The 9 discrete motion primitives were too coarse for the delicate balance required by the bipedal walker. Continuous control allowed for micro-adjustments essential for stability.
3.  **Reward Hacking**: Custom rewards (e.g., Gait Bonus) unintentionally encouraged local optima like "kneeling and twitching" to maximize stability while gaining minor bonuses. The original reward is well-balanced for this task.

## 5. Conclusion
**"Don't Reinvent the Wheel."** The standard BipedalWalker-v3 environment is well-designed. Future improvements should focus on **Curriculum Learning** or **Domain Randomization** on top of this stable baseline, rather than modifying the core physics or reward structure excessively.
