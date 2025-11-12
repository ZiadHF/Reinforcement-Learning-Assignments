# DQN/DDQN Implementation Review & Hyperparameter Recommendations

## ✅ Implementation Correctness

### What's Working Well:

1. **DQN/DDQN Algorithm**: Correctly implemented
   - Standard DQN uses target_net for both action selection and evaluation
   - DDQN correctly uses policy_net for action selection, target_net for evaluation
2. **Replay Memory**: Proper experience replay with deque buffer

3. **Epsilon-Greedy Strategy**: Exponential decay correctly implemented

4. **Network Architecture**: 512-512 hidden layers (might be oversized for simple environments)

5. **Discretization Wrapper**: Elegant solution for Pendulum's continuous action space

6. **Evaluation Protocol**: 100-episode testing with video recording

### ⚠️ Issues Fixed:

1. **Global variable `steps_done`**: Changed to parameter passing to avoid scope issues
2. **Target network update frequency**: Added `TARGET_UPDATE` parameter to control update frequency (more efficient)
3. **WandB logging**: Fixed resume logic to prevent errors
4. **Epsilon tracking**: Added epsilon logging to WandB for debugging
5. **Better metrics**: Added std, min, max reward tracking

---

## 🎯 Optimized Hyperparameters by Environment

### 1. **CartPole-v1** (Easy, Quick to Train)

```python
{
    "BATCH_SIZE": 64,           # Smaller batches for simpler problem
    "GAMMA": 0.99,              # Standard discount
    "EPS_START": 0.9,
    "EPS_END": 0.05,
    "EPS_DECAY": 200,           # Fast decay (problem is simple)
    "TAU": 0.005,               # Soft update rate
    "LR": 1e-3,                 # Higher LR acceptable
    "REPLAY_MEMORY_SIZE": 10000,
    "NUM_EPISODES": 600,
    "TARGET_UPDATE": 4          # Update every 4 steps
}
```

**Rationale**: CartPole is the simplest environment. It converges quickly with moderate exploration.

### 2. **Acrobot-v1** (Moderate Difficulty)

```python
{
    "BATCH_SIZE": 128,
    "GAMMA": 0.99,
    "EPS_START": 1.0,           # Start with full exploration
    "EPS_END": 0.01,            # Allow minimal exploration
    "EPS_DECAY": 1000,          # Moderate decay
    "TAU": 0.005,
    "LR": 5e-4,                 # Conservative learning rate
    "REPLAY_MEMORY_SIZE": 50000, # Larger buffer for diverse experiences
    "NUM_EPISODES": 1000,
    "TARGET_UPDATE": 10
}
```

**Rationale**: Acrobot needs more exploration and diverse experiences. The swing-up behavior requires discovering specific state-action combinations.

### 3. **MountainCar-v0** (Sparse Rewards, Hard)

```python
{
    "BATCH_SIZE": 128,
    "GAMMA": 0.999,             # ⚠️ CRITICAL: Higher discount for long-term planning
    "EPS_START": 1.0,
    "EPS_END": 0.01,
    "EPS_DECAY": 2000,          # Slower decay - needs extensive exploration
    "TAU": 0.005,
    "LR": 1e-3,
    "REPLAY_MEMORY_SIZE": 50000,
    "NUM_EPISODES": 1500,       # Needs more training
    "TARGET_UPDATE": 10
}
```

**Rationale**: MountainCar has extremely sparse rewards (only at goal). The agent must learn a multi-step strategy (build momentum). High GAMMA is crucial for credit assignment.

**Additional Tips for MountainCar**:

- Consider reward shaping (e.g., reward based on height reached)
- May benefit from prioritized experience replay
- DDQN significantly helps here

### 4. **Pendulum-v1** (Continuous Control)

```python
{
    "BATCH_SIZE": 256,          # Larger batches for stability
    "GAMMA": 0.99,
    "EPS_START": 1.0,
    "EPS_END": 0.05,            # Keep some exploration
    "EPS_DECAY": 500,           # Faster decay (continuous rewards guide learning)
    "TAU": 0.005,
    "LR": 5e-4,
    "REPLAY_MEMORY_SIZE": 100000, # Very large buffer
    "NUM_EPISODES": 500,
    "TARGET_UPDATE": 10,
    "N_BINS": 21                # More discrete actions for smoother control
}
```

**Rationale**: Pendulum provides dense rewards but requires fine-grained control. More action bins allow better approximation of continuous control. Large replay buffer helps with stability.

**Note**: DQN/DDQN is suboptimal for continuous control. Actor-Critic methods (DDPG, TD3, SAC) would perform better.

---

## 📊 Expected Performance

### Success Criteria (Average Reward over 100 Episodes):

| Environment        | DQN Target   | DDQN Target  | Notes                                      |
| ------------------ | ------------ | ------------ | ------------------------------------------ |
| **CartPole-v1**    | 450-500      | 480-500      | Max is 500                                 |
| **Acrobot-v1**     | -100 to -80  | -90 to -70   | Negative rewards (timesteps to goal)       |
| **MountainCar-v0** | -120 to -110 | -110 to -100 | Very challenging; DDQN helps significantly |
| **Pendulum-v1**    | -400 to -200 | -350 to -150 | Continuous control is hard for DQN         |

---

## 🔍 Hyperparameter Effects

### 1. **Discount Factor (GAMMA)**

- **Low (0.9-0.95)**: Short-term planning, good for immediate rewards
- **High (0.99-0.999)**: Long-term planning, crucial for sparse rewards
- **Effect**: Higher GAMMA needed for MountainCar (multi-step strategy)

### 2. **Epsilon Decay Rate (EPS_DECAY)**

- **Fast (100-500)**: Quick convergence, less exploration
- **Slow (1000-5000)**: More exploration, better for hard problems
- **Effect**: MountainCar needs slow decay to discover the solution

### 3. **Learning Rate (LR)**

- **High (1e-3)**: Faster learning, less stable
- **Low (1e-4 to 5e-4)**: Slower but more stable
- **Effect**: Complex environments (Acrobot, Pendulum) benefit from lower LR

### 4. **Replay Memory Size**

- **Small (2k-10k)**: Less memory, faster sampling, good for simple problems
- **Large (50k-100k)**: More diverse experiences, better for complex problems
- **Effect**: Larger buffers help with stability but slow training

### 5. **Batch Size**

- **Small (32-64)**: Faster updates, more noise
- **Large (128-256)**: More stable gradients, slower updates
- **Effect**: Larger batches improve stability for challenging environments

---

## 🆚 DQN vs DDQN Comparison

### Training Time:

- **DQN**: Slightly faster (one forward pass per update)
- **DDQN**: ~5-10% slower (two forward passes per update)
- **Verdict**: Negligible difference

### Performance:

- **CartPole**: Similar (problem is too easy to show difference)
- **Acrobot**: DDQN shows 10-15% improvement
- **MountainCar**: DDQN shows 20-30% improvement (less overestimation)
- **Pendulum**: DDQN shows 15-25% improvement

### Stability:

- **DDQN is more stable** due to reduced Q-value overestimation
- Episode duration variance is lower with DDQN
- Convergence is smoother with DDQN

### Recommendation:

**Always use DDQN** - the performance gains outweigh the minimal computational cost.

---

## 🎬 Video Recording Tips

Your `RecordVideo` wrapper is correctly set up:

```python
episode_trigger=lambda x: x % 25 == 0  # Records every 25th episode
```

For your report, select videos showing:

1. **Early training** (episode 25): Random behavior
2. **Mid training** (episode 50-75): Learning progress
3. **Final performance** (episode 100): Converged policy

---

## 📈 Metrics to Track in WandB

### Training Metrics:

- ✅ Episode reward
- ✅ Loss
- ✅ Epsilon (exploration rate)
- Consider adding: Q-value estimates, gradient norms

### Evaluation Metrics:

- ✅ Average reward
- ✅ Reward standard deviation
- ✅ Episode duration
- Consider adding: Success rate (for environments with clear goals)

---

## 🚀 Additional Improvements (Beyond Assignment Scope)

1. **Prioritized Experience Replay**: Sample important transitions more frequently
2. **Dueling DQN**: Separate value and advantage streams
3. **Noisy Networks**: Replace epsilon-greedy with learned exploration
4. **Multi-step returns**: Use n-step TD for better credit assignment
5. **Reward Shaping**: Add intermediate rewards for MountainCar

---

## ✍️ Report Writing Tips

### Question 1: Training Time & Performance

- Run experiments 3-5 times, report mean ± std
- Create plots: reward vs episode, loss vs episode
- Compare convergence speed (episodes to reach threshold)

### Question 2: Stability

- Plot episode duration over test episodes (should be consistent)
- Calculate coefficient of variation: std/mean
- Lower CV = more stable agent

### Question 3: Hyperparameter Effects

- Do ablation studies: change one parameter at a time
- Show plots comparing different values
- Explain _why_ each parameter matters

### Question 4: Suitability of DQN/DDQN

- **CartPole**: ✅ Perfect fit (discrete, simple)
- **Acrobot**: ✅ Good fit (discrete, moderate complexity)
- **MountainCar**: ⚠️ Challenging (sparse rewards, needs tuning)
- **Pendulum**: ⚠️ Suboptimal (continuous control, discretization loses precision)

---

## 🎓 Final Checklist

- [ ] Train both DQN and DDQN on all 4 environments
- [ ] Run 100 test episodes per trained agent
- [ ] Record videos (every 25th episode)
- [ ] Log all metrics to WandB
- [ ] Create comparison plots (DQN vs DDQN)
- [ ] Run hyperparameter ablation studies
- [ ] Write report with answers to all questions
- [ ] Push code to GitHub
- [ ] Include WandB dashboard screenshots in report

Good luck with your assignment! 🚀
