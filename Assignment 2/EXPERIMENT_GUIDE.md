# Quick Experiment Guide

## 🚀 Running the Notebook

1. **Make sure all dependencies are installed:**

```bash
pip install torch gymnasium wandb numpy
```

2. **Set up your WandB API key** in `key.txt`

3. **Run all cells** - The main training loop will:
   - Train DQN on all 4 environments
   - Train DDQN on all 4 environments
   - Evaluate each trained agent for 100 episodes
   - Save videos every 25 episodes
   - Log everything to WandB

## 📊 What to Expect

### Training Time Estimates (on CPU):

- **CartPole-v1**: ~5-10 minutes
- **Acrobot-v1**: ~15-20 minutes
- **MountainCar-v0**: ~25-35 minutes
- **Pendulum-v1**: ~20-30 minutes

**Total: ~2-3 hours for complete run (all environments, both algorithms)**

### Expected Results:

#### CartPole-v1

- DQN: ~450-500 average reward
- DDQN: ~480-500 average reward
- Both should converge by episode 300-400

#### Acrobot-v1

- DQN: ~-100 to -80 average reward
- DDQN: ~-90 to -70 average reward
- Converges around episode 500-700

#### MountainCar-v0 (Hardest!)

- DQN: ~-120 to -110 average reward
- DDQN: ~-110 to -100 average reward
- May need full 1500 episodes to converge
- **If it doesn't converge**: Try increasing GAMMA to 0.9995 or EPS_DECAY to 3000

#### Pendulum-v1

- DQN: ~-400 to -200 average reward
- DDQN: ~-350 to -150 average reward
- Converges around episode 300-400
- **Note**: Lower (less negative) is better!

## 🔧 Troubleshooting

### If training is too slow:

- Reduce NUM_EPISODES (but results may be worse)
- Use GPU if available (automatically detected)
- Reduce REPLAY_MEMORY_SIZE

### If MountainCar doesn't learn:

- Increase GAMMA to 0.9995
- Increase EPS_DECAY to 3000
- Increase NUM_EPISODES to 2500
- Consider adding reward shaping:
  ```python
  # In the training loop, replace the reward line with:
  if env_name == "MountainCar-v0":
      # Reward based on height (position)
      shaped_reward = reward + 300 * (abs(observation[0]) - 0.5)
      reward = torch.tensor([shaped_reward], device=device)
  ```

### If WandB fails:

- Set `USE_WANDB = False` in first cell
- Training will continue without logging

### If videos aren't recording:

- Check that `videos/` folder is created
- Ensure you have write permissions
- On Windows, may need to install: `pip install moviepy`

## 📈 Hyperparameter Tuning Guide

### To test different hyperparameters:

1. **Modify the config in the second cell**:

```python
env_configs = {
    "CartPole-v1": {
        "LR": 1e-3,  # <-- Change this
        "EPS_DECAY": 200,  # <-- Or this
        # ... etc
    }
}
```

2. **Delete the saved model** to retrain:

```bash
del DQN_CartPole-v1_policy.pth
del DDQN_CartPole-v1_policy.pth
```

3. **Re-run the training cell**

### Recommended experiments for the report:

#### Experiment 1: Effect of Learning Rate

Test LR: [1e-4, 5e-4, 1e-3, 5e-3]

- Expected: Higher LR = faster but less stable

#### Experiment 2: Effect of Discount Factor

Test GAMMA: [0.95, 0.99, 0.995, 0.999]

- Expected: Higher GAMMA = better long-term planning

#### Experiment 3: Effect of Epsilon Decay

Test EPS_DECAY: [100, 500, 1000, 2000]

- Expected: Slower decay = more exploration

#### Experiment 4: Effect of Batch Size

Test BATCH_SIZE: [32, 64, 128, 256]

- Expected: Larger batches = more stable but slower

#### Experiment 5: Effect of Replay Memory

Test REPLAY_MEMORY_SIZE: [5000, 10000, 50000, 100000]

- Expected: Larger buffer = more diverse experiences

## 📝 Generating Report Figures

### From WandB:

1. Go to your project: https://wandb.ai/your-username/RL-Assignment2-FINAL
2. For each run, export plots:
   - Training reward curve
   - Loss curve
   - Evaluation metrics table

### From Code:

Add this after training to save plots locally:

```python
import matplotlib.pyplot as plt

# Plot training rewards
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards_train)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title(f'{algo_name} on {env_name} - Training Rewards')
plt.savefig(f'{algo_name}_{env_name}_training.png')
plt.close()
```

## 🎯 Report Structure Template

### Section 1: Introduction

- Brief explanation of DQN/DDQN
- Assignment objectives

### Section 2: Implementation

- Network architecture
- Hyperparameters used (use the table from IMPLEMENTATION_REVIEW.md)
- Training procedure

### Section 3: Results (For Each Environment)

**CartPole-v1:**

- Q1: Training time comparison (DQN: X min, DDQN: Y min)
- Q1: Performance comparison (include reward curves)
- Q2: Stability analysis (episode duration plot with std)
- Q3: Hyperparameter effects (ablation study plots)
- Q4: Suitability discussion (DQN is perfect for CartPole because...)

**Repeat for Acrobot-v1, MountainCar-v0, Pendulum-v1**

### Section 4: Overall Comparison

- Summary table of all results
- DQN vs DDQN comparison across all environments
- Lessons learned

### Section 5: Conclusion

- Best hyperparameters found
- When to use DQN vs DDQN
- Limitations and future work

## 🎬 Video Selection for Report

For each environment, include 2-3 videos:

1. **Before training** (or early training): Shows random behavior
2. **After training**: Shows learned policy
3. **(Optional) Failure case**: If any test episodes failed

Recommended tool for video to GIF (for LaTeX):

```bash
ffmpeg -i video.mp4 -vf "fps=10,scale=320:-1:flags=lanczos" output.gif
```

## 📚 References to Cite

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature.
2. Van Hasselt, H., et al. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.
3. Brockman, G., et al. (2016). "OpenAI Gym." arXiv preprint.
4. PyTorch DQN Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Good luck! 🚀
