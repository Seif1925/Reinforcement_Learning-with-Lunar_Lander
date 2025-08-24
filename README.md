# Deep Q-Network (DQN) for LunarLander-v3

This project implements a **Deep Q-Network (DQN)** using PyTorch to solve the [LunarLander-v3](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) environment from OpenAI Gymnasium.  
The agent learns to land the lunar module safely by interacting with the environment, storing experiences, and improving its Q-value estimates through experience replay and target networks.

---

## 📌 Key Features
- **Environment**: LunarLander-v3 with RGB rendering and video recording.
- **Replay Buffer**: Stores past transitions for training stability.
- **Target Network**: A separate Q-network updated slowly to stabilize training.
- **Soft Updates**: Gradual update of target network parameters.
- **Epsilon-Greedy Strategy**: Balances exploration and exploitation.
- **Training Frequency**: Updates occur every few steps instead of every step for stability.
- **Visualization**: Plots training rewards and average loss per episode.
- **Video Recording**: Saves gameplay videos at selected episodes.
- **Model Saving**: Trained weights are saved as a `.pth` file.

---

## ⚙️ Main Parameters
- `REPLAY_MEMORY_SIZE = 100_000` → Maximum number of stored transitions.
- `TRAIN_FREQUENCY = 4` → Train the agent every 4 steps.
- `MINI_BATCH_SIZE = 100` → Sample size from replay buffer for training.
- `DISCOUNT = 0.99` → Discount factor for future rewards.
- `EPSILON_DECAY = 0.995` and `MIN_EPSILON = 0.01` → Controls exploration vs exploitation.
- `LEARNING_RATE = 5e-4` → Learning rate for Adam optimizer.
- `INTERPOLATION_PARAMETER = 1e-3` → Soft update factor for the target network.

---

## 🏗️ Code Structure

### 1. **DQN Network**
```python
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        ...
    def forward(self, x):
        ...
