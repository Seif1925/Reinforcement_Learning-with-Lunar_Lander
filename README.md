
# 🚀 Deep Q-Network (DQN) for LunarLander-v3

This project implements a **Deep Q-Network (DQN)** using **PyTorch** and **Gymnasium** to solve the **LunarLander-v3** environment.  
The agent learns to land a lunar module safely by interacting with the environment, storing past experiences, and updating its neural network.

---

## 📌 1. Setup

```python
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np
import random
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device set to: {device}")
```

---

## 📌 2. Hyperparameters

```python
REPLAY_MEMORY_SIZE = 100_000
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 1_000
DISCOUNT = 0.99
MINI_BATCH_SIZE = 100
INTERPOLATION_PARAMETER = 1e-3  
TRAIN_FREQUENCY = 4  
LEARNING_RATE = 5e-4  
```

- `TRAIN_FREQUENCY = 4` → Train every 4 steps (not every step) to improve stability.  
- `EPSILON_DECAY` → Controls how quickly the agent shifts from exploration to exploitation.  
- `INTERPOLATION_PARAMETER` → Factor for **soft updating** the target network.  

---

## 📌 3. The Q-Network

A **neural network** approximates Q-values for each action.  

```python
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(DQNNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

---

## 📌 4. The Agent

The agent manages training, memory, and action selection.  

```python
class DQNAgent:
    def __init__(self):
        self.model = DQNNetwork(state_shape, num_actions).to(device)
        self.target_model = DQNNetwork(state_shape, num_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.t_step = 0
        self.losses = []
    
    def train(self, done):
        # Train every few steps to improve stability
        self.t_step = (self.t_step + 1) % TRAIN_FREQUENCY
        if self.t_step != 0:
            return
        ...
```

---

## 📌 5. Training Loop

```python
for episode in range(EPISODES):
    current_state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(current_state, epsilon)
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done)
        
        current_state = new_state
        total_reward += reward
```

---

## 📌 6. Visualization

```python
# Plot rewards
plt.plot(episode_rewards, color='blue', alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Rewards per Episode")
plt.show()
```

---

## 📦 Outputs

- `videos/` → contains recorded gameplay episodes  
- `dqn_lunarlander_model.pth` → saved PyTorch model weights  
- Training plots for rewards and losses  

---

## 🚀 Run

```bash
pip install torch gymnasium[box2d] matplotlib numpy
python dqn_lunarlander.py
```
