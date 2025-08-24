
# Deep Q-Network (DQN)

## 📌 Introduction
This project demonstrates the **Deep Q-Network (DQN)** algorithm, one of the foundational algorithms in **Deep Reinforcement Learning**.  
DQN was introduced by DeepMind to teach agents how to act optimally in an environment using **neural networks** to approximate the Q-value function.

The key idea is:
- The agent interacts with the environment.
- It collects experiences `(state, action, reward, next_state, done)`.
- These experiences are stored in a **Replay Buffer**.
- A **neural network** predicts the Q-values for each action.
- The agent updates its knowledge using the **Bellman equation**.

---

## 🎯 What This Project Does
The agent is trained to solve an environment (e.g., **LunarLander**).  
Over episodes, it learns how to maximize its total reward.  

▶️ Below you can watch a short video (`media/video.mp4`) showing how the agent improves from episode 0 up to episode 1000.

---

## 📊 Training Results

During training, we track two important metrics:

1. **Reward per Episode**  
   Shows how much reward the agent gets on average each episode.


2. **Loss per Episode**  
   Measures how well the neural network is learning to approximate Q-values.

   ![Loss per Episode](media/lossAndReward.png)

---

## 🧠 Key Concepts & Functions

Here are some of the **main functions** in the DQN code, explained step by step in *Jupyter-like cells*.

### 🔹 Soft Update
```python
def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
```
**Explanation:**  
- Ensures the **target network** slowly follows the **local network**.  
- Helps stabilize training.  
- `tau` controls the update speed (e.g., `0.001`).

---

### 🔹 Epsilon-Greedy Action Selection
```python
def act(self, state, eps=0.1):
    if random.random() > eps:
        with torch.no_grad():
            q_values = self.qnetwork_local(state)
        return np.argmax(q_values.cpu().data.numpy())
    else:
        return random.choice(np.arange(self.action_size))
```
**Explanation:**  
- With probability **1 - eps** → choose the **best action**.  
- With probability **eps** → choose a **random action**.  
- This balances **exploration** and **exploitation**.

---

### 🔹 Replay Buffer
```python
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        return random.sample(self.memory, k=self.batch_size)
```
**Explanation:**  
- Stores past experiences.  
- Sampling random minibatches breaks correlation and improves learning stability.

---

## 📂 Project Structure
```
project/
│── main.py            # Training script
│── dqn_agent.py       # DQN agent implementation
│── model.py           # Neural network model
│── media/
│   ├── reward.png     # Reward per episode plot
│   ├── loss.png       # Loss per episode plot
│   └── video.mp4      # Training progress video
│── README.md          # This file
```

---

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install torch numpy matplotlib gym
   ```
2. Train the agent:
   ```bash
   python main.py
   ```
3. View results in the `media/` folder.

---

## 📺 Training Video
👉 [Watch Training Progress](media/video.mp4)

---
