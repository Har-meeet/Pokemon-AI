import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=5000)
        self.batch_size = 64

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        """Copies weights from model to target model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
    
    def select_action_from_mask(self, state, valid_indices):
        if np.random.rand() < self.epsilon:
            return random.choice(valid_indices)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze().numpy()

        masked_q = np.full_like(q_values, -np.inf)
        for idx in valid_indices:
            masked_q[idx] = q_values[idx]

        return int(np.argmax(masked_q))


    def remember(self, state, action, reward, next_state, done):
        """Stores experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Trains model using replay memory."""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        state_batch = torch.tensor(np.array([exp[0] for exp in minibatch]), dtype=torch.float32)
        action_batch = torch.tensor([exp[1] for exp in minibatch], dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(np.array([exp[2] for exp in minibatch]), dtype=torch.float32)
        next_state_batch = torch.tensor(np.array([exp[3] for exp in minibatch]), dtype=torch.float32)
        done_batch = torch.tensor(np.array([exp[4] for exp in minibatch]), dtype=torch.float32)

        # Compute Q targets
        with torch.no_grad():
            max_next_q_values = self.target_model(next_state_batch).max(1)[0]
            q_targets = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)

        # Compute Q values for taken actions
        q_values = self.model(state_batch).gather(1, action_batch).squeeze()

        # Compute loss
        loss = self.loss_fn(q_values, q_targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reduce epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def plot_q_values(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state_tensor).detach().numpy().flatten()
        
        plt.bar(range(len(q_values)), q_values)
        plt.xlabel("Actions")
        plt.ylabel("Q-Values")
        plt.title("Q-Value Distribution")
        plt.show()