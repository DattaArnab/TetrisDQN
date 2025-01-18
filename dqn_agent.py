import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from game import Game

import random

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 20 * 10, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 20 * 10)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, device):
        self.device = device
        self.model = DQN().to(device)
        self.target_model = DQN().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.memory = deque(maxlen=50000)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.gamma = 0.99
        self.batch_size = 32

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(4)
        
        # Convert state to correct format
        state = np.array(state)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(state).argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays first
        states = np.array(states)
        next_states = np.array(next_states)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Add gradient clipping
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())