import torch
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class PerformanceQNet(nn.Module): #performance QNet

    def __init__(self, input_size, hidden_size, output_size):
        super(PerformanceQNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def save(self, file_name='employee_performance_model.pth'): #save model
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        print(f"model saved to {file_name}")
    
    def load(self, file_name='employee_performance_model.pth'): #load model
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            print(f"model loaded from {file_name}")
            return True
        return False


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done): #train step
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
       
        pred = self.model(state)
        
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class PerformanceAgent:
    def __init__(self, state_size=6, action_size=4): #initialize agent
        self.n_episodes = 0
        self.epsilon = 0  
        self.gamma = 0.9  #discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        
        #state are avg_hours, std_hours, min_hours, max_hours, punctuality_rate, consistency
        self.model = PerformanceQNet(state_size, 256, action_size)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
    def get_action(self, state, training=True): #get action. random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_episodes
        final_move = [0, 0, 0, 0]
        
        if training and random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
    
        return final_move
    
    def remember(self, state, action, reward, next_state, done): #store experience in replay memory
        self.memory.append((state, action, reward, next_state, done))
        
    def train_long_memory(self): #train on batch of experiences from memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done): #train on single experience
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def evaluate_employee(self, state): #evaluate employee performance
        action = self.get_action(state, training=False)
        action_idx = action.index(1)
        return self._get_performance_label(action_idx)
    
    def _get_performance_label(self, action): #convert action to performance label
        labels = {
            0: "Poor",
            1: "Average",
            2: "Good",
            3: "Excellent"
        }
        return labels.get(action, "Unknown")
    
    