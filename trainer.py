import os
import numpy as np
import torch
import torch.nn as nn
from data_processor import AttendanceDataProcessor
from environment import EmployeePerformanceEnv
from agent import PerformanceAgent


class PerformanceTrainer: #trainer class
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.processor = AttendanceDataProcessor(data_directory)
        self.env = None
        self.agent = None
        self.is_trained = False
        
    def initialize_agent(self): #initialize agent
        if self.agent is None:
            self.agent = PerformanceAgent(state_size=6, action_size=4)
            # Try to load existing model
            if self.agent.model.load():
                self.is_trained = True
                print("loaded existing trained model")
            else:
                print("train from scratch")
        
        if self.env is None:
            self.env = EmployeePerformanceEnv(self.processor)
    
    def train_initial_model(self, n_episodes=100):
        print("\n" + "="*60)
        print("initial model training")
        print("="*60)
        
        #process attendance data
        self.processor.process_attendance_data()
        
        if self.processor.employee_data.empty:
            print("no data available for training")
            return False
        
        print(f"processed data for {len(self.processor.employee_data)} employees")
        
        #initialize agent and environment
        self.initialize_agent()
        
        #training loop
        print(f"\nstarting RL training for {n_episodes} episodes...\n")
        total_score = 0
        record = float('-inf')
        
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_score = 0
            
            #train on all employees
            for _ in range(self.env.get_employee_count()):
                action = self.agent.get_action(state, training=True)
                next_state, reward, done = self.env.step(action)
                episode_score += reward
                
                self.agent.train_short_memory(state, action, reward, next_state, done)
                self.agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                if done:
                    break
            
            #train long memory
            self.agent.n_episodes += 1
            self.agent.train_long_memory()
            
            if episode_score > record:
                record = episode_score
                self.agent.model.save()
            
            total_score += episode_score 
            mean_score = total_score / (episode + 1)
            
            if (episode + 1) % 10 == 0 or episode == 0: 
                print(f"Episode {episode + 1}/{n_episodes} | Score: {episode_score:.1f} | Mean: {mean_score:.1f} | Record: {record:.1f}")
        
        print("\n" + "="*60)
        print("initial Training Complete")
        print(f"Best Score: {record:.1f}")
        print(f"Final Mean Score: {mean_score:.1f}")
        print("="*60 + "\n")
        
        self.is_trained = True
        return True
    
    def train_incremental(self, employee_name, correct_label, n_episodes=100): #train incremental for specfic employee
        print("\n" + "="*60)
        print("targeted training for specific employee")
        print("="*60)
        
        if self.agent is None:
            print("agent not initialized")
            return False
        
        if not self.is_trained:
            print("model not trained yet")
            return False
        
        #get the specific employee's state
        self.processor.process_attendance_data()
        state = self.processor.get_employee_state(employee_name)
        
        if state is None:
            print(f"employee '{employee_name}' not found!")
            return False
        
        label_names = {0: "Poor", 1: "Average", 2: "Good", 3: "Excellent"}
        print(f"✓ Training specifically on: {employee_name}")
        print(f"✓ Correct label: {label_names[correct_label]}")
        print(f"Starting targeted training for {n_episodes} episodes...\n")
        
        #use supervised learning approach
        state_tensor = torch.tensor(state, dtype=torch.float32)
        target = torch.zeros(4, dtype=torch.float32)
        target[correct_label] = 1.0
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.agent.model.parameters(), lr=0.01)
        
        best_loss = float('inf')
        
        for episode in range(n_episodes):
            #forward pass
            output = self.agent.model(state_tensor)
            
            #calculate loss
            loss = criterion(output, target)
            
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #get current prediction
            with torch.no_grad():
                prediction = torch.argmax(output).item()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                self.agent.model.save()
            
            if (episode + 1) % 10 == 0 or episode == 0:
                status = "✓" if prediction == correct_label else "✗"
                print(f"Episode {episode + 1}/{n_episodes} | Loss: {loss.item():.4f} | Prediction: {label_names[prediction]} {status}")
        
        print("\n" + "="*60)
        print("targeted training complete")
        print(f"  Best Loss: {best_loss:.4f}")
        print(f"  Model learned: {employee_name} → {label_names[correct_label]}")
        print("="*60 + "\n")
        
        return True
    
    def get_agent(self): #get agent
        if self.agent is None:
            self.initialize_agent()
        return self.agent
    
    def get_processor(self): #get data processor
        return self.processor
    
    def is_model_trained(self): #check if model is trained
        return self.is_trained
