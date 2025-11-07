import numpy as np
import pandas as pd
import os

class EmployeePerformanceEnv: #environment for employee performance evaluation
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.employees = data_processor.get_all_employees()
        self.current_step = 0
        self.current_employee = None
        self.human_feedback = self._load_human_feedback()
        
    def reset(self): #reset the environment 
        self.current_step = 0
        if self.employees:
            self.current_employee = self.employees[self.current_step]
            return self.data_processor.get_employee_state(self.current_employee)
        return np.zeros(6, dtype=np.float32)
    
    def get_state(self, employee_name): #get state for a specific employee
        return self.data_processor.get_employee_state(employee_name)
    
    def calculate_reward(self, predicted_action_idx, employee_name): #calculate reward based on predicted action and real value
        true_label = self.get_ground_truth_label(employee_name)
        
        if true_label is None:
            return 0
        
        #get +10 for correct prediction and -abs(diff)*5 for wrong prediction
        if predicted_action_idx == true_label:
            return 10.0 
        else:
            return -abs(predicted_action_idx - true_label) * 5
    
    def step(self, action_one_hot):
        if self.current_step >= len(self.employees):
            return np.zeros(6, dtype=np.float32), 0, True
        
        predicted_action = action_one_hot.index(1) if 1 in action_one_hot else 0
        reward = self.calculate_reward(predicted_action, self.current_employee)
        
        self.current_step += 1
        done = self.current_step >= len(self.employees)
        
        if not done:
            self.current_employee = self.employees[self.current_step]
            next_state = self.data_processor.get_employee_state(self.current_employee)
        else:
            next_state = np.zeros(6, dtype=np.float32)
        
        return next_state, reward, done
    
    def get_employee_count(self): #get total number of employees
        return len(self.employees)
    
    def _load_human_feedback(self): #load human feedback
        feedback_file = 'human_feedback.csv'
        if os.path.exists(feedback_file):
            try:
                df = pd.read_csv(feedback_file) #create a dictionary of employee name with true label
                feedback_dict = {}
                for _, row in df.iterrows():
                    emp_name = str(row['employee_name']).lower()
                    true_label = int(row['true_label'])
                    feedback_dict[emp_name] = true_label
                return feedback_dict
            except Exception as e:
                print(f"can not load human feedback: {e}")
                return {}
        return {}
    
    def get_ground_truth_label(self, employee_name): #get truth label for an employee
        emp_name_lower = str(employee_name).lower()
        
        if emp_name_lower in self.human_feedback: #check if human feedback exists for the employee
            return self.human_feedback[emp_name_lower]
        
        #if doesnot find then back to rule based label
        return self.data_processor.get_performance_label(employee_name)