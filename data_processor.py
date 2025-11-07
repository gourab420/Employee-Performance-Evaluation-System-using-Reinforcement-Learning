import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
import json

class AttendanceDataProcessor:
    def __init__(self, data_directory):
        self.data_directory=data_directory
        self.employee_data=None
        
    def read_csv_files(self):
        if os.path.isfile(self.data_directory):  
            return pd.read_csv(self.data_directory)
        elif os.path.isdir(self.data_directory): 
            all_data = []
            csv_files = glob.glob(os.path.join(self.data_directory, '*.csv'))
            
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    all_data.append(df)
                    print(f"loaded: {os.path.basename(file)}")
                except Exception as e:
                    print(f"error to loading {file}: {e}")
                    
            return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        return pd.DataFrame()
    
    def calculate_work_hours(self, row): #calculate total work time without break time
        try:
            start_time = datetime.strptime(row['Sign In'], '%H:%M')
            end_time = datetime.strptime(row['Sign Out'], '%H:%M')
            break_start = datetime.strptime(row['Break Start'], '%H:%M')
            break_end = datetime.strptime(row['Break End'], '%H:%M')
            
            work_time = (end_time - start_time).total_seconds() / 3600
            break_time = (break_end - break_start).total_seconds() / 3600
            
            return work_time - break_time
        except:
            return 0
    
    def calculate_punctuality(self, sign_in_time, threshold='09:00'): #check the employee come office under 9 am
        try:
            if ':' in str(sign_in_time):
                time_obj = pd.to_datetime(sign_in_time, format='%H:%M').time()
                threshold_obj = pd.to_datetime(threshold, format='%H:%M').time()
                return 1 if time_obj <= threshold_obj else 0
        except:
            return 0
        return 0
    
    
    
    def process_attendance_data(self): #process attendacce data and calculate metrics
        df = self.read_csv_files()
        if df.empty:
            return pd.DataFrame()
        
        df.columns = df.columns.str.strip()

        if 'Employee Name' in df.columns:
            df['employee_name'] = df['Employee Name']
        elif 'Employee' in df.columns:
            df['employee_name'] = df['Employee']
        
        #calculate punctuality
        df['is_punctual'] = df['Sign In'].apply(self.calculate_punctuality)
        
        #group by employee and calculate metrics
        employee_metrics = df.groupby('employee_name').agg({
            'Total Work Hours': ['mean', 'std', 'min', 'max', 'count'],
            'is_punctual': 'mean'
        }).reset_index()

        employee_metrics.columns = [
            'employee_name', 'avg_work_hours', 'std_work_hours', 
            'min_work_hours', 'max_work_hours', 'days_present', 
            'punctuality_rate'
        ]

        employee_metrics['std_work_hours'] = employee_metrics['std_work_hours'].fillna(0)
        
        employee_metrics['consistency_score'] = 1 / (1 + employee_metrics['std_work_hours'])
        
        self.employee_data = employee_metrics
        return employee_metrics
    
    def get_employee_state(self, employee_name): #get state vactor for spcefic employee
        if self.employee_data is None:
            return None
        
        employee = self.employee_data[self.employee_data['employee_name'] == employee_name]
        if employee.empty:
            return None
        
        #state vector: [avg_hours, std_hours, min_hours, max_hours, punctuality_rate, consistency]
        state = np.array([
            employee['avg_work_hours'].values[0],
            employee['std_work_hours'].values[0],
            employee['min_work_hours'].values[0],
            employee['max_work_hours'].values[0],
            employee['punctuality_rate'].values[0],
            employee['consistency_score'].values[0]
        ], dtype=np.float32)
        
        return state
    
    def get_all_employees(self): #get list of all employees
        if self.employee_data is None:
            return []
        return self.employee_data['employee_name'].tolist()
    
    def get_performance_label(self, employee_name): 
        if self.employee_data is None:
            return None
        
        employee = self.employee_data[self.employee_data['employee_name'] == employee_name]
        if employee.empty:
            return None
        
        avg_hours = employee['avg_work_hours'].values[0]
        punctuality = employee['punctuality_rate'].values[0]
        consistency = employee['consistency_score'].values[0]
        
        # define performance categories
        if avg_hours >= 8.0 and punctuality >= 0.9 and consistency >= 0.7:
            return 3  # excellent
        elif avg_hours >= 7.5 and punctuality >= 0.75:
            return 2  # good
        elif avg_hours >= 6.5 and punctuality >= 0.5:
            return 1  # average
        else:
            return 0  # poor
    
    def save_human_feedback(self, employee_name, true_label, metrics): #save human feedback
        feedback_file = 'human_feedback.csv'
        feedback_data = {
            'employee_name': employee_name,
            'true_label': true_label,
            'avg_work_hours': metrics['average_work_hours'],  
            'std_work_hours': 0,  
            'min_work_hours': metrics['min_work_hours'],
            'max_work_hours': metrics['max_work_hours'],
            'punctuality_rate': metrics['punctuality_rate'],
            'consistency_score': metrics['consistency_score'],
            'timestamp': datetime.now().isoformat()
        }
        
        # add to csv file
        df = pd.DataFrame([feedback_data])
        if os.path.exists(feedback_file):
            df.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            df.to_csv(feedback_file, mode='w', header=True, index=False)
        
        print(f"feedback saved for {employee_name}")
        return True
    
    def load_human_feedback(self): #human feedback to override automatic labels
        feedback_file = 'human_feedback.csv'
        if os.path.exists(feedback_file):
            feedback_df = pd.read_csv(feedback_file)
            return feedback_df
        return pd.DataFrame()