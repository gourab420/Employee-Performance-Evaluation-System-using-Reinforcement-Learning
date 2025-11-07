import os
import pandas as pd
from trainer import PerformanceTrainer

class EmployeeAttendanceSystem:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.trainer = PerformanceTrainer(data_directory)
        self.processor = self.trainer.get_processor()
        self.agent = None
        
    def train_initial_model(self, n_episodes=100): #model train setup
        success = self.trainer.train_initial_model(n_episodes)
        if success:
            self.agent = self.trainer.get_agent()
        return success
    
    def train_with_feedback(self, employee_name, correct_label, n_episodes=100): #retrain the model with new value
        success = self.trainer.train_incremental(employee_name, correct_label, n_episodes)
        if success:
            self.agent = self.trainer.get_agent()
        return success
        
    def evaluate_employee(self, employee_name): #evaluate specfic employee performance
        if self.agent is None:
            print("system not trained yet")
            return None
        
        state = self.processor.get_employee_state(employee_name)
        
        if state is None:
            print(f"employee '{employee_name}' not found!")
            return None
        
        #compute AI prediction 
        performance = self.agent.evaluate_employee(state)
        true_label_idx = self.processor.get_performance_label(employee_name)
        true_labels = {0: "Poor", 1: "Average", 2: "Good", 3: "Excellent"}
        
        employee_data = self.processor.employee_data[
            self.processor.employee_data['employee_name'] == employee_name
        ].iloc[0]
        
        #attempt to loaf and apply prior human feedback override
        human_override = None
        try:
            feedback_path = os.path.join(os.path.dirname(__file__), 'human_feedback.csv')
            if os.path.exists(feedback_path):
                df_fb = pd.read_csv(feedback_path)
                #normalize to string compare to be safe
                match = df_fb[df_fb['employee_name'].astype(str).str.lower() == str(employee_name).lower()]
                if not match.empty:
                    last = match.iloc[-1]
                    if 'true_label' in last:
                        human_override = int(last['true_label'])
                    elif 'correct_label' in last:
                        human_override = int(last['correct_label'])
                    elif 'label' in last:
                        human_override = int(last['label'])
        
        except Exception as e:
            #if feedback loading has issues then do not fail evaluation 
            human_override = None
        
        final_label = performance
        label_source = 'AI Prediction'
        if human_override is not None:
            #map index back to label name space used by agent outputs if needed
            final_label = {0: 'Poor', 1: 'Average', 2: 'Good', 3: 'Excellent'}.get(human_override, performance)
            label_source = 'Human Override'
        
        return {
            'name': employee_name,
            'prediction': final_label,
            'prediction_source': label_source,
            'ai_prediction': performance,
            'ground_truth': true_labels.get(true_label_idx, 'Unknown'),
            'metrics': {
                'average_work_hours': float(employee_data['avg_work_hours']),
                'min_work_hours': float(employee_data['min_work_hours']),
                'max_work_hours': float(employee_data['max_work_hours']),
                'punctuality_rate': float(employee_data['punctuality_rate']),
                'consistency_score': float(employee_data['consistency_score']),
                'days_present': int(employee_data['days_present'])
            }
        }


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "dummy_dataset")
    if not os.path.exists(data_dir):
        print(f"error: directory {data_dir} not found!")
        return
    
    print(f"loading data from: {data_dir}")
    print("this system will load ALL CSV files from the directory.")
    print("you can add new CSV files daily and they will be included.\n")
    
    #initialize the system
    system = EmployeeAttendanceSystem(data_dir)
    
    #check if model already exists
    if system.trainer.is_model_trained():
        print("trained model found! Loading existing model...")
        system.agent = system.trainer.get_agent()
    else:
        print("training the model on initial startup...")
        system.train_initial_model(n_episodes=100)
    
    #interactive employee evaluation
    while True:
        print("\n" + "-"*50)
        employee_name = input("enter employee name to evaluate (or 'quit' to exit): ").strip()
        if employee_name.lower() == 'quit':
            break
        
        if not employee_name:
            print("enter a valid employee name")
            continue
            
        result = system.evaluate_employee(employee_name)
        if result:
            print("\n" + "-"*50)
            print(f"Employee: {result['name']}")
            print("-"*50)
            print(f"AI Prediction: {result.get('ai_prediction', result['prediction'])}")
            print(f"Rule-Based Label: {result['ground_truth']}")
            if result.get('prediction_source') == 'Human Override':
                print(f"✓ Human Feedback Override: {result['prediction']}")
            print("\nMetrics:")
            print(f"  Avg Work Hours: {result['metrics']['average_work_hours']:.2f}")
            print(f"  Min/Max Hours: {result['metrics']['min_work_hours']:.2f} / {result['metrics']['max_work_hours']:.2f}")
            print(f"  Punctuality Rate: {result['metrics']['punctuality_rate']*100:.1f}%")
            print(f"  Consistency Score: {result['metrics']['consistency_score']:.2f}")
            print(f"  Days Present: {result['metrics']['days_present']}")
            print("-"*50)
            
            #ask for human feedback
            feedback = input("\n do you agree with the AI prediction? (y/n/skip): ").strip().lower()
            
            if feedback == 'n':
                print("\n what is the correct performance level?")
                print("0 = Poor | 1 = Average | 2 = Good | 3 = Excellent")
                correct_label = input("Enter number (0-3): ").strip()
                
                if correct_label in ['0', '1', '2', '3']:
                    label_names = {0: "Poor", 1: "Average", 2: "Good", 3: "Excellent"}
                    correct_label = int(correct_label)
                    print(f"\n you labeled '{employee_name}' as: {label_names[correct_label]}")
                    
                    #save feedback
                    system.processor.save_human_feedback(
                        employee_name, 
                        correct_label, 
                        result['metrics']
                    )
                    print("your feedback has been saved!")
                    
                    #auto retrain with new feedback
                    print("\n auto-retraining model with your feedback")
                    print(" (training only on this employee for 100 episodes...)")
                    system.train_with_feedback(employee_name, correct_label, n_episodes=100)
                    print("model retrained! evaluating the same employee again to show improvements...\n")
                    
                    #re-evaluate to show improvement
                    result_improved = system.evaluate_employee(employee_name)
                    if result_improved:
                        print("-"*50)
                        print(f"Employee: {result_improved['name']} (RE-EVALUATED)")
                        print("-"*50)
                        print(f"AI Prediction: {result_improved.get('ai_prediction', result_improved['prediction'])} ← IMPROVED!")
                        print(f"Rule-Based Label: {result_improved['ground_truth']}")
                        if result_improved.get('prediction_source') == 'Human Override':
                            print(f"human feedback override: {result_improved['prediction']}")
                        print("-"*50)
                        print("model learned from your feedback")
                else:
                    print("invalid input")
            elif feedback == 'y':
                print("the model is learning well")
            elif feedback == 'skip':
                print("skipped feedback for this employee")


if __name__ == "__main__":
    main()
