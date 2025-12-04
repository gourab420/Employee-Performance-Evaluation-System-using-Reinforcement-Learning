import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from trainer import PerformanceTrainer
from data_processor import AttendanceDataProcessor

# Page configuration
st.set_page_config(
    page_title="Employee Performance Evaluation - RL System",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = None

# Header
st.markdown('<div class="main-header">👔 Employee Performance Evaluation System</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Powered by Reinforcement Learning & Deep Q-Network</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("⚙️ System Control")
    
    # Data source configuration
    st.subheader("📁 Data Source")
    data_path = st.text_input(
        "Dataset folder path:",
        value=r"dummy_dataset"
    )
    
    st.markdown("---")
    
    # System initialization
    st.subheader("🚀 Initialize System")
    if st.button("🔄 Load Data & Initialize", type="primary", use_container_width=True):
        if not os.path.exists(data_path):
            st.error(f"❌ Path not found: {data_path}")
        else:
            with st.spinner("Loading data and initializing system..."):
                try:
                    st.session_state.trainer = PerformanceTrainer(data_path)
                    st.session_state.trainer.initialize_agent()
                    st.session_state.processor = st.session_state.trainer.get_processor()
                    st.session_state.processor.process_attendance_data()
                    st.session_state.system_initialized = True
                    st.success("✅ System initialized successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    if st.session_state.system_initialized:
        st.success("✅ System Ready")
        
        # Training section
        st.markdown("---")
        st.subheader("🎓 Training")
        
        if not st.session_state.trainer.is_trained:
            n_episodes = st.slider("Training Episodes:", 10, 500, 100, 10)
            if st.button("🏋️ Train Model", type="primary", use_container_width=True):
                with st.spinner(f"Training for {n_episodes} episodes..."):
                    success = st.session_state.trainer.train_initial_model(n_episodes)
                    if success:
                        st.success("✅ Training completed!")
                        st.rerun()
        else:
            st.info("✅ Model already trained")
    else:
        st.warning("⚠️ System not initialized")
    
    st.markdown("---")
    st.markdown("**About:**")
    st.markdown("""
    This system uses Deep Q-Learning to evaluate employee performance based on:
    - Work hours
    - Punctuality
    - Consistency
    - Attendance patterns
    """)

# Main content
if not st.session_state.system_initialized:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("""
        ### 👋 Welcome to the Employee Performance Evaluation System!
        
        **Getting Started:**
        1. Configure your data source in the sidebar
        2. Click "Load Data & Initialize"
        3. Train the model if needed
        4. Start evaluating employees!
        
        Use the sidebar controls to begin. 👈
        """)
        
else:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "👤 Evaluate Employee", "📈 Analytics", "📝 Feedback History"])
    
    # Tab 1: Dashboard
    with tab1:
        st.header("📊 System Dashboard")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_employees = len(st.session_state.processor.get_all_employees())
            st.metric("Total Employees", total_employees, "Active")
        
        with col2:
            training_status = "✅ Trained" if st.session_state.trainer.is_trained else "❌ Not Trained"
            st.metric("Model Status", training_status)
        
        with col3:
            if st.session_state.trainer.agent:
                device = st.session_state.trainer.agent.device
                st.metric("Compute Device", str(device).upper())
            else:
                st.metric("Compute Device", "N/A")
        
        with col4:
            feedback_path = os.path.join('feedback', 'human_feedback.csv')
            if os.path.exists(feedback_path):
                feedback_df = pd.read_csv(feedback_path)
                feedback_count = len(feedback_df)
            else:
                feedback_count = 0
            st.metric("Human Feedbacks", feedback_count)
        
        st.markdown("---")
        
        # Employee data table
        st.subheader("👥 Employee Overview")
        if st.session_state.processor.employee_data is not None:
            df = st.session_state.processor.employee_data.copy()
            
            # Format the dataframe
            df_display = df.copy()
            df_display['punctuality_rate'] = (df_display['punctuality_rate'] * 100).round(1).astype(str) + '%'
            df_display['avg_work_hours'] = df_display['avg_work_hours'].round(2)
            df_display['consistency_score'] = df_display['consistency_score'].round(3)
            
            # Display table
            st.dataframe(
                df_display[['employee_name', 'avg_work_hours', 'min_work_hours', 'max_work_hours', 
                    'punctuality_rate', 'consistency_score', 'days_present']],
                use_container_width=True,
                height=400
            )
        else:
            st.warning("No employee data available")
    
    # Tab 2: Evaluate Employee
    with tab2:
        st.header("👤 Evaluate Employee Performance")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Select Employee")
            employees = st.session_state.processor.get_all_employees()
            
            if employees:
                selected_employee = st.selectbox(
                    "Choose an employee to evaluate:",
                    employees,
                    key="employee_selector"
                )
                
                if st.button("🔍 Evaluate Performance", type="primary", use_container_width=True):
                    if st.session_state.trainer.agent:
                        state = st.session_state.processor.get_employee_state(selected_employee)
                        if state is not None:
                            # Get prediction
                            performance = st.session_state.trainer.agent.evaluate_employee(state)
                            true_label_idx = st.session_state.processor.get_performance_label(selected_employee)
                            true_labels = {0: "Poor", 1: "Average", 2: "Good", 3: "Excellent"}
                            
                            employee_data = st.session_state.processor.employee_data[
                                st.session_state.processor.employee_data['employee_name'] == selected_employee
                            ].iloc[0]
                            
                            st.session_state.evaluation_result = {
                                'name': selected_employee,
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
                            st.rerun()
                    else:
                        st.error("Agent not initialized. Please train the model first.")
            else:
                st.warning("No employees found in the dataset")
        
        with col2:
            if st.session_state.evaluation_result:
                result = st.session_state.evaluation_result
                st.subheader("📋 Evaluation Results")
                
                # Performance prediction
                st.markdown(f"### Employee: **{result['name']}**")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**🤖 AI Prediction:**")
                    prediction_color = {
                        "Poor": "#dc3545",
                        "Average": "#ffc107",
                        "Good": "#17a2b8",
                        "Excellent": "#28a745"
                    }
                    color = prediction_color.get(result['ai_prediction'], "#666")
                    st.markdown(f"<h2 style='color: {color};'>{result['ai_prediction']}</h2>", unsafe_allow_html=True)
                
                with col_b:
                    st.markdown("**📊 Rule-Based Label:**")
                    color = prediction_color.get(result['ground_truth'], "#666")
                    st.markdown(f"<h2 style='color: {color};'>{result['ground_truth']}</h2>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Metrics
                st.markdown("**📈 Performance Metrics:**")
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Average Work Hours", f"{result['metrics']['average_work_hours']:.2f}")
                    st.metric("Punctuality Rate", f"{result['metrics']['punctuality_rate']*100:.1f}%")
                    st.metric("Days Present", result['metrics']['days_present'])
                
                with metric_col2:
                    st.metric("Min Work Hours", f"{result['metrics']['min_work_hours']:.2f}")
                    st.metric("Max Work Hours", f"{result['metrics']['max_work_hours']:.2f}")
                    st.metric("Consistency Score", f"{result['metrics']['consistency_score']:.3f}")
                
                # Human feedback section
                st.markdown("---")
                st.subheader("💬 Provide Feedback")
                
                feedback_option = st.radio(
                    "Do you agree with the AI prediction?",
                    ["✅ Yes, it's correct", "❌ No, it's wrong", "⏭️ Skip"],
                    key="feedback_radio"
                )
                
                if feedback_option == "❌ No, it's wrong":
                    st.markdown("**What is the correct performance level?**")
                    correct_label_name = st.selectbox(
                        "Select correct label:",
                        ["Poor", "Average", "Good", "Excellent"],
                        key="correct_label_select"
                    )
                    
                    label_map = {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3}
                    correct_label = label_map[correct_label_name]
                    
                    if st.button("💾 Save Feedback & Retrain", type="primary", use_container_width=True):
                        # Save feedback
                        st.session_state.processor.save_human_feedback(
                            result['name'],
                            correct_label,
                            result['metrics']
                        )
                        st.success("✅ Feedback saved!")
                        
                        # Retrain
                        with st.spinner("🔄 Retraining model with your feedback..."):
                            success = st.session_state.trainer.train_incremental(
                                result['name'],
                                correct_label,
                                n_episodes=100
                            )
                            if success:
                                st.success("✅ Model retrained successfully!")
                                st.balloons()
                
                elif feedback_option == "✅ Yes, it's correct":
                    st.success("👍 Great! The model is learning well!")
    
    # Tab 3: Analytics
    with tab3:
        st.header("📈 Performance Analytics")
        
        if st.session_state.processor.employee_data is not None:
            df = st.session_state.processor.employee_data.copy()
            
            # Distribution charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Work hours distribution
                fig_hours = px.histogram(
                    df,
                    x='avg_work_hours',
                    nbins=20,
                    title='Average Work Hours Distribution',
                    labels={'avg_work_hours': 'Average Work Hours'},
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig_hours, use_container_width=True)
            
            with col2:
                # Punctuality distribution
                fig_punct = px.histogram(
                    df,
                    x='punctuality_rate',
                    nbins=20,
                    title='Punctuality Rate Distribution',
                    labels={'punctuality_rate': 'Punctuality Rate'},
                    color_discrete_sequence=['#764ba2']
                )
                st.plotly_chart(fig_punct, use_container_width=True)
            
            # Scatter plot
            st.subheader("Work Hours vs Punctuality")
            fig_scatter = px.scatter(
                df,
                x='avg_work_hours',
                y='punctuality_rate',
                size='days_present',
                hover_data=['employee_name'],
                title='Employee Performance Matrix',
                labels={
                    'avg_work_hours': 'Average Work Hours',
                    'punctuality_rate': 'Punctuality Rate'
                },
                color='consistency_score',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Top performers
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top 5 by Work Hours")
                top_hours = df.nlargest(5, 'avg_work_hours')[['employee_name', 'avg_work_hours']]
                st.dataframe(top_hours, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("⭐ Top 5 by Punctuality")
                top_punct = df.nlargest(5, 'punctuality_rate')[['employee_name', 'punctuality_rate']]
                top_punct['punctuality_rate'] = (top_punct['punctuality_rate'] * 100).round(1).astype(str) + '%'
                st.dataframe(top_punct, use_container_width=True, hide_index=True)
        else:
            st.warning("No data available for analytics")
    
    # Tab 4: Feedback History
    with tab4:
        st.header("📝 Human Feedback History")
        
        feedback_path = os.path.join('feedback', 'human_feedback.csv')
        if os.path.exists(feedback_path):
            feedback_df = pd.read_csv(feedback_path)
            
            if not feedback_df.empty:
                st.success(f"📊 Total Feedbacks: {len(feedback_df)}")
                
                # Show recent feedback
                st.subheader("Recent Feedback")
                
                # Add label names
                label_map = {0: "Poor", 1: "Average", 2: "Good", 3: "Excellent"}
                feedback_df['performance_label'] = feedback_df['true_label'].map(label_map)
                
                display_cols = ['employee_name', 'performance_label', 'avg_work_hours', 
                               'punctuality_rate', 'consistency_score']
                
                if all(col in feedback_df.columns for col in display_cols):
                    st.dataframe(
                        feedback_df[display_cols].tail(20),
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.dataframe(feedback_df.tail(20), use_container_width=True, height=400)
                
                # Feedback statistics
                st.subheader("📊 Feedback Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Label distribution
                    label_counts = feedback_df['true_label'].value_counts().sort_index()
                    label_counts.index = label_counts.index.map(label_map)
                    
                    fig_labels = px.pie(
                        values=label_counts.values,
                        names=label_counts.index,
                        title='Feedback Label Distribution',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig_labels, use_container_width=True)
                
                with col2:
                    # Employees with most feedback
                    employee_feedback_count = feedback_df['employee_name'].value_counts().head(10)
                    
                    fig_emp = px.bar(
                        x=employee_feedback_count.values,
                        y=employee_feedback_count.index,
                        orientation='h',
                        title='Employees with Most Feedback',
                        labels={'x': 'Feedback Count', 'y': 'Employee'},
                        color_discrete_sequence=['#667eea']
                    )
                    st.plotly_chart(fig_emp, use_container_width=True)
                
                # Download feedback data
                st.download_button(
                    label="📥 Download Feedback CSV",
                    data=feedback_df.to_csv(index=False),
                    file_name='human_feedback_export.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            else:
                st.info("No feedback records yet. Start evaluating employees!")
        else:
            st.info("No feedback file found. Feedback will be saved after first evaluation.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🤖 Powered by Deep Q-Learning | Built with Streamlit</p>
    <p>Employee Performance Evaluation System © 2024</p>
</div>

""", unsafe_allow_html=True)

