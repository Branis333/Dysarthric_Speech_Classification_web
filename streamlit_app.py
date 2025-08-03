"""
Streamlit Dashboard for Dysarthric Speech Classification
A modern, interactive web interface for the ML pipeline
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import time
import os
from datetime import datetime, timedelta
import io

# Configure Streamlit page
st.set_page_config(
    page_title="Dysarthric Speech Classification",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .prediction-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    
    .prediction-card.dysarthric {
        border-left-color: #dc3545;
    }
    
    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# API Base URL
API_BASE_URL = "https://dysarthric-speech-classification-web.onrender.com"

# Helper functions
def call_api(endpoint, method="GET", data=None, files=None):
    """Make API calls with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, data=data, timeout=120)
            else:
                response = requests.post(url, json=data, timeout=120)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure the FastAPI server is running on localhost:8000"
    except requests.exceptions.Timeout:
        return None, "API request timed out"
    except Exception as e:
        return None, f"Error: {str(e)}"

def load_metrics():
    """Load model metrics from API"""
    metrics, error = call_api("/metrics")
    if error:
        return None, error
    return metrics, None

def load_model_info():
    """Load model information"""
    info, error = call_api("/model-info")
    if error:
        return None, error
    return info, None

def load_system_metrics():
    """Load system metrics"""
    metrics, error = call_api("/system-metrics")
    if error:
        return None, error
    return metrics, None

def check_health():
    """Check API health"""
    health, error = call_api("/health")
    if error:
        return None, error
    return health, None

# Main Dashboard
def main_dashboard():
    """Main dashboard layout"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ Dysarthric Speech Classification Dashboard</h1>
        <p>AI-Powered Speech Analysis & Model Monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["📊 Dashboard", "🎵 Audio Classification", "🔄 Model Management", "📈 Analytics", "⚙️ Settings"]
    )
    
    # Check API health first - SIMPLIFIED VERSION
    health, health_error = check_health()
    
    # Simple status indicator instead of detailed error
    if health_error:
        st.info("🔄 System is starting up...")
    else:
        st.success("✅ API Connected")
    
    # Display current page
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "🎵 Audio Classification":
        show_audio_classification()
    elif page == "🔄 Model Management":
        show_model_management()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "⚙️ Settings":
        show_settings()

def show_dashboard():
    """Show main dashboard with metrics"""
    
    # Load data
    metrics, metrics_error = load_metrics()
    model_info, info_error = load_model_info()
    system_metrics, system_error = load_system_metrics()
    
    # Model Status Section
    st.header("🏥 System Health")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if metrics and not metrics_error:
            accuracy = metrics.get('accuracy', 0) * 100
            st.metric("🎯 Accuracy", f"{accuracy:.1f}%", delta=None)
        else:
            st.metric("🎯 Accuracy", "N/A", delta=None)
    
    with col2:
        if metrics and not metrics_error:
            f1_score = metrics.get('f1_score', 0) * 100
            st.metric("📊 F1 Score", f"{f1_score:.1f}%", delta=None)
        else:
            st.metric("📊 F1 Score", "N/A", delta=None)
    
    with col3:
        if model_info and not info_error:
            uptime = model_info.get('uptime_formatted', 'N/A')
            st.metric("⏰ Uptime", uptime, delta=None)
        else:
            st.metric("⏰ Uptime", "N/A", delta=None)
    
    with col4:
        if system_metrics and not system_error:
            cpu_usage = system_metrics.get('cpu_percent', 0)
            st.metric("💻 CPU Usage", f"{cpu_usage:.1f}%", delta=None)
        else:
            st.metric("💻 CPU Usage", "N/A", delta=None)
    
    # Performance Metrics Chart
    st.header("📈 Model Performance")
    
    if metrics and not metrics_error:
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance metrics bar chart
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            metric_values = [
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0)
            ]
            
            fig_performance = go.Figure(data=[
                go.Bar(x=metric_names, y=metric_values, 
                      marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ])
            fig_performance.update_layout(
                title="Model Performance Metrics",
                xaxis_title="Metrics",
                yaxis_title="Score",
                showlegend=False
            )
            st.plotly_chart(fig_performance, use_container_width=True)
        
        with col2:
            # Confusion Matrix
            cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Control', 'Dysarthric'],
                y=['Control', 'Dysarthric'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 20}
            ))
            
            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual"
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("📊 Model metrics will appear here when the API is connected")
    
    # System Resources
    st.header("💻 System Resources")
    
    if system_metrics and not system_error:
        col1, col2 = st.columns(2)
        
        with col1:
            # CPU and Memory usage
            cpu_percent = system_metrics.get('cpu_percent', 0)
            memory_percent = system_metrics.get('memory_percent', 0)
            
            st.progress(cpu_percent / 100)
            st.write(f"🔵 CPU Usage: {cpu_percent:.1f}%")
            
            st.progress(memory_percent / 100)
            st.write(f"🟡 Memory Usage: {memory_percent:.1f}%")
        
        with col2:
            # Resource usage pie chart
            fig_resources = go.Figure(data=[go.Pie(
                labels=['CPU', 'Memory', 'Available'],
                values=[cpu_percent, memory_percent, 100 - ((cpu_percent + memory_percent) / 2)]
            )])
            fig_resources.update_layout(title="System Resource Distribution")
            st.plotly_chart(fig_resources, use_container_width=True)
    else:
        st.info("💻 System metrics will appear here when the API is connected")

def show_audio_classification():
    """Show audio classification interface"""
    
    st.header("🎵 Audio Classification")
    st.write("Upload audio files to classify as Control (Healthy) or Dysarthric speech")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose audio files",
        type=['wav', 'mp3', 'flac'],
        accept_multiple_files=True,
        help="Upload WAV, MP3, or FLAC audio files"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        
        if st.button("🔍 Classify Audio", type="primary"):
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Prepare file for API
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Call prediction API
                result, error = call_api("/predict", method="POST", files=files)
                
                if result:
                    results.append({
                        'filename': uploaded_file.name,
                        'prediction': result.get('predicted_class', 'Unknown'),
                        'confidence': result.get('confidence', 0),
                        'probability_dysarthric': result.get('probability_dysarthric', 0),
                        'probability_control': result.get('probability_control', 0)
                    })
                else:
                    results.append({
                        'filename': uploaded_file.name,
                        'error': error
                    })
            
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.header("🎯 Classification Results")
            
            for result in results:
                if 'error' in result:
                    st.error(f"❌ {result['filename']}: {result['error']}")
                else:
                    prediction = result['prediction']
                    confidence = result['confidence']
                    
                    # Color coding
                    if prediction == "Dysarthric":
                        color = "🔴"
                        card_class = "dysarthric"
                    else:
                        color = "🟢"
                        card_class = "control"
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="prediction-card {card_class}">
                            <h4>{color} {result['filename']}</h4>
                            <p><strong>Classification:</strong> {prediction}</p>
                            <p><strong>Confidence:</strong> {confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Confidence gauge
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = confidence * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Confidence"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "green"}
                                ]
                            }
                        ))
                        fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_gauge, use_container_width=True)

def show_model_management():
    """Show model management interface"""
    
    st.header("🔄 Model Management")
    
    # Model retraining section
    st.subheader("🎯 Model Retraining")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Training Epochs", min_value=5, max_value=50, value=20)
        
        if st.button("🚀 Start Retraining", type="primary"):
            with st.spinner("Starting model retraining..."):
                data = {"epochs": epochs}
                result, error = call_api("/retrain", method="POST", data=data)
                
                if result:
                    st.success("✅ Retraining started successfully!")
                    st.info("Check the server console for training progress.")
                else:
                    st.error(f"❌ Retraining failed: {error}")
    
    with col2:
        st.info("""
        **Retraining Tips:**
        - More epochs = longer training but potentially better performance
        - Monitor system resources during training
        - Backup current model before retraining
        """)
    
    # Training data upload
    st.subheader("📁 Upload Training Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_label = st.selectbox(
            "Data Label",
            ["control", "dysarthric"],
            help="Select whether the audio files are from control (healthy) or dysarthric speakers"
        )
        
        training_files = st.file_uploader(
            "Upload Training Audio Files",
            type=['wav', 'mp3', 'flac'],
            accept_multiple_files=True,
            key="training_upload"
        )
        
        if training_files and st.button("📤 Upload Training Data"):
            with st.spinner("Uploading training data..."):
                files = [("files", (f.name, f.getvalue(), f.type)) for f in training_files]
                data = {"label": data_label}
                
                result, error = call_api("/upload-training-data", method="POST", files=files, data=data)
                
                if result:
                    st.success(f"✅ Uploaded {len(training_files)} files for {data_label} class")
                else:
                    st.error(f"❌ Upload failed: {error}")
    
    with col2:
        st.info("""
        **Data Upload Guidelines:**
        - Use high-quality audio files (16kHz recommended)
        - Ensure balanced datasets (similar number of control and dysarthric samples)
        - Each file should be 2-5 seconds long
        """)
    
    # Model information
    st.subheader("📋 Current Model Information")
    
    model_info, error = load_model_info()
    
    if model_info and not error:
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                "Model Path": model_info.get('model_path', 'N/A'),
                "Input Shape": model_info.get('input_shape', 'N/A'),
                "Classes": model_info.get('classes', 'N/A')
            })
        
        with col2:
            if 'performance' in model_info:
                perf = model_info['performance']
                st.json({
                    "Accuracy": f"{perf.get('accuracy', 0):.3f}",
                    "F1 Score": f"{perf.get('f1_score', 0):.3f}",
                    "Precision": f"{perf.get('precision', 0):.3f}",
                    "Recall": f"{perf.get('recall', 0):.3f}"
                })
    else:
        st.info("📋 Model information will appear here when the API is connected")

def show_analytics():
    """Show analytics and visualizations"""
    
    st.header("📈 Analytics Dashboard")
    
    # Load visualization data
    viz_data, error = call_api("/visualization-data")
    
    if viz_data and not error:
        # Data counts
        st.subheader("📊 Training Data Distribution")
        
        if 'data_counts' in viz_data:
            data_counts = viz_data['data_counts']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                control_count = data_counts.get('train_control', 0)
                st.metric("🟢 Control Samples", control_count)
            
            with col2:
                dysarthric_count = data_counts.get('train_dysarthric', 0)
                st.metric("🔴 Dysarthric Samples", dysarthric_count)
            
            with col3:
                total_count = control_count + dysarthric_count
                st.metric("📁 Total Samples", total_count)
            
            # Data distribution pie chart
            if total_count > 0:
                fig_dist = go.Figure(data=[go.Pie(
                    labels=['Control', 'Dysarthric'],
                    values=[control_count, dysarthric_count],
                    marker_colors=['#28a745', '#dc3545']
                )])
                fig_dist.update_layout(title="Training Data Distribution")
                st.plotly_chart(fig_dist, use_container_width=True)
    
    # Model performance timeline (sample data)
    st.subheader("📈 Model Performance Timeline")
    
    # Create sample timeline data
    base_date = datetime(2024, 1, 1)
    dates = [(base_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10)]
    
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=dates,
        y=[0.90, 0.91, 0.89, 0.93, 0.92, 0.94, 0.93, 0.95, 0.94, 0.96],
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='blue')
    ))
    fig_timeline.add_trace(go.Scatter(
        x=dates,
        y=[0.85, 0.86, 0.84, 0.88, 0.87, 0.89, 0.88, 0.90, 0.89, 0.91],
        mode='lines+markers',
        name='F1 Score',
        line=dict(color='orange')
    ))
    fig_timeline.update_layout(
        title="Model Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Score"
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)

def show_settings():
    """Show settings and configuration"""
    
    st.header("⚙️ Settings & Configuration")
    
    # API Configuration
    st.subheader("🔗 API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_url = st.text_input("API Base URL", value=API_BASE_URL)
        
        if st.button("🔍 Test Connection"):
            health, error = check_health()
            if health:
                st.success("✅ API connection successful!")
                st.json(health)
            else:
                st.error(f"❌ Connection failed: {error}")
    
    with col2:
        st.info("""
        **API Status:**
        - Health endpoint: `/health`
        - Metrics endpoint: `/metrics`
        - Prediction endpoint: `/predict`
        """)
    
    # Model Configuration
    st.subheader("🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
        st.selectbox("Model Version", ["v1.0", "v1.1", "v2.0"])
        st.checkbox("Enable Auto-retraining")
    
    with col2:
        st.slider("Performance Alert Threshold", 0.0, 1.0, 0.8, 0.05)
        st.slider("Data Refresh Interval (seconds)", 10, 300, 30)
        st.checkbox("Enable Notifications")
    
    # System Information
    st.subheader("💻 System Information")
    
    system_info = {
        "Streamlit Version": st.__version__,
        "Python Version": "3.11+",
        "Platform": "Windows/Linux/MacOS",
        "Dependencies": "Streamlit, Plotly, Requests"
    }
    
    st.json(system_info)

# Run the app
if __name__ == "__main__":
    main_dashboard()