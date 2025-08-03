# 🎙️ Dysarthric Speech Classification - Streamlit Dashboard

A modern, interactive web interface for dysarthric speech classification using Streamlit and FastAPI.

## ✨ Features

### 🎯 Main Dashboard
- **Real-time model metrics** (Accuracy, F1-Score, Precision, Recall)
- **System health monitoring** (CPU, Memory, Uptime)
- **Interactive confusion matrix**
- **Performance charts and visualizations**

### 🎵 Audio Classification
- **Drag-and-drop audio upload** (WAV, MP3, FLAC)
- **Batch processing** for multiple files
- **Real-time confidence scores**
- **Visual classification results**

### 🔄 Model Management
- **Model retraining interface** with epoch selection
- **Training data upload** for both control and dysarthric samples
- **Model information and performance metrics**
- **Progress monitoring**

### 📈 Analytics
- **Training data distribution charts**
- **Performance timeline visualization**
- **Historical metrics tracking**
- **Interactive data exploration**

### ⚙️ Settings & Configuration
- **API connection testing**
- **Model configuration options**
- **System information display**
- **Threshold adjustments**

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```powershell
# Run the automated setup script
python setup_and_run.py

# Choose option 2: "Install and run complete system"
```

### Option 2: Manual Setup
```powershell
# 1. Install Streamlit dependencies
pip install -r streamlit_requirements.txt

# 2. Install backend dependencies (without TensorFlow)
pip install -r requirements.txt

# 3. Start FastAPI backend (in one terminal)
python main.py

# 4. Start Streamlit dashboard (in another terminal)
streamlit run streamlit_app.py
```

### Option 3: Streamlit Only (for UI testing)
```powershell
# Install only Streamlit dependencies
pip install -r streamlit_requirements.txt

# Run dashboard (backend features will show connection errors)
python run_dashboard.py
```

## 📋 Dependencies

### Core Requirements (streamlit_requirements.txt)
- **streamlit** >= 1.28.0 - Web framework
- **plotly** >= 5.15.0 - Interactive charts
- **pandas** >= 2.0.0 - Data manipulation
- **numpy** >= 1.24.0 - Numerical computing
- **requests** >= 2.31.0 - API communication

### Backend Requirements (requirements.txt)
- **fastapi** - REST API framework
- **uvicorn** - ASGI server
- **scikit-learn** - Machine learning utilities
- **librosa** - Audio processing
- **psutil** - System monitoring

**Note**: TensorFlow is not included in default requirements due to installation issues. Install manually if needed:
```powershell
pip install tensorflow==2.13.0
```

## 🌐 Accessing the Dashboard

Once running, access the applications at:
- **Streamlit Dashboard**: http://localhost:8501
- **FastAPI Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🎯 Usage Guide

### 1. Audio Classification
1. Go to "🎵 Audio Classification" page
2. Upload WAV, MP3, or FLAC files
3. Click "🔍 Classify Audio"
4. View results with confidence scores

### 2. Model Monitoring
1. Visit "📊 Dashboard" for real-time metrics
2. Check system health and performance
3. Monitor CPU/Memory usage

### 3. Model Retraining
1. Go to "🔄 Model Management"
2. Upload training data (control/dysarthric)
3. Set training epochs
4. Start retraining process

### 4. Analytics
1. Visit "📈 Analytics" for data insights
2. View training data distribution
3. Analyze performance trends

## 🔧 Configuration

### API Settings
- Default API URL: `http://localhost:8000`
- Configurable in "⚙️ Settings" page
- Test connection with built-in health check

### Model Settings
- Confidence thresholds
- Performance alert levels
- Auto-retraining options

## 🐛 Troubleshooting

### Common Issues

**1. "Cannot connect to API" Error**
```
Solution: Start the FastAPI backend first
Command: python main.py
```

**2. "Module not found" Errors**
```
Solution: Install missing dependencies
Command: pip install -r streamlit_requirements.txt
```

**3. TensorFlow Import Errors**
```
Solution: The Streamlit dashboard doesn't require TensorFlow
Only the FastAPI backend needs it for model operations
```

**4. Port Already in Use**
```
Solution: Change ports in the startup commands
Streamlit: streamlit run streamlit_app.py --server.port 8502
FastAPI: uvicorn main:app --port 8001
```

### Debug Mode
Run with debug information:
```powershell
streamlit run streamlit_app.py --logger.level debug
```

## 📁 File Structure
```
├── streamlit_app.py           # Main Streamlit application
├── streamlit_requirements.txt # Streamlit dependencies
├── run_dashboard.py          # Simple dashboard launcher
├── setup_and_run.py         # Complete setup script
├── main.py                  # FastAPI backend
├── requirements.txt         # Backend dependencies
└── src/                    # Backend modules
    ├── model.py
    ├── preprocessing.py
    └── prediction.py
```

## 🎨 Customization

### Themes and Styling
The dashboard uses custom CSS for styling. Modify the `st.markdown()` sections in `streamlit_app.py` to customize:
- Colors and gradients
- Card layouts
- Metrics display
- Chart appearances

### Adding New Features
1. Add new pages in the `main_dashboard()` function
2. Create corresponding `show_*()` functions
3. Implement API calls for data fetching
4. Add visualizations with Plotly

## 🔒 Security Notes

- Default setup runs on localhost only
- No authentication implemented (development use)
- For production, add proper authentication and HTTPS
- Validate file uploads and API inputs

## 🤝 Contributing

To add new features:
1. Fork the repository
2. Create a feature branch
3. Add Streamlit components in `streamlit_app.py`
4. Test with the FastAPI backend
5. Submit a pull request

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review console logs for errors
3. Test API connectivity first
4. Verify all dependencies are installed

---

**Happy Speech Classification! 🎙️✨**
