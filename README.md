# 🎙️ Dysarthric Speech Classification

AI-powered web app that classifies speech as healthy or dysarthric using deep learning.
### 1. Youtube Demo Link
```bash
https://drive.google.com/file/d/1ALEKzl1EvZCmWdqfjLZFVjq1WmLvnS56/view?usp=sharing
```

## ⚡ Quick Setup

### 1. Install & Run
```bash
git clone <repository>
cd Dysarthric_Speech_Classification_web
pip install -r requirements.txt
```

### 2. Start the App
```bash
#  Full dashboard (FastAPI )
# Terminal 1:
uvicorn main:app --reload 

# Terminal 2:
streamlit run streamlit_app.py
```

### 3. Access App
- **Dashboard**: http://localhost:8501 (Streamlit)
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🎯 What It Does

- Upload audio files (WAV, MP3, FLAC)
- Get instant classification: **Healthy** or **Dysarthric**
- View confidence scores and predictions
- Monitor system performance

## 🔧 Key Features

✅ **95%+ Accuracy** - Advanced CNN model  
✅ **Real-time Predictions** - Upload and get results instantly  
✅ **Easy Retraining** - One-click model updates  
✅ **Web Dashboard** - No coding required  

## 📱 How to Use

1. **Start the app**: See step 2 above
2. **Open dashboard**: Go to http://localhost:8501
3. **Upload audio**: Drag & drop your audio file
4. **Get results**: See classification and confidence

## 🐛 Problems?

| Issue | Solution |
|-------|----------|
| Model not found | Run the Jupyter notebook first |
| App won't start | Check if ports 8000 & 8501 are free |
| Audio upload fails | Use WAV files under 10MB |
| Dashboard won't load | Start FastAPI first, then Streamlit |

## 📋 Requirements

- Python 3.8+
- 4GB RAM
- Windows/Mac/Linux

That's it! 🚀
