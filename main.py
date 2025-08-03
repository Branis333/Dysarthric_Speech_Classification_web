"""
FastAPI application for Dysarthric Speech Classification.
Provides endpoints for model prediction, training, monitoring, and management.
"""

import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=DeprecationWarning)

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import uvicorn
import json
import shutil
import tempfile
import psutil
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import logging
from pathlib import Path

# Import custom modules
from src.prediction import SpeechClassifier, ModelValidator
from src.model import DysarthriaModel
from src.preprocessing import AudioPreprocessor, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dysarthric Speech Classification API",
    description="ML Pipeline for classifying dysarthric vs healthy speech patterns",
    version="1.0.0"
)

# Global variables
model_path = "saved_models/best_dysarthria_model.h5"
classifier = None
validator = None
model_uptime_start = datetime.now()

# Create necessary directories
os.makedirs("saved_models", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global classifier, validator
    
    try:
        logger.info("Starting application...")
        
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            classifier = SpeechClassifier(model_path)
            validator = ModelValidator(classifier)
            logger.info("Model loaded successfully on startup")
        else:
            logger.warning(f"Model file not found at {model_path}")
            logger.info("Application will run without model - limited functionality")
            # Don't create classifier if model doesn't exist
            
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.info("Application starting with limited functionality...")
        # Continue startup even if model loading fails

# Health and Monitoring Endpoints

@app.get("/")
async def home():
    """API home page with basic information."""
    return {
        "message": "🎙️ Dysarthric Speech Classification API",
        "status": "online",
        "version": "1.0.0",
        "description": "AI-powered speech analysis for dysarthria classification",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict": "/predict",
            "model_info": "/model-info",
            "system_metrics": "/system-metrics"
        },
        "usage": "Upload audio files to /predict endpoint for classification",
        "supported_formats": ["WAV", "MP3", "FLAC"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global classifier, validator
    
    if classifier is None or validator is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "starting", 
                "message": "Model is loading or not available",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    try:
        health_status = validator.health_check()
        status_code = 200 if health_status.get("overall_status", False) else 503
        return JSONResponse(status_code=status_code, content=health_status)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics."""
    try:
        metrics_file = "model_metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        else:
            return {
                "message": "No metrics available yet",
                "default_metrics": {
                    "accuracy": 0.95,
                    "f1_score": 0.95,
                    "model_type": "CNN with L2 Regularization"
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading metrics: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get model information and architecture."""
    global classifier
    
    if classifier is None:
        return {
            "status": "Model not loaded",
            "architecture": "CNN with L2 Regularization",
            "input_shape": [62, 129, 1],
            "classes": ["Control (Healthy)", "Dysarthric"],
            "model_path": model_path,
            "message": "Model will be loaded on first prediction request"
        }
    
    try:
        model_info = classifier.get_model_info()
        
        # Add uptime information
        uptime = datetime.now() - model_uptime_start
        model_info['uptime_seconds'] = uptime.total_seconds()
        model_info['uptime_formatted'] = str(uptime).split('.')[0]
        
        return model_info
    except Exception as e:
        return {
            "error": f"Error getting model info: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/system-metrics")
async def get_system_metrics():
    """Get system resource metrics."""
    try:
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "timestamp": datetime.now().isoformat(),
            "status": "online"
        }
    except Exception as e:
        return {
            "error": f"Error getting system metrics: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# Prediction Endpoints

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """Predict dysarthria from uploaded audio file."""
    global classifier
    
    if classifier is None:
        # Try to load classifier on demand
        try:
            if os.path.exists(model_path):
                classifier = SpeechClassifier(model_path)
                logger.info("Model loaded on demand for prediction")
            else:
                raise HTTPException(status_code=503, detail="Model file not found. Please train the model first.")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model loading failed: {str(e)}")
    
    # Validate file type
    if not file.filename.endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload WAV, MP3, or FLAC files.")
    
    tmp_file_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name
        
        # Make prediction
        result = classifier.predict_audio_file(tmp_file_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict dysarthria for multiple audio files."""
    global classifier
    
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Too many files. Maximum 10 files allowed.")
    
    results = []
    temp_files = []
    
    try:
        # Save all files temporarily
        for file in files:
            if not file.filename.endswith(('.wav', '.mp3', '.flac')):
                results.append({
                    "filename": file.filename,
                    "error": "Invalid file type"
                })
                continue
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                temp_files.append((tmp_file.name, file.filename))
        
        # Make predictions
        for tmp_path, original_filename in temp_files:
            try:
                result = classifier.predict_audio_file(tmp_path)
                result["filename"] = original_filename
                results.append(result)
            except Exception as e:
                results.append({
                    "filename": original_filename,
                    "error": str(e)
                })
        
        return {"predictions": results, "total_files": len(files)}
        
    finally:
        # Clean up all temporary files
        for tmp_path, _ in temp_files:
            try:
                os.unlink(tmp_path)
            except:
                pass

# Training and Management Endpoints

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks,
                       control_path: str = Form("./M_Con"),
                       dysarthric_path: str = Form("./M_Dys"),
                       max_files: int = Form(50),
                       epochs: int = Form(20)):
    """Trigger model retraining."""
    
    def retrain_task():
        try:
            logger.info("Starting model retraining task...")
            
            # Initialize components
            preprocessor = AudioPreprocessor()
            data_loader = DataLoader(preprocessor)
            
            # Load data
            all_files, all_labels = data_loader.load_dataset(
                control_path, dysarthric_path, max_files
            )
            
            if len(all_files) == 0:
                logger.error("No data files found for retraining")
                return
            
            # Split data
            from sklearn.model_selection import train_test_split
            files_temp, files_test, labels_temp, labels_test = train_test_split(
                all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
            )
            files_train, files_val, labels_train, labels_val = train_test_split(
                files_temp, labels_temp, test_size=0.25, random_state=42, stratify=labels_temp
            )
            
            # Create datasets
            train_dataset = data_loader.create_dataset_generator(files_train, labels_train)
            val_dataset = data_loader.create_dataset_generator(files_val, labels_val)
            test_dataset = data_loader.create_dataset_generator(files_test, labels_test)
            
            # Get input shape
            sample_batch = next(iter(train_dataset))
            input_shape = sample_batch[0].shape[1:]
            
            # Initialize model
            dysarthria_model = DysarthriaModel(input_shape)
            
            # Train model
            dysarthria_model.train_model(train_dataset, val_dataset, epochs=epochs)
            
            # Evaluate - FIX: Only expect 2 values from evaluate
            metrics = dysarthria_model.evaluate_model(test_dataset, labels_test)
            
            # Save retrained model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_model_path = dysarthria_model.save_model(f"retrained_model_{timestamp}.h5")
            
            # Save metrics
            with open(f"retrain_metrics_{timestamp}.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Model retraining completed. New F1 score: {metrics.get('f1_score', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
    
    background_tasks.add_task(retrain_task)
    return {"message": "Retraining started in background", "timestamp": datetime.now().isoformat()}

@app.post("/upload-training-data")
async def upload_training_data(files: List[UploadFile] = File(...), 
                              label: str = Form(...)):
    """Upload new training data."""
    
    if label not in ["control", "dysarthric"]:
        raise HTTPException(status_code=400, detail="Label must be 'control' or 'dysarthric'")
    
    upload_dir = f"data/train/{label}"
    os.makedirs(upload_dir, exist_ok=True)
    
    saved_files = []
    
    try:
        for file in files:
            if not file.filename.endswith(('.wav', '.mp3', '.flac')):
                continue
            
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
        
        return {
            "message": f"Uploaded {len(saved_files)} files for {label} class",
            "files": saved_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

# Data Visualization Endpoints

@app.get("/visualization-data")
async def get_visualization_data():
    """Get data for dashboard visualizations."""
    try:
        # Load metrics if available
        metrics = {}
        if os.path.exists("model_metrics.json"):
            with open("model_metrics.json", 'r') as f:
                metrics = json.load(f)
        
        # Get training data counts
        train_control_count = 0
        train_dysarthric_count = 0
        
        try:
            if os.path.exists("data/train/control"):
                train_control_count = len([f for f in os.listdir("data/train/control") if f.endswith('.wav')])
            if os.path.exists("data/train/dysarthric"):
                train_dysarthric_count = len([f for f in os.listdir("data/train/dysarthric") if f.endswith('.wav')])
        except:
            pass
        
        # System metrics
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
        
        return {
            "model_metrics": metrics,
            "data_counts": {
                "train_control": train_control_count,
                "train_dysarthric": train_dysarthric_count
            },
            "system_metrics": system_metrics,
            "uptime": (datetime.now() - model_uptime_start).total_seconds()
        }
        
    except Exception as e:
        return {
            "error": f"Error getting visualization data: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# File Management Endpoints

@app.get("/list-models")
async def list_models():
    """List all available models."""
    models_dir = "saved_models"
    if not os.path.exists(models_dir):
        return {"models": []}
    
    models = []
    try:
        for file in os.listdir(models_dir):
            if file.endswith('.h5'):
                file_path = os.path.join(models_dir, file)
                file_stats = os.stat(file_path)
                models.append({
                    "filename": file,
                    "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                })
    except Exception as e:
        logger.error(f"Error listing models: {e}")
    
    return {"models": models}

@app.post("/switch-model")
async def switch_model(model_filename: str = Form(...)):
    """Switch to a different model."""
    global classifier, validator, model_uptime_start
    
    new_model_path = os.path.join("saved_models", model_filename)
    
    if not os.path.exists(new_model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    try:
        classifier = SpeechClassifier(new_model_path)
        validator = ModelValidator(classifier)
        model_uptime_start = datetime.now()
        
        return {"message": f"Successfully switched to {model_filename}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error switching model: {str(e)}")


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
