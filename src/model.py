"""
Model creation, training, and management for dysarthric speech classification.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from typing import Tuple, Dict, Optional, List
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DysarthriaModel:
    def __init__(self, input_shape: Tuple[int, int, int], model_save_path: str = "saved_models"):
        """
        Initialize the dysarthria classification model.
        
        Args:
            input_shape: Shape of input spectrograms
            model_save_path: Directory to save models
        """
        self.input_shape = input_shape
        self.model_save_path = model_save_path
        self.model = None
        self.history = None
        self.metrics = {}
        
        # Create save directory
        os.makedirs(model_save_path, exist_ok=True)
    
    def create_model(self) -> Sequential:
        """
        Create the optimized CNN model architecture.
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=self.input_shape,
                   kernel_regularizer=l2(0.01)),
            MaxPooling2D(2, 2),
            Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(0.01)),
            MaxPooling2D(2, 2),
            Dropout(0.3),
            Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Flatten(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile with Adam optimizer - FIXED: Only use 'accuracy' for compatibility
        optimizer_adam = Adam(learning_rate=0.0005)
        model.compile(
            optimizer=optimizer_adam,
            loss='binary_crossentropy',
            metrics=['accuracy']  # Only use accuracy - matches notebook exactly
        )
        
        self.model = model
        logger.info("Model created and compiled successfully")
        return model
    
    def train_model(self, train_dataset, 
                   val_dataset,
                   epochs: int = 20, 
                   patience: int = 5) -> Dict:
        """
        Train the model with early stopping.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Maximum number of epochs
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.create_model()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            os.path.join(self.model_save_path, 'best_model_checkpoint.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train model
        logger.info("Starting model training...")
        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        logger.info("Model training completed")
        return self.history.history
    
    def evaluate_model(self, test_dataset, 
                      test_labels: List[int]) -> Dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            test_dataset: Test dataset
            test_labels: True test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Get model predictions - FIXED: Only expect loss and accuracy
        test_loss, test_acc = self.model.evaluate(test_dataset, verbose=0)
        
        # Get predictions for detailed metrics
        predictions = []
        probabilities = []
        
        for batch_X, batch_y in test_dataset:
            batch_pred = self.model.predict(batch_X, verbose=0)
            predictions.extend([1 if pred > 0.5 else 0 for pred in batch_pred])
            probabilities.extend(batch_pred.flatten())
        
        predictions = np.array(predictions)
        test_labels = np.array(test_labels)
        
        # Calculate metrics manually using sklearn - FIXED
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, zero_division=0)
        recall = recall_score(test_labels, predictions, zero_division=0)
        f1 = f1_score(test_labels, predictions, zero_division=0)
        cm = confusion_matrix(test_labels, predictions)
        
        self.metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'test_loss': float(test_loss),
            'confusion_matrix': cm.tolist(),
            'training_epochs': len(self.history.history['loss']) if self.history else 0,
            'model_architecture': 'CNN with L2 Regularization',
            'optimizer': 'Adam',
            'learning_rate': 0.0005,
            'evaluation_time': datetime.now().isoformat()
        }
        
        logger.info(f"Model evaluation completed - F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
        return self.metrics
    
    def save_model(self, filename: str = "best_dysarthria_model.h5") -> str:
        """
        Save the trained model with compatibility options.
        
        Args:
            filename: Model filename
            
        Returns:
            Full path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        model_path = os.path.join(self.model_save_path, filename)
        
        try:
            # Save in SavedModel format for better compatibility
            if filename.endswith('.h5'):
                # For .h5 files, use legacy format with save_format parameter
                self.model.save(model_path, save_format='h5')
            else:
                # For other formats, use SavedModel format
                self.model.save(model_path)
                
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            # Fallback: save weights only
            weights_path = model_path.replace('.h5', '_weights.h5')
            self.model.save_weights(weights_path)
            logger.info(f"Model weights saved to {weights_path}")
            model_path = weights_path
        
        # Save model info
        model_info = {
            'model_path': model_path,
            'input_shape': list(self.input_shape),
            'classes': ['Control (Healthy)', 'Dysarthric'],
            'preprocessing': {
                'sample_rate': 16000,
                'duration': 2.0,
                'frame_length': 512,
                'frame_step': 256
            },
            'performance': self.metrics,
            'saved_time': datetime.now().isoformat()
        }
        
        info_path = os.path.join(self.model_save_path, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return model_path
    
    def load_model(self, model_path: str):
        """
        Load a saved model with compatibility handling.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded Keras model
        """
        try:
            # Try loading the model directly first
            self.model = load_model(model_path, compile=False)
            
            # Recompile the model with current Keras version - FIXED
            optimizer_adam = Adam(learning_rate=0.0005)
            self.model.compile(
                optimizer=optimizer_adam,
                loss='binary_crossentropy',
                metrics=['accuracy']  # Only use accuracy - matches notebook
            )
            
            logger.info(f"Model loaded and recompiled from {model_path}")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            logger.info("Creating new model instead...")
            
            # If loading fails, create a new model
            return self.create_model()
    
    def predict(self, audio_data: np.ndarray) -> Tuple[int, float]:
        """
        Make prediction on audio data.
        
        Args:
            audio_data: Preprocessed audio spectrogram
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or train model first.")
        
        prediction_prob = self.model.predict(audio_data, verbose=0)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        confidence = float(prediction_prob if prediction == 1 else 1 - prediction_prob)
        
        return prediction, confidence
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary.
        
        Returns:
            Model summary as string
        """
        if self.model is None:
            return "No model available"
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)

class ModelRetrainer:
    def __init__(self, model: DysarthriaModel):
        """
        Initialize model retrainer.
        
        Args:
            model: DysarthriaModel instance
        """
        self.model = model
        self.retrain_threshold = 0.1  # Trigger retraining if performance drops by 10%
        self.baseline_f1 = None
    
    def set_baseline_performance(self, f1_score: float):
        """
        Set baseline F1 score for retraining triggers.
        
        Args:
            f1_score: Baseline F1 score
        """
        self.baseline_f1 = f1_score
        logger.info(f"Baseline F1 score set to {f1_score:.4f}")
    
    def should_retrain(self, current_f1: float) -> bool:
        """
        Check if model should be retrained based on performance degradation.
        
        Args:
            current_f1: Current F1 score
            
        Returns:
            True if retraining is needed
        """
        if self.baseline_f1 is None:
            return False
        
        performance_drop = self.baseline_f1 - current_f1
        should_retrain = performance_drop > self.retrain_threshold
        
        if should_retrain:
            logger.warning(f"Performance drop detected: {performance_drop:.4f}. Retraining recommended.")
        
        return should_retrain
    
    def retrain_model(self, train_dataset,
                     val_dataset,
                     test_dataset,
                     test_labels: List[int]) -> Dict:
        """
        Retrain the model with new data.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            test_labels: Test labels
            
        Returns:
            New metrics after retraining
        """
        logger.info("Starting model retraining...")
        
        # Create new model instance
        self.model.create_model()
        
        # Train model
        self.model.train_model(train_dataset, val_dataset)
        
        # Evaluate
        new_metrics = self.model.evaluate_model(test_dataset, test_labels)
        
        # Save retrained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model.save_model(f"retrained_model_{timestamp}.h5")
        
        # Update baseline if performance improved
        if new_metrics['f1_score'] > (self.baseline_f1 or 0):
            self.set_baseline_performance(new_metrics['f1_score'])
        
        logger.info(f"Model retraining completed. New F1 score: {new_metrics['f1_score']:.4f}")
        return new_metrics
