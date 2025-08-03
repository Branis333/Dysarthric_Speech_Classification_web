"""
Prediction module for dysarthric speech classification.
Handles audio file upload, preprocessing, and model inference.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional
import logging
from .preprocessing import AudioPreprocessor
from .model import DysarthriaModel
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechClassifier:
    def __init__(self, model_path: str):
        """
        Initialize speech classifier with trained model.
        
        Args:
            model_path: Path to trained model file
        """
        self.model_path = model_path
        self.preprocessor = AudioPreprocessor()
        self.model = None
        self.class_names = ['Control (Healthy)', 'Dysarthric']
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model with compatibility handling."""
        try:
            if os.path.exists(self.model_path):
                # Try loading with compile=False to avoid compatibility issues
                try:
                    self.model = tf.keras.models.load_model(self.model_path, compile=False)
                    
                    # Recompile the model with current Keras version - FIXED
                    from tensorflow.keras.optimizers import Adam
                    optimizer_adam = Adam(learning_rate=0.0005)
                    self.model.compile(
                        optimizer=optimizer_adam,
                        loss='binary_crossentropy',
                        metrics=['accuracy']  # Only use accuracy - matches notebook
                    )
                    
                    logger.info(f"Model loaded and recompiled successfully from {self.model_path}")
                    
                except Exception as load_error:
                    logger.warning(f"Failed to load model: {load_error}")
                    logger.info("Creating new model from scratch...")
                    
                    # Create a new model if loading fails
                    self._create_fallback_model()
                
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Creating new model from scratch...")
                self._create_fallback_model()
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _create_fallback_model(self):
        """Create a new model when loading fails."""
        # Create a new DysarthriaModel instance
        input_shape = (62, 129, 1)  # Default spectrogram shape
        dysarthria_model = DysarthriaModel(input_shape)
        self.model = dysarthria_model.create_model()
        logger.info("Created new fallback model - will need training before use")
    
    def predict_audio_file(self, file_path: str) -> Dict:
        """
        Predict dysarthria from audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess audio
            audio_data = self.preprocessor.preprocess_single_file(file_path)
            
            # Make prediction
            prediction_prob = self.model.predict(audio_data, verbose=0)[0][0]
            prediction = 1 if prediction_prob > 0.5 else 0
            confidence = float(prediction_prob if prediction == 1 else 1 - prediction_prob)
            
            result = {
                'prediction': prediction,
                'predicted_class': self.class_names[prediction],
                'confidence': confidence,
                'probability_dysarthric': float(prediction_prob),
                'probability_control': float(1 - prediction_prob),
                'file_processed': file_path
            }
            
            logger.info(f"Prediction completed for {file_path}: {self.class_names[prediction]} ({confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def predict_audio_data(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        Predict dysarthria from audio array.
        
        Args:
            audio_array: Raw audio data
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # If sample rate is different, we would need to resample
            # For now, assuming input is already at 16kHz
            
            # Convert to tensor and preprocess
            audio_tensor = tf.constant(audio_array, dtype=tf.float32)
            
            # Ensure correct length (2 seconds at 16kHz = 32000 samples)
            target_length = 32000
            if len(audio_tensor) > target_length:
                audio_tensor = audio_tensor[:target_length]
            else:
                padding = target_length - len(audio_tensor)
                audio_tensor = tf.concat([tf.zeros(padding, dtype=tf.float32), audio_tensor], 0)
            
            # Create spectrogram
            spectrogram = tf.signal.stft(audio_tensor, frame_length=512, frame_step=256)
            spectrogram = tf.abs(spectrogram)
            spectrogram = spectrogram[::2, ::2]  # Downsample
            spectrogram = tf.expand_dims(spectrogram, axis=2)
            spectrogram = np.expand_dims(spectrogram.numpy(), axis=0)
            
            # Make prediction
            prediction_prob = self.model.predict(spectrogram, verbose=0)[0][0]
            prediction = 1 if prediction_prob > 0.5 else 0
            confidence = float(prediction_prob if prediction == 1 else 1 - prediction_prob)
            
            result = {
                'prediction': prediction,
                'predicted_class': self.class_names[prediction],
                'confidence': confidence,
                'probability_dysarthric': float(prediction_prob),
                'probability_control': float(1 - prediction_prob),
                'audio_length': len(audio_array),
                'sample_rate': sample_rate
            }
            
            logger.info(f"Prediction completed: {self.class_names[prediction]} ({confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def batch_predict(self, file_paths: list) -> list:
        """
        Predict dysarthria for multiple audio files.
        
        Args:
            file_paths: List of audio file paths
            
        Returns:
            List of prediction results
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.predict_audio_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append({
                    'file_processed': file_path,
                    'error': str(e)
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {'error': 'No model loaded'}
        
        # Get model summary
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        
        return {
            'model_path': self.model_path,
            'input_shape': list(self.model.input_shape[1:]),
            'output_shape': list(self.model.output_shape[1:]),
            'total_params': self.model.count_params(),
            'architecture_summary': '\n'.join(summary_list),
            'class_names': self.class_names
        }

class ModelValidator:
    def __init__(self, classifier: SpeechClassifier):
        """
        Initialize model validator.
        
        Args:
            classifier: SpeechClassifier instance
        """
        self.classifier = classifier
    
    def validate_prediction_format(self, prediction: Dict) -> bool:
        """
        Validate prediction result format.
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            True if format is valid
        """
        required_keys = ['prediction', 'predicted_class', 'confidence', 
                        'probability_dysarthric', 'probability_control']
        
        return all(key in prediction for key in required_keys)
    
    def validate_confidence_scores(self, prediction: Dict) -> bool:
        """
        Validate confidence scores are in valid range.
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            True if confidence scores are valid
        """
        try:
            prob_dys = prediction['probability_dysarthric']
            prob_ctrl = prediction['probability_control']
            confidence = prediction['confidence']
            
            # Check probabilities sum to 1
            prob_sum = abs(prob_dys + prob_ctrl - 1.0)
            if prob_sum > 0.01:  # Allow small floating point errors
                return False
            
            # Check confidence is in [0, 1]
            if not (0 <= confidence <= 1):
                return False
            
            # Check probabilities are in [0, 1]
            if not (0 <= prob_dys <= 1 and 0 <= prob_ctrl <= 1):
                return False
            
            return True
            
        except (KeyError, TypeError):
            return False
    
    def health_check(self) -> Dict:
        """
        Perform health check on the model and classifier.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            'model_loaded': self.classifier.model is not None,
            'model_path_exists': os.path.exists(self.classifier.model_path),
            'preprocessor_initialized': self.classifier.preprocessor is not None,
            'class_names_defined': len(self.classifier.class_names) == 2,
            'timestamp': float(tf.timestamp().numpy())
        }
        
        # Test prediction with dummy data if model is loaded
        if health_status['model_loaded']:
            try:
                # Create dummy spectrogram data
                dummy_input = np.random.random((1, 63, 63, 1)).astype(np.float32)
                _ = self.classifier.model.predict(dummy_input, verbose=0)
                health_status['model_inference_working'] = True
            except Exception as e:
                health_status['model_inference_working'] = False
                health_status['inference_error'] = str(e)
        
        health_status['overall_status'] = all([
            health_status['model_loaded'],
            health_status['model_path_exists'],
            health_status['preprocessor_initialized'],
            health_status['class_names_defined'],
            health_status.get('model_inference_working', False)
        ])
        
        return health_status
