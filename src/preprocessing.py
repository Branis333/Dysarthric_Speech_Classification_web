"""
Data preprocessing module for dysarthric speech classification.
Handles audio file loading, feature extraction, and data preparation.
"""

import os
import librosa
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    def __init__(self, sample_rate: int = 16000, duration: float = 2.0, 
                 frame_length: int = 512, frame_step: int = 256):
        """
        Initialize audio preprocessor with configuration parameters.
        
        Args:
            sample_rate: Target sample rate for audio files (16000 Hz)
            duration: Duration to extract from audio files (2.0 seconds)
            frame_length: Frame length for STFT (512 - matches notebook)
            frame_step: Frame step for STFT (256 - matches notebook)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.frame_length = frame_length
        self.frame_step = frame_step
        # Use exact target length from notebook: 32000 samples
        self.target_length = 32000  # Fixed to match notebook exactly
        
    def load_wav_16k_mono(self, filename: str):
        """
        Load a WAV file, convert to float tensor, resample to 16kHz mono.
        
        Args:
            filename: Path to the audio file
            
        Returns:
            Audio tensor
        """
        try:
            audio, sr = librosa.load(filename, sr=self.sample_rate, 
                                   mono=True, duration=self.duration)
            wav = tf.constant(audio, dtype=tf.float32)
            return wav
        except Exception as e:
            logger.error(f"Error loading audio file {filename}: {e}")
            raise
    
    def preprocess_audio_for_nn(self, file_path: str, label: Optional[int] = None):
        """
        Convert audio to spectrogram for neural network (matches notebook exactly).
        
        Args:
            file_path: Path to audio file
            label: Optional label for the audio
            
        Returns:
            Tuple of (spectrogram, label)
        """
        try:
            wav = self.load_wav_16k_mono(file_path)
            
            # Fixed length (32000 samples = 2 seconds at 16kHz) - EXACT match to notebook
            if len(wav) > 32000:
                wav = wav[:32000]
            else:
                padding = 32000 - len(wav)
                wav = tf.concat([tf.zeros(padding, dtype=tf.float32), wav], 0)
            
            # Create spectrogram - EXACT match to notebook parameters
            spectrogram = tf.signal.stft(wav, frame_length=512, frame_step=256)
            spectrogram = tf.abs(spectrogram)
            
            # Add temporary channel dimension for tf.image.resize compatibility
            spectrogram = tf.expand_dims(spectrogram, axis=-1)

            # Resize explicitly to (62, 129)
            spectrogram = tf.image.resize(spectrogram, [62, 129])

            # Remove temporary dimension and add final channel dimension
            spectrogram = tf.squeeze(spectrogram, axis=-1)        
            spectrogram = tf.expand_dims(spectrogram, axis=-1)
            
            return spectrogram, label
            
        except Exception as e:
            logger.error(f"Error preprocessing audio {file_path}: {e}")
            raise
    
    def preprocess_single_file(self, file_path: str) -> np.ndarray:
        """
        Preprocess a single audio file for prediction.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Preprocessed spectrogram as numpy array
        """
        spectrogram, _ = self.preprocess_audio_for_nn(file_path)
        return np.expand_dims(spectrogram.numpy(), axis=0)

class DataLoader:
    def __init__(self, preprocessor: AudioPreprocessor):
        """
        Initialize data loader with audio preprocessor.
        
        Args:
            preprocessor: AudioPreprocessor instance
        """
        self.preprocessor = preprocessor
    
    def get_file_list(self, directory: str) -> List[str]:
        """
        Get list of wav files from directory recursively.
        
        Args:
            directory: Directory path to search
            
        Returns:
            List of wav file paths
        """
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return []
        
        wav_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(wav_files)} wav files in {directory}")
        return wav_files
    
    def load_dataset(self, control_path: str, dysarthric_path: str, 
                    max_files_per_class: int = 50) -> Tuple[List[str], List[int]]:
        """
        Load dataset from control and dysarthric directories.
        
        Args:
            control_path: Path to control (healthy) audio files
            dysarthric_path: Path to dysarthric audio files
            max_files_per_class: Maximum files to use per class
            
        Returns:
            Tuple of (all_files, all_labels)
        """
        control_files = self.get_file_list(control_path)[:max_files_per_class]
        dysarthric_files = self.get_file_list(dysarthric_path)[:max_files_per_class]
        
        # Create labels and file lists
        all_files = control_files + dysarthric_files
        all_labels = [0] * len(control_files) + [1] * len(dysarthric_files)
        
        logger.info(f"Dataset loaded: {len(all_files)} total files "
                   f"({len(control_files)} control, {len(dysarthric_files)} dysarthric)")
        
        return all_files, all_labels
    
    def create_dataset_generator(self, files: List[str], labels: List[int], 
                               batch_size: int = 8):
        """
        Create TensorFlow dataset generator.
        
        Args:
            files: List of file paths
            labels: List of corresponding labels
            batch_size: Batch size for dataset
            
        Returns:
            TensorFlow dataset
        """
        def data_generator():
            for file_path, label in zip(files, labels):
                try:
                    spectrogram, processed_label = self.preprocessor.preprocess_audio_for_nn(
                        file_path, label)
                    yield spectrogram.numpy(), float(processed_label)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue
        
        # Get sample to determine output signature
        sample_spec, sample_label = self.preprocessor.preprocess_audio_for_nn(files[0], labels[0])
        
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=sample_spec.shape, dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            )
        )
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
