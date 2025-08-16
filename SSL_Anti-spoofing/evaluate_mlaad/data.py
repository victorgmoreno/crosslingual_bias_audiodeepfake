"""
Dataset handling for MLAAD evaluation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
import csv
from tqdm import tqdm
import glob

from config import AUDIO_CUT_LENGTH, SAMPLE_RATE

class MLAADDataset(Dataset):
    def __init__(self, base_dir, language=None, max_files=None):
        """
        Dataset for MLAAD evaluation
        
        Args:
            base_dir: Base directory of MLAAD dataset
            language: Optional, specific language to evaluate
            max_files: Optional, maximum number of files to include per model
        """
        self.base_dir = base_dir
        self.audio_files = []
        self.metadata = []
        self.cut = AUDIO_CUT_LENGTH

        # Find languages to process
        languages = []
        if language:
            if os.path.exists(os.path.join(base_dir, language)):
                languages.append(language)
            else:
                print(f"Language directory {language} not found")
                return
        else:
            # Process all languages
            for lang_dir in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, lang_dir)):
                    languages.append(lang_dir)
        
        print(f"Processing languages: {', '.join(languages)}")
        
        # Increase CSV field size limit
        csv.field_size_limit(1000000)  # Set to a much higher value
        
        # Process each language
        for lang in tqdm(languages, desc="Loading languages"):
            lang_dir = os.path.join(base_dir, lang)
            lang_files = 0
            
            # Process each model directory
            for model_dir in os.listdir(lang_dir):
                model_path = os.path.join(lang_dir, model_dir)
                if not os.path.isdir(model_path):
                    continue
                
                # Look for audio files directly
                audio_files = glob.glob(os.path.join(model_path, "*.wav"))
                
                if not audio_files:
                    continue
                
                # Take at most max_files from this model
                if max_files:
                    audio_files = audio_files[:max_files]
                
                # Add files to the dataset
                for audio_path in audio_files:
                    self.audio_files.append(audio_path)
                    
                    # Extract basic metadata from path
                    filename = os.path.basename(audio_path)
                    
                    # Store basic metadata
                    self.metadata.append({
                        'path': audio_path,
                        'language': lang,
                        'model_name': model_dir,
                        'architecture': "unknown",  # We don't have this without the meta.csv
                        'transcript': ""
                    })
                    
                    lang_files += 1
            
            print(f"  Added {lang_files} files for language {lang}")
        
        print(f"Total: Loaded {len(self.audio_files)} audio files for evaluation")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        """
        Get preprocessed audio and metadata for the given index
        """
        audio_path = self.audio_files[index]
        metadata = self.metadata[index]
        
        # Load and preprocess audio
        try:
            x, fs = librosa.load(audio_path, sr=SAMPLE_RATE)
            # Pad or truncate to fixed length
            if len(x) >= self.cut:
                x = x[:self.cut]
            else:
                # Pad with zeros
                x = np.pad(x, (0, self.cut - len(x)), 'constant')
                
            x_inp = torch.FloatTensor(x)
            
            return x_inp, metadata
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            # Return a dummy tensor in case of error
            return torch.zeros(self.cut), metadata