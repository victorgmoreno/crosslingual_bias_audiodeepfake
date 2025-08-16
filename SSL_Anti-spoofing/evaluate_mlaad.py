#!/usr/bin/env python
"""
Evaluates ASVspoof 5 model on MLAAD dataset.
"""
import os
import sys
import json
import torch
import numpy as np
import librosa
import argparse
from tqdm import tqdm

# Add original ASVspoof model path to system path
SSL_ANTISPOOFING_PATH = '/home/victor.moreno/dl-29_backup/spoof/SSL_Anti-spoofing'
sys.path.insert(0, SSL_ANTISPOOFING_PATH)

# Constants
AUDIO_LENGTH = 64600  # ~4 seconds at 16kHz
SAMPLE_RATE = 16000
BATCH_SIZE = 32

def load_asvspoof_model(model_path, device):
    """
    Load ASVspoof 5 model.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    print(f"Loading model from {model_path}...")
    
    try:
        # Import model from original directory
        from model import Model
        
        # Create dummy args
        class Args:
            pass
        args = Args()
        
        # Load model
        model = Model(args, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run this script from the SSL_Anti-spoofing directory.")
        sys.exit(1)

def preprocess_audio(audio_path):
    """
    Preprocess audio file.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Preprocessed audio tensor
    """
    try:
        # Load audio
        x, fs = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Pad or truncate to fixed length
        if len(x) >= AUDIO_LENGTH:
            x = x[:AUDIO_LENGTH]
        else:
            x = np.pad(x, (0, AUDIO_LENGTH - len(x)), 'constant')
        
        return torch.FloatTensor(x)
    except Exception as e:
        print(f"Error preprocessing audio {audio_path}: {e}")
        return None

def process_language_batch(model, files, device, language, results):
    """
    Process a batch of files for a language.
    
    Args:
        model: ASVspoof model
        files: List of files to process
        device: Device to run inference on
        language: Language being processed
        results: Results dictionary to update
    """
    batch_audio = []
    batch_paths = []
    batch_models = []
    
    # Preprocess audio files
    for file_info in files:
        audio_path = file_info['path']
        audio_tensor = preprocess_audio(audio_path)
        
        if audio_tensor is not None:
            batch_audio.append(audio_tensor)
            batch_paths.append(audio_path)
            batch_models.append(file_info['model'])
    
    if not batch_audio:
        return
    
    # Create batch tensor
    batch_tensor = torch.stack(batch_audio).to(device)
    
    # Run inference
    with torch.no_grad():
        batch_out = model(batch_tensor)
        batch_scores = torch.softmax(batch_out, dim=1)[:, 1].cpu().numpy()
    
    # Save results
    for i, (path, model_name, score) in enumerate(zip(batch_paths, batch_models, batch_scores)):
        results.append({
            'path': path,
            'language': language,
            'model': model_name,
            'score': float(score)  # Probability of being spoof
        })

def evaluate_mlaad(model_path, catalog_path, output_path, max_files_per_lang=None):
    """
    Evaluate ASVspoof 5 model on MLAAD dataset.
    
    Args:
        model_path: Path to model checkpoint
        catalog_path: Path to dataset catalog
        output_path: Path to save results
        max_files_per_lang: Maximum files to process per language
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_asvspoof_model(model_path, device)
    
    # Load dataset catalog
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Evaluate each language
    results = []
    
    for language, lang_info in tqdm(catalog['languages'].items(), desc="Evaluating languages"):
        print(f"Processing language: {language} ({lang_info['count']} files)")
        
        # Get files for this language
        lang_files = lang_info['files']
        
        # Limit files if specified
        if max_files_per_lang is not None and max_files_per_lang > 0:
            lang_files = lang_files[:max_files_per_lang]
        
        # Process in batches
        for i in range(0, len(lang_files), BATCH_SIZE):
            batch_files = lang_files[i:i+BATCH_SIZE]
            process_language_batch(model, batch_files, device, language, results)
    
    # Save results
    results_path = os.path.join(output_path, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation completed. Results saved to: {results_path}")
    print(f"Total files evaluated: {len(results)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate ASVspoof 5 model on MLAAD dataset")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--catalog_path', type=str, required=True, help='Path to dataset catalog')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--max_files_per_lang', type=int, default=None, help='Maximum files to process per language')
    
    args = parser.parse_args()
    evaluate_mlaad(args.model_path, args.catalog_path, args.output_path, args.max_files_per_lang)

if __name__ == "__main__":
    main()