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
from collections import OrderedDict

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
        
        # Initialize model
        model = Model(args, device)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle DataParallel model
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # Remove 'module.' prefix
            else:
                name = k
            new_state_dict[name] = v
        
        # Load state dict
        model.load_state_dict(new_state_dict)
        model = model.to(device)
        model.eval()
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
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
        
        # Ensure minimum length for convolution
        min_length = 320  # This is larger than the kernel size (10)
        if len(x) < min_length:
            x = np.pad(x, (0, min_length - len(x)), 'constant')
        
        # Pad or truncate to fixed length
        if len(x) >= AUDIO_LENGTH:
            x = x[:AUDIO_LENGTH]
        else:
            # Pad with zeros
            x = np.pad(x, (0, AUDIO_LENGTH - len(x)), 'constant')
        
        # For wav2vec models, input should be [batch_size, sequence_length]
        # No need for extra channel dimension
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
    
    # Run inference - handle potential errors
    try:
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
    except Exception as e:
        print(f"Error processing batch: {e}")
        print(f"Audio tensor shape: {batch_tensor.shape}")
        # Try processing one by one to salvage some results
        for i, (tensor, path, model_name) in enumerate(zip(batch_audio, batch_paths, batch_models)):
            try:
                with torch.no_grad():
                    single_tensor = tensor.unsqueeze(0).to(device)
                    single_out = model(single_tensor)
                    single_score = torch.softmax(single_out, dim=1)[0, 1].cpu().item()
                
                results.append({
                    'path': path,
                    'language': language,
                    'model': model_name,
                    'score': float(single_score)
                })
            except Exception as e_inner:
                print(f"Error processing file {path}: {e_inner}")

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
    
    # Create progress bar for languages
    languages = list(catalog['languages'].items())
    pbar = tqdm(total=len(languages), desc="Evaluating languages")
    
    # Process languages
    for i, (language, lang_info) in enumerate(languages):
        print(f"Processing language: {language} ({lang_info['count']} files)")
        
        # Get files for this language
        lang_files = lang_info['files']
        
        # Limit files if specified
        if max_files_per_lang is not None and max_files_per_lang > 0:
            lang_files = lang_files[:max_files_per_lang]
        
        # Create progress bar for batches
        num_batches = (len(lang_files) + BATCH_SIZE - 1) // BATCH_SIZE
        batch_pbar = tqdm(total=num_batches, desc=f"Processing {language}", leave=False)
        
        # Process in batches
        for i in range(0, len(lang_files), BATCH_SIZE):
            batch_files = lang_files[i:i+BATCH_SIZE]
            process_language_batch(model, batch_files, device, language, results)
            batch_pbar.update(1)
        
        # Close batch progress bar
        batch_pbar.close()
        
        # Update language progress bar
        pbar.update(1)
        
        # Save intermediate results
        interim_results_path = os.path.join(output_path, f'interim_results_{language}.json')
        with open(interim_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Interim results saved to: {interim_results_path}")
    
    # Close language progress bar
    pbar.close()
    
    # Save final results
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