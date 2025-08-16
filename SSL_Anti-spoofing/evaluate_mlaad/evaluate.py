"""
Core evaluation logic for running model inference on MLAAD dataset.
"""

import os
import sys
import torch
import json
from tqdm import tqdm

from model_loader import load_asvspoof_model
from config import SSL_ANTISPOOFING_PATH

# Add path to the original model code
sys.path.append(SSL_ANTISPOOFING_PATH)
from model import Model

def load_model(model_path, device):
    """
    Load the ASVspoof model
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    model = load_asvspoof_model(model_path, device)
    model.eval()
    return model

def evaluate_dataset(model, dataloader, device):
    """
    Run inference on dataset
    
    Args:
        model: Loaded model
        dataloader: Dataset loader
        device: Device to run inference on
    
    Returns:
        List of results with metadata and scores
    """
    results = []
    
    # Evaluate
    print("Starting evaluation...")
    with torch.no_grad():
        for batch_x, batch_metadata in tqdm(dataloader, desc="Evaluating"):
            print(f"type(batch_metadata): {type(batch_metadata)}")
            print(f"batch_metadata: {batch_metadata}")
            
            try:
                first_metadata = {
                    'path': batch_metadata['path'][0],
                    'language': batch_metadata['language'][0],
                    'model_name': batch_metadata['model_name'][0],
                    'architecture': batch_metadata['architecture'][0]
                }
                print(f"First metadata: {first_metadata}")
            except Exception as e:
                print(f"Error accessing first metadata: {e}")

            batch_x = batch_x.to(device)
            
            # Get model predictions
            batch_out = model(batch_x)
            batch_scores = torch.softmax(batch_out, dim=1)[:, 1].cpu().numpy()  # Probability of being spoof
            
            # Save results
            for i, score in enumerate(batch_scores):
                metadata = {
                    'path': batch_metadata['path'][i],
                    'language': batch_metadata['language'][i],
                    'model_name': batch_metadata['model_name'][i],
                    'architecture': batch_metadata['architecture'][i]
                }
                result = {
                    'path': metadata['path'],
                    'language': metadata['language'],
                    'model_name': metadata['model_name'],
                    'architecture': metadata['architecture'],
                    'score': float(score)
                }
                results.append(result)

    return results

def save_results(results, output_path):
    """
    Save evaluation results to JSON
    
    Args:
        results: Evaluation results
        output_path: Path to save results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {output_path}")
    print(f"Total results saved: {len(results)}")
    print(f"First result: {results[0]}")
    print(f"Last result: {results[-1]}")
    print(f"First 5 results: {results[:5]}")
    print(f"Last 5 results: {results[-5:]}")
