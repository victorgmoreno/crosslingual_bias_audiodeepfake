#!/usr/bin/env python
"""
Organizes the MLAAD dataset for ASVspoof 5 model evaluation.
"""
import os
import sys
import glob
import json
import argparse
from tqdm import tqdm

def organize_mlaad_dataset(mlaad_path, output_path):
    """
    Organizes MLAAD dataset for evaluation.
    
    Args:
        mlaad_path: Path to MLAAD dataset
        output_path: Path to save organized dataset catalog
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Dictionary to store dataset organization
    dataset_catalog = {
        'languages': {},
        'total_files': 0
    }
    
    # Find all language directories
    try:
        languages = [lang_dir for lang_dir in os.listdir(mlaad_path) 
                    if os.path.isdir(os.path.join(mlaad_path, lang_dir))]
    except Exception as e:
        print(f"Error accessing MLAAD dataset directory: {e}")
        print(f"Please check that the path exists and is accessible: {mlaad_path}")
        sys.exit(1)
    
    # Process each language
    for lang in tqdm(languages, desc="Processing languages"):
        lang_dir = os.path.join(mlaad_path, lang)
        lang_files = []
        
        # Process each model directory
        for model_dir in os.listdir(lang_dir):
            model_path = os.path.join(lang_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
            
            # Find all audio files
            audio_files = glob.glob(os.path.join(model_path, "*.wav"))
            
            if not audio_files:
                continue
            
            # Add files to language catalog
            for audio_path in audio_files:
                lang_files.append({
                    'path': audio_path,
                    'model': model_dir
                })
        
        # Add language to catalog
        dataset_catalog['languages'][lang] = {
            'count': len(lang_files),
            'files': lang_files
        }
        dataset_catalog['total_files'] += len(lang_files)
        
        print(f"  Found {len(lang_files)} files for language: {lang}")
    
    # Save catalog
    catalog_path = os.path.join(output_path, 'mlaad_catalog.json')
    with open(catalog_path, 'w') as f:
        json.dump(dataset_catalog, f, indent=2)
    
    print(f"Dataset catalog saved to: {catalog_path}")
    print(f"Total languages: {len(dataset_catalog['languages'])}")
    print(f"Total files: {dataset_catalog['total_files']}")
    
    return dataset_catalog

def main():
    parser = argparse.ArgumentParser(description="Organize MLAAD dataset for evaluation")
    parser.add_argument('--mlaad_path', type=str, required=True, help='Path to MLAAD dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save organized dataset')
    
    args = parser.parse_args()
    organize_mlaad_dataset(args.mlaad_path, args.output_path)

if __name__ == "__main__":
    main()