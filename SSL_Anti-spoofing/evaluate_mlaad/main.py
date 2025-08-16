#!/usr/bin/env python
"""
Main entry point for MLAAD evaluation.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

from config import MODEL_PATH, MLAAD_PATH, OUTPUT_DIR, BATCH_SIZE, NUM_WORKERS
from data import MLAADDataset
from evaluate import load_model, evaluate_dataset, save_results
from visualize import generate_statistics, print_summary, create_visualizations, save_statistics

def main():
    """
    Main function to run MLAAD evaluation
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate ASVspoof model on MLAAD dataset')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to the model checkpoint')
    parser.add_argument('--mlaad_path', type=str, default=MLAAD_PATH, help='Path to the MLAAD dataset')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Path to save results')
    parser.add_argument('--language', type=str, default=None, help='Specific language to evaluate (optional)')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum files per model (optional)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load dataset
    dataset = MLAADDataset(args.mlaad_path, args.language, args.max_files)
    if len(dataset) == 0:
        print("No files found for evaluation. Exiting.")
        return
        
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Evaluate
    results = evaluate_dataset(model, dataloader, device)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    save_results(results, results_path)
    
    # Generate statistics and visualizations
    df, lang_stats, arch_stats = generate_statistics(results)
    
    # Print summary
    print_summary(df, lang_stats, arch_stats)
    
    # Save statistics
    save_statistics(lang_stats, arch_stats, args.output_dir)
    
    # Create visualizations
    create_visualizations(df, lang_stats, arch_stats, args.output_dir)
    
    print(f"\nEvaluation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()