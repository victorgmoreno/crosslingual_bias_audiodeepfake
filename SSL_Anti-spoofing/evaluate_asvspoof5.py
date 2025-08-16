#!/usr/bin/env python
"""
Script to compute EER for ASVspoof5. 
Usage:
$: python PATH_TO_SCORE_FILE PATH_TO_PROTOCOL_FILE 
 
 -PATH_TO_SCORE_FILE: path to the score file 
 -PATH_TO_PROTOCOL_FILE: path to the protocol file
Example:
$: python evaluate_asvspoof5.py scores.txt /path/to/ASVspoof5.eval.track_1.tsv
"""

import sys, os.path
import numpy as np
import pandas as pd
import eval_metric_LA as em

if len(sys.argv) != 3:
    print("CHECK: invalid input arguments. Please read the instruction below:")
    print(__doc__)
    exit(1)

submit_file = sys.argv[1]
protocol_file = sys.argv[2]

def eval_to_score_file(score_file, protocol_file):
    # Load protocol file
    try:
        protocol_data = pd.read_csv(protocol_file, sep='\t', header=None, names=['protocol_entry'])
        print(f"Loaded protocol with {len(protocol_data)} entries")
        
        # Parse the protocol data by splitting each line into parts
        protocol_entries = []
        file_ids = []
        labels = []
        
        # First, extract the crucial information from each protocol entry
        for idx, row in protocol_data.iterrows():
            parts = row['protocol_entry'].split()
            protocol_entries.append(parts)
            file_ids.append(parts[1])  # Extract file ID (second element in split)
            if len(parts) > 7:  # Make sure there are enough parts
                label_value = parts[7]
                # Convert all attack labels (A17-A32) to 'spoof'
                if label_value.startswith('A') and label_value != 'bonafide':
                    labels.append('spoof')
                else:
                    labels.append(label_value)
            else:
                labels.append('unknown')
        
        # Create a new DataFrame with the information we need
        protocol_df = pd.DataFrame({
            'file_id': file_ids,
            'label': labels
        })
        
        print(f"Protocol parsed with {len(protocol_df)} entries")
        print(f"Label distribution: {protocol_df['label'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"Error processing protocol file: {e}")
        exit(1)
    
    # Load submission scores
    try:
        submission_scores = pd.read_csv(score_file, sep=' ', header=None, names=['file_id', 'score'])
        print(f"Loaded scores with {len(submission_scores)} entries")
        
    except Exception as e:
        print(f"Error loading score file: {e}")
        exit(1)
    
    # Merge submission scores with protocol data
    try:
        merged_data = pd.merge(
            submission_scores, 
            protocol_df,
            on='file_id',  
            how='inner'
        )
        
        print(f"Successfully merged data with {len(merged_data)} entries")
        
        # Extract bonafide and spoof scores
        bona_cm = merged_data[merged_data['label'] == 'bonafide']['score'].values
        spoof_cm = merged_data[merged_data['label'] == 'spoof']['score'].values
        
        print(f"Found {len(bona_cm)} bonafide and {len(spoof_cm)} spoof entries")
        
        if len(bona_cm) == 0 or len(spoof_cm) == 0:
            print("WARNING: Either bonafide or spoof entries are empty. Check protocol parsing.")
            return None
        
        # Calculate EER
        eer_cm, threshold = em.compute_eer(bona_cm, spoof_cm)
        
        # Calculate EER with negated scores (to check for label swapping)
        eer_cm_neg, threshold_neg = em.compute_eer(-bona_cm, -spoof_cm)
        
        # Print results
        out_data = f"EER: {eer_cm*100:.2f}%\n"
        print(out_data, end="")
        
        # Check if negating scores improves performance
        if eer_cm_neg < eer_cm:
            print(
                f'CHECK: we negated your scores and achieved a lower EER. Before: {eer_cm*100:.2f}% - Negated: {eer_cm_neg*100:.2f}% - your class labels are swapped during training.')
        
        if eer_cm == eer_cm_neg:
            print(
                'WARNING: your classifier might not work correctly, we checked if negating your scores gives different EER - it does not. Are all values the same?')
        
        return eer_cm
    except Exception as e:
        print(f"Error during merging or evaluation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    # Check if input files exist
    if not os.path.isfile(submit_file):
        print(f"{submit_file} doesn't exist")
        exit(1)
        
    if not os.path.isfile(protocol_file):
        print(f"{protocol_file} doesn't exist")
        exit(1)
    
    # Evaluate
    _ = eval_to_score_file(submit_file, protocol_file)