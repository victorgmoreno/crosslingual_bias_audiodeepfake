#!/usr/bin/env python
"""
Script to analyze MAILABS evaluation results.
Since MAILABS only contains bonafide samples, we analyze score distribution
and model behavior on genuine speech.

Usage:
python analyze_mailabs_results.py score_file.txt [csv_file.csv]

Arguments:
- score_file.txt: Simple score file (filename score format)
- csv_file.csv: Optional CSV file with metadata for detailed analysis
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_scores(score_file):
    """Load scores from simple text file format"""
    scores_data = []
    with open(score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                filename, score = parts[0], float(parts[1])
                scores_data.append({'filename': filename, 'score': score})
    
    return pd.DataFrame(scores_data)


def analyze_score_distribution(scores_df):
    """Analyze the distribution of scores"""
    scores = scores_df['score'].values
    
    print("=== SCORE DISTRIBUTION ANALYSIS ===")
    print(f"Number of samples: {len(scores)}")
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Median score: {np.median(scores):.4f}")
    print(f"Standard deviation: {np.std(scores):.4f}")
    print(f"Min score: {np.min(scores):.4f}")
    print(f"Max score: {np.max(scores):.4f}")
    print(f"25th percentile: {np.percentile(scores, 25):.4f}")
    print(f"75th percentile: {np.percentile(scores, 75):.4f}")
    print()
    
    # Check score interpretation
    print("=== SCORE INTERPRETATION ===")
    print("In spoofing detection models:")
    print("- Higher scores typically indicate SPOOFED samples")
    print("- Lower scores typically indicate BONAFIDE samples")
    print()
    
    # Since these are all bonafide samples, analyze how well the model performs
    high_score_threshold = np.percentile(scores, 90)  # Top 10% of scores
    low_score_threshold = np.percentile(scores, 10)   # Bottom 10% of scores
    
    high_scores = scores[scores >= high_score_threshold]
    low_scores = scores[scores <= low_score_threshold]
    
    print(f"Samples with HIGH scores (>{high_score_threshold:.3f}): {len(high_scores)} ({len(high_scores)/len(scores)*100:.1f}%)")
    print(f"Samples with LOW scores (<{low_score_threshold:.3f}): {len(low_scores)} ({len(low_scores)/len(scores)*100:.1f}%)")
    print()
    
    # Expected behavior analysis
    if np.mean(scores) < 5.0:  # Assuming scores are typically in 0-10 range
        print("✓ GOOD: Most scores are relatively low, suggesting model correctly")
        print("  identifies these bonafide samples as genuine speech.")
    else:
        print("⚠ WARNING: Many scores are high, suggesting model may be")
        print("  incorrectly flagging bonafide samples as spoofed.")
    print()


def analyze_by_metadata(scores_df, csv_file):
    """Analyze scores by metadata categories"""
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found. Skipping metadata analysis.")
        return
    
    # Load metadata
    metadata_df = pd.read_csv(csv_file)
    
    # Extract filename from file_path for merging
    metadata_df['filename'] = metadata_df['file_name']
    
    # Merge with scores
    merged_df = scores_df.merge(metadata_df, on='filename', how='left')
    
    print("=== ANALYSIS BY METADATA ===")
    
    # Analysis by language
    if 'language' in merged_df.columns:
        print("\n--- Scores by Language ---")
        lang_stats = merged_df.groupby('language')['score'].agg(['count', 'mean', 'std'])
        print(lang_stats)
    
    # Analysis by gender
    if 'gender' in merged_df.columns:
        print("\n--- Scores by Gender ---")
        gender_stats = merged_df.groupby('gender')['score'].agg(['count', 'mean', 'std'])
        print(gender_stats)
    
    # Analysis by speaker
    if 'speaker' in merged_df.columns:
        print("\n--- Scores by Speaker (top 10) ---")
        speaker_stats = merged_df.groupby('speaker')['score'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
        print(speaker_stats.head(10))
    
    # Analysis by duration
    if 'duration' in merged_df.columns:
        print("\n--- Scores by Duration Range ---")
        merged_df['duration_range'] = pd.cut(merged_df['duration'], 
                                           bins=[0, 3, 6, 10, float('inf')], 
                                           labels=['0-3s', '3-6s', '6-10s', '10s+'])
        duration_stats = merged_df.groupby('duration_range')['score'].agg(['count', 'mean', 'std'])
        print(duration_stats)


def create_visualizations(scores_df, csv_file=None):
    """Create visualizations of the results"""
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('seaborn')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MAILABS Evaluation Results Analysis', fontsize=16)
    
    scores = scores_df['score'].values
    
    # Histogram
    axes[0, 0].hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
    axes[0, 0].axvline(np.median(scores), color='green', linestyle='--', label=f'Median: {np.median(scores):.3f}')
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot
    axes[0, 1].boxplot(scores, vert=True)
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Score Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # QQ plot to check normality
    stats.probplot(scores, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_scores = np.sort(scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1, 1].plot(sorted_scores, cumulative, linewidth=2, color='navy')
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution Function')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.dirname(sys.argv[1]) if len(sys.argv) > 1 else '.'
    plot_path = os.path.join(output_dir, 'mailabs_analysis_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_path}")
    
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_mailabs_results.py score_file.txt [csv_file.csv]")
        sys.exit(1)
    
    score_file = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(score_file):
        print(f"Score file {score_file} not found!")
        sys.exit(1)
    
    # Load scores
    print(f"Loading scores from: {score_file}")
    scores_df = load_scores(score_file)
    
    if len(scores_df) == 0:
        print("No scores found in the file!")
        sys.exit(1)
    
    # Analyze score distribution
    analyze_score_distribution(scores_df)
    
    # Analyze by metadata if CSV provided
    if csv_file:
        analyze_by_metadata(scores_df, csv_file)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(scores_df, csv_file)
    
    print("\n=== SUMMARY ===")
    print("Since MAILABS contains only bonafide samples, we cannot compute")
    print("traditional spoofing metrics (EER, tDCF). However, we can analyze:")
    print("1. Score distribution - are scores generally low (good) or high (concerning)?")
    print("2. Model consistency across different speakers/languages/durations")
    print("3. Potential bias or outliers in the model predictions")
    print("\nFor a complete evaluation, you would need both bonafide and spoofed samples.")


if __name__ == "__main__":
    main()