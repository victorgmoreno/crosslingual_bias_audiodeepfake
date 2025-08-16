#!/usr/bin/env python
"""
Analysis script for M-AILABS evaluation results.

Usage:
python analyze_results.py --results_csv RESULTS_CSV_PATH [--output_dir OUTPUT_DIR]

Example:
python analyze_results.py \
    --results_csv /home/victor.moreno/dl-29_backup/spoof/dataset/code/mailabs_evaluation_results.csv \
    --output_dir /home/victor.moreno/dl-29_backup/spoof/dataset/code/analysis_results
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def setup_plot_style():
    """Set up plotting style for consistent visuals"""
    plt.style.use('seaborn-whitegrid')
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['savefig.dpi'] = 300

def analyze_language_bias(df, output_dir):
    """Analyze and visualize bias by language"""
    print("Analyzing language bias...")
    
    # Group by language and compute statistics
    lang_stats = df.groupby('language').agg({
        'antispoofing_score': ['mean', 'std', 'median', 'count'],
        'is_false_positive': ['mean']
    })
    lang_stats.columns = ['mean_score', 'std_score', 'median_score', 'count', 'false_positive_rate']
    lang_stats = lang_stats.reset_index().sort_values('false_positive_rate', ascending=False)
    
    # Save language statistics to CSV
    lang_stats.to_csv(os.path.join(output_dir, 'language_statistics.csv'), index=False)
    
    # Plot false positive rates by language
    plt.figure(figsize=(14, 8))
    bars = sns.barplot(x='language', y='false_positive_rate', data=lang_stats, palette='viridis')
    
    # Add counts on top of bars
    for i, bar in enumerate(bars.patches):
        bars.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"n={int(lang_stats.iloc[i]['count'])}",
            ha='center',
            va='bottom',
            rotation=45,
            fontsize=8
        )
    
    plt.axhline(y=df['is_false_positive'].mean(), color='r', linestyle='--', label='Overall FPR')
    plt.title('False Positive Rate by Language')
    plt.ylabel('False Positive Rate')
    plt.xlabel('Language')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'language_false_positive_rates.png'))
    
    # Plot mean scores by language
    plt.figure(figsize=(14, 8))
    bars = sns.barplot(x='language', y='mean_score', data=lang_stats, palette='viridis')
    
    # Add error bars
    for i, bar in enumerate(bars.patches):
        bars.errorbar(
            x=i, 
            y=lang_stats.iloc[i]['mean_score'], 
            yerr=lang_stats.iloc[i]['std_score'],
            fmt='none', 
            color='black', 
            capsize=5
        )
    
    plt.axhline(y=df['antispoofing_score'].mean(), color='r', linestyle='--', label='Overall Mean')
    plt.title('Mean Antispoofing Scores by Language')
    plt.ylabel('Mean Score')
    plt.xlabel('Language')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'language_mean_scores.png'))
    
    return lang_stats

def analyze_gender_bias(df, output_dir):
    """Analyze and visualize bias by gender"""
    print("Analyzing gender bias...")
    
    # Group by gender and compute statistics
    gender_stats = df.groupby('gender').agg({
        'antispoofing_score': ['mean', 'std', 'median', 'count'],
        'is_false_positive': ['mean']
    })
    gender_stats.columns = ['mean_score', 'std_score', 'median_score', 'count', 'false_positive_rate']
    gender_stats = gender_stats.reset_index().sort_values('false_positive_rate', ascending=False)
    
    # Save gender statistics to CSV
    gender_stats.to_csv(os.path.join(output_dir, 'gender_statistics.csv'), index=False)
    
    # Plot false positive rates by gender
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x='gender', y='false_positive_rate', data=gender_stats, palette='Set2')
    
    # Add counts on top of bars
    for i, bar in enumerate(bars.patches):
        bars.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"n={int(gender_stats.iloc[i]['count'])}",
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.axhline(y=df['is_false_positive'].mean(), color='r', linestyle='--', label='Overall FPR')
    plt.title('False Positive Rate by Gender')
    plt.ylabel('False Positive Rate')
    plt.xlabel('Gender')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gender_false_positive_rates.png'))
    
    # Plot mean scores by gender
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x='gender', y='mean_score', data=gender_stats, palette='Set2')
    
    # Add error bars
    for i, bar in enumerate(bars.patches):
        bars.errorbar(
            x=i, 
            y=gender_stats.iloc[i]['mean_score'], 
            yerr=gender_stats.iloc[i]['std_score'],
            fmt='none', 
            color='black', 
            capsize=5
        )
    
    plt.axhline(y=df['antispoofing_score'].mean(), color='r', linestyle='--', label='Overall Mean')
    plt.title('Mean Antispoofing Scores by Gender')
    plt.ylabel('Mean Score')
    plt.xlabel('Gender')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gender_mean_scores.png'))
    
    # Statistical tests between genders
    if len(gender_stats) >= 2:
        genders = gender_stats['gender'].values
        print("\nStatistical tests between genders:")
        
        for i in range(len(genders)):
            for j in range(i+1, len(genders)):
                gender1 = genders[i]
                gender2 = genders[j]
                
                scores1 = df[df['gender'] == gender1]['antispoofing_score'].values
                scores2 = df[df['gender'] == gender2]['antispoofing_score'].values
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(scores1, scores2, equal_var=False)
                print(f"{gender1} vs {gender2}: t={t_stat:.4f}, p={p_value:.4f}")
                
                # Effect size (Cohen's d)
                mean1, mean2 = np.mean(scores1), np.mean(scores2)
                std1, std2 = np.std(scores1), np.std(scores2)
                pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                cohens_d = (mean1 - mean2) / pooled_std
                print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
    
    return gender_stats

def analyze_language_gender_interaction(df, output_dir):
    """Analyze interaction between language and gender"""
    print("Analyzing language and gender interaction...")
    
    # Group by the combined category
    interaction_stats = df.groupby(['language', 'gender']).agg({
        'antispoofing_score': ['mean', 'std', 'count'],
        'is_false_positive': ['mean']
    })
    interaction_stats.columns = ['mean_score', 'std_score', 'count', 'false_positive_rate']
    interaction_stats = interaction_stats.reset_index()
    
    # Filter out combinations with too few samples
    min_samples = 10
    interaction_stats = interaction_stats[interaction_stats['count'] >= min_samples]
    
    if interaction_stats.empty:
        print("Not enough data for language-gender interaction analysis")
        return
    
    # Save interaction statistics to CSV
    interaction_stats.to_csv(os.path.join(output_dir, 'language_gender_interaction.csv'), index=False)
    
    # Plot interaction of false positive rates
    plt.figure(figsize=(16, 10))
    interaction_plot = sns.barplot(
        x='language', 
        y='false_positive_rate', 
        hue='gender', 
        data=interaction_stats,
        palette='Set2'
    )
    
    plt.title('False Positive Rate by Language and Gender')
    plt.ylabel('False Positive Rate')
    plt.xlabel('Language')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Gender')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'language_gender_fpr_interaction.png'))
    
    # Heatmap of false positive rates
    pivot_table = interaction_stats.pivot_table(
        index='language', 
        columns='gender', 
        values='false_positive_rate'
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        pivot_table, 
        annot=True, 
        cmap='viridis', 
        cbar_kws={'label': 'False Positive Rate'},
        fmt='.3f'
    )
    plt.title('False Positive Rate by Language and Gender')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'language_gender_fpr_heatmap.png'))

def analyze_duration_effects(df, output_dir):
    """Analyze how audio duration affects scores"""
    print("Analyzing effects of audio duration...")
    
    # Check if duration column exists
    if 'duration' not in df.columns:
        print("Duration column not found, skipping duration analysis")
        return
    
    # Remove any outliers in duration
    q1 = df['duration'].quantile(0.25)
    q3 = df['duration'].quantile(0.75)
    iqr = q3 - q1
    df_filtered = df[(df['duration'] >= q1 - 1.5 * iqr) & 
                     (df['duration'] <= q3 + 1.5 * iqr)]
    
    # Plot scatter plot of duration vs score
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_filtered, 
        x='duration', 
        y='antispoofing_score', 
        alpha=0.3,
        hue='is_false_positive'
    )
    
    # Add trend line
    sns.regplot(
        data=df_filtered, 
        x='duration', 
        y='antispoofing_score', 
        scatter=False, 
        line_kws={'color': 'red'}
    )
    
    # Calculate correlation
    corr, p_value = stats.pearsonr(df_filtered['duration'], df_filtered['antispoofing_score'])
    plt.title(f'Relationship Between Audio Duration and Score\nCorrelation: {corr:.3f} (p={p_value:.3f})')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Antispoofing Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_score_relationship.png'))
    
    # Bin durations and analyze mean scores by bin
    df_filtered['duration_bin'] = pd.cut(
        df_filtered['duration'], 
        bins=10, 
        labels=[f"{i+1}" for i in range(10)]
    )
    
    duration_bin_stats = df_filtered.groupby('duration_bin').agg({
        'antispoofing_score': ['mean', 'std', 'count'],
        'is_false_positive': ['mean']
    })
    duration_bin_stats.columns = ['mean_score', 'std_score', 'count', 'false_positive_rate']
    duration_bin_stats = duration_bin_stats.reset_index()
    
    # Plot false positive rate by duration bin
    plt.figure(figsize=(14, 8))
    bars = sns.barplot(
        x='duration_bin', 
        y='false_positive_rate', 
        data=duration_bin_stats,
        palette='viridis'
    )
    
    plt.axhline(y=df_filtered['is_false_positive'].mean(), color='r', linestyle='--', label='Overall FPR')
    plt.title('False Positive Rate by Duration Bin')
    plt.ylabel('False Positive Rate')
    plt.xlabel('Duration Bin (1=shortest, 10=longest)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_bin_fpr.png'))

def main():
    parser = argparse.ArgumentParser(description="Analyze M-AILABS evaluation results")
    parser.add_argument('--results_csv', type=str, required=True, 
                        help='Path to the evaluation results CSV file')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    create_directory(args.output_dir)
    
    # Set up matplotlib style
    setup_plot_style()
    
    # Load results
    print(f"Loading results from {args.results_csv}")
    df = pd.read_csv(args.results_csv)
    
    # Add is_false_positive column if it doesn't exist
    if 'is_false_positive' not in df.columns and 'antispoofing_score' in df.columns:
        # For M-AILABS, all samples are bonafide, so false positive when predicted as spoof
        threshold = 0.5
        df['is_false_positive'] = (df['antispoofing_score'] < threshold).astype(int)
    
    # Generate overall statistics
    print("\nOverall Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Average antispoofing score: {df['antispoofing_score'].mean():.4f}")
    print(f"Overall false positive rate: {df['is_false_positive'].mean():.4f}")
    
    # Perform analyses
    lang_stats = analyze_language_bias(df, args.output_dir)
    gender_stats = analyze_gender_bias(df, args.output_dir)
    analyze_language_gender_interaction(df, args.output_dir)
    analyze_duration_effects(df, args.output_dir)
    
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()