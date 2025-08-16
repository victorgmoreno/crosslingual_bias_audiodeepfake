#!/usr/bin/env python
"""
Analyzes evaluation results and extracts metrics.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def analyze_results(results_path, output_path):
    """
    Analyze evaluation results and extract metrics.
    
    Args:
        results_path: Path to evaluation results
        output_path: Path to save analysis
    """
    # Check if results file exists
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        print("Please run the evaluation step first to generate results.")
        return
    
    # Load results
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading results file: {e}")
        return
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    total_files = len(df)
    avg_score = df['score'].mean()
    high_conf_count = (df['score'] > 0.9).sum()
    high_conf_ratio = high_conf_count / total_files
    
    print("\n=== OVERALL STATISTICS ===")
    print(f"Total files evaluated: {total_files}")
    print(f"Average spoofing score: {avg_score:.4f}")
    print(f"High confidence detections (score > 0.9): {high_conf_count} ({high_conf_ratio * 100:.2f}%)")
    
    # Language statistics
    lang_stats = df.groupby('language')['score'].agg(['count', 'mean', 'std']).reset_index()
    lang_stats = lang_stats.sort_values('count', ascending=False)
    
    # Calculate high confidence ratio by language
    high_conf_by_lang = {}
    for lang in lang_stats['language']:
        lang_files = df[df['language'] == lang]
        high_conf = (lang_files['score'] > 0.9).sum()
        high_conf_by_lang[lang] = high_conf / len(lang_files)
    
    lang_stats['high_conf_ratio'] = lang_stats['language'].map(high_conf_by_lang)
    
    print("\n=== LANGUAGE STATISTICS ===")
    print(lang_stats.to_string(index=False))
    
    # Model statistics by grouping models across languages
    model_stats = df.groupby('model')['score'].agg(['count', 'mean', 'std']).reset_index()
    model_stats = model_stats.sort_values('count', ascending=False)
    
    # Calculate high confidence ratio by model
    high_conf_by_model = {}
    for model in model_stats['model']:
        model_files = df[df['model'] == model]
        high_conf = (model_files['score'] > 0.9).sum()
        high_conf_by_model[model] = high_conf / len(model_files)
    
    model_stats['high_conf_ratio'] = model_stats['model'].map(high_conf_by_model)
    
    print("\n=== MODEL STATISTICS (TOP 10) ===")
    print(model_stats.head(10).to_string(index=False))
    
    # Save statistics
    lang_stats.to_csv(os.path.join(output_path, 'language_statistics.csv'), index=False)
    model_stats.to_csv(os.path.join(output_path, 'model_statistics.csv'), index=False)
    
    # Create summary document
    summary = {
        'overall': {
            'total_files': total_files,
            'avg_score': float(avg_score),
            'high_conf_count': int(high_conf_count),
            'high_conf_ratio': float(high_conf_ratio)
        },
        'by_language': {
            lang: {
                'count': int(stats['count']),
                'avg_score': float(stats['mean']),
                'std_dev': float(stats['std']),
                'high_conf_ratio': float(stats['high_conf_ratio'])
            }
            for lang, stats in lang_stats.set_index('language').iterrows()
        },
        'by_model': {
            model: {
                'count': int(stats['count']),
                'avg_score': float(stats['mean']),
                'std_dev': float(stats['std']),
                'high_conf_ratio': float(stats['high_conf_ratio'])
            }
            for model, stats in model_stats.head(10).set_index('model').iterrows()
        }
    }
    
    with open(os.path.join(output_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create visualizations
    vis_dir = os.path.join(output_path, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['score'], bins=50, kde=True)
    plt.title('Distribution of Spoofing Scores')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'score_distribution.png'), dpi=300)
    
    # 2. Scores by language
    plt.figure(figsize=(14, 8))
    sns.barplot(x='language', y='mean', data=lang_stats.head(15), yerr=lang_stats.head(15)['std'])
    plt.title('Average Spoofing Score by Language (Top 15)')
    plt.xlabel('Language')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'language_scores.png'), dpi=300)
    
    # 3. High confidence ratio by language
    plt.figure(figsize=(14, 8))
    sns.barplot(x='language', y='high_conf_ratio', data=lang_stats.head(15))
    plt.title('High Confidence Detection Ratio by Language (Top 15)')
    plt.xlabel('Language')
    plt.ylabel('Ratio of High Confidence Detections (Score > 0.9)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'language_high_conf.png'), dpi=300)
    
    # 4. Score distribution by language (boxplot)
    plt.figure(figsize=(14, 10))
    top_languages = lang_stats.head(10)['language'].tolist()
    sns.boxplot(x='language', y='score', data=df[df['language'].isin(top_languages)])
    plt.title('Score Distribution by Language (Top 10)')
    plt.xlabel('Language')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'language_score_distribution.png'), dpi=300)
    
    print(f"Analysis completed. Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument('--results_path', type=str, required=True, help='Path to evaluation results')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save analysis')
    
    args = parser.parse_args()
    analyze_results(args.results_path, args.output_path)

if __name__ == "__main__":
    main()