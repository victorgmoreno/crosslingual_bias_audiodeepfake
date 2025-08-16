"""
Visualization and reporting functions for MLAAD evaluation results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_statistics(results):
    """
    Generate statistics from evaluation results
    
    Args:
        results: List of evaluation results
    
    Returns:
        DataFrame with results, language stats, and architecture stats
    """
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Analysis by language
    lang_stats = df.groupby('language')['score'].agg(['count', 'mean', 'std']).reset_index()
    lang_stats['high_conf_ratio'] = df[df['score'] > 0.9].groupby('language').size() / lang_stats['count']
    lang_stats = lang_stats.sort_values('count', ascending=False)
    
    # Analysis by architecture
    arch_stats = df.groupby('architecture')['score'].agg(['count', 'mean', 'std']).reset_index()
    arch_stats['high_conf_ratio'] = df[df['score'] > 0.9].groupby('architecture').size() / arch_stats['count']
    arch_stats = arch_stats.sort_values('count', ascending=False)
    
    return df, lang_stats, arch_stats

def print_summary(df, lang_stats, arch_stats):
    """
    Print summary statistics
    
    Args:
        df: DataFrame with results
        lang_stats: Language statistics
        arch_stats: Architecture statistics
    """
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total files evaluated: {len(df)}")
    print(f"Average spoofing score: {df['score'].mean():.4f}")
    print(f"High confidence detections (score > 0.9): {(df['score'] > 0.9).sum()} ({(df['score'] > 0.9).sum() / len(df) * 100:.2f}%)")
    
    print("\n=== LANGUAGE STATISTICS (TOP 10) ===")
    print(lang_stats.head(10).to_string(index=False))
    
    print("\n=== ARCHITECTURE STATISTICS (TOP 10) ===")
    print(arch_stats.head(10).to_string(index=False))

def create_visualizations(df, lang_stats, arch_stats, output_dir):
    """
    Create visualizations from evaluation results
    
    Args:
        df: DataFrame with results
        lang_stats: Language statistics
        arch_stats: Architecture statistics
        output_dir: Directory to save visualizations
    """
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot by language
    plt.figure(figsize=(14, 8))
    sns.barplot(x='language', y='mean', data=lang_stats.head(15), yerr=lang_stats.head(15)['std'])
    plt.title('Average Spoofing Score by Language (Top 15)')
    plt.xlabel('Language')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'language_scores.png'), dpi=300)
    
    # Score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['score'], bins=50, kde=True)
    plt.title('Distribution of Spoofing Scores')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'score_distribution.png'), dpi=300)
    
    # Plot by architecture
    plt.figure(figsize=(14, 8))
    sns.barplot(x='architecture', y='mean', data=arch_stats.head(10), yerr=arch_stats.head(10)['std'])
    plt.title('Average Spoofing Score by Architecture (Top 10)')
    plt.xlabel('Architecture')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'architecture_scores.png'), dpi=300)
    
    # High confidence ratio by language
    plt.figure(figsize=(14, 8))
    sns.barplot(x='language', y='high_conf_ratio', data=lang_stats.head(15))
    plt.title('High Confidence Detection Ratio by Language (Top 15)')
    plt.xlabel('Language')
    plt.ylabel('Ratio of High Confidence Detections (Score > 0.9)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'language_high_conf.png'), dpi=300)
    
    print(f"Visualizations saved to {vis_dir}")

def save_statistics(lang_stats, arch_stats, output_dir):
    """
    Save statistics to CSV files
    
    Args:
        lang_stats: Language statistics
        arch_stats: Architecture statistics
        output_dir: Directory to save statistics
    """
    # Save to CSV
    lang_stats.to_csv(os.path.join(output_dir, 'language_stats.csv'), index=False)
    arch_stats.to_csv(os.path.join(output_dir, 'architecture_stats.csv'), index=False)
    
    print(f"Statistics saved to {output_dir}")