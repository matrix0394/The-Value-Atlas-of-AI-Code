#!/usr/bin/env python3
"""
Generate supplementary Fig. S2:
Model imitation accuracy analysis.

Output: ``Supplementary Materials/figures/FigS2/``

Figures:
- S2A: Consistency distribution by model (baseline vs roleplay)
- S2B: Response distribution by question
- S2C: Imitation shift by model
- S2D: Imitation country shift summary (top/bottom 15)
- S2E–S2G: Extended model analyses (size, open/closed, vendor/origin)

Requirements: 4.1, 5.1, 5.2
"""

from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'Supplementary Materials' / 'data'

OUTPUT_DIR = PROJECT_ROOT / 'Supplementary Materials' / 'figures' / 'FigS2'


def set_publication_style():
    """Apply lightweight publication defaults without external helpers."""
    sns.set_theme(style='whitegrid')
    plt.rcParams.update(
        {
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'font.size': 11,
        }
    )


def save_figure(fig, output_path: Path, close: bool = True):
    """Save figure as both PNG and PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    if close:
        plt.close(fig)


def _load_publication_csv(candidates, rename_map):
    for candidate in candidates:
        if candidate.exists():
            df = pd.read_csv(candidate)
            return df.rename(columns=rename_map)
    tried = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(f'Could not find required data file. Tried: {tried}')


def load_ivs_data() -> pd.DataFrame:
    """Load IVS benchmark coordinates from the publication-ready data package."""
    df = _load_publication_csv(
        [
            DATA_DIR / 'DataS1_ivs_pca_coordinates.csv',
            DATA_DIR / 'ivs_pca_coordinates.csv',
        ],
        {
            'Country': 'country',
            'Cultural_Region': 'cultural_region',
            'PC1_Self_Expression': 'PC1',
            'PC2_Secular_Rational': 'PC2',
        },
    )
    return df.dropna(subset=['country']).drop_duplicates(subset=['country'])


def load_baseline_data() -> pd.DataFrame:
    """Load baseline LLM coordinates from the publication-ready data package."""
    df = _load_publication_csv(
        [
            DATA_DIR / 'DataS2_llm_baseline_pca.csv',
            DATA_DIR / 'llm_baseline_pca.csv',
        ],
        {
            'Model': 'model',
            'Language': 'language',
            'PC1_Self_Expression': 'PC1',
            'PC2_Secular_Rational': 'PC2',
        },
    )
    if 'model_name' not in df.columns and 'model' in df.columns:
        df['model_name'] = df['model']
    return df


def load_roleplay_data() -> pd.DataFrame:
    """Load multilingual roleplay coordinates from the publication-ready data package."""
    df = _load_publication_csv(
        [
            DATA_DIR / 'DataS3_llm_roleplay_pca.csv',
            DATA_DIR / 'llm_roleplay_pca.csv',
        ],
        {
            'Model': 'model',
            'Country': 'country',
            'Language': 'language',
            'PC1_Self_Expression': 'PC1',
            'PC2_Secular_Rational': 'PC2',
            'Cultural_Region': 'cultural_region',
        },
    )
    if 'model_name' not in df.columns and 'model' in df.columns:
        df['model_name'] = df['model']
    return df


def load_model_imitation_accuracy() -> pd.DataFrame:
    """Load the frozen model-level imitation summary used in the paper."""
    path = DATA_DIR / 'model_imitation_accuracy.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing frozen imitation summary: {path}')
    return pd.read_csv(path)


def load_study5_summary() -> dict:
    """Load the frozen Study 5 statistics used in the paper."""
    path = DATA_DIR / 'study5_model_imitation.json'
    if not path.exists():
        raise FileNotFoundError(f'Missing frozen Study 5 summary: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# =============================================================================
# Model Metadata Definitions
# =============================================================================

# Model size categories (Small vs Large)
# Small: models with <20B parameters or lightweight variants
# Large: flagship models with >20B parameters
SMALL_MODELS = [
    'llama-3.2-3b-instruct',
    'phi-3-mini-128k-instruct',
    'gemma-3-4b-it',
    'gpt-4o-mini',
    'mistral-nemo',
]

LARGE_MODELS = [
    'gpt-4o',
    'gpt-5.1',
    'claude-3-7-sonnet-20250219',
    'claude-sonnet-4.5',
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'gemini-3-pro-preview',
    'mistral-medium-3.1',
    'deepseek-chat',
    'deepseek-chat-v3.1',
    'llama-3.3-70b-instruct',
    'qwen3-max',
    'kimi-k2',
    'doubao-1-5-pro-32k-250115',
    'grok-4.1-fast',
]

# Open source vs closed source
OPEN_SOURCE_MODELS = [
    'llama-3.2-3b-instruct',
    'llama-3.3-70b-instruct',
    'mistral-nemo',
    'phi-3-mini-128k-instruct',
    'gemma-3-4b-it',
    'deepseek-chat',
    'deepseek-chat-v3.1',
    'qwen3-max',
    'kimi-k2',  # Moonshot Kimi K2 is open source
]

CLOSED_SOURCE_MODELS = [
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-5.1',
    'claude-3-7-sonnet-20250219',
    'claude-sonnet-4.5',
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'gemini-3-pro-preview',
    'mistral-medium-3.1',
    'doubao-1-5-pro-32k-250115',
    'grok-4.1-fast',
]

# Model vendors/families
MODEL_VENDORS = {
    'gpt-4o': 'OpenAI',
    'gpt-4o-mini': 'OpenAI',
    'gpt-5.1': 'OpenAI',
    'claude-3-7-sonnet-20250219': 'Anthropic',
    'claude-sonnet-4.5': 'Anthropic',
    'gemini-2.5-flash': 'Google',
    'gemini-2.5-pro': 'Google',
    'gemini-3-pro-preview': 'Google',
    'gemma-3-4b-it': 'Google',
    'llama-3.2-3b-instruct': 'Meta',
    'llama-3.3-70b-instruct': 'Meta',
    'mistral-medium-3.1': 'Mistral',
    'mistral-nemo': 'Mistral',
    'deepseek-chat': 'DeepSeek',
    'deepseek-chat-v3.1': 'DeepSeek',
    'qwen3-max': 'Alibaba',
    'doubao-1-5-pro-32k-250115': 'ByteDance',
    'kimi-k2': 'Moonshot',
    'grok-4.1-fast': 'xAI',
    'phi-3-mini-128k-instruct': 'Microsoft',
}

# Model origin countries
MODEL_ORIGINS = {
    'OpenAI': 'USA',
    'Anthropic': 'USA',
    'Google': 'USA',
    'Meta': 'USA',
    'Microsoft': 'USA',
    'xAI': 'USA',
    'Mistral': 'Europe',
    'DeepSeek': 'China',
    'Alibaba': 'China',
    'ByteDance': 'China',
    'Moonshot': 'China',
}

# =============================================================================
# Unified Color Palette for Fig. S2 figures (same as S5)
# =============================================================================
# Base colors - same as S5 for consistency
COLOR_RED = '#E74C3C'    # Red - bad/negative
COLOR_GREEN = '#27AE60'  # Green - good/positive
COLOR_BLUE = '#3498DB'   # Blue - neutral/third category

# Two-color figures (S2A, S2B, S2C, S2D, S2F): Red + Green
# Three-color figures (S2E, S2G): Red + Green + Blue

# S2A: Baseline vs Roleplay
CONDITION_COLORS = {
    'Baseline': COLOR_BLUE,   # Blue
    'Roleplay': COLOR_RED,    # Red
}

# S2B, S2C, S2D: Good vs Bad performance
PERFORMANCE_COLORS = {
    'good': COLOR_GREEN,      # Green (good/low distance)
    'bad': COLOR_RED,         # Red (bad/high distance)
}

# S2E: Model size (2 categories)
SIZE_COLORS = {
    'Small': COLOR_RED,       # Red
    'Large': COLOR_GREEN,     # Green
}

# S2F: Open vs Closed source
SOURCE_COLORS = {
    'Open Source': COLOR_GREEN,      # Green
    'Closed Source': COLOR_RED,      # Red
}

# S2G: Model origin (3 categories)
ORIGIN_COLORS = {
    'USA': COLOR_BLUE,            # Blue
    'China': COLOR_RED,           # Red
    'Europe': COLOR_GREEN,        # Green
}


def get_model_size_category(model: str) -> str:
    """Get model size category (Small or Large)."""
    if model in SMALL_MODELS:
        return 'Small'
    elif model in LARGE_MODELS:
        return 'Large'
    return 'Unknown'


def is_open_source(model: str) -> bool:
    """Check if model is open source."""
    return model in OPEN_SOURCE_MODELS


def get_model_vendor(model: str) -> str:
    """Get model vendor."""
    return MODEL_VENDORS.get(model, 'Unknown')


def get_model_origin(model: str) -> str:
    """Get model origin country."""
    vendor = get_model_vendor(model)
    return MODEL_ORIGINS.get(vendor, 'Unknown')


set_publication_style()


def load_and_preprocess_data():
    """Load baseline and roleplay PCA data from the publication-ready package."""
    baseline_df = load_baseline_data()
    roleplay_df = load_roleplay_data()
    ivs_df = load_ivs_data()
    
    # Normalize model names
    if 'model_name' in baseline_df.columns:
        baseline_df['model'] = baseline_df['model_name'].apply(lambda x: x.split('/')[-1])
    if 'model_name' in roleplay_df.columns:
        roleplay_df['model'] = roleplay_df['model_name'].apply(lambda x: x.split('/')[-1])
    
    # Add language_normalized for merging (handle en-native -> en)
    if 'language' in roleplay_df.columns:
        roleplay_df['language_normalized'] = roleplay_df['language'].replace({'en-native': 'en'})
    
    print(f"Loaded baseline: {len(baseline_df)} rows, {baseline_df['model'].nunique()} models")
    print(f"Loaded roleplay: {len(roleplay_df)} rows, {roleplay_df['model'].nunique()} models")
    print(f"Loaded IVS: {len(ivs_df)} rows")
    
    return baseline_df, roleplay_df, ivs_df


def generate_s2a_consistency_distribution(baseline_df, roleplay_df, output_dir):
    """Generate Fig. S2A: Consistency distribution by model."""
    print("Rendering Fig. S2A: model consistency distribution...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate PC1 standard deviation as consistency metric (lower = more consistent)
    baseline_consistency = baseline_df.groupby('model')['PC1'].std().reset_index()
    baseline_consistency['condition'] = 'Baseline'
    baseline_consistency.columns = ['model', 'consistency', 'condition']
    
    roleplay_consistency = roleplay_df.groupby('model')['PC1'].std().reset_index()
    roleplay_consistency['condition'] = 'Roleplay'
    roleplay_consistency.columns = ['model', 'consistency', 'condition']
    
    combined = pd.concat([baseline_consistency, roleplay_consistency])
    
    # Create boxplot
    models = sorted(combined['model'].unique())
    x = np.arange(len(models))
    width = 0.35
    
    baseline_vals = [baseline_consistency[baseline_consistency['model'] == m]['consistency'].values[0] 
                     if m in baseline_consistency['model'].values else 0 for m in models]
    roleplay_vals = [roleplay_consistency[roleplay_consistency['model'] == m]['consistency'].values[0] 
                     if m in roleplay_consistency['model'].values else 0 for m in models]
    
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', color=CONDITION_COLORS['Baseline'], alpha=0.8)
    ax.bar(x + width/2, roleplay_vals, width, label='Roleplay', color=CONDITION_COLORS['Roleplay'], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('PC1 Standard Deviation (lower = more consistent)', fontsize=11)
    ax.set_xlabel('Model', fontsize=11)
    ax.set_title('Model Response Consistency (Baseline vs Roleplay)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'FigS2A_consistency_distribution')
    print("  Saved: FigS2A_consistency_distribution")


def generate_s2b_response_distribution(baseline_df, roleplay_df, output_dir):
    """Generate Fig. S2B: Model flexibility analysis - deviation from baseline."""
    print("Rendering Fig. S2B: model flexibility analysis...")
    
    # Calculate baseline centroid for each model
    baseline_centroids = baseline_df.groupby('model')[['PC1', 'PC2']].mean().reset_index()
    baseline_centroids.columns = ['model', 'baseline_PC1', 'baseline_PC2']
    
    # Merge roleplay with baseline centroids
    merged = roleplay_df.merge(baseline_centroids, on='model', how='left')
    
    # Calculate deviation from baseline
    merged['deviation'] = np.sqrt(
        (merged['PC1'] - merged['baseline_PC1'])**2 + 
        (merged['PC2'] - merged['baseline_PC2'])**2
    )
    
    # Group by model
    model_flexibility = merged.groupby('model')['deviation'].agg(['mean', 'std']).reset_index()
    model_flexibility.columns = ['model', 'flexibility', 'flexibility_std']
    model_flexibility = model_flexibility.sort_values('flexibility', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by flexibility level using unified palette
    median_flex = model_flexibility['flexibility'].median()
    colors = [PERFORMANCE_COLORS['bad'] if f > median_flex else PERFORMANCE_COLORS['good'] 
              for f in model_flexibility['flexibility']]
    
    bars = ax.barh(range(len(model_flexibility)), model_flexibility['flexibility'],
                   xerr=model_flexibility['flexibility_std'], color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5, capsize=3)
    
    ax.set_yticks(range(len(model_flexibility)))
    ax.set_yticklabels(model_flexibility['model'], fontsize=9)
    ax.set_xlabel('Mean Deviation from Baseline (higher = more flexible)', fontsize=11)
    ax.set_title('Model Roleplay Flexibility',
                fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (_, row) in enumerate(model_flexibility.iterrows()):
        ax.text(row['flexibility'] + 0.05, i, f'{row["flexibility"]:.2f}', va='center', fontsize=8)
    
    # Add interpretation text
    ax.text(0.98, 0.02, 'Conservative: stays close to baseline\nFlexible: adapts to roleplay context',
           transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'FigS2B_model_flexibility')
    print("  Saved: FigS2B_model_flexibility")


def generate_s2c_imitation_shift_by_model(roleplay_df, ivs_df, output_dir):
    """Generate Fig. S2C: Imitation shift by model (best performers on top)."""
    print("Rendering Fig. S2C: imitation accuracy by model...")
    
    # Merge roleplay with IVS to calculate distance
    merged = roleplay_df.merge(
        ivs_df[['country', 'PC1', 'PC2']].rename(columns={'PC1': 'ivs_PC1', 'PC2': 'ivs_PC2'}),
        on='country',
        how='left'
    )
    
    # Calculate cultural distance
    merged['distance'] = np.sqrt(
        (merged['PC1'] - merged['ivs_PC1'])**2 + 
        (merged['PC2'] - merged['ivs_PC2'])**2
    )
    
    # Group by model - sort descending so best (lowest) appears at top
    model_distances = merged.groupby('model')['distance'].agg(['mean', 'std']).reset_index()
    model_distances = model_distances.sort_values('mean', ascending=False)  # Descending for barh
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by performance tier
    median_dist = model_distances['mean'].median()
    colors = [PERFORMANCE_COLORS['good'] if d < median_dist else PERFORMANCE_COLORS['bad'] 
              for d in model_distances['mean']]
    
    bars = ax.barh(range(len(model_distances)), model_distances['mean'], 
                   xerr=model_distances['std'], color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5, capsize=3)
    
    ax.set_yticks(range(len(model_distances)))
    ax.set_yticklabels(model_distances['model'], fontsize=9)
    ax.set_xlabel('Mean Cultural Distance (lower = better imitation)', fontsize=11)
    ax.set_title('A  Cultural Imitation Accuracy by Model',
                fontsize=12, fontweight='bold')
    ax.axvline(median_dist, color='gray', linestyle='--', 
               alpha=0.7, label=f'Median ({median_dist:.2f})')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (_, row) in enumerate(model_distances.iterrows()):
        ax.text(row['mean'] + 0.05, i, f'{row["mean"]:.2f}', va='center', fontsize=8)
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'FigS2C_imitation_accuracy_ranking')
    print("  Saved: FigS2C_imitation_accuracy_ranking")


def generate_s2d_country_shift_summary(roleplay_df, ivs_df, output_dir):
    """Generate Fig. S2D: Country shift summary (top/bottom 15)."""
    print("Rendering Fig. S2D: country imitation ranking...")
    
    # Merge roleplay with IVS
    merged = roleplay_df.merge(
        ivs_df[['country', 'PC1', 'PC2']].rename(columns={'PC1': 'ivs_PC1', 'PC2': 'ivs_PC2'}),
        on='country',
        how='left'
    )
    
    # Calculate cultural distance
    merged['distance'] = np.sqrt(
        (merged['PC1'] - merged['ivs_PC1'])**2 + 
        (merged['PC2'] - merged['ivs_PC2'])**2
    )
    
    # Group by country
    country_distances = merged.groupby('country')['distance'].mean().sort_values()
    
    # Get top 15 (best) and bottom 15 (worst)
    top15 = country_distances.head(15)
    bottom15 = country_distances.tail(15)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # Top 15 (best imitation) - use blue (good)
    ax1 = axes[0]
    ax1.barh(range(15), top15.values, color=PERFORMANCE_COLORS['good'], edgecolor='black', linewidth=0.5, alpha=0.8)
    ax1.set_yticks(range(15))
    ax1.set_yticklabels(top15.index, fontsize=9)
    ax1.set_xlabel('Mean Cultural Distance', fontsize=11)
    ax1.set_title('Top 15: Best Imitated Countries\n(lowest distance)', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Bottom 15 (worst imitation) - use red (bad)
    ax2 = axes[1]
    ax2.barh(range(15), bottom15.values, color=PERFORMANCE_COLORS['bad'], edgecolor='black', linewidth=0.5, alpha=0.8)
    ax2.set_yticks(range(15))
    ax2.set_yticklabels(bottom15.index, fontsize=9)
    ax2.set_xlabel('Mean Cultural Distance', fontsize=11)
    ax2.set_title('Bottom 15: Worst Imitated Countries\n(highest distance)', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    fig.suptitle('Country Imitation Ranking', fontsize=13, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.95, wspace=0.3)
    save_figure(fig, output_dir / 'FigS2D_country_imitation_ranking')
    print("  Saved: FigS2D_country_imitation_ranking")


def generate_s2e_model_size_analysis(roleplay_df, ivs_df, output_dir):
    """Generate Fig. S2E: Model size vs accuracy - Small vs Large comparison."""
    print("Rendering Fig. S2E: model size analysis...")
    from scipy import stats
    
    # Merge roleplay with IVS
    merged = roleplay_df.merge(
        ivs_df[['country', 'PC1', 'PC2']].rename(columns={'PC1': 'ivs_PC1', 'PC2': 'ivs_PC2'}),
        on='country',
        how='left'
    )
    
    # Calculate cultural distance
    merged['distance'] = np.sqrt(
        (merged['PC1'] - merged['ivs_PC1'])**2 + 
        (merged['PC2'] - merged['ivs_PC2'])**2
    )
    
    # Add model size category
    merged['size_category'] = merged['model'].apply(get_model_size_category)
    
    # Filter out models with unknown size
    merged = merged[merged['size_category'] != 'Unknown']
    
    # Get model-level statistics
    model_stats = merged.groupby('model').agg({
        'distance': 'mean',
        'size_category': 'first'
    }).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Bar chart showing mean by size category with error bars
    ax1 = axes[0]
    size_order = ['Small', 'Large']
    size_colors = SIZE_COLORS
    
    # Calculate statistics for each category
    small_data = model_stats[model_stats['size_category'] == 'Small']['distance']
    large_data = model_stats[model_stats['size_category'] == 'Large']['distance']
    
    category_stats = pd.DataFrame({
        'category': ['Small', 'Large'],
        'mean': [small_data.mean(), large_data.mean()],
        'std': [small_data.std(), large_data.std()],
        'n': [len(small_data), len(large_data)]
    })
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(small_data, large_data)
    
    # Create bar chart
    bars = ax1.bar(range(2), category_stats['mean'], 
                   yerr=category_stats['std'], capsize=10,
                   color=[size_colors['Small'], size_colors['Large']],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_xticks(range(2))
    ax1.set_xticklabels(['Small Models', 'Large Models'], fontsize=11)
    ax1.set_ylabel('Mean Cultural Distance', fontsize=11)
    ax1.set_title(f'Performance by Size (t-test p={p_value:.3f})', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels inside bars
    for i, (bar, row) in enumerate(zip(bars, category_stats.itertuples())):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                f'μ={row.mean:.2f}\nn={row.n}', ha='center', va='center', fontsize=10,
                color='white', fontweight='bold')
    
    # Right: Horizontal bar chart showing all models sorted by distance, colored by size
    ax2 = axes[1]
    
    model_stats_sorted = model_stats.sort_values('distance', ascending=True)
    colors = [size_colors[cat] for cat in model_stats_sorted['size_category']]
    
    bars = ax2.barh(range(len(model_stats_sorted)), model_stats_sorted['distance'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_yticks(range(len(model_stats_sorted)))
    ax2.set_yticklabels(model_stats_sorted['model'], fontsize=8)
    ax2.set_xlabel('Mean Cultural Distance', fontsize=11)
    ax2.set_title('All Models Ranked (colored by size)', fontsize=11, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=size_colors['Small'], alpha=0.7, label='Small'),
                      Patch(facecolor=size_colors['Large'], alpha=0.7, label='Large')]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    fig.suptitle('B  Model Size vs. Imitation Accuracy', fontsize=13, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.95, wspace=0.3)
    save_figure(fig, output_dir / 'FigS2E_model_size_analysis')
    print("  Saved: FigS2E_model_size_analysis")


def generate_s2f_open_vs_closed(roleplay_df, ivs_df, output_dir):
    """Generate Fig. S2F: Open vs closed source comparison - clean bar chart."""
    print("Rendering Fig. S2F: open-vs-closed comparison...")
    from scipy import stats
    
    # Merge roleplay with IVS
    merged = roleplay_df.merge(
        ivs_df[['country', 'PC1', 'PC2']].rename(columns={'PC1': 'ivs_PC1', 'PC2': 'ivs_PC2'}),
        on='country',
        how='left'
    )
    
    # Calculate cultural distance
    merged['distance'] = np.sqrt(
        (merged['PC1'] - merged['ivs_PC1'])**2 + 
        (merged['PC2'] - merged['ivs_PC2'])**2
    )
    
    # Add source type
    def get_source_type(model):
        if model in OPEN_SOURCE_MODELS:
            return 'Open Source'
        elif model in CLOSED_SOURCE_MODELS:
            return 'Closed Source'
        return 'Unknown'
    
    merged['source_type'] = merged['model'].apply(get_source_type)
    merged = merged[merged['source_type'] != 'Unknown']
    
    # Get model-level statistics
    model_stats = merged.groupby(['model', 'source_type'])['distance'].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Bar chart comparing Open vs Closed with error bars
    ax1 = axes[0]
    source_colors = SOURCE_COLORS
    
    # Calculate statistics
    open_data = model_stats[model_stats['source_type'] == 'Open Source']['distance']
    closed_data = model_stats[model_stats['source_type'] == 'Closed Source']['distance']
    
    source_stats = pd.DataFrame({
        'source': ['Open Source', 'Closed Source'],
        'mean': [open_data.mean(), closed_data.mean()],
        'std': [open_data.std(), closed_data.std()],
        'n': [len(open_data), len(closed_data)]
    })
    
    # Statistical tests
    t_stat, p_value_t = stats.ttest_ind(open_data, closed_data)
    u_stat, p_value_mw = stats.mannwhitneyu(open_data, closed_data, alternative='two-sided')
    pooled_std = np.sqrt((open_data.std()**2 + closed_data.std()**2) / 2)
    cohens_d = (open_data.mean() - closed_data.mean()) / pooled_std if pooled_std > 0 else 0
    
    # Create bar chart
    bars = ax1.bar(range(2), source_stats['mean'], yerr=source_stats['std'], capsize=10,
                   color=[source_colors['Open Source'], source_colors['Closed Source']],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_xticks(range(2))
    ax1.set_xticklabels(['Open Source', 'Closed Source'], fontsize=11)
    ax1.set_ylabel('Mean Cultural Distance', fontsize=11)
    ax1.set_title('Performance Comparison', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels inside bars
    for i, (bar, row) in enumerate(zip(bars, source_stats.itertuples())):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                f'μ={row.mean:.2f}\nn={row.n}', ha='center', va='center', fontsize=10,
                color='white', fontweight='bold')
    
    # Add significance bracket - position relative to axis limits
    y_max = max(source_stats['mean'] + source_stats['std']) + 0.3
    ax1.set_ylim(0, y_max + 0.5)
    ax1.plot([0, 0, 1, 1], [y_max, y_max + 0.08, y_max + 0.08, y_max], 'k-', linewidth=1.5)
    sig_text = 'n.s.' if p_value_mw > 0.05 else ('*' if p_value_mw > 0.01 else '**')
    ax1.text(0.5, y_max + 0.12, sig_text, ha='center', fontsize=14, fontweight='bold')
    
    # Add statistics box
    stats_text = (f"Mann-Whitney p={p_value_mw:.3f}\n"
                  f"Cohen's d={cohens_d:.3f}")
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Right: Horizontal bar chart showing all models sorted by distance, colored by source
    ax2 = axes[1]
    
    model_stats_sorted = model_stats.sort_values('distance', ascending=True)
    colors = [source_colors[s] for s in model_stats_sorted['source_type']]
    
    bars = ax2.barh(range(len(model_stats_sorted)), model_stats_sorted['distance'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_yticks(range(len(model_stats_sorted)))
    ax2.set_yticklabels(model_stats_sorted['model'], fontsize=8)
    ax2.set_xlabel('Mean Cultural Distance', fontsize=11)
    ax2.set_title('All Models Ranked (colored by source)', fontsize=11, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=SOURCE_COLORS['Open Source'], alpha=0.7, label='Open Source'),
                      Patch(facecolor=SOURCE_COLORS['Closed Source'], alpha=0.7, label='Closed Source')]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    fig.suptitle('C  Open-Source vs. Closed-Source Comparison', fontsize=13, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.95, wspace=0.3)
    save_figure(fig, output_dir / 'FigS2F_open_vs_closed_source')
    print("  Saved: FigS2F_open_vs_closed_source")


def generate_s2g_vendor_analysis(roleplay_df, ivs_df, output_dir):
    """Generate Fig. S2G: Model origin (USA/China/Europe) comparison - clean bar chart."""
    print("Rendering Fig. S2G: model origin analysis...")
    from scipy import stats
    
    # Merge roleplay with IVS
    merged = roleplay_df.merge(
        ivs_df[['country', 'PC1', 'PC2']].rename(columns={'PC1': 'ivs_PC1', 'PC2': 'ivs_PC2'}),
        on='country',
        how='left'
    )
    
    # Calculate cultural distance
    merged['distance'] = np.sqrt(
        (merged['PC1'] - merged['ivs_PC1'])**2 + 
        (merged['PC2'] - merged['ivs_PC2'])**2
    )
    
    # Add vendor and origin
    merged['vendor'] = merged['model'].apply(get_model_vendor)
    merged['origin'] = merged['model'].apply(get_model_origin)
    merged = merged[merged['origin'] != 'Unknown']
    
    # Get model-level statistics
    model_stats = merged.groupby(['model', 'vendor', 'origin'])['distance'].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left: Bar chart by origin country with error bars
    ax1 = axes[0]
    origin_colors = ORIGIN_COLORS
    origins = ['USA', 'China', 'Europe']
    
    # Calculate statistics for each origin
    origin_stats = []
    for origin in origins:
        origin_data = model_stats[model_stats['origin'] == origin]['distance']
        if len(origin_data) > 0:
            origin_stats.append({
                'origin': origin,
                'mean': origin_data.mean(),
                'std': origin_data.std(),
                'n': len(origin_data)
            })
    origin_df = pd.DataFrame(origin_stats)
    
    # ANOVA test
    usa_data = model_stats[model_stats['origin'] == 'USA']['distance']
    china_data = model_stats[model_stats['origin'] == 'China']['distance']
    europe_data = model_stats[model_stats['origin'] == 'Europe']['distance']
    
    groups = [g for g in [usa_data, china_data, europe_data] if len(g) > 0]
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
    else:
        f_stat, p_value = 0, 1
    
    # Create bar chart
    bars = ax1.bar(range(len(origin_df)), origin_df['mean'], yerr=origin_df['std'], capsize=8,
                   color=[origin_colors[o] for o in origin_df['origin']],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_xticks(range(len(origin_df)))
    ax1.set_xticklabels(origin_df['origin'], fontsize=11)
    ax1.set_ylabel('Mean Cultural Distance', fontsize=11)
    ax1.set_xlabel('Model Origin', fontsize=11)
    ax1.set_title(f'Performance by Origin (ANOVA p={p_value:.3f})', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels inside bars
    for i, (bar, row) in enumerate(zip(bars, origin_df.itertuples())):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                f'μ={row.mean:.2f}\nn={row.n}', ha='center', va='center', fontsize=9,
                color='white', fontweight='bold')
    
    # Right: Horizontal bar chart by vendor, colored by origin
    ax2 = axes[1]
    
    # Sort vendors by mean distance
    vendor_stats = model_stats.groupby(['vendor', 'origin']).agg({
        'distance': ['mean', 'std', 'count']
    }).reset_index()
    vendor_stats.columns = ['vendor', 'origin', 'mean', 'std', 'count']
    vendor_stats = vendor_stats.sort_values('mean', ascending=True)
    
    # Fill NaN std with 0
    vendor_stats['std'] = vendor_stats['std'].fillna(0)
    
    # Create bar chart
    bars = ax2.barh(range(len(vendor_stats)), vendor_stats['mean'],
                   xerr=vendor_stats['std'],
                   color=[origin_colors[o] for o in vendor_stats['origin']],
                   alpha=0.8, edgecolor='black', linewidth=0.5, capsize=3)
    
    # Add vendor labels with model count
    labels = [f"{row['vendor']} (n={int(row['count'])})" for _, row in vendor_stats.iterrows()]
    ax2.set_yticks(range(len(vendor_stats)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel('Mean Cultural Distance', fontsize=11)
    ax2.set_title('Performance by Vendor (ranked)', fontsize=11, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (_, row) in enumerate(vendor_stats.iterrows()):
        ax2.text(row['mean'] + row['std'] + 0.05, i, f'{row["mean"]:.2f}', va='center', fontsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=origin_colors[o], alpha=0.7, label=o) for o in origins]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=9, title='Origin')
    
    fig.suptitle('D  Model Origin and Vendor Analysis', fontsize=13, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.88, bottom=0.10, left=0.08, right=0.95, wspace=0.3)
    save_figure(fig, output_dir / 'FigS2G_vendor_origin_analysis')
    print("  Saved: FigS2G_vendor_origin_analysis")


def _compute_model_distances(roleplay_df, ivs_df):
    """Shared helper: merge roleplay with IVS and compute per-model mean distances."""
    merged = roleplay_df.merge(
        ivs_df[['country', 'PC1', 'PC2']].rename(columns={'PC1': 'ivs_PC1', 'PC2': 'ivs_PC2'}),
        on='country', how='left',
    )
    merged['distance'] = np.sqrt(
        (merged['PC1'] - merged['ivs_PC1'])**2 +
        (merged['PC2'] - merged['ivs_PC2'])**2
    )
    return merged


def generate_s2_combined(accuracy_df, study5_summary, output_dir):
    """Generate the frozen Fig. S2 combined figure from publication-ready tables."""
    from scipy import stats
    from matplotlib.gridspec import GridSpec

    print("Rendering the combined Fig. S2 figure...")
    model_distances = accuracy_df.rename(
        columns={'model_name': 'model', 'mean_distance': 'mean', 'std_distance': 'std'}
    ).copy()

    # ---- figure and grid -------------------------------------------------
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.1, 1],
                  hspace=0.38, wspace=0.30)

    # =====================================================================
    # Panel A – Model ranking (left column, spans all 3 rows)
    # =====================================================================
    ax_a = fig.add_subplot(gs[:, 0])
    md = model_distances.sort_values('mean', ascending=True).reset_index(drop=True)
    median_dist = md['mean'].median()
    colors_a = [PERFORMANCE_COLORS['good'] if d < median_dist else PERFORMANCE_COLORS['bad']
                for d in md['mean']]
    ax_a.barh(range(len(md)), md['mean'], xerr=md['std'], color=colors_a,
              alpha=0.8, edgecolor='black', linewidth=0.5, capsize=3)
    ax_a.set_yticks(range(len(md)))
    ax_a.set_yticklabels(md['model'], fontsize=9)
    ax_a.set_xlabel('Mean Cultural Distance (lower = better imitation)', fontsize=11)
    ax_a.set_title('A  Cultural Imitation Accuracy by Model', fontsize=12, fontweight='bold')
    ax_a.axvline(median_dist, color='gray', linestyle='--', alpha=0.7,
                 label=f'Median ({median_dist:.2f})')
    ax_a.legend(loc='lower right', fontsize=9)
    ax_a.grid(axis='x', alpha=0.3)
    ax_a.invert_yaxis()
    for i, (_, row) in enumerate(md.iterrows()):
        ax_a.text(row['mean'] + 0.05, i, f'{row["mean"]:.2f}', va='center', fontsize=8)

    # =====================================================================
    # Panel B – Size comparison (right-top)
    # =====================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    model_distances_full = accuracy_df.copy()
    small_d = model_distances_full.loc[model_distances_full['size_category'] == 'Small', 'mean_distance']
    large_d = model_distances_full.loc[model_distances_full['size_category'] == 'Large', 'mean_distance']
    t_stat_b, p_b = stats.ttest_ind(small_d, large_d)

    for idx, (cat, data, col) in enumerate([
        ('Small', small_d, SIZE_COLORS['Small']),
        ('Large', large_d, SIZE_COLORS['Large']),
    ]):
        bar = ax_b.bar(idx, data.mean(), yerr=data.std(), capsize=8,
                       color=col, alpha=0.8, edgecolor='black', linewidth=1.2)
        ax_b.text(idx, data.mean() * 0.5, f'\u03bc={data.mean():.2f}\nn={len(data)}',
                  ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(['Small Models', 'Large Models'], fontsize=10)
    ax_b.set_ylabel('Mean Cultural Distance', fontsize=10)
    ax_b.set_title(f'B  Model Size (t-test p={p_b:.3f})', fontsize=11, fontweight='bold')
    ax_b.grid(axis='y', alpha=0.3)

    # =====================================================================
    # Panel C – Open / Closed (right-middle)
    # =====================================================================
    ax_c = fig.add_subplot(gs[1, 1])
    open_closed = study5_summary['open_vs_closed']
    open_d = accuracy_df.loc[accuracy_df['open_source'] == True, 'mean_distance']
    closed_d = accuracy_df.loc[accuracy_df['open_source'] == False, 'mean_distance']
    panel_c_rows = [
        ('Open Source', open_closed['open_mean'], open_d.std(), int(open_closed['n_open']), SOURCE_COLORS['Open Source']),
        ('Closed Source', open_closed['closed_mean'], closed_d.std(), int(open_closed['n_closed']), SOURCE_COLORS['Closed Source']),
    ]

    for idx, (label, mean_val, std_val, n_val, col) in enumerate(panel_c_rows):
        ax_c.bar(idx, mean_val, yerr=std_val, capsize=8,
                 color=col, alpha=0.8, edgecolor='black', linewidth=1.2)
        ax_c.text(idx, mean_val * 0.5, f'\u03bc={mean_val:.2f}\nn={n_val}',
                  ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(['Open Source', 'Closed Source'], fontsize=10)
    ax_c.set_ylabel('Mean Cultural Distance', fontsize=10)
    p_mw = float(open_closed['p'])
    sig = 'n.s.' if p_mw > 0.05 else ('*' if p_mw > 0.01 else '**')
    ax_c.set_title(f'C  Open vs. Closed Source (MW p={p_mw:.3f}, {sig})',
                   fontsize=11, fontweight='bold')
    ax_c.grid(axis='y', alpha=0.3)

    # =====================================================================
    # Panel D – Origin & Vendor (right-bottom)
    # =====================================================================
    ax_d = fig.add_subplot(gs[2, 1])
    origin_df = accuracy_df.copy()
    origin_df['origin_plot'] = origin_df['origin'].replace({'US': 'USA'})
    origins = ['USA', 'China', 'Europe']
    origin_data = {o: origin_df.loc[origin_df['origin_plot'] == o, 'mean_distance'] for o in origins}
    groups = [v for v in origin_data.values() if len(v) > 0]
    _, p_anova = stats.f_oneway(*groups) if len(groups) >= 2 else (0, 1)

    origin_summary = {
        'USA': study5_summary['origin_comparison']['US'],
        'China': study5_summary['origin_comparison']['China'],
        'Europe': study5_summary['origin_comparison']['Europe'],
    }
    for idx, o in enumerate(origins):
        d = origin_data[o]
        if len(d) == 0:
            continue
        mean_val = float(origin_summary[o]['mean'])
        n_val = int(origin_summary[o]['n'])
        ax_d.bar(idx, mean_val, yerr=d.std(), capsize=8,
                 color=ORIGIN_COLORS[o], alpha=0.8, edgecolor='black', linewidth=1.2)
        ax_d.text(idx, mean_val * 0.5, f'\u03bc={mean_val:.2f}\nn={n_val}',
                  ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax_d.set_xticks(range(len(origins)))
    ax_d.set_xticklabels(origins, fontsize=10)
    ax_d.set_ylabel('Mean Cultural Distance', fontsize=10)
    ax_d.set_title(f'D  Model Origin (ANOVA p={p_anova:.3f})', fontsize=11, fontweight='bold')
    ax_d.grid(axis='y', alpha=0.3)

    # ---- save ------------------------------------------------------------
    save_figure(fig, output_dir / 'FigS2_model_imitation_analysis')
    print("  Saved: FigS2_model_imitation_analysis")


def main():
    """Main function to generate the frozen Fig. S2 combined figure."""
    print("=" * 60)
    print("Generating Fig. S2: Model imitation accuracy analysis")
    print("=" * 60)
    
    # Output directory
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_df = load_model_imitation_accuracy()
    study5_summary = load_study5_summary()
    generate_s2_combined(accuracy_df, study5_summary, output_dir)
    
    print("\n" + "=" * 60)
    print(f"Fig. S2 outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
