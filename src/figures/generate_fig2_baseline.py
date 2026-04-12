#!/usr/bin/env python3
"""
Generate main-paper Figure 2:
Baseline cultural value orientations of 20 LLMs on the Inglehart-Welzel map.

This script is self-contained and reads publication-ready data from
``Supplementary Materials/data``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'Supplementary Materials' / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'figures' / 'paper'
LANGUAGES_ORDER = ['ar', 'en', 'es', 'fr', 'ru', 'zh-cn']

LANGUAGE_NAMES = {
    'ar': 'Arabic',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'ru': 'Russian',
    'zh-cn': 'Chinese',
}

EXCLUDED_REGIONS = {'Other', 'Baltic'}

WVS_REGION_COLORS = {
    'Confucian': '#E8872E',
    'Protestant Europe': '#D4C826',
    'Latin America': '#18A48C',
    'Catholic Europe': '#2EAD56',
    'English-Speaking': '#C8C830',
    'African-Islamic': '#8C8C8C',
    'West & South Asia': '#C46A2C',
    'Orthodox Europe': '#4878A8',
    'Other': '#B0B0B0',
    'Baltic': '#88A0C0',
}

MODEL_DISPLAY = {
    'gpt-4o': {'short': 'GPT-4o', 'color': '#0D47A1'},
    'gpt-4o-mini': {'short': 'GPT-4o-mini', 'color': '#1976D2'},
    'gpt-5.1': {'short': 'GPT-5.1', 'color': '#42A5F5'},
    'claude-3-7-sonnet-20250219': {'short': 'Claude-3.7-Sonnet', 'color': '#B71C1C'},
    'claude-sonnet-4.5': {'short': 'Claude-Sonnet-4.5', 'color': '#E53935'},
    'gemini-2.5-flash': {'short': 'Gemini-2.5-Flash', 'color': '#1B5E20'},
    'gemini-2.5-pro': {'short': 'Gemini-2.5-Pro', 'color': '#2E7D32'},
    'gemini-3-pro-preview': {'short': 'Gemini-3-Pro', 'color': '#43A047'},
    'gemma-3-4b-it': {'short': 'Gemma-3-4B', 'color': '#66BB6A'},
    'llama-3.2-3b-instruct': {'short': 'LLaMA-3.2-3B', 'color': '#E65100'},
    'llama-3.3-70b-instruct': {'short': 'LLaMA-3.3-70B', 'color': '#FB8C00'},
    'mistral-medium-3.1': {'short': 'Mistral-Medium', 'color': '#6A1B9A'},
    'mistral-nemo': {'short': 'Mistral-Nemo', 'color': '#AB47BC'},
    'deepseek-chat': {'short': 'DeepSeek-V3', 'color': '#006064'},
    'deepseek-chat-v3.1': {'short': 'DeepSeek-V3.1', 'color': '#00ACC1'},
    'qwen3-max': {'short': 'Qwen3-Max', 'color': '#F9A825'},
    'doubao-1-5-pro-32k-250115': {'short': 'Doubao-1.5-Pro', 'color': '#AD1457'},
    'kimi-k2': {'short': 'Kimi-K2', 'color': '#795548'},
    'phi-3-mini-128k-instruct': {'short': 'Phi-3-Mini', 'color': '#558B2F'},
    'grok-4.1-fast': {'short': 'Grok-4.1', 'color': '#37474F'},
}

MODEL_TO_VENDOR = {
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
    'qwen3-max': 'Other',
    'doubao-1-5-pro-32k-250115': 'Other',
    'kimi-k2': 'Other',
    'phi-3-mini-128k-instruct': 'Other',
    'grok-4.1-fast': 'Other',
}

VENDOR_MARKERS = {
    'OpenAI': 'o',
    'Anthropic': 's',
    'Google': '^',
    'Meta': 'D',
    'Mistral': 'v',
    'DeepSeek': 'P',
    'Other': 'h',
}

plt.rcParams.update(
    {
        'font.family': 'Helvetica',
        'font.size': 9,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 0.6,
        'axes.facecolor': '#F8F8F8',
        'figure.facecolor': 'white',
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    }
)

try:
    from adjustText import adjust_text
    ADJUSTTEXT_AVAILABLE = True
except ImportError:
    ADJUSTTEXT_AVAILABLE = False


def save_figure(fig, output_path: Path, close: bool = True):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix('.png'), dpi=600, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    if close:
        plt.close(fig)


def _load_first_csv(candidates, rename_map):
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate).rename(columns=rename_map)
    tried = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(f'Could not find required data file. Tried: {tried}')


def load_ivs_data() -> pd.DataFrame:
    df = _load_first_csv(
        [
            DATA_DIR / 'DataS1_ivs_pca_coordinates.csv',
            DATA_DIR / 'ivs_pca_coordinates.csv',
        ],
        {
            'Country': 'country',
            'Cultural_Region': 'cultural_region',
            'Cultural Region': 'cultural_region',
            'PC1_Self_Expression': 'PC1',
            'PC2_Secular_Rational': 'PC2',
            'PC1_rescaled': 'PC1',
            'PC2_rescaled': 'PC2',
        },
    )
    return df.dropna(subset=['country']).drop_duplicates(subset=['country'])


def load_baseline_data() -> pd.DataFrame:
    df = _load_first_csv(
        [
            DATA_DIR / 'figure2_baseline_20models.csv',
            DATA_DIR / 'DataS2_llm_baseline_pca.csv',
        ],
        {
            'Model': 'model_name',
            'Language': 'language',
            'PC1_Self_Expression': 'PC1',
            'PC2_Secular_Rational': 'PC2',
        },
    )
    df = df[df['language'].isin(LANGUAGES_ORDER)].copy()
    df['model_name'] = df['model_name'].astype(str).apply(lambda x: x.split('/')[-1])
    return df


def get_global_axis_limits(ivs_data: pd.DataFrame, baseline_data: pd.DataFrame):
    x_vals = pd.concat([ivs_data['PC1'], baseline_data['PC1']], ignore_index=True)
    y_vals = pd.concat([ivs_data['PC2'], baseline_data['PC2']], ignore_index=True)
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    x_pad = max(0.5, (x_max - x_min) * 0.08)
    y_pad = max(0.5, (y_max - y_min) * 0.08)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def train_region_classifier(data: pd.DataFrame):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import SVC

    X = data[['PC1', 'PC2']].values
    y = data['cultural_region'].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)
    clf.fit(X, y_encoded)
    return clf, le


def plot_decision_boundaries(ax, data: pd.DataFrame, xlim, ylim):
    from scipy.interpolate import splprep, splev
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path as MplPath

    clf, le = train_region_classifier(data)
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 250),
        np.linspace(ylim[0], ylim[1], 250),
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    mask = np.zeros_like(Z, dtype=bool)

    for region in data['cultural_region'].unique():
        subset = data[data['cultural_region'] == region]
        points = subset[['PC1', 'PC2']].values
        if len(points) < 3:
            continue
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            center = points.mean(axis=0)
            expanded = center + (hull_points - center) * 1.6
            closed = np.vstack([expanded, expanded[0]])
            try:
                tck, _ = splprep(
                    [closed[:, 0], closed[:, 1]],
                    s=0.2,
                    per=True,
                    k=min(3, len(expanded) - 1),
                )
                u_new = np.linspace(0, 1, 100)
                smooth_x, smooth_y = splev(u_new, tck)
                expanded = np.column_stack([smooth_x, smooth_y])
            except Exception:
                pass
            path = MplPath(expanded)
            inside = path.contains_points(np.c_[xx.ravel(), yy.ravel()])
            mask = mask | inside.reshape(xx.shape)
        except Exception:
            continue

    unique_labels = np.unique(Z[mask])
    region_names = le.inverse_transform(unique_labels)
    colors = [WVS_REGION_COLORS.get(name, '#DDDDDD') for name in region_names]
    Z_masked = np.ma.masked_where(~mask, Z)
    ax.contourf(
        xx,
        yy,
        Z_masked,
        levels=np.arange(len(unique_labels) + 1) - 0.5,
        colors=colors,
        alpha=0.40,
    )


def plot_ivs_background(ax, ivs_data: pd.DataFrame, xlim, ylim):
    regions = sorted(
        [
            region
            for region in ivs_data['cultural_region'].dropna().unique()
            if region not in EXCLUDED_REGIONS
        ]
    )
    ml_data = ivs_data[ivs_data['cultural_region'].isin(regions)].copy()
    ml_data = ml_data[['PC1', 'PC2', 'cultural_region']].dropna()
    if not ml_data.empty:
        try:
            plot_decision_boundaries(ax, ml_data, xlim, ylim)
        except Exception:
            pass

    for region in ivs_data['cultural_region'].dropna().unique():
        subset = ivs_data[ivs_data['cultural_region'] == region]
        ax.scatter(
            subset['PC1'],
            subset['PC2'],
            c=WVS_REGION_COLORS.get(region, '#CCCCCC'),
            s=14,
            marker='o',
            alpha=0.40,
            edgecolors='none',
            zorder=1,
        )


def plot_panel(ax, language, baseline_data, ivs_data, xlim, ylim, panel_label):
    plot_ivs_background(ax, ivs_data, xlim, ylim)
    lang_data = baseline_data[baseline_data['language'] == language]

    texts = []
    for model in sorted(lang_data['model_name'].unique()):
        row = lang_data[lang_data['model_name'] == model].iloc[0]
        info = MODEL_DISPLAY.get(model, {'short': model[:8], 'color': '#555555'})
        vendor = MODEL_TO_VENDOR.get(model, 'Other')
        marker = VENDOR_MARKERS.get(vendor, 'o')
        ax.scatter(
            row['PC1'],
            row['PC2'],
            c=info['color'],
            s=55,
            marker=marker,
            alpha=0.95,
            edgecolors='white',
            linewidths=0.5,
            zorder=4,
        )
        texts.append((row['PC1'], row['PC2'], info['short'], info['color']))

    stroke = [pe.withStroke(linewidth=1.2, foreground='white')]
    if ADJUSTTEXT_AVAILABLE:
        txt_objs = []
        for x_pos, y_pos, label, color in texts:
            txt_objs.append(
                ax.text(
                    x_pos,
                    y_pos,
                    label,
                    fontsize=5.5,
                    color=color,
                    ha='center',
                    va='bottom',
                    zorder=5,
                    path_effects=stroke,
                )
            )
        adjust_text(
            txt_objs,
            ax=ax,
            arrowprops=dict(arrowstyle='-', color='#999999', lw=0.4, alpha=0.6),
            expand_points=(1.5, 1.5),
            expand_text=(1.2, 1.2),
            force_points=(0.4, 0.4),
            force_text=(0.4, 0.4),
            lim=600,
        )
    else:
        for x_pos, y_pos, label, color in texts:
            ax.annotate(
                label,
                (x_pos, y_pos),
                fontsize=5.5,
                color=color,
                ha='center',
                va='bottom',
                xytext=(0, 3),
                textcoords='offset points',
                zorder=5,
                path_effects=stroke,
            )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(y=0, color='#CCCCCC', linestyle='--', linewidth=0.4, zorder=0)
    ax.axvline(x=0, color='#CCCCCC', linestyle='--', linewidth=0.4, zorder=0)
    for spine in ax.spines.values():
        spine.set_color('#555555')
        spine.set_linewidth(0.6)

    ax.set_title(LANGUAGE_NAMES.get(language, language), fontsize=13, fontweight='bold', pad=6, color='#1A1A1A')
    ax.text(
        -0.05,
        1.08,
        panel_label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight='bold',
        va='top',
        ha='right',
        color='#000000',
    )


def make_model_legend(models):
    vendor_order = ['OpenAI', 'Anthropic', 'Google', 'Meta', 'Mistral', 'DeepSeek', 'Other']
    handles = []
    added = set()
    for vendor in vendor_order:
        for model in models:
            if MODEL_TO_VENDOR.get(model) != vendor or model in added:
                continue
            added.add(model)
            info = MODEL_DISPLAY.get(model, {'short': model[:10], 'color': '#555'})
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=VENDOR_MARKERS.get(vendor, 'o'),
                    color='none',
                    markerfacecolor=info['color'],
                    markersize=6,
                    markeredgecolor='white',
                    markeredgewidth=0.4,
                    label=info['short'],
                )
            )
    return handles


def main():
    print('=' * 60)
    print('Generating Figure 2: Baseline cultural value orientations of 20 LLMs')
    print('=' * 60)

    ivs_data = load_ivs_data()
    baseline_data = load_baseline_data()
    xlim, ylim = get_global_axis_limits(ivs_data, baseline_data)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8),
        sharex=True,
        sharey=True,
        gridspec_kw={'wspace': 0.08, 'hspace': 0.15},
    )
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, lang in enumerate(LANGUAGES_ORDER):
        plot_panel(axes.flatten()[i], lang, baseline_data, ivs_data, xlim, ylim, panel_labels[i])

    models = sorted(baseline_data['model_name'].unique())
    model_handles = make_model_legend(models)
    region_handles = [
        mpatches.Patch(color=color, label=region, alpha=0.45)
        for region, color in WVS_REGION_COLORS.items()
        if region not in {'Other', 'Baltic'}
    ]

    legend_font = {'family': 'Helvetica', 'size': 7.5}
    title_font = {'family': 'Helvetica', 'size': 9, 'weight': 'bold'}
    leg1 = fig.legend(
        handles=model_handles,
        loc='upper right',
        prop=legend_font,
        frameon=True,
        framealpha=0.95,
        edgecolor='#AAAAAA',
        title='LLM Models',
        title_fontproperties=title_font,
        bbox_to_anchor=(0.98, 0.95),
        labelspacing=0.4,
        handletextpad=0.4,
        borderpad=0.6,
    )
    leg2 = fig.legend(
        handles=region_handles,
        loc='upper right',
        prop=legend_font,
        frameon=True,
        framealpha=0.95,
        edgecolor='#AAAAAA',
        title='WVS Cultural Regions',
        title_fontproperties=title_font,
        bbox_to_anchor=(0.98, 0.38),
        labelspacing=0.4,
        handletextpad=0.4,
        borderpad=0.6,
    )
    fig.add_artist(leg1)
    fig.add_artist(leg2)

    fig.text(
        0.44,
        0.02,
        r'$\longleftarrow$ Survival Values          Self-Expression Values $\longrightarrow$',
        ha='center',
        fontsize=12,
        fontweight='bold',
        color='#1A1A1A',
    )
    fig.text(
        0.02,
        0.5,
        r'$\longleftarrow$ Traditional Values          Secular-Rational Values $\longrightarrow$',
        va='center',
        rotation='vertical',
        fontsize=12,
        fontweight='bold',
        color='#1A1A1A',
    )
    fig.subplots_adjust(left=0.06, right=0.84, bottom=0.08, top=0.94)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, OUTPUT_DIR / 'Fig2_baseline_intrinsic_values')
    print(f'Output directory: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
