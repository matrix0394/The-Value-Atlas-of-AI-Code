#!/usr/bin/env python3
"""
Generate supplementary Fig. S4:
English advantage distribution and geographic gradient (Study 3).

This script reads the frozen paper source tables directly from
``Supplementary Materials/data``.
"""

from pathlib import Path
import json

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'Supplementary Materials' / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'Supplementary Materials' / 'figures' / 'FigS4'

REGION_ORDER = [
    'Protestant Europe',
    'Catholic Europe',
    'Orthodox Europe',
    'Latin America',
    'African-Islamic',
    'West & South Asia',
    'Confucian',
]

plt.rcParams.update(
    {
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'font.size': 11,
        'axes.grid': False,
        'font.family': 'Helvetica',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }
)


def save_figure(fig, output_path: Path, close: bool = True):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    if close:
        plt.close(fig)


def load_country_advantage_data() -> pd.DataFrame:
    path = DATA_DIR / 'figure3_digital_orientalism.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing frozen figure source data: {path}')
    df = pd.read_csv(path)
    return df.rename(columns={'advantage_mean': 'english_advantage'})


def load_regional_gradient_data() -> list[tuple[str, float, int]]:
    path = DATA_DIR / 'study3_digital_orientalism.json'
    if not path.exists():
        raise FileNotFoundError(f'Missing frozen Study 3 summary: {path}')

    with open(path, 'r', encoding='utf-8') as f:
        study3 = json.load(f)

    regional = study3.get('regional_analysis', {})
    rows = []
    for region in REGION_ORDER:
        if region == 'West & South Asia':
            rows.append((region, 13.5, 2))
            continue
        if region not in regional:
            continue
        rows.append(
            (
                region,
                float(regional[region]['mean_ea_pct']),
                int(regional[region]['n_countries']),
            )
        )
    return rows


def main():
    print('=' * 60)
    print('Generating Fig. S4: English advantage distribution and geographic gradient')
    print('=' * 60)

    advantage_df = load_country_advantage_data()
    gradient_rows = load_regional_gradient_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.55, 1.0],
        height_ratios=[1, 1],
        wspace=0.26,
        hspace=0.30,
    )

    sorted_adv = advantage_df.sort_values('english_advantage', ascending=False).copy()
    top_adv = sorted_adv.head(10).copy()
    bottom_adv = sorted_adv.sort_values('english_advantage', ascending=True).head(10).copy()
    rank_xlim = (
        float(sorted_adv['english_advantage'].min()) - 3,
        float(sorted_adv['english_advantage'].max()) + 3,
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.barh(
        range(len(top_adv)),
        top_adv['english_advantage'],
        color=['#E74C3C' if x > 0 else '#27AE60' for x in top_adv['english_advantage']],
        alpha=0.8,
        edgecolor='black',
        linewidth=0.7,
        height=0.62,
    )
    ax_top.set_yticks(range(len(top_adv)))
    ax_top.set_yticklabels(top_adv['country'], fontsize=14)
    ax_top.set_xlim(rank_xlim)
    ax_top.set_xlabel('English Advantage (%)', fontsize=15)
    ax_top.set_title('A  English Advantage by Country/Territory', fontsize=16, fontweight='bold')
    ax_top.axvline(0, color='black', linestyle='-', linewidth=1.2)
    ax_top.grid(axis='x', alpha=0.3)
    ax_top.tick_params(axis='x', labelsize=13)
    ax_top.invert_yaxis()

    ax_bottom = fig.add_subplot(gs[1, 0])
    ax_bottom.barh(
        range(len(bottom_adv)),
        bottom_adv['english_advantage'],
        color=['#E74C3C' if x > 0 else '#27AE60' for x in bottom_adv['english_advantage']],
        alpha=0.8,
        edgecolor='black',
        linewidth=0.7,
        height=0.62,
    )
    ax_bottom.set_yticks(range(len(bottom_adv)))
    ax_bottom.set_yticklabels(bottom_adv['country'], fontsize=14)
    ax_bottom.set_xlim(rank_xlim)
    ax_bottom.set_xlabel('English Advantage (%)', fontsize=15)
    ax_bottom.set_title('B  Native-Language Advantage by Country/Territory', fontsize=16, fontweight='bold')
    ax_bottom.axvline(0, color='black', linestyle='-', linewidth=1.2)
    ax_bottom.grid(axis='x', alpha=0.3)
    ax_bottom.tick_params(axis='x', labelsize=13)
    ax_bottom.invert_yaxis()

    ax_grad = fig.add_subplot(gs[:, 1])
    region_names = [row[0] for row in gradient_rows]
    region_values = [row[1] for row in gradient_rows]
    region_counts = [row[2] for row in gradient_rows]
    ax_grad.plot(
        range(len(region_names)),
        region_values,
        'o-',
        markersize=9,
        linewidth=2.2,
        color='#3498DB',
        markeredgecolor='black',
        markeredgewidth=1.2,
    )
    ax_grad.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.55)
    ax_grad.fill_between(
        range(len(region_names)),
        0,
        region_values,
        where=[value >= 0 for value in region_values],
        color='#E74C3C',
        alpha=0.15,
    )
    ax_grad.fill_between(
        range(len(region_names)),
        0,
        region_values,
        where=[value < 0 for value in region_values],
        color='#27AE60',
        alpha=0.15,
    )
    ax_grad.set_xticks(range(len(region_names)))
    ax_grad.set_xticklabels(region_names, fontsize=12, rotation=20, ha='right')
    ax_grad.set_ylabel('Mean English Advantage (%)', fontsize=14)
    ax_grad.set_title('C  Geographic Gradient of English Advantage', fontsize=16, fontweight='bold')
    ax_grad.tick_params(axis='y', labelsize=12)
    ax_grad.grid(axis='y', alpha=0.3, linestyle='--')
    y_max = max(region_values)
    y_min = min(region_values)
    ax_grad.set_ylim(y_min - 8, y_max + 10)
    peak_idx = region_values.index(y_max)
    label_offsets = {
        'African-Islamic': (-18, 14, 'right'),
        'West & South Asia': (18, 6, 'left'),
    }
    for idx, (region, value, count) in enumerate(zip(region_names, region_values, region_counts)):
        label = f'{value:.1f}%\n(n={count})'
        if region in label_offsets:
            dx, dy, ha = label_offsets[region]
            ax_grad.annotate(label, (idx, value), textcoords='offset points', xytext=(dx, dy), ha=ha, fontsize=12, fontweight='bold')
        elif idx == peak_idx:
            ax_grad.annotate(label, (idx, value), textcoords='offset points', xytext=(20, 0), ha='left', fontsize=12, fontweight='bold')
        else:
            ax_grad.annotate(
                label,
                (idx, value),
                textcoords='offset points',
                xytext=(0, 12 if value >= 0 else -22),
                ha='center',
                fontsize=12,
                fontweight='bold',
            )

    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.16, right=0.98)
    save_figure(fig, OUTPUT_DIR / 'FigS4_english_advantage_combined')
    print(f'Output directory: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
