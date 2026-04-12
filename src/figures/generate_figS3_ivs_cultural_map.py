"""
Generate Fig. S3: Inglehart-Welzel cultural map of 112 countries and
territories from the Integrated Values Survey.

Self-contained version for the paper reproducibility package. Matches the
visual style of main-paper Figure 2 (SVM decision-boundary fills, WVS-official
colour palette).

Output: Supplementary Materials/figures/FigS3/FigS3_IVS_cultural_map.{png,pdf}
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'Supplementary Materials' / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'Supplementary Materials' / 'figures' / 'FigS3'

try:
    from adjustText import adjust_text
    ADJUSTTEXT_AVAILABLE = True
except ImportError:
    ADJUSTTEXT_AVAILABLE = False

EXCLUDED_REGIONS = {'Other', 'Baltic'}
POINT_SIZE_S3 = 80
FONT_SIZE_COUNTRY_LABEL = 7

WVS_REGION_COLORS = {
    'Confucian':         '#E8872E',
    'Protestant Europe': '#D4C826',
    'Latin America':     '#18A48C',
    'Catholic Europe':   '#2EAD56',
    'English-Speaking':  '#C8C830',
    'African-Islamic':   '#8C8C8C',
    'West & South Asia': '#C46A2C',
    'Orthodox Europe':   '#4878A8',
    'Other':             '#B0B0B0',
    'Baltic':            '#88A0C0',
}

REGION_COLORS = dict(WVS_REGION_COLORS)

plt.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': 9,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.6,
    'axes.facecolor': '#F8F8F8',
    'figure.facecolor': 'white',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})


def save_figure(fig, output_path: Path, close: bool = True):
    """Save figure as both PNG and PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    if close:
        plt.close(fig)


def load_ivs_data() -> pd.DataFrame:
    """Load IVS benchmark coordinates from the publication-ready data package."""
    candidates = [
        DATA_DIR / 'DataS1_ivs_pca_coordinates.csv',
        DATA_DIR / 'ivs_pca_coordinates.csv',
    ]
    for candidate in candidates:
        if candidate.exists():
            df = pd.read_csv(candidate)
            df = df.rename(
                columns={
                    'Country': 'country',
                    'Cultural_Region': 'cultural_region',
                    'PC1_Self_Expression': 'PC1',
                    'PC2_Secular_Rational': 'PC2',
                }
            )
            return df.dropna(subset=['country']).drop_duplicates(subset=['country'])
    tried = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(f'Could not find IVS PCA data. Tried: {tried}')


def get_region_color(region: str) -> str:
    if pd.isna(region):
        return REGION_COLORS.get('Other', '#DDDDDD')
    return REGION_COLORS.get(region, '#DDDDDD')


def train_region_classifier(data: pd.DataFrame, method: str = 'svm', **kwargs):
    """Train a classifier to predict cultural regions from PC coordinates."""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import SVC

    X = data[['PC1', 'PC2']].values
    y = data['cultural_region'].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    if method != 'svm':
        raise ValueError(f'Unsupported method: {method}')

    clf = SVC(
        kernel=kwargs.get('kernel', 'rbf'),
        C=kwargs.get('C', 10.0),
        gamma=kwargs.get('gamma', 'scale'),
        probability=True,
    )
    clf.fit(X, y_encoded)
    return clf, le


def create_decision_boundary_mesh(xlim, ylim, resolution: int = 200):
    """Create a mesh grid for decision boundary visualization."""
    x_min, x_max = xlim
    y_min, y_max = ylim
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    xx, yy = np.meshgrid(
        np.linspace(x_min - x_padding, x_max + x_padding, resolution),
        np.linspace(y_min - y_padding, y_max + y_padding, resolution),
    )
    return xx, yy


def generate_ml_boundaries(
    data: pd.DataFrame,
    region_colors: dict,
    method: str = 'svm',
    resolution: int = 200,
    **kwargs,
):
    """Generate ML-based decision boundaries for cultural regions."""
    del region_colors  # Kept for signature compatibility.
    clf, le = train_region_classifier(data, method=method, **kwargs)
    xlim = (data['PC1'].min(), data['PC1'].max())
    ylim = (data['PC2'].min(), data['PC2'].max())
    xx, yy = create_decision_boundary_mesh(xlim, ylim, resolution)
    return clf, le, xx, yy


def plot_decision_boundaries_masked(
    ax,
    clf,
    le,
    xx,
    yy,
    data: pd.DataFrame,
    region_colors: dict,
    alpha: float = 0.3,
    mask_padding: float = 0.3,
    smooth_boundary: bool = True,
):
    """Plot decision boundaries only around actual data points."""
    from scipy.interpolate import splprep, splev
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path as MplPath

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    mask = np.zeros_like(Z, dtype=bool)

    for region in data['cultural_region'].unique():
        if pd.isna(region):
            continue
        subset = data[data['cultural_region'] == region]
        points = subset[['PC1', 'PC2']].values
        if len(points) < 3:
            continue
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            center = points.mean(axis=0)
            expanded_hull = center + (hull_points - center) * (1 + mask_padding)

            if smooth_boundary and len(expanded_hull) >= 4:
                try:
                    closed_hull = np.vstack([expanded_hull, expanded_hull[0]])
                    tck, _ = splprep(
                        [closed_hull[:, 0], closed_hull[:, 1]],
                        s=0.2,
                        per=True,
                        k=min(3, len(expanded_hull) - 1),
                    )
                    u_new = np.linspace(0, 1, 100)
                    smooth_x, smooth_y = splev(u_new, tck)
                    expanded_hull = np.column_stack([smooth_x, smooth_y])
                except Exception:
                    pass

            path = MplPath(expanded_hull)
            inside = path.contains_points(np.c_[xx.ravel(), yy.ravel()])
            mask = mask | inside.reshape(xx.shape)
        except Exception:
            continue

    Z_masked = np.ma.masked_where(~mask, Z)
    unique_labels = np.unique(Z[mask])
    region_names = le.inverse_transform(unique_labels)
    colors = [region_colors.get(name, '#DDDDDD') for name in region_names]

    return ax.contourf(
        xx,
        yy,
        Z_masked,
        levels=np.arange(len(unique_labels) + 1) - 0.5,
        colors=colors,
        alpha=alpha,
    )


def should_include_region(region: str) -> bool:
    if pd.isna(region):
        return False
    return region not in EXCLUDED_REGIONS


def plot_svm_region_fills(ax, ivs_data):
    """SVM decision-boundary shaded regions (same as Fig 2)."""
    try:
        regions = sorted([r for r in ivs_data['cultural_region'].unique()
                          if r not in EXCLUDED_REGIONS and pd.notna(r)])
        ml_data = ivs_data[ivs_data['cultural_region'].isin(regions)].copy()
        ml_data = ml_data[['PC1', 'PC2', 'cultural_region']].dropna()

        clf, le, xx, yy = generate_ml_boundaries(
            ml_data, WVS_REGION_COLORS, method='svm', resolution=250)

        plot_decision_boundaries_masked(
            ax, clf, le, xx, yy, ml_data, WVS_REGION_COLORS,
            alpha=0.40, mask_padding=0.6)
        print("  SVM region fills plotted")
    except Exception as e:
        print(f"  SVM boundaries unavailable ({e}); using scatter only")


def generate_figs1_cultural_map(ivs_data: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 11))

    regions = sorted([r for r in ivs_data['cultural_region'].unique()
                      if pd.notna(r) and should_include_region(r)])

    plot_svm_region_fills(ax, ivs_data)

    stroke = [pe.withStroke(linewidth=2, foreground='white')]
    texts = []
    for region in regions:
        subset = ivs_data[ivs_data['cultural_region'] == region]
        color = WVS_REGION_COLORS.get(region, get_region_color(region))

        ax.scatter(
            subset['PC1'], subset['PC2'],
            c=color, s=POINT_SIZE_S3, marker='o', alpha=0.85,
            edgecolors='white', linewidths=0.6, zorder=2
        )

        for _, row in subset.iterrows():
            country = row['country']
            if pd.notna(country) and not str(country).startswith('Code_'):
                text = ax.text(
                    row['PC1'], row['PC2'], country,
                    fontsize=FONT_SIZE_COUNTRY_LABEL,
                    ha='center', va='bottom', zorder=3,
                    path_effects=stroke
                )
                texts.append(text)

    if ADJUSTTEXT_AVAILABLE and len(texts) > 0:
        print(f"  Adjusting {len(texts)} labels...")
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle='-', color='#999999', lw=0.4, alpha=0.5),
            expand_points=(1.5, 1.5),
            force_points=(0.5, 0.5),
            force_text=(0.4, 0.4),
            lim=800
        )

    ax.axhline(y=0, color='#CCCCCC', linestyle='--', linewidth=0.5, zorder=0)
    ax.axvline(x=0, color='#CCCCCC', linestyle='--', linewidth=0.5, zorder=0)

    pc1_min, pc1_max = ivs_data['PC1'].min(), ivs_data['PC1'].max()
    pc2_min, pc2_max = ivs_data['PC2'].min(), ivs_data['PC2'].max()
    padding = 0.4
    ax.set_xlim(pc1_min - padding, pc1_max + padding)
    ax.set_ylim(pc2_min - padding, pc2_max + padding)

    for spine in ax.spines.values():
        spine.set_color('#555555')
        spine.set_linewidth(0.6)

    fig.text(0.48, 0.02,
             r'$\longleftarrow$ Survival Values'
             r'          Self-Expression Values $\longrightarrow$',
             ha='center', fontsize=12, fontweight='bold', color='#1a1a1a')
    fig.text(0.02, 0.5,
             r'$\longleftarrow$ Traditional Values'
             r'          Secular-Rational Values $\longrightarrow$',
             va='center', rotation='vertical', fontsize=12,
             fontweight='bold', color='#1a1a1a')

    region_handles = [
        mpatches.Patch(color=WVS_REGION_COLORS.get(r, '#CCC'), label=r, alpha=0.55)
        for r in regions
    ]
    legend_font = {'family': 'Helvetica', 'size': 9}
    title_font = {'family': 'Helvetica', 'size': 10, 'weight': 'bold'}
    ax.legend(
        handles=region_handles, loc='upper left',
        prop=legend_font, frameon=True, framealpha=0.95,
        edgecolor='#AAAAAA', title='WVS Cultural Region',
        title_fontproperties=title_font,
        labelspacing=0.5, borderpad=0.6
    )

    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.07, top=0.96)

    save_figure(fig, output_dir / 'FigS3_IVS_cultural_map')
    print("Saved: FigS3_IVS_cultural_map")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Generating Fig. S3: IVS Cultural Map")
    print("=" * 60)
    
    # Load data
    ivs_data = load_ivs_data()
    print(f"Loaded {len(ivs_data)} countries from IVS data")
    
    # Generate figure
    generate_figs1_cultural_map(ivs_data, OUTPUT_DIR)
    
    print()
    print("=" * 60)
    print(f"Done! Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
