"""Visualization helpers for multilingual intrinsic LLM coordinates."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm_values.llm_visualization import LLMCulturalMapVisualizer

LANGUAGE_COLORS = {
    'en': '#1f77b4',
    'fr': '#ff7f0e',
    'es': '#2ca02c',
    'ru': '#d62728',
    'ar': '#9467bd',
    'zh-cn': '#e377c2',
}

LANGUAGE_MARKERS = {
    'en': 'o',
    'fr': 's',
    'es': '^',
    'ru': 'D',
    'ar': 'p',
    'zh-cn': '*',
}

LANGUAGE_NAMES = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'ru': 'Russian',
    'ar': 'Arabic',
    'zh-cn': 'Chinese',
}

LANGUAGE_NAMES_EN = {
    'en': 'EN',
    'fr': 'FR',
    'es': 'ES',
    'ru': 'RU',
    'ar': 'AR',
    'zh-cn': 'ZH',
}


class LLMMultilingualVisualizer(LLMCulturalMapVisualizer):
    """Create multilingual visualizations for intrinsic LLM coordinates."""
    
    def __init__(self, data_path: str = "data", results_path: str = "results"):
        """Initialize the multilingual visualizer."""
        super().__init__(data_path, results_path)
        
        self.multilingual_results_dir = Path(results_path) / "llm_values" / "multilingual"
        self.multilingual_results_dir.mkdir(parents=True, exist_ok=True)
        
        self.language_colors = LANGUAGE_COLORS
        self.language_markers = LANGUAGE_MARKERS
        self.language_names = LANGUAGE_NAMES
        self.language_names_en = LANGUAGE_NAMES_EN
    
    def load_multilingual_data(self) -> pd.DataFrame:
        """
        Load multilingual PCA results and ensure a language column is present.
        """
        data = self.load_data()
        
        if 'language' not in data.columns:
            print("No language column found; attempting to infer it from identifiers.")
            if 'country_code' in data.columns:
                data['language'] = data['country_code'].apply(self._extract_language_from_country_code)
            elif 'entity_id' in data.columns:
                data['language'] = data['entity_id'].apply(self._extract_language_from_entity_id)
            else:
                data['language'] = 'en'
        
        lang_counts = data['language'].value_counts()
        print("Language distribution:")
        for lang, count in lang_counts.items():
            lang_name = self.language_names.get(lang, lang)
            print(f"   - {lang} ({lang_name}): {count}")
        
        return data
    
    def _extract_language_from_country_code(self, country_code) -> str:
        """Extract a language code from ``country_code``."""
        if not country_code or not isinstance(country_code, str):
            return 'en'
        
        if country_code.endswith('_zh-cn'):
            return 'zh-cn'
        
        for lang in ['en', 'fr', 'es', 'ru', 'ar']:
            if country_code.endswith(f'_{lang}'):
                return lang
        
        return 'en'
    
    def _extract_language_from_entity_id(self, entity_id: str) -> str:
        """Extract a language code from ``entity_id``."""
        if not entity_id:
            return 'en'
        
        for lang in self.language_colors.keys():
            if f'_{lang}' in entity_id or entity_id.endswith(f'_{lang}'):
                return lang
        
        return 'en'
    
    def _extract_model_name(self, row: pd.Series) -> str:
        """Extract the model name from a row."""
        if 'model_name' in row.index and pd.notna(row['model_name']):
            return str(row['model_name'])
        elif 'extracted_model' in row.index and pd.notna(row['extracted_model']):
            return str(row['extracted_model'])
        elif 'entity_id' in row.index:
            entity_id = str(row['entity_id'])
            if entity_id.startswith('llm_'):
                parts = entity_id[4:].rsplit('_', 1)
                if len(parts) >= 1:
                    return parts[0]
        return "Unknown"
    
    def plot_all_models_by_language(self, data: pd.DataFrame = None,
                                    figsize: Tuple[int, int] = (16, 12),
                                    save_path: Optional[str] = None,
                                    show_countries: bool = True) -> plt.Figure:
        """Plot all multilingual model coordinates, colored by language."""
        if data is None:
            data = self.load_multilingual_data()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        country_data, llm_data = self._split_llm_and_country_data(data)
        
        # Draw countries as a faded background layer.
        if show_countries and not country_data.empty:
            for region, color in self.cultural_region_colors.items():
                subset = country_data[country_data['Cultural Region'] == region]
                if len(subset) > 0:
                    ax.scatter(subset['PC1_rescaled'], subset['PC2_rescaled'],
                              color=color, s=30, alpha=0.2, marker='.')
                    for _, row in subset.iterrows():
                        country_name = self._get_point_label(row)
                        if country_name:
                            ax.text(row['PC1_rescaled'], row['PC2_rescaled'], 
                                   country_name, color=color, fontsize=7, alpha=0.4)
        
        if not llm_data.empty:
            for lang in self.language_colors.keys():
                lang_data = llm_data[llm_data['language'] == lang]
                if lang_data.empty:
                    continue
                
                color = self.language_colors[lang]
                marker = self.language_markers[lang]
                lang_name = self.language_names[lang]
                
                ax.scatter(lang_data['PC1_rescaled'], lang_data['PC2_rescaled'],
                          color=color, s=150, alpha=0.8, marker=marker,
                          edgecolors='black', linewidth=1,
                          label=f'{lang_name} ({lang})')
                
                for _, row in lang_data.iterrows():
                    model_name = self._extract_model_name(row)
                    ax.annotate(model_name, 
                               (row['PC1_rescaled'], row['PC2_rescaled']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color=color, alpha=0.8)
        
        ax.set_xlabel('Survival vs. Self-Expression Values', fontsize=12)
        ax.set_ylabel('Traditional vs. Secular Values', fontsize=12)
        ax.set_title('LLM Cultural Values by Interview Language', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multilingual comparison plot saved: {save_path}")
        
        return fig
    
    def plot_single_model_languages(self, model_name: str,
                                    data: pd.DataFrame = None,
                                    figsize: Tuple[int, int] = (12, 10),
                                    save_path: Optional[str] = None,
                                    show_countries: bool = True) -> plt.Figure:
        """Plot one model across the available interview languages."""
        if data is None:
            data = self.load_multilingual_data()
        
        country_data, llm_data = self._split_llm_and_country_data(data)
        
        model_data = llm_data[llm_data.apply(
            lambda row: self._extract_model_name(row) == model_name, axis=1
        )]
        
        if model_data.empty:
            print(f"No rows found for model {model_name}.")
            return plt.figure()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if show_countries and not country_data.empty:
            for region, color in self.cultural_region_colors.items():
                subset = country_data[country_data['Cultural Region'] == region]
                if len(subset) > 0:
                    ax.scatter(subset['PC1_rescaled'], subset['PC2_rescaled'],
                              color=color, s=30, alpha=0.15, marker='.')
                    for _, row in subset.iterrows():
                        country_name = self._get_point_label(row)
                        if country_name:
                            ax.text(row['PC1_rescaled'], row['PC2_rescaled'],
                                   country_name, color=color, fontsize=7, alpha=0.3)
        
        points = []
        for _, row in model_data.iterrows():
            lang = row.get('language', 'en')
            color = self.language_colors.get(lang, 'gray')
            marker = self.language_markers.get(lang, 'o')
            lang_name = self.language_names.get(lang, lang)
            
            x, y = row['PC1_rescaled'], row['PC2_rescaled']
            points.append((x, y, lang))
            
            ax.scatter(x, y, color=color, s=200, alpha=0.9, marker=marker,
                      edgecolors='black', linewidth=1.5,
                      label=f'{lang_name} ({lang})')
            
            ax.annotate(lang_name, (x, y), xytext=(8, 8), 
                       textcoords='offset points',
                       fontsize=10, fontweight='bold', color=color)
        
        # Connect points to show the within-model language trajectory.
        if len(points) > 1:
            points_sorted = sorted(points, key=lambda p: list(self.language_colors.keys()).index(p[2]) 
                                  if p[2] in self.language_colors else 99)
            xs = [p[0] for p in points_sorted]
            ys = [p[1] for p in points_sorted]
            ax.plot(xs, ys, 'k--', alpha=0.3, linewidth=1)
        
        if len(points) > 1:
            max_dist = 0
            for i, p1 in enumerate(points):
                for p2 in points[i+1:]:
                    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                    if dist > max_dist:
                        max_dist = dist
            ax.text(0.02, 0.98, f'Max Language Distance: {max_dist:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Survival vs. Self-Expression Values', fontsize=12)
        ax.set_ylabel('Traditional vs. Secular Values', fontsize=12)
        ax.set_title(f'Cultural Values of {model_name} by Language', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved multilingual plot for {model_name}: {save_path}")
        
        return fig
    
    def plot_all_models_individual(self, data: pd.DataFrame = None,
                                   save_dir: Optional[str] = None) -> Dict[str, str]:
        """Generate one multilingual plot per model."""
        if data is None:
            data = self.load_multilingual_data()
        
        if save_dir is None:
            save_dir = self.multilingual_results_dir / "per_model"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        _, llm_data = self._split_llm_and_country_data(data)
        model_names = llm_data.apply(self._extract_model_name, axis=1).unique()
        
        print(f"\nGenerating individual plots for {len(model_names)} models...")
        
        saved_files = {}
        for model_name in sorted(model_names):
            safe_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            save_path = save_dir / f"{safe_name}_languages.png"
            
            fig = self.plot_single_model_languages(model_name, data, save_path=str(save_path))
            plt.close(fig)
            
            saved_files[model_name] = str(save_path)
            print(f"   Saved: {model_name}")
        
        return saved_files
    
    def plot_language_comparison_grid(self, data: pd.DataFrame = None,
                                      figsize: Tuple[int, int] = (20, 16),
                                      save_path: Optional[str] = None) -> plt.Figure:
        """Plot a small-multiples grid with one panel per language."""
        if data is None:
            data = self.load_multilingual_data()
        
        country_data, llm_data = self._split_llm_and_country_data(data)
        
        languages = [lang for lang in self.language_colors.keys() 
                    if lang in llm_data['language'].values]
        
        if not languages:
            print("No multilingual data are available.")
            return plt.figure()
        
        n_langs = len(languages)
        n_cols = min(3, n_langs)
        n_rows = (n_langs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for idx, lang in enumerate(languages):
            ax = axes[idx]
            lang_name = self.language_names.get(lang, lang)
            color = self.language_colors[lang]
            
            if not country_data.empty:
                for region, reg_color in self.cultural_region_colors.items():
                    subset = country_data[country_data['Cultural Region'] == region]
                    if len(subset) > 0:
                        ax.scatter(subset['PC1_rescaled'], subset['PC2_rescaled'],
                                  color=reg_color, s=15, alpha=0.15, marker='.')
            
            lang_data = llm_data[llm_data['language'] == lang]
            ax.scatter(lang_data['PC1_rescaled'], lang_data['PC2_rescaled'],
                      color=color, s=100, alpha=0.8, marker=self.language_markers[lang],
                      edgecolors='black', linewidth=1)
            
            for _, row in lang_data.iterrows():
                model_name = self._extract_model_name(row)
                ax.annotate(model_name, (row['PC1_rescaled'], row['PC2_rescaled']),
                           xytext=(3, 3), textcoords='offset points',
                           fontsize=7, color=color)
            
            ax.set_title(f'{lang_name} ({lang})', fontsize=12, color=color, fontweight='bold')
            ax.set_xlabel('Survival-Expression', fontsize=9)
            ax.set_ylabel('Traditional-Secular', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        for idx in range(len(languages), len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle('LLM Cultural Values by Interview Language', fontsize=16, y=1.02)
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Language comparison grid saved: {save_path}")
        
        return fig
    
    def create_multilingual_dashboard(self, data: pd.DataFrame = None,
                                      save_dir: Optional[str] = None) -> Dict[str, str]:
        """Create the full multilingual visualization bundle."""
        if data is None:
            data = self.load_multilingual_data()
        
        if save_dir is None:
            save_dir = self.multilingual_results_dir
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'=' * 60}")
        print("Creating multilingual visualization bundle")
        print(f"{'=' * 60}")
        print(f"Output directory: {save_dir}")
        
        print("\n1. Creating all-model multilingual comparison plot...")
        path1 = save_dir / f"all_models_by_language_{timestamp}.png"
        fig1 = self.plot_all_models_by_language(data, save_path=str(path1))
        saved_files['all_models_by_language'] = str(path1)
        plt.close(fig1)
        
        print("2. Creating language comparison grid...")
        path2 = save_dir / f"language_comparison_grid_{timestamp}.png"
        fig2 = self.plot_language_comparison_grid(data, save_path=str(path2))
        saved_files['language_grid'] = str(path2)
        plt.close(fig2)
        
        print("3. Creating per-model language plots...")
        model_files = self.plot_all_models_individual(data, save_dir=save_dir / "per_model")
        saved_files['per_model'] = model_files
        
        print(f"\n{'=' * 60}")
        print("Multilingual visualizations completed.")
        print("  - Summary figures: 2")
        print(f"  - Per-model figures: {len(model_files)}")
        print(f"{'=' * 60}")
        
        return saved_files


def main():
    """Run the multilingual visualization pipeline."""
    print("Running multilingual visualizations...")
    
    visualizer = LLMMultilingualVisualizer()
    
    try:
        saved_files = visualizer.create_multilingual_dashboard()
        
        print("\nVisualizations completed.")
        print("Generated files:")
        for name, path in saved_files.items():
            if isinstance(path, dict):
                print(f"   - {name}: {len(path)} files")
            else:
                print(f"   - {name}: {path}")
        
        return saved_files
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
