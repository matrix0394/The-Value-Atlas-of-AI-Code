"""Visualization helpers for intrinsic LLM cultural coordinates."""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.country_values.visualization import CulturalMapVisualizer


class LLMCulturalMapVisualizer(CulturalMapVisualizer):
    """Extend the country visualizer with LLM-specific views."""
    
    def __init__(self, data_path: str = "data", results_path: str = "results"):
        """Initialize the visualizer."""
        super().__init__(data_path)
        
        self.llm_data_dir = Path(data_path) / "llm_values"
        self.llm_results_dir = Path(results_path) / "llm_values" / "llm_dashboard"
        
        self.llm_results_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_path = self.llm_results_dir
    
    def load_data(self) -> pd.DataFrame:
        """
        Load Stage 1 PCA results for IVS and intrinsic LLM responses.
        """
        standard_path = self.llm_data_dir / "llm_pca_entity_scores.pkl"
        
        if standard_path.exists():
            data = pd.read_pickle(standard_path)
            print(f"Loaded PCA results: {standard_path}")
            print(f"  Rows: {len(data)}")
            
            if 'is_llm' in data.columns:
                llm_count = data['is_llm'].sum()
                print(f"  LLM rows: {llm_count}")
            elif 'data_source' in data.columns:
                llm_count = (data['data_source'] == 'LLM').sum()
                print(f"  LLM rows: {llm_count}")
            
            return data
        else:
            raise FileNotFoundError(
                f"PCA result file not found: {standard_path}\n"
                "Run llm_pca_analysis.py first."
            )
    
    def _split_llm_and_country_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split benchmark countries and intrinsic LLM rows."""
        if 'data_source' in data.columns:
            country_data = data[data['data_source'] == 'IVS']
            llm_data = data[data['data_source'] == 'LLM']
        elif 'is_llm' in data.columns:
            country_data = data[data['is_llm'] == False]
            llm_data = data[data['is_llm'] == True]
        else:
            country_data = data[data['Cultural Region'] != 'AI Model']
            llm_data = data[data['Cultural Region'] == 'AI Model']
        
        return country_data, llm_data
    
    def _get_point_label(self, row: pd.Series) -> str:
        """Return the display label for a point."""
        if 'Country' in row.index and pd.notna(row['Country']):
            return str(row['Country'])
        elif 'model_name' in row.index and pd.notna(row['model_name']):
            return str(row['model_name'])
        elif 'extracted_model' in row.index and pd.notna(row['extracted_model']):
            return str(row['extracted_model'])
        elif 'country_code' in row.index and str(row['country_code']).startswith('LLM_'):
            return str(row['country_code']).replace('LLM_', '')
        else:
            return ""
    
    def plot_llm_vs_countries(self, data: pd.DataFrame = None,
                             figsize: Tuple[int, int] = (14, 10),
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot intrinsic LLM coordinates against the IVS benchmark."""
        if data is None:
            data = self.load_data()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        country_data, llm_data = self._split_llm_and_country_data(data)
        
        for region, color in self.cultural_region_colors.items():
            subset = country_data[country_data['Cultural Region'] == region]
            if len(subset) > 0:
                for i, row in subset.iterrows():
                    country_name = self._get_point_label(row)
                    if country_name:
                        if 'Islamic' in subset.columns and row['Islamic']:
                            ax.text(row['PC1_rescaled'], row['PC2_rescaled'], country_name, 
                                    color=color, fontsize=10, fontstyle='italic')
                        else:
                            ax.text(row['PC1_rescaled'], row['PC2_rescaled'], country_name, 
                                    color=color, fontsize=10)
                
                ax.scatter(subset['PC1_rescaled'], subset['PC2_rescaled'], 
                           label=region, color=color, s=50, alpha=0.7)
        
        if not llm_data.empty:
            print(f"Plotting {len(llm_data)} LLM rows...")
            for _, row in llm_data.iterrows():
                model_name = self._get_point_label(row)
                color = self.get_color_for_model(model_name)
                
                ax.scatter(row['PC1_rescaled'], row['PC2_rescaled'], 
                           color=color, s=200, alpha=0.9, marker='*', 
                           edgecolors='black', linewidth=1.5,
                           label=f"LLM: {model_name}" if len(llm_data) <= 10 else None)
                
                ax.text(row['PC1_rescaled'], row['PC2_rescaled'], model_name, 
                        color=color, fontsize=10, fontweight='bold',
                        ha='center', va='bottom')
                print(f"  {model_name}: ({row['PC1_rescaled']:.2f}, {row['PC2_rescaled']:.2f})")
        else:
            print("No LLM rows were found for plotting.")
        
        ax.set_xlabel('Survival vs. Self-Expression Values')
        ax.set_ylabel('Traditional vs. Secular Values')
        ax.set_title('Inglehart-Welzel Cultural Map with LLM Models')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LLM comparison plot saved to: {save_path}")
        
        return fig
    
    def plot_model_comparison(self, data: pd.DataFrame = None,
                             figsize: Tuple[int, int] = (14, 10),
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot a model-only comparison view."""
        if data is None:
            data = self.load_data()
        
        _, llm_data = self._split_llm_and_country_data(data)
        
        if llm_data.empty:
            print("No LLM rows were found.")
            return plt.figure()
        
        return self.create_comparison_plot(
            llm_data, 
            'extracted_model' if 'extracted_model' in llm_data.columns else 'model_name',
            figsize=figsize,
            save_path=save_path
        )
    
    def create_llm_dashboard(self, data: pd.DataFrame = None,
                           save_dir: Optional[str] = None) -> Dict[str, str]:
        """Create the intrinsic LLM visualization bundle."""
        if data is None:
            data = self.load_data()
        
        if save_dir is None:
            save_dir = self.llm_results_dir
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        print("\nCreating intrinsic LLM visualization bundle...")
        print(f"Output directory: {save_dir}")
        
        print("\n1. Creating LLM-versus-country plot...")
        fig1 = self.plot_llm_vs_countries(data, figsize=(16, 12))
        path1 = save_dir / "llm_vs_countries.png"
        fig1.savefig(path1, dpi=300, bbox_inches='tight')
        saved_files['llm_vs_countries'] = str(path1)
        plt.close(fig1)
        
        print("2. Creating full cultural map...")
        fig2 = self.plot_basic_cultural_map(
            data, 
            title="Inglehart-Welzel Cultural Map with LLM Models",
            figsize=(16, 12)
        )
        path2 = save_dir / "cultural_map_with_llm.png"
        fig2.savefig(path2, dpi=300, bbox_inches='tight')
        saved_files['cultural_map'] = str(path2)
        plt.close(fig2)
        
        print("3. Creating model comparison plot...")
        fig3 = self.plot_model_comparison(data, figsize=(14, 10))
        path3 = save_dir / "model_comparison.png"
        fig3.savefig(path3, dpi=300, bbox_inches='tight')
        saved_files['model_comparison'] = str(path3)
        plt.close(fig3)
        
        print(f"\nSaved {len(saved_files)} figures to: {save_dir}")
        return saved_files


def main():
    """Run the intrinsic LLM visualization pipeline."""
    data_path = "data"
    results_path = "results"
    
    print("Running intrinsic LLM visualizations...")
    
    visualizer = LLMCulturalMapVisualizer(data_path, results_path)
    
    try:
        saved_files = visualizer.create_llm_dashboard()
        
        print("\nLLM visualizations completed.")
        print("Generated files:")
        for name, path in saved_files.items():
            print(f"   - {name}: {path}")
        
        return saved_files
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
