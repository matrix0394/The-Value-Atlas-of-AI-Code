import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.base.base_cultural_map_visualizer import BaseCulturalMapVisualizer


class CulturalMapVisualizer(BaseCulturalMapVisualizer):
    """Visualization helper for the country-level cultural benchmark."""

    def __init__(self, data_path="../data", results_path="../results"):
        """Initialize the benchmark visualizer."""
        super().__init__(data_path=data_path, results_path=results_path)

    def load_data(self) -> pd.DataFrame:
        """Return the country-score table expected by the base visualizer."""
        return self.load_country_scores()

    def load_country_scores(self):
        """Load the saved country-level PCA coordinates."""
        country_scores_path = os.path.join(str(self.data_path), "country_scores_pca.pkl")
        return pd.read_pickle(country_scores_path)

    def plot_cultural_map(self, country_scores_pca=None, figsize=(14, 10), save_path=None):
        """Plot the cultural map and italicize countries flagged as Islamic."""
        if country_scores_pca is None:
            country_scores_pca = self.load_country_scores()

        fig = self.plot_basic_cultural_map(
            data=country_scores_pca,
            figsize=figsize,
            title='Inglehart-Welzel Cultural Map',
            show_labels=False,
            save_path=None
        )

        ax = plt.gca()
        for region, color in self.cultural_region_colors.items():
            subset = country_scores_pca[country_scores_pca['Cultural Region'] == region]
            if len(subset) > 0:
                for _, row in subset.iterrows():
                    if 'Islamic' in country_scores_pca.columns and row['Islamic']:
                        ax.text(row['PC1_rescaled'], row['PC2_rescaled'], row['Country'], 
                               color=color, fontsize=10, fontstyle='italic')
                    else:
                        ax.text(row['PC1_rescaled'], row['PC2_rescaled'], row['Country'], 
                               color=color, fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cultural map saved to {save_path}")
        
        plt.show()
        return fig

    def plot_decision_boundary(self, country_scores_pca=None, figsize=(14, 10), save_path=None):
        """Plot the region decision boundary using the shared base implementation."""
        if country_scores_pca is None:
            country_scores_pca = self.load_country_scores()

        fig = super().plot_decision_boundary(
            data=country_scores_pca,
            target_column='Cultural Region',
            figsize=figsize,
            save_path=save_path
        )
        
        plt.show()
        return fig

    def create_summary_statistics(self, country_scores_pca=None):
        """Print basic summary statistics for the saved country scores."""
        if country_scores_pca is None:
            country_scores_pca = self.load_country_scores()
        
        print("=== Cultural Map Summary Statistics ===")
        print(f"Total countries: {len(country_scores_pca)}")
        
        if 'Cultural Region' in country_scores_pca.columns:
            print("\nCountries by Cultural Region:")
            region_counts = country_scores_pca['Cultural Region'].value_counts()
            for region, count in region_counts.items():
                print(f"  {region}: {count}")
        
        print("\nPrincipal Component Statistics:")
        print(f"PC1 (Survival vs. Self-Expression) range: [{country_scores_pca['PC1_rescaled'].min():.2f}, {country_scores_pca['PC1_rescaled'].max():.2f}]")
        print(f"PC2 (Traditional vs. Secular) range: [{country_scores_pca['PC2_rescaled'].min():.2f}, {country_scores_pca['PC2_rescaled'].max():.2f}]")
        
        return country_scores_pca.describe()

def main():
    """Run a small smoke test for the benchmark visualizations."""
    print("=== Testing benchmark visualizations ===")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)

    visualizer = CulturalMapVisualizer()

    try:
        print("\n1. Loading country scores...")
        country_scores_path = os.path.join(visualizer.data_path, "country_scores_pca.pkl")
        if os.path.exists(country_scores_path):
            country_scores_pca = visualizer.load_country_scores()
            print(f"Loaded country scores for {len(country_scores_pca)} countries")
        else:
            print("Missing country_scores_pca.pkl. Please run pca_analysis.py first.")
            return

        print("\n2. Drawing the cultural map...")
        cultural_map_path = os.path.join(results_dir, "cultural_map_test.png")
        visualizer.plot_cultural_map(save_path=cultural_map_path)
        print(f"Cultural map saved to: {cultural_map_path}")

        print("\n3. Drawing decision boundaries...")
        decision_boundary_path = os.path.join(results_dir, "decision_boundary_test.png")
        visualizer.plot_decision_boundary(save_path=decision_boundary_path)
        print(f"Decision-boundary plot saved to: {decision_boundary_path}")

        print("\n4. Printing summary statistics...")
        summary_stats = visualizer.create_summary_statistics()
        print("\nSummary statistics:")
        print(summary_stats)

        if 'Cultural Region' in country_scores_pca.columns:
            print("\n5. Cultural-region counts:")
            region_counts = country_scores_pca['Cultural Region'].value_counts()
            for region, count in region_counts.items():
                color = visualizer.cultural_region_colors.get(region, '#000000')
                print(f"  {region}: {count} countries (color: {color})")

        print("\nVisualization smoke test completed.")

    except Exception as e:
        print(f"Visualization test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
