"""Country-level PCA analysis for the IVS cultural benchmark."""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.base.base_pca_analyzer import BasePCAAnalyzer


class CorePCAAnalyzer(BasePCAAnalyzer):
    """PCA wrapper for the IVS benchmark pipeline."""

    def __init__(self, data_path: str = "data/country_values"):
        """Initialize the benchmark PCA analyzer."""
        super().__init__(data_path)

    def load_additional_data(self) -> pd.DataFrame:
        """No auxiliary table is required for the benchmark PCA."""
        return pd.DataFrame()

    def combine_data(self) -> pd.DataFrame:
        """Prepare the IVS table used for PCA estimation."""
        if self.ivs_df is None:
            raise ValueError("Please load the IVS data first")

        self.combined_data = self.prepare_ivs_data()
        self.combined_data['data_source'] = 'IVS'

        self.combined_data = self.prepare_country_codes_for_merge(self.combined_data)

        print(f"Prepared IVS data: {len(self.combined_data)} rows")
        return self.combined_data

    def calculate_country_scores(self) -> pd.DataFrame:
        """Calculate country-level scores while preserving the legacy interface."""
        return self.calculate_entity_scores(['country_code'])

    def save_results(self, entity_scores=None, prefix="core_pca"):
        """Save outputs using the legacy file names expected elsewhere in the project."""
        if self.pca_results is not None:
            valid_data_path = self.data_path / "valid_data.pkl"
            self.pca_results.to_pickle(valid_data_path)
            print(f"Saved valid_data to: {valid_data_path}")

        if entity_scores is not None:
            country_scores_path = self.data_path / "country_scores_pca.pkl"
            entity_scores.to_pickle(country_scores_path)
            print(f"Saved country_scores_pca to: {country_scores_path}")

            country_scores_json = self.data_path / "country_scores_pca.json"
            entity_scores.to_json(country_scores_json, orient='records', indent=2)
            print(f"Saved country_scores_pca.json to: {country_scores_json}")

        if hasattr(self, 'ppca_model') and self.ppca_model is not None:
            pca_model_path = self.data_path / "pca_model_fixed.pkl"
            self.save_pca_model(pca_model_path)
            print(f"Saved fixed PCA model to: {pca_model_path}")


def main():
    """Run the benchmark PCA pipeline."""
    data_path = "data/country_values"

    print("Running benchmark PCA analysis...")

    analyzer = CorePCAAnalyzer(data_path=data_path)

    try:
        country_scores = analyzer.run_full_analysis()

        print("\nBenchmark PCA analysis completed.")
        print(f"Generated PCA scores for {len(country_scores)} countries")

        if 'PC1_rescaled' in country_scores.columns:
            print(f"PC1 range: [{country_scores['PC1_rescaled'].min():.2f}, {country_scores['PC1_rescaled'].max():.2f}]")
        if 'PC2_rescaled' in country_scores.columns:
            print(f"PC2 range: [{country_scores['PC2_rescaled'].min():.2f}, {country_scores['PC2_rescaled'].max():.2f}]")
        if 'Cultural Region' in country_scores.columns:
            print("\nCultural-region counts:")
            print(country_scores['Cultural Region'].value_counts())

        return country_scores

    except Exception as e:
        print(f"Benchmark PCA analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
