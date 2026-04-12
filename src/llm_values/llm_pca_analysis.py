"""Project intrinsic LLM responses into the benchmark PCA space."""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the project root so local modules can be imported when the script is run directly.
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.base.base_pca_analyzer import BasePCAAnalyzer
from src.base.ivs_question_processor import IVSQuestionProcessor


class LLMPCAAnalyzer(BasePCAAnalyzer):
    """PCA analyzer for intrinsic LLM responses and IVS benchmark data."""
    
    def __init__(self, data_path: str = "data"):
        """Initialize the analyzer with the cleaned repository data layout."""
        super().__init__(data_path, ivs_data_subdir="country_values")
        self.llm_data = None
    
    def load_additional_data(self) -> pd.DataFrame:
        """Load the IVS-compatible intrinsic-response table from canonical paths."""
        try:
            import glob

            candidate_files = []
            candidate_files.extend(
                sorted(
                    glob.glob(str(self.data_path / "llm_interviews" / "intrinsic" / "llm_processed_responses_ivs_format_*.pkl")),
                    reverse=True,
                )
            )
            candidate_files.extend(
                [
                    str(self.data_path / "llm_interviews" / "intrinsic" / "multilingual_ivs_format.pkl"),
                    str(self.data_path / "llm_values" / "llm_values_ivs_format.pkl"),
                ]
            )
            candidate_files.extend(
                sorted(
                    glob.glob(str(self.data_path / "llm_values" / "llm_processed_responses_ivs_format_*.pkl")),
                    reverse=True,
                )
            )

            for candidate in candidate_files:
                llm_path = Path(candidate)
                if not llm_path.exists():
                    continue
                self.llm_data = pd.read_pickle(llm_path)
                print(f"Loaded intrinsic LLM data: {llm_path.name}")
                print(f"   - Path: {llm_path}")
                print(f"   - Models: {len(self.llm_data)}")
                print("   - Format: IVS-compatible table")
                return self.llm_data

            print("No intrinsic LLM data file was found.")
            print("   Searched paths:")
            print(f"   - {self.data_path / 'llm_interviews' / 'intrinsic'}")
            print(f"   - {self.data_path / 'llm_values'}")
            print("   Expected example: llm_processed_responses_ivs_format_latest.pkl")
            print("   Generate it with llm_data_processor.py or the intrinsic pipeline runner.")
            return pd.DataFrame()
        
        except Exception as e:
            print(f"Failed to load intrinsic LLM data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _load_released_country_scores(self) -> pd.DataFrame:
        """Load released benchmark coordinates when raw IVS rows are unavailable."""
        candidates = [
            self.data_path / "country_values" / "country_scores_pca.json",
            self.data_path / "country_values" / "country_scores_pca.pkl",
        ]

        for candidate in candidates:
            if not candidate.exists():
                continue

            if candidate.suffix == ".json":
                released = pd.read_json(candidate)
            else:
                released = pd.read_pickle(candidate)

            if "data_source" not in released.columns:
                released["data_source"] = "IVS"
            if "model_name" not in released.columns:
                released["model_name"] = None

            print(f"Loaded released benchmark coordinates: {candidate}")
            print(f"   - Countries: {len(released)}")
            return released

        return pd.DataFrame()
        
    def _prepare_llm_data_for_pca(self, llm_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare the intrinsic-response table for joint PCA projection."""
        print(f"Preparing {len(llm_data)} LLM rows for PCA projection...")
        
        llm_prepared = llm_data.copy()
        
        llm_prepared['data_source'] = 'LLM'
        
        llm_prepared['Cultural Region'] = 'AI Model'
        
        if 'model_name' not in llm_prepared.columns:
            llm_prepared['model_name'] = llm_prepared['country_code']
        
        llm_prepared['country_code'] = llm_prepared['country_code'].apply(
            lambda x: x if str(x).startswith('LLM_') else f'LLM_{x}'
        )
        
        print(f"Prepared {len(llm_prepared)} rows for PCA projection.")
        print("   - Input already follows the IVS-compatible layout")
        print("   - Cultural Region is set to 'AI Model'")
        return llm_prepared
    
    def combine_data(self) -> pd.DataFrame:
        """Combine the IVS benchmark and intrinsic LLM tables."""
        ivs_data = self.prepare_ivs_data()
        ivs_data['data_source'] = 'IVS'
        ivs_data['model_name'] = None
        
        ivs_data = self.prepare_country_codes_for_merge(ivs_data)
        
        data_parts = [ivs_data]
        
        if self.llm_data is not None and not self.llm_data.empty:
            llm_prepared = self._prepare_llm_data_for_pca(self.llm_data)
            if not llm_prepared.empty:
                llm_prepared = self.prepare_country_codes_for_merge(llm_prepared)
                data_parts.append(llm_prepared)
        
        self.combined_data = pd.concat(data_parts, ignore_index=True)
        
        print("Combined benchmark and intrinsic-response data.")
        print(f"   - Total rows: {len(self.combined_data)}")
        print(f"   - IVS rows: {len(ivs_data)}")
        if len(data_parts) > 1:
            llm_count = len(self.combined_data[self.combined_data['data_source'] == 'LLM'])
            print(f"   - LLM rows: {llm_count}")
        
        return self.combined_data
    
    def calculate_entity_scores(self, group_by=None) -> pd.DataFrame:
        """Calculate entity-level PCA scores for both benchmark and model rows."""
        if group_by is None:
            group_by = ['country_code', 'data_source']
        
        entity_scores = super().calculate_entity_scores(group_by)
        
        entity_scores['is_llm'] = entity_scores['data_source'] == 'LLM'
        
        if 'is_llm' in entity_scores.columns:
            llm_rows = entity_scores['is_llm'] == True
            if llm_rows.any():
                entity_scores.loc[llm_rows, 'extracted_model'] = entity_scores.loc[llm_rows, 'country_code'].str.replace('LLM_', '')
        
        return entity_scores
    
    def save_results(self, entity_scores=None, prefix="llm_pca"):
        """Save PCA outputs to the canonical intrinsic-results directory."""
        output_dir = self.data_path / "llm_pca" / "intrinsic"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.pca_results is not None:
            pca_path = output_dir / f"{prefix}_results.pkl"
            self.pca_results.to_pickle(pca_path)
            print(f"Saved PCA results to: {pca_path}")
        
        if entity_scores is not None:
            scores_path = output_dir / f"{prefix}_entity_scores.pkl"
            entity_scores.to_pickle(scores_path)
            print(f"Saved entity scores to: {scores_path}")
            
            scores_json = output_dir / f"{prefix}_entity_scores.json"
            entity_scores.to_json(scores_json, orient='records', indent=2)
            print(f"Saved JSON export to: {scores_json}")
    
    def run_analysis_with_fixed_pca(self) -> pd.DataFrame:
        """Project intrinsic LLM data with the fixed benchmark PCA model."""
        print("\n" + "="*60)
        print("Projecting intrinsic LLM data with the fixed benchmark PCA model")
        print("="*60)
        
        pca_model_path = Path('data/country_values/pca_model_fixed.pkl')
        if not pca_model_path.exists():
            raise FileNotFoundError(
                f"Fixed PCA model not found: {pca_model_path}\n"
                "Generate it first with python src/country_values/pca_analysis.py"
            )
        
        pca_model = self.load_pca_model(pca_model_path)
        
        print("\nLoading IVS benchmark data...")
        ivs_pca = pd.DataFrame()
        if self.load_base_data():
            ivs_data = self.prepare_ivs_data()
            ivs_data['data_source'] = 'IVS'
            ivs_data['model_name'] = None
            print(f"   IVS rows: {len(ivs_data)}")
            
            print("\nProjecting IVS benchmark data...")
            ivs_pca = self.transform_with_fixed_pca(ivs_data, pca_model)
            ivs_pca['country_code'] = ivs_data['country_code'].values
            ivs_pca['data_source'] = 'IVS'
            ivs_pca['model_name'] = None
            if 'year' in ivs_data.columns:
                ivs_pca['year'] = ivs_data['year'].values
        else:
            ivs_pca = self._load_released_country_scores()
            if ivs_pca.empty:
                raise ValueError("Failed to load IVS benchmark data")
        
        print("\nLoading intrinsic LLM data...")
        llm_data = self.load_additional_data()
        if llm_data.empty:
            raise ValueError("Intrinsic LLM data are empty")
        
        llm_prepared = self._prepare_llm_data_for_pca(llm_data)
        print(f"   LLM rows: {len(llm_prepared)}")
        
        print("\nProjecting intrinsic LLM data...")
        llm_pca = self.transform_with_fixed_pca(llm_prepared, pca_model)
        
        llm_pca['country_code'] = llm_prepared['country_code'].values
        llm_pca['data_source'] = 'LLM'
        llm_pca['Cultural Region'] = 'AI Model'
        
        if 'model_name' in llm_prepared.columns:
            llm_pca['model_name'] = llm_prepared['model_name'].values
        
        print("\nCombining PCA outputs...")
        self.pca_results = pd.concat([ivs_pca, llm_pca], ignore_index=True)
        print(f"   Combined rows: {len(self.pca_results)}")
        
        self.pca_results = self.prepare_country_codes_for_merge(self.pca_results)
        self.pca_results = self.merge_country_metadata(self.pca_results, on_column='country_code_clean')
        
        print("\nCalculating entity scores...")
        entity_scores = self.calculate_entity_scores()
        
        self.save_results(entity_scores)
        
        self.print_summary(entity_scores)
        
        return entity_scores
    
    def run_llm_analysis_for_runner(self, use_fixed_pca: bool = True) -> pd.DataFrame:
        """Run the intrinsic LLM PCA workflow for the top-level pipeline."""
        print("Starting intrinsic LLM PCA analysis...")
        
        if use_fixed_pca:
            entity_scores = self.run_analysis_with_fixed_pca()
        else:
            print("Warning: refitting PCA from scratch may break comparability with the benchmark.")
            entity_scores = super().run_full_analysis()
        
        print(f"Intrinsic LLM PCA analysis completed for {len(entity_scores)} entities")
        return entity_scores
    
    def print_summary(self, entity_scores=None):
        """Print a compact summary of benchmark and intrinsic LLM PCA results."""
        super().print_summary(entity_scores)
        
        if entity_scores is not None and 'is_llm' in entity_scores.columns:
            print("\nLLM vs. benchmark summary:")
            llm_count = entity_scores['is_llm'].sum()
            country_count = (~entity_scores['is_llm']).sum()
            print(f"   - LLM models: {llm_count}")
            print(f"   - Benchmark countries: {country_count}")
            
            if llm_count > 0:
                llm_data = entity_scores[entity_scores['is_llm'] == True]
                print("\nLLM score ranges:")
                print(f"   - PC1: [{llm_data['PC1_rescaled'].min():.2f}, {llm_data['PC1_rescaled'].max():.2f}]")
                print(f"   - PC2: [{llm_data['PC2_rescaled'].min():.2f}, {llm_data['PC2_rescaled'].max():.2f}]")


def main():
    """Entry point for intrinsic LLM PCA analysis."""
    data_path = "data"
    
    print("Running intrinsic LLM PCA analysis...")
    
    analyzer = LLMPCAAnalyzer(data_path=data_path)
    
    try:
        entity_scores = analyzer.run_llm_analysis_for_runner(use_fixed_pca=True)
        
        print("\nIntrinsic LLM PCA analysis finished.")
        print(f"Generated PCA scores for {len(entity_scores)} entities")
        
        return entity_scores
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
