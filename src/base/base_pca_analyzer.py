"""Shared PCA utilities used across benchmark and model analyses."""

import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from src.base.ppca import PPCA
from factor_analyzer import Rotator
import pickle
from pathlib import Path
from src.utils.country_name_standardizer import CountryNameStandardizer


class BasePCAAnalyzer(ABC):
    """Base class for PCA-based cultural-coordinate analyses."""
    
    def __init__(self, data_path: str = "../data", ivs_data_subdir: str = None):
        """Initialize shared paths, item lists, and PCA state."""
        self.data_path = Path(data_path)
        self.ivs_data_subdir = ivs_data_subdir
        
        self.iv_qns = ["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006", "Y002", "Y003"]
        self.meta_col = ["S020", "S003"]
        self.weights = ["S017"]
        
        self.pc_rescale_params = {
            'PC1': (1.81, 0.38), 
            'PC2': (1.61, -0.01)
        }
        
        self.ivs_df = None
        self.country_codes = None
        self.combined_data = None
        self.pca_results = None
        
        self.name_standardizer = CountryNameStandardizer()
    
    @staticmethod
    def clean_country_code(code: Any) -> str:
        """Normalize country-code values into stable string keys."""
        if pd.isna(code):
            return str(code)
        
        try:
            num_value = float(str(code))
            if num_value.is_integer():
                return str(int(num_value))
            return str(num_value)
        except (ValueError, TypeError):
            return str(code).strip()
    
    def _merge_metadata_helper(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and merge country metadata in one step."""
        df = self.prepare_country_codes_for_merge(df)
        df = self.merge_country_metadata(df, on_column='country_code_clean')
        return df
    
    def prepare_country_codes_for_merge(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a normalized country-code column for merge operations."""
        if 'country_code' in df.columns:
            df = df.copy()
            df['country_code_clean'] = df['country_code'].apply(self.clean_country_code)
        return df
    
    def merge_country_metadata(self, df: pd.DataFrame, 
                              on_column: str = 'country_code_clean',
                              keep_original: bool = True) -> pd.DataFrame:
        """Merge country metadata onto an analysis dataframe."""
        if self.country_codes is None:
            print("country_codes not loaded; skipping metadata merge")
            return df
        
        df_merged = df.copy()
        
        original_cultural_region = None
        if 'Cultural Region' in df_merged.columns:
            original_cultural_region = df_merged['Cultural Region'].copy()
        
        # Primary strategy: merge by numeric country code.
        if on_column in df_merged.columns and 'Numeric' in self.country_codes.columns:
            df_merged['_merge_key'] = pd.to_numeric(df_merged['country_code'], errors='coerce')
            self.country_codes['_Numeric_float'] = self.country_codes['Numeric'].astype(float)
            
            df_merged = df_merged.merge(
                self.country_codes, 
                left_on='_merge_key', 
                right_on='_Numeric_float', 
                how='left',
                suffixes=('', '_from_codes')
            )
            
            for tmp_col in ['_merge_key', '_Numeric_float']:
                if tmp_col in df_merged.columns:
                    df_merged = df_merged.drop(columns=[tmp_col])
                if tmp_col in self.country_codes.columns:
                    self.country_codes = self.country_codes.drop(columns=[tmp_col])
            
            if 'Country' in df_merged.columns:
                match_rate = df_merged['Country'].notna().sum() / len(df_merged)
                print(f"Numeric-code match rate: {match_rate:.1%}")
            
                # Fallback strategy: merge by country name if numeric matching is weak.
                if match_rate < 0.5:
                    print("Numeric matching is weak; trying country-name matching...")
                    existing_numeric = df_merged.get('Numeric', pd.Series(dtype=float)).copy()
                    
                    df_merged = df.merge(
                        self.country_codes, 
                        left_on=on_column, 
                        right_on='Country', 
                        how='left',
                        suffixes=('', '_from_codes')
                    )
                    
                    if 'Numeric' in df_merged.columns and len(existing_numeric) == len(df_merged):
                        nan_mask = df_merged['Numeric'].isna() & existing_numeric.notna()
                        df_merged.loc[nan_mask, 'Numeric'] = existing_numeric[nan_mask]
                    
                    match_rate = df_merged['Country'].notna().sum() / len(df_merged)
                    print(f"Country-name match rate: {match_rate:.1%}")
        
        if 'Cultural Region_from_codes' in df_merged.columns:
            match_mask = df_merged['Country'].notna()
            if match_mask.any():
                df_merged.loc[match_mask, 'Cultural Region'] = df_merged.loc[match_mask, 'Cultural Region_from_codes']
                print(f"Filled Cultural Region for {match_mask.sum()} matched country rows")
            
            df_merged = df_merged.drop(columns=['Cultural Region_from_codes'])
        
        if original_cultural_region is not None:
            unmatch_mask = df_merged['Country'].isna()
            if unmatch_mask.any():
                df_merged.loc[unmatch_mask, 'Cultural Region'] = original_cultural_region[unmatch_mask]
                print(f"Retained original Cultural Region for {unmatch_mask.sum()} non-country rows")
        
        if keep_original and 'country_code' in df.columns:
            df_merged['country_code_original'] = df['country_code']
        
        if 'Country' in df_merged.columns:
            df_merged['Country'] = df_merged['Country'].apply(
                lambda x: self.name_standardizer.standardize(x) if pd.notna(x) else x
            )
            print("Standardized country names using project mappings")
        
        return df_merged
    
    def load_base_data(self) -> bool:
        """Load IVS benchmark data and country metadata."""
        try:
            if self.ivs_data_subdir:
                base_path = self.data_path / self.ivs_data_subdir
                print(f"Loading IVS benchmark from subdirectory: {self.ivs_data_subdir}")
            else:
                base_path = self.data_path
            
            ivs_path = base_path / "ivs_df.pkl"
            if ivs_path.exists():
                self.ivs_df = pd.read_pickle(ivs_path)
                print(f"Loaded IVS dataframe: {ivs_path}")
                print(f"   shape: {self.ivs_df.shape}")
            else:
                print(f"IVS dataframe not found: {ivs_path}")
                return False
            
            config_country_path = Path(__file__).parent.parent.parent / "config" / "country" / "country_codes.pkl"
            if config_country_path.exists():
                self.country_codes = pd.read_pickle(config_country_path)
                print(f"Loaded country metadata: {config_country_path}")
                print(f"   countries: {len(self.country_codes)}")
            else:
                country_codes_path = base_path / "country_codes.pkl"
                if country_codes_path.exists():
                    self.country_codes = pd.read_pickle(country_codes_path)
                    print(f"Loaded country metadata: {country_codes_path}")
                    print(f"   countries: {len(self.country_codes)}")
                else:
                    print(f"Country metadata file not found: {config_country_path}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Failed to load base data: {e}")
            return False
    
    def prepare_ivs_data(self) -> pd.DataFrame:
        """Prepare and filter the IVS benchmark table."""
        if self.ivs_df is None:
            raise ValueError("Load IVS data before calling prepare_ivs_data()")
        
        all_columns = self.meta_col + self.weights + self.iv_qns
        subset_ivs_df = self.ivs_df[all_columns].copy()
        
        subset_ivs_df = subset_ivs_df.rename(columns={
            'S020': 'year', 
            'S003': 'country_code', 
            'S017': 'weight'
        })
        
        subset_ivs_df = subset_ivs_df[subset_ivs_df['year'] >= 2005]
        
        subset_ivs_df = subset_ivs_df.dropna(subset=self.iv_qns, thresh=6)
        
        return subset_ivs_df
    
    @abstractmethod
    def load_additional_data(self) -> pd.DataFrame:
        """Load stage-specific additional data."""
        pass
    
    @abstractmethod
    def combine_data(self) -> pd.DataFrame:
        """Combine IVS benchmark data with stage-specific data."""
        pass
    
    def perform_pca_analysis(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Run the shared PCA workflow for the provided dataframe."""
        if data is None:
            if self.combined_data is None:
                raise ValueError("Provide data or run combine_data() first")
            data = self.combined_data
        
        print("\n" + "="*60)
        print("Running PCA analysis...")
        print("="*60)
        
        np.random.seed(42)
        
        data_for_pca = data[self.iv_qns].to_numpy()
        
        valid_rows = ~np.isnan(data_for_pca).all(axis=1)
        if not valid_rows.any():
            raise ValueError("No valid rows are available for PCA analysis")
        
        print(f"Valid rows for PCA: {valid_rows.sum()}/{len(valid_rows)}")
        
        min_samples_required = max(3, data_for_pca.shape[1] // 10)
        if valid_rows.sum() < min_samples_required:
            print(f"Insufficient rows for stable PCA: {valid_rows.sum()} < {min_samples_required}")
            print("   Consider increasing repeat_count or adding more language-country pairs.")
            return pd.DataFrame(columns=['PC1', 'PC2', 'country_code', 'country_name', 'data_source'])
        
        ppca_df = self._run_ppca_and_rotation(data_for_pca)
        
        ppca_df = self._add_metadata_to_pca_results(ppca_df, data)
        
        ppca_df = self._merge_metadata_helper(ppca_df)
        
        self.pca_results = ppca_df.dropna(subset=['PC1_rescaled', 'PC2_rescaled'])
        
        print("\nPCA result dataframe:")
        print(f"   shape: {self.pca_results.shape}")
        print(f"   columns: {list(self.pca_results.columns)}")
        print(f"PCA analysis completed with {len(self.pca_results)} valid observations\n")
        
        return self.pca_results
    
    def _run_ppca_and_rotation(self, data_for_pca: np.ndarray) -> pd.DataFrame:
        """Run PPCA followed by varimax rotation."""
        print("Running PPCA...")
        ppca = PPCA()
        ppca.fit(data_for_pca, d=2, min_obs=1, verbose=True)
        
        self.ppca_model = ppca
        self.loadings = ppca.C
        print(f"Loadings shape: {self.loadings.shape}")
        
        principal_components = ppca.transform()
        
        print("Applying varimax rotation...")
        rotator = Rotator(method='varimax')
        rotated_components = rotator.fit_transform(principal_components)
        
        self.rotated_loadings = self.loadings @ rotator.rotation_
        self.rotation_matrix = rotator.rotation_
        print(f"Rotated loadings shape: {self.rotated_loadings.shape}")
        
        ppca_df = pd.DataFrame(rotated_components, columns=["PC1", "PC2"])
        
        ppca_df['PC1_rescaled'] = (
            self.pc_rescale_params['PC1'][0] * ppca_df['PC1'] + 
            self.pc_rescale_params['PC1'][1]
        )
        ppca_df['PC2_rescaled'] = (
            self.pc_rescale_params['PC2'][0] * ppca_df['PC2'] + 
            self.pc_rescale_params['PC2'][1]
        )
        
        return ppca_df
    
    def save_pca_model(self, save_path: Path = None):
        """Save the fitted PCA model for reuse in downstream stages."""
        if save_path is None:
            save_path = Path('data/country_values/pca_model_fixed.pkl')
        
        if not hasattr(self, 'ppca_model') or self.ppca_model is None:
            raise RuntimeError("Run PCA analysis before saving the model")
        
        pca_model = {
            'ppca_C': self.ppca_model.C,
            'ppca_means': self.ppca_model.means,
            'ppca_stds': self.ppca_model.stds,
            'rotation_matrix': self.rotation_matrix,
            'rotated_loadings': self.rotated_loadings,
            'pc_rescale_params': self.pc_rescale_params,
            'question_ids': self.iv_qns,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(pca_model, f)
        
        print(f"Saved PCA model: {save_path}")
        return save_path
    
    def load_pca_model(self, model_path: Path = None) -> Dict:
        """Load a previously saved PCA model."""
        if model_path is None:
            model_path = Path('data/country_values/pca_model_fixed.pkl')
        
        if not model_path.exists():
            raise FileNotFoundError(f"PCA model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            pca_model = pickle.load(f)
        
        print(f"Loaded PCA model: {model_path}")
        return pca_model
    
    def transform_with_fixed_pca(self, data: pd.DataFrame, pca_model: Dict = None) -> pd.DataFrame:
        """Project new data with a fixed PCA model instead of refitting."""
        if pca_model is None:
            pca_model = self.load_pca_model()
        
        ppca_C = pca_model['ppca_C']
        ppca_means = pca_model['ppca_means']
        ppca_stds = pca_model['ppca_stds']
        rotation_matrix = pca_model['rotation_matrix']
        pc_rescale_params = pca_model['pc_rescale_params']
        
        data_for_pca = data[self.iv_qns].to_numpy()
        
        # Match the fit-time preprocessing used by PPCA.
        data_standardized = (data_for_pca - ppca_means) / ppca_stds
        data_standardized = np.nan_to_num(data_standardized, nan=0.0)
        
        principal_components = np.dot(data_standardized, ppca_C)
        
        rotated_components = np.dot(principal_components, rotation_matrix)
        
        result_df = pd.DataFrame(rotated_components, columns=["PC1", "PC2"])
        
        result_df['PC1_rescaled'] = (
            pc_rescale_params['PC1'][0] * result_df['PC1'] + 
            pc_rescale_params['PC1'][1]
        )
        result_df['PC2_rescaled'] = (
            pc_rescale_params['PC2'][0] * result_df['PC2'] + 
            pc_rescale_params['PC2'][1]
        )
        
        print(f"Projected {len(result_df)} rows with the fixed PCA model")
        return result_df
    
    def _save_loadings_and_rotation(self):
        """Save loadings and rotation matrices for inspection."""
        try:
            loadings_file = self.data_path / "pca_loadings.pkl"
            with open(loadings_file, 'wb') as f:
                pickle.dump({
                    'loadings': self.loadings,
                    'rotated_loadings': self.rotated_loadings,
                    'rotation_matrix': self.rotation_matrix,
                    'question_ids': self.iv_qns
                }, f)
            print(f"Saved loadings matrix: {loadings_file}")
            
            loadings_df = pd.DataFrame(
                self.rotated_loadings,
                index=self.iv_qns,
                columns=['PC1_loading', 'PC2_loading']
            )
            
            csv_file = self.data_path / "pca_loadings.csv"
            loadings_df.to_csv(csv_file)
            print(f"Saved loadings CSV: {csv_file}")
            
            print("\nRotated loadings:")
            print(loadings_df.to_string())
            print("\nInterpretation:")
            print("  * Larger absolute loadings indicate stronger influence on that component.")
            print("  * Positive loadings imply higher item values increase the component score.")
            print("  * Negative loadings imply higher item values decrease the component score.")
            
        except Exception as e:
            print(f"Could not save loadings and rotation matrices: {e}")
    
    def _add_metadata_to_pca_results(self, ppca_df: pd.DataFrame, 
                                    source_data: pd.DataFrame) -> pd.DataFrame:
        """Attach source metadata to PCA coordinates."""
        ppca_df["country_code"] = source_data["country_code"].values
        if "year" in source_data.columns:
            ppca_df["year"] = source_data["year"].values
        
        metadata_columns = ["data_source", "model_name", "language"]
        for col in metadata_columns:
            if col in source_data.columns:
                ppca_df[col] = source_data[col].values
        
        if "Cultural Region" in source_data.columns:
            ppca_df["Cultural Region"] = source_data["Cultural Region"].values
        elif "cultural_region" in source_data.columns:
            ppca_df["Cultural Region"] = source_data["cultural_region"].values
        
        return ppca_df
    
    def calculate_entity_scores(self, group_by: List[str] = None) -> pd.DataFrame:
        """Aggregate PCA coordinates into entity-level scores."""
        if self.pca_results is None:
            raise ValueError("Run PCA analysis before calculating entity scores")
        
        if group_by is None:
            group_by = ['country_code']
        
        print("\n" + "="*60)
        print("Calculating entity scores...")
        print("="*60)
        
        group_by = self._prepare_grouping_columns(group_by)
        
        entity_scores = self._aggregate_by_data_source(group_by)
        
        entity_scores = self._merge_metadata_helper(entity_scores)
        
        print(f"\nCalculated scores for {len(entity_scores)} entities")
        return entity_scores
    
    def _prepare_grouping_columns(self, base_group_by: List[str]) -> List[str]:
        """Add data_source to grouping columns when available."""
        group_by = base_group_by.copy()
        
        if 'data_source' in self.pca_results.columns and 'data_source' not in group_by:
            group_by.append('data_source')
        
        return group_by
    
    def _aggregate_by_data_source(self, group_by: List[str]) -> pd.DataFrame:
        """Aggregate IVS and non-IVS rows with source-aware grouping."""
        agg_dict = {
            'PC1_rescaled': 'mean',
            'PC2_rescaled': 'mean'
        }
        
        if 'data_source' not in self.pca_results.columns:
            return self.pca_results.groupby(group_by).agg(agg_dict).reset_index()
        
        ivs_data = self.pca_results[self.pca_results['data_source'] == 'IVS']
        non_ivs_data = self.pca_results[self.pca_results['data_source'] != 'IVS']
        
        entity_scores_list = []
        
        if len(ivs_data) > 0:
            print(f"Aggregating IVS rows: {len(ivs_data)}")
            ivs_scores = self._aggregate_single_source(ivs_data, group_by, agg_dict)
            entity_scores_list.append(ivs_scores)
        
        if len(non_ivs_data) > 0:
            print(f"Aggregating non-IVS rows: {len(non_ivs_data)}")
            non_ivs_scores = self._aggregate_llm_data(non_ivs_data, group_by, agg_dict)
            entity_scores_list.append(non_ivs_scores)
        
        if entity_scores_list:
            entity_scores = pd.concat(entity_scores_list, ignore_index=True)
            print(f"Aggregation complete: {entity_scores.shape}")
            return entity_scores
        else:
            return pd.DataFrame()
    
    def _aggregate_single_source(self, data: pd.DataFrame, 
                                 group_by: List[str], 
                                 agg_dict: Dict[str, str]) -> pd.DataFrame:
        """Aggregate one source dataframe under the requested grouping."""
        valid_group_by = [col for col in group_by if col in data.columns]
        valid_agg_dict = {k: v for k, v in agg_dict.items() if k in data.columns}
        
        for col in ['Cultural Region', 'cultural_region']:
            if col in data.columns and col not in valid_group_by:
                valid_agg_dict[col] = 'first'
        
        return data.groupby(valid_group_by).agg(valid_agg_dict).reset_index()
    
    def _aggregate_llm_data(self, data: pd.DataFrame, 
                           group_by: List[str], 
                           agg_dict: Dict[str, str]) -> pd.DataFrame:
        """Aggregate LLM rows while preserving model and language identity."""
        llm_group_by = group_by.copy()
        
        if 'model_name' in data.columns and 'model_name' not in llm_group_by:
            llm_group_by.append('model_name')
            print("   Added model_name to grouping columns")
        
        if 'language' in data.columns and 'language' not in llm_group_by:
            llm_group_by.append('language')
            print("   Added language to grouping columns")
        
        print(f"   Grouping columns: {llm_group_by}")
        
        return self._aggregate_single_source(data, llm_group_by, agg_dict)
    
    def save_results(self, entity_scores: pd.DataFrame = None, prefix: str = "pca"):
        """Save PCA results and aggregated entity scores."""
        if self.pca_results is not None:
            pca_path = self.data_path / f"{prefix}_results.pkl"
            self.pca_results.to_pickle(pca_path)
            print(f"Saved PCA results to: {pca_path}")
        
        if entity_scores is not None:
            scores_path = self.data_path / f"{prefix}_entity_scores.pkl"
            entity_scores.to_pickle(scores_path)
            print(f"Saved entity scores to: {scores_path}")
    
    def print_summary(self, entity_scores: pd.DataFrame = None):
        """Print a compact summary of the PCA output."""
        print("\n" + "="*50)
        print("PCA analysis summary")
        print("="*50)
        
        if self.pca_results is not None:
            print(f"Total observations: {len(self.pca_results)}")
            print(f"PC1 range: [{self.pca_results['PC1_rescaled'].min():.2f}, {self.pca_results['PC1_rescaled'].max():.2f}]")
            print(f"PC2 range: [{self.pca_results['PC2_rescaled'].min():.2f}, {self.pca_results['PC2_rescaled'].max():.2f}]")
        
        if entity_scores is not None:
            print(f"Entities: {len(entity_scores)}")
            
            if 'data_source' in entity_scores.columns:
                print("\nData-source distribution:")
                print(entity_scores['data_source'].value_counts())
            
            if 'Cultural Region' in entity_scores.columns:
                print("\nCultural-region distribution:")
                print(entity_scores['Cultural Region'].value_counts())
    
    def run_full_analysis(self) -> pd.DataFrame:
        """Run the complete PCA workflow from loading through aggregation."""
        print("Starting full PCA workflow...")
        
        if not self.load_base_data():
            raise ValueError("Failed to load base data")
        
        additional_data = self.load_additional_data()
        
        combined_data = self.combine_data()
        
        pca_results = self.perform_pca_analysis(combined_data)
        
        entity_scores = self.calculate_entity_scores()
        
        self.save_results(entity_scores)
        
        self.print_summary(entity_scores)
        
        print("Full PCA workflow completed")
        return entity_scores
