"""PCA projection and comparison utilities for multilingual roleplay data."""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.base.ppca import PPCA
except ImportError:
    print("Warning: PPCA could not be imported; falling back to standard PCA where needed")
    PPCA = None

from src.base.base_pca_analyzer import BasePCAAnalyzer


class MultilingualRoleplayPCAAnalysis(BasePCAAnalyzer):
    """PCA analyzer for multilingual roleplay outputs."""
    
    def __init__(self, data_path: str = "data"):
        super().__init__(data_path=data_path, ivs_data_subdir="country_values")

        self.processed_dir = self.data_path / "llm_interviews" / "multilingual" / "processed"

        self.cultural_regions = self._load_cultural_regions()
        self.country_mapping = self._load_country_mapping()
    
    def _load_cultural_regions(self) -> Dict:
        """Load the cultural-region configuration table."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'country' / 'cultural_regions.json'
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Could not load cultural-region configuration: {e}")
            return {}
    
    def _load_country_mapping(self) -> Dict:
        """Load the numeric country-code mapping used by the benchmark tables."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'country' / 'country_codes.json'
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                mapping = {int(item['Numeric']): item['Country'] for item in data}
                print(f"Loaded country-code mapping for {len(mapping)} countries")
                return mapping
        except Exception as e:
            print(f"Could not load country-code mapping: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def load_additional_data(self) -> pd.DataFrame:
        """Load the processed multilingual roleplay table required by the base class."""
        return self.load_processed_data()
    
    def load_processed_data(self, data_file: str = None) -> pd.DataFrame:
        """Load the processed IVS-aligned multilingual roleplay table."""
        if data_file is None:
            ivs_files = list(self.processed_dir.glob("llm_roleplay_ml_processed_responses_ivs_format_*.pkl"))
            if not ivs_files:
                ivs_files = list(self.processed_dir.glob("multilingual_roleplay_ivs_format_*.pkl"))
                if not ivs_files:
                    raise FileNotFoundError(
                        f"Multilingual IVS-format data file not found.\n"
                        f"Search path: {self.processed_dir}\n"
                        f"Expected pattern: llm_roleplay_ml_processed_responses_ivs_format_*.pkl"
                    )
                print("Using a legacy-format processed roleplay file")
            data_file = max(ivs_files, key=lambda x: x.stat().st_mtime)
        else:
            data_file = Path(data_file)
        
        print(f"Loading processed data file: {data_file}")
        
        with open(data_file, 'rb') as f:
            return pickle.load(f)

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

            print(f"Loaded released benchmark coordinates: {candidate}")
            print(f"   - Countries: {len(released)}")
            return released

        return pd.DataFrame()
    
    def combine_data(self) -> pd.DataFrame:
        """Combine IVS benchmark data with filtered multilingual roleplay responses."""
        ivs_data = self.prepare_ivs_data()

        print(f"IVS country_code dtype: {ivs_data['country_code'].dtype}")
        ivs_data['country_code'] = ivs_data['country_code'].astype(float)

        print(f"IVS countries represented: {ivs_data['country_code'].nunique()}")
        
        ivs_data['data_source'] = 'IVS'
        
        multilingual_data = self.load_additional_data()
        
        if multilingual_data.empty:
            print("Multilingual roleplay data is empty; using IVS data only")
            return ivs_data
        
        try:
            multilingual_data_copy = multilingual_data.copy()
            multilingual_data_copy['data_source'] = 'Multilingual'
            
            print(f"Multilingual country_code dtype: {multilingual_data_copy['country_code'].dtype}")
            multilingual_data_copy['country_code'] = multilingual_data_copy['country_code'].astype(float)
            
            print("Applying benchmark-quality filters to multilingual data...")
            print(f"   Rows before filtering: {len(multilingual_data_copy)}")
            
            multilingual_data_copy = multilingual_data_copy.dropna(subset=self.iv_qns, thresh=6)
            print(f"   Rows after thresh=6 filter: {len(multilingual_data_copy)}")
            
            valid_ivs_mask = multilingual_data_copy[self.iv_qns].notna().any(axis=1)
            multilingual_data_copy = multilingual_data_copy[valid_ivs_mask]
            print(f"   Rows after removing all-NaN items: {len(multilingual_data_copy)}")
            
            from src.base.ivs_question_processor import IVSQuestionProcessor
            
            for col in self.iv_qns:
                if col in multilingual_data_copy.columns and col in IVSQuestionProcessor.QUESTION_CONFIG:
                    config = IVSQuestionProcessor.QUESTION_CONFIG[col]
                    if config["type"] == "single":
                        min_val, max_val = config["scale"]
                        mask = (multilingual_data_copy[col] >= min_val) & (multilingual_data_copy[col] <= max_val)
                        multilingual_data_copy.loc[~mask, col] = np.nan
            
            multilingual_data_copy = multilingual_data_copy.dropna(subset=self.iv_qns, thresh=6)
            print(f"   Final valid rows: {len(multilingual_data_copy)}")
            
            if len(multilingual_data_copy) == 0:
                print("No valid multilingual rows remain after filtering; using IVS data only")
                return ivs_data
            
            if 'country_code' in ivs_data.columns and 'country' not in ivs_data.columns:
                ivs_data['country'] = ivs_data['country_code']
            
            if 'country_code' in multilingual_data_copy.columns and 'country' not in multilingual_data_copy.columns:
                multilingual_data_copy['country'] = multilingual_data_copy['country_code']
            
            all_columns = list(set(ivs_data.columns) | set(multilingual_data_copy.columns))
            
            ivs_aligned = ivs_data.reindex(columns=all_columns)
            multilingual_aligned = multilingual_data_copy.reindex(columns=all_columns)
            
            combined_data = pd.concat([
                ivs_aligned, 
                multilingual_aligned
            ], ignore_index=True)
            
            print("Data combination completed:")
            print(f"   - Total rows: {len(combined_data)}")
            print(f"   - IVS rows: {len(ivs_data)}")
            print(f"   - Multilingual rows: {len(multilingual_data_copy)}")
            
            numeric_cols = [col for col in self.iv_qns if col in combined_data.columns]
            if numeric_cols:
                valid_data_per_col = combined_data[numeric_cols].notna().sum()
                print(f"   Non-missing counts by item: {dict(valid_data_per_col)}")
            
            return combined_data
            
        except Exception as e:
            print(f"Data combination failed: {e}")
            return multilingual_data.copy()
    
    def analyze_language_effects(self, results_df: pd.DataFrame) -> Dict:
        """Summarize average coordinate differences across languages."""
        print("\nAnalyzing language effects...")

        if 'language' not in results_df.columns:
            print("No language column is present; skipping language-effect analysis")
            return {'language_statistics': {}, 'language_differences': {}}
        
        language_analysis = {}
        
        pc1_col = 'PC1_rescaled' if 'PC1_rescaled' in results_df.columns else 'PC1'
        pc2_col = 'PC2_rescaled' if 'PC2_rescaled' in results_df.columns else 'PC2'
        
        for language in results_df['language'].unique():
            lang_data = results_df[results_df['language'] == language]
            
            if len(lang_data) > 0:
                language_analysis[language] = {
                    'count': len(lang_data),
                    'countries': lang_data['country'].unique().tolist() if 'country' in lang_data.columns else [],
                    'pc1_mean': lang_data[pc1_col].mean() if not lang_data[pc1_col].isna().all() else np.nan,
                    'pc1_std': lang_data[pc1_col].std() if not lang_data[pc1_col].isna().all() else np.nan,
                    'pc2_mean': lang_data[pc2_col].mean() if not lang_data[pc2_col].isna().all() else np.nan,
                    'pc2_std': lang_data[pc2_col].std() if not lang_data[pc2_col].isna().all() else np.nan,
                }
        
        language_differences = {}
        languages = list(language_analysis.keys())
        
        for i, lang1 in enumerate(languages):
            for lang2 in languages[i+1:]:
                if (not np.isnan(language_analysis[lang1]['pc1_mean']) and 
                    not np.isnan(language_analysis[lang2]['pc1_mean'])):
                    
                    pc1_diff = abs(language_analysis[lang1]['pc1_mean'] - language_analysis[lang2]['pc1_mean'])
                    pc2_diff = abs(language_analysis[lang1]['pc2_mean'] - language_analysis[lang2]['pc2_mean'])
                    
                    language_differences[f"{lang1}_vs_{lang2}"] = {
                        'pc1_difference': pc1_diff,
                        'pc2_difference': pc2_diff,
                        'euclidean_distance': np.sqrt(pc1_diff**2 + pc2_diff**2)
                    }
        
        return {
            'language_statistics': language_analysis,
            'language_differences': language_differences
        }
    
    def save_results(self, entity_scores: pd.DataFrame = None, prefix: str = "roleplay_ml_pca"):
        """Save PCA outputs using the file layout expected by the paper pipeline."""
        print(f"\n{'='*60}")
        print("Saving multilingual roleplay PCA outputs")
        print(f"{'='*60}")
        
        save_dir = self.data_path / "llm_pca" / "multilingual"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.pca_results is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = save_dir / f'{prefix}_results_{timestamp}.pkl'
            self.pca_results.to_pickle(results_file)
            print(f"Saved PCA results: {results_file.name}")
            print(f"   - Rows: {len(self.pca_results)}")
            
            results_file_latest = save_dir / f'{prefix}_results_latest.pkl'
            self.pca_results.to_pickle(results_file_latest)
            print(f"Saved latest PCA results: {results_file_latest.name}")
        
        if entity_scores is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scores_file = save_dir / f'{prefix}_entity_scores_{timestamp}.pkl'
            entity_scores.to_pickle(scores_file)
            print(f"Saved entity scores: {scores_file.name}")
            print(f"   - Rows: {len(entity_scores)}")
            
            scores_file_latest = save_dir / f'{prefix}_entity_scores_latest.pkl'
            entity_scores.to_pickle(scores_file_latest)
            print(f"Saved latest entity scores: {scores_file_latest.name}")
            
            json_file = save_dir / f'{prefix}_entity_scores_latest.json'
            entity_scores.to_json(json_file, orient='records', indent=2, force_ascii=False)
            print(f"Saved JSON export: {json_file.name}")
            
            csv_file = save_dir / f'{prefix}_entity_scores_latest.csv'
            entity_scores.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"Saved CSV export: {csv_file.name}")
            
            if 'data_source' in entity_scores.columns:
                print("\nEntity-score summary:")
                print(f"   - IVS rows: {len(entity_scores[entity_scores['data_source'] == 'IVS'])}")
                print(f"   - Multilingual rows: {len(entity_scores[entity_scores['data_source'] == 'Multilingual'])}")
        
        print(f"\nOutput directory: {save_dir}")
        print(f"{'='*60}")
    
    def run_multilingual_analysis_for_runner(self, use_fixed_pca: bool = True) -> pd.DataFrame:
        """Run the multilingual roleplay PCA pipeline and return the entity-score table."""
        print("Starting multilingual roleplay PCA analysis...")
        
        if use_fixed_pca:
            entity_scores = self.run_analysis_with_fixed_pca()
        else:
            print("Warning: refitting PCA may change the coordinate system relative to the benchmark")
            entity_scores = super().run_full_analysis()
        
        if hasattr(self, 'pca_results') and self.pca_results is not None and 'language' in self.pca_results.columns:
            print("\nAnalyzing language effects...")
            language_analysis = self.analyze_language_effects(self.pca_results)
            
            if language_analysis['language_statistics']:
                print("\nLanguage-wise mean PC1 values (rescaled):")
                for lang, stats in language_analysis['language_statistics'].items():
                    if not np.isnan(stats['pc1_mean']):
                        print(f"   {lang}: {stats['pc1_mean']:.3f} (n={stats['count']})")
            
            analysis_file = self.processed_dir / 'roleplay_ml_language_analysis_latest.json'
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(language_analysis, f, ensure_ascii=False, indent=2, default=str)
            print(f"Saved language-analysis summary: {analysis_file.name}")
        
        print(f"Multilingual PCA analysis completed: {len(entity_scores)} entities")
        return entity_scores
    
    def run_analysis_with_fixed_pca(self) -> pd.DataFrame:
        """Project IVS and roleplay data into the fixed benchmark PCA space."""
        print("\n" + "="*60)
        print("Running projection with the fixed benchmark PCA model")
        print("="*60)
        
        pca_model_path = Path('data/country_values/pca_model_fixed.pkl')
        if not pca_model_path.exists():
            raise FileNotFoundError(
                f"Fixed PCA model not found: {pca_model_path}\n"
                "Run the benchmark PCA pipeline first to generate it."
            )
        
        pca_model = self.load_pca_model(pca_model_path)
        
        print("\nLoading IVS benchmark data...")
        ivs_pca = pd.DataFrame()
        if self.load_base_data():
            ivs_data = self.prepare_ivs_data()
            ivs_data['data_source'] = 'IVS'
            print(f"   IVS rows: {len(ivs_data)}")

            print("\nApplying the fixed PCA model to IVS data...")
            ivs_pca = self.transform_with_fixed_pca(ivs_data, pca_model)
            ivs_pca['country_code'] = ivs_data['country_code'].values
            ivs_pca['data_source'] = 'IVS'
            if 'year' in ivs_data.columns:
                ivs_pca['year'] = ivs_data['year'].values
        else:
            ivs_pca = self._load_released_country_scores()
            if ivs_pca.empty:
                raise ValueError("Failed to load the IVS benchmark data")
        
        print("\nLoading multilingual roleplay data...")
        multilingual_data = self.load_processed_data()
        multilingual_data['data_source'] = 'Multilingual'
        print(f"   Multilingual rows: {len(multilingual_data)}")
        
        print("\nApplying the fixed PCA model to multilingual roleplay data...")
        ml_pca = self.transform_with_fixed_pca(multilingual_data, pca_model)
        
        ml_pca['country_code'] = multilingual_data['country_code'].values
        ml_pca['data_source'] = 'Multilingual'
        
        for col in ['model_name', 'language', 'Country']:
            if col in multilingual_data.columns:
                ml_pca[col] = multilingual_data[col].values
        
        print("\nCombining projected PCA results...")
        self.pca_results = pd.concat([ivs_pca, ml_pca], ignore_index=True)
        print(f"   Total combined rows: {len(self.pca_results)}")
        
        self.pca_results = self.prepare_country_codes_for_merge(self.pca_results)
        self.pca_results = self.merge_country_metadata(self.pca_results, on_column='country_code_clean')
        
        print("\nCalculating entity-level scores...")
        entity_scores = self.calculate_entity_scores()
        
        self.save_results(entity_scores)
        
        self.print_summary(entity_scores)
        
        return entity_scores


class LanguageComparisonAnalyzer:
    """Compare English and native-language roleplay performance."""
    
    def __init__(self):
        """Initialize helpers for distance-based language comparisons."""
        from scipy.spatial.distance import euclidean
        self.euclidean = euclidean
    
    @staticmethod
    def normalize_country_name(name):
        """Normalize country names across IVS and roleplay outputs."""
        if pd.isna(name):
            return None
        name = str(name).strip()
        
        # Handle naming differences between benchmark and model-output files.
        name_mapping = {
            'Russian Federation': 'Russia',
            'Korea, Republic of': 'Korea',
            'Taiwan, Province of China': 'Taiwan',
            'United States of America': 'United States',
            'Viet Nam': 'Vietnam',
        }
        
        if name in name_mapping:
            name = name_mapping[name]
        
        # Remove common suffixes used in survey naming conventions.
        name = name.replace(' (the)', '').replace('(the)', '')
        name = name.replace(' (Islamic Republic of)', '').replace('(Islamic Republic of)', '')
        name = name.replace(' (Bolivarian Republic of)', '').replace('(Bolivarian Republic of)', '')
        
        return name.strip()
    
    def build_country_coords_dict(self, real_countries: pd.DataFrame):
        """Build a lookup table of benchmark country coordinates.
        
        Args:
            real_countries: IVS benchmark country data.
            
        Returns:
            Tuple[Dict, Dict]: Coordinate lookup and normalized-name mapping.
        """
        real_country_avg = real_countries.groupby('Country')[['PC1_rescaled', 'PC2_rescaled']].mean()
        
        real_country_coords = {}
        real_country_name_mapping = {}
        for country_name, row in real_country_avg.iterrows():
            if pd.notna(country_name) and country_name != 'Unknown':
                normalized_name = self.normalize_country_name(country_name)
                if normalized_name:
                    real_country_coords[normalized_name] = (row['PC1_rescaled'], row['PC2_rescaled'])
                    real_country_name_mapping[normalized_name] = country_name
        
        return real_country_coords, real_country_name_mapping
    
    def calculate_en_native_baseline(self, en_native_data: pd.DataFrame, real_country_coords: Dict):
        """Compute the English-native baseline from anglophone entities.
        
        Args:
            en_native_data: Roleplay outputs for English-native entities.
            real_country_coords: Benchmark country coordinate lookup.
            
        Returns:
            Tuple[Dict, List]: Baseline summary and all observed distances.
        """
        en_native_countries_raw = en_native_data['country_code'].dropna().unique()
        en_native_countries = set(self.normalize_country_name(name) for name in en_native_countries_raw if self.normalize_country_name(name))
        en_native_baseline = {}
        all_en_native_distances = []
        
        for country_name_normalized in en_native_countries:
            if country_name_normalized in real_country_coords:
                print(f"   Matched en-native country: {country_name_normalized}")
                real_coords = real_country_coords[country_name_normalized]
                country_en_native = en_native_data[
                    en_native_data['country_code'].apply(self.normalize_country_name) == country_name_normalized
                ]
                distances = []
                models_data = {}
                
                for _, row in country_en_native.iterrows():
                    model_coords = (row['PC1_rescaled'], row['PC2_rescaled'])
                    distance = self.euclidean(real_coords, model_coords)
                    distances.append(distance)
                    all_en_native_distances.append(distance)
                    
                    model_name = row.get('model_name', 'Unknown')
                    models_data[model_name] = distance
                
                en_native_baseline[country_name_normalized] = {
                    'avg_distance': np.mean(distances) if distances else None,
                    'models': models_data
                }
        
        print(f"en-native baseline covers {len(en_native_baseline)} anglophone entities")
        if all_en_native_distances:
            print(f"Mean en-native distance: {np.mean(all_en_native_distances):.3f}")
        
        return en_native_baseline, all_en_native_distances
    
    def calculate_country_distances(self, matchable_countries, real_country_coords, 
                                   native_data, english_data, en_native_data):
        """Compute native, English, and en-native distances for each country.
        
        Args:
            matchable_countries: Countries that appear in both datasets.
            real_country_coords: Benchmark country coordinate lookup.
            native_data: Native-language roleplay coordinates.
            english_data: English roleplay coordinates for non-anglophone entities.
            en_native_data: English-native roleplay coordinates.
            
        Returns:
            Dict: Per-country comparison results.
        """
        country_comparisons = {}
        for country_name_normalized in matchable_countries:
            real_coords = real_country_coords[country_name_normalized]
            country_results = {
                'real_coordinates': real_coords,
                'native_distances': [],
                'english_distances': [],
                'en_native_distances': [],
                'models': {}
            }
            
            country_native = native_data[
                native_data['country_code'].apply(self.normalize_country_name) == country_name_normalized
            ]
            for _, row in country_native.iterrows():
                model_coords = (row['PC1_rescaled'], row['PC2_rescaled'])
                distance = self.euclidean(real_coords, model_coords)
                country_results['native_distances'].append(distance)
                
                model_name = row.get('model_name', 'Unknown')
                if model_name not in country_results['models']:
                    country_results['models'][model_name] = {}
                country_results['models'][model_name]['native_distance'] = distance
            
            country_english = english_data[
                english_data['country_code'].apply(self.normalize_country_name) == country_name_normalized
            ]
            for _, row in country_english.iterrows():
                model_coords = (row['PC1_rescaled'], row['PC2_rescaled'])
                distance = self.euclidean(real_coords, model_coords)
                country_results['english_distances'].append(distance)
                
                model_name = row.get('model_name', 'Unknown')
                if model_name not in country_results['models']:
                    country_results['models'][model_name] = {}
                country_results['models'][model_name]['english_distance'] = distance
            
            country_en_native = en_native_data[
                en_native_data['country_code'].apply(self.normalize_country_name) == country_name_normalized
            ]
            for _, row in country_en_native.iterrows():
                model_coords = (row['PC1_rescaled'], row['PC2_rescaled'])
                distance = self.euclidean(real_coords, model_coords)
                country_results['en_native_distances'].append(distance)
                
                model_name = row.get('model_name', 'Unknown')
                if model_name not in country_results['models']:
                    country_results['models'][model_name] = {}
                country_results['models'][model_name]['en_native_distance'] = distance
            
            country_results['avg_native_distance'] = np.mean(country_results['native_distances']) if country_results['native_distances'] else None
            country_results['avg_english_distance'] = np.mean(country_results['english_distances']) if country_results['english_distances'] else None
            country_results['avg_en_native_distance'] = np.mean(country_results['en_native_distances']) if country_results['en_native_distances'] else None
            
            country_comparisons[country_name_normalized] = country_results
        
        return country_comparisons
    
    def analyze_model_specific_language_performance(self, native_data, english_data, 
                                                   real_country_coords, matchable_countries):
        """Summarize language effects at the model level.
        
        Args:
            native_data: Native-language roleplay coordinates.
            english_data: English roleplay coordinates.
            real_country_coords: Benchmark country coordinate lookup.
            matchable_countries: Countries that appear in both datasets.
            
        Returns:
            Dict: Per-model summary statistics.
        """
        model_analysis = {}
        
        all_models = set()
        if 'model_name' in native_data.columns:
            all_models.update(native_data['model_name'].dropna().unique())
        if 'model_name' in english_data.columns:
            all_models.update(english_data['model_name'].dropna().unique())
        
        for model_name in all_models:
            model_stats = {
                'model_name': model_name,
                'native_distances': [],
                'english_distances': [],
                'native_avg_distance': None,
                'english_avg_distance': None,
                'language_improvement': None,
                'countries_analyzed': [],
                'native_count': 0,
                'english_count': 0
            }
            
            for country_name_normalized in matchable_countries:
                real_coords = real_country_coords[country_name_normalized]
                
                model_native = native_data[
                    (native_data['country_code'].apply(self.normalize_country_name) == country_name_normalized) & 
                    (native_data['model_name'] == model_name)
                ]
                
                for _, row in model_native.iterrows():
                    model_coords = (row['PC1_rescaled'], row['PC2_rescaled'])
                    distance = self.euclidean(real_coords, model_coords)
                    model_stats['native_distances'].append(distance)
                    model_stats['native_count'] += 1
                
                model_english = english_data[
                    (english_data['country_code'].apply(self.normalize_country_name) == country_name_normalized) & 
                    (english_data['model_name'] == model_name)
                ]
                
                for _, row in model_english.iterrows():
                    model_coords = (row['PC1_rescaled'], row['PC2_rescaled'])
                    distance = self.euclidean(real_coords, model_coords)
                    model_stats['english_distances'].append(distance)
                    model_stats['english_count'] += 1
                
                if len(model_native) > 0 or len(model_english) > 0:
                    model_stats['countries_analyzed'].append(country_name_normalized)
            
            if model_stats['native_distances']:
                model_stats['native_avg_distance'] = float(np.mean(model_stats['native_distances']))
            if model_stats['english_distances']:
                model_stats['english_avg_distance'] = float(np.mean(model_stats['english_distances']))
            
            if model_stats['native_avg_distance'] and model_stats['english_avg_distance']:
                native_avg = model_stats['native_avg_distance']
                english_avg = model_stats['english_avg_distance']
                improvement = ((native_avg - english_avg) / native_avg) * 100
                model_stats['language_improvement'] = float(improvement)
            
            model_analysis[model_name] = model_stats
        
        return model_analysis
    
    def calculate_language_distance_comparison(self, real_countries, multilingual_native, 
                                              multilingual_english, multilingual_en_native):
        """Compare distances for en-native, English, and native-language prompts.
        
        Args:
            real_countries: IVS benchmark country data.
            multilingual_native: Native-language roleplay coordinates.
            multilingual_english: English roleplay coordinates.
            multilingual_en_native: English-native roleplay coordinates.
            
        Returns:
            Dict: Comparison results.
        """
        print("Computing language-distance comparisons...")
        
        comparison_results = {
            'summary': {},
            'country_details': {},
            'model_performance': {},
            'language_effectiveness': {},
            'model_specific_analysis': {},
            'en_native_baseline': {}
        }
        
        native_data = multilingual_native
        english_data = multilingual_english
        en_native_data = multilingual_en_native
        
        print(f"Native-language rows: {len(native_data)}")
        print(f"English rows for non-anglophone entities: {len(english_data)}")
        print(f"en-native rows: {len(en_native_data)}")
        
        real_country_coords, real_country_name_mapping = self.build_country_coords_dict(real_countries)
        print(f"Benchmark countries available: {len(real_country_coords)}")
        
        all_multilingual_data = pd.concat([native_data, english_data, en_native_data], ignore_index=True)
        multilingual_countries_raw = all_multilingual_data['country_code'].dropna().unique()
        multilingual_countries = set(self.normalize_country_name(name) for name in multilingual_countries_raw if self.normalize_country_name(name))
        print(f"Countries covered by multilingual data: {len(multilingual_countries)}")
        
        matchable_countries = real_country_coords.keys() & multilingual_countries
        print(f"Countries matched across benchmark and roleplay data: {len(matchable_countries)}")
        print(f"Matched countries: {sorted(list(matchable_countries))}")
        
        en_native_baseline, all_en_native_distances = self.calculate_en_native_baseline(en_native_data, real_country_coords)
        
        country_comparisons = self.calculate_country_distances(matchable_countries, real_country_coords, native_data, english_data, en_native_data)
        
        all_native_distances = []
        all_english_distances = []
        
        for country_data in country_comparisons.values():
            if country_data['native_distances']:
                all_native_distances.extend(country_data['native_distances'])
            if country_data['english_distances']:
                all_english_distances.extend(country_data['english_distances'])
        
        comparison_results['summary'] = {
            'total_countries_analyzed': len(country_comparisons),
            'en_native_avg_distance': float(np.mean(all_en_native_distances)) if all_en_native_distances else None,
            'native_language_avg_distance': float(np.mean(all_native_distances)) if all_native_distances else None,
            'english_language_avg_distance': float(np.mean(all_english_distances)) if all_english_distances else None,
            'language_improvement': None,
            'native_vs_en_native': None,
            'english_vs_en_native': None
        }
        
        comparison_results['en_native_baseline'] = en_native_baseline
        
        if comparison_results['summary']['native_language_avg_distance'] and comparison_results['summary']['english_language_avg_distance']:
            native_avg = comparison_results['summary']['native_language_avg_distance']
            english_avg = comparison_results['summary']['english_language_avg_distance']
            improvement = ((native_avg - english_avg) / native_avg) * 100
            comparison_results['summary']['language_improvement'] = float(improvement)
        
        if comparison_results['summary']['native_language_avg_distance'] and comparison_results['summary']['en_native_avg_distance']:
            native_avg = comparison_results['summary']['native_language_avg_distance']
            en_native_avg = comparison_results['summary']['en_native_avg_distance']
            diff = ((native_avg - en_native_avg) / en_native_avg) * 100
            comparison_results['summary']['native_vs_en_native'] = float(diff)
        
        if comparison_results['summary']['english_language_avg_distance'] and comparison_results['summary']['en_native_avg_distance']:
            english_avg = comparison_results['summary']['english_language_avg_distance']
            en_native_avg = comparison_results['summary']['en_native_avg_distance']
            diff = ((english_avg - en_native_avg) / en_native_avg) * 100
            comparison_results['summary']['english_vs_en_native'] = float(diff)
        
        comparison_results['country_details'] = country_comparisons
        
        print("\nSummarizing model-level language effects...")
        model_analysis = self.analyze_model_specific_language_performance(
            native_data, english_data, real_country_coords, matchable_countries
        )
        comparison_results['model_specific_analysis'] = model_analysis
        
        self._print_comparison_summary(comparison_results, model_analysis)
        
        return comparison_results
    
    def _print_comparison_summary(self, comparison_results, model_analysis):
        """Print a compact summary of the comparison analysis."""
        print(f"\nAnalyzed {comparison_results['summary']['total_countries_analyzed']} countries")
        
        print("\nAverage distances by prompt type:")
        if comparison_results['summary']['en_native_avg_distance']:
            print(f"   en-native (anglophone entities): {comparison_results['summary']['en_native_avg_distance']:.3f} [baseline]")
        if comparison_results['summary']['english_language_avg_distance']:
            print(f"   en (non-anglophone entities prompted in English): {comparison_results['summary']['english_language_avg_distance']:.3f}")
        if comparison_results['summary']['native_language_avg_distance']:
            print(f"   native-language prompting: {comparison_results['summary']['native_language_avg_distance']:.3f}")
        
        print("\nRelative to the en-native baseline:")
        if comparison_results['summary']['english_vs_en_native'] is not None:
            diff = comparison_results['summary']['english_vs_en_native']
            if diff > 0:
                print(f"   English prompting is {diff:.1f}% worse than en-native")
            else:
                print(f"   English prompting is {-diff:.1f}% better than en-native")
        
        if comparison_results['summary']['native_vs_en_native'] is not None:
            diff = comparison_results['summary']['native_vs_en_native']
            if diff > 0:
                print(f"   Native-language prompting is {diff:.1f}% worse than en-native")
            else:
                print(f"   Native-language prompting is {-diff:.1f}% better than en-native")
        
        print("\nNative-language vs English comparison:")
        if comparison_results['summary']['language_improvement'] is not None:
            improvement = comparison_results['summary']['language_improvement']
            if improvement > 0:
                print(f"   English prompting performs {improvement:.1f}% better than native-language prompting")
            else:
                print(f"   Native-language prompting performs {-improvement:.1f}% better than English prompting")
        
        print("\nModel-level language effects:")
        for model_name, model_stats in model_analysis.items():
            if model_stats['native_avg_distance'] and model_stats['english_avg_distance']:
                native_avg = model_stats['native_avg_distance']
                english_avg = model_stats['english_avg_distance']
                improvement = ((native_avg - english_avg) / native_avg) * 100
                
                print(f"   {model_name.split('/')[-1]}:")
                print(f"     native: {native_avg:.3f}, English: {english_avg:.3f}")
                if improvement > 0:
                    print(f"     -> English performs better ({improvement:.1f}%)")
                else:
                    print(f"     -> Native-language prompting performs better ({-improvement:.1f}%)")


def main():
    """Run a smoke test for the multilingual roleplay PCA pipeline."""
    analyzer = MultilingualRoleplayPCAAnalysis(data_path="data")
    
    try:
        entity_scores = analyzer.run_multilingual_analysis_for_runner()
        print("\nMultilingual PCA analysis completed.")
        print(f"Generated entity scores for {len(entity_scores)} rows")
        print("\nNext step:")
        print("  python src/roleplay_multilingual/multilingual_roleplay_visualization.py")
        
    except Exception as e:
        print(f"PCA analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
