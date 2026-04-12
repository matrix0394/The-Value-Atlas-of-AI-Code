"""Process multilingual interview outputs into IVS-compatible analysis tables."""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.base.ivs_question_processor import IVSQuestionProcessor


UN_LANGUAGE_NAMES_ZH = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "ar": "Arabic",
    "zh-cn": "Simplified Chinese"
}


class LLMMultilingualDataProcessor:
    """Convert cached multilingual interviews into analysis-ready tables."""
    
    def __init__(self, data_path: str = "data"):
        """Initialize the processor."""
        self.data_path = Path(data_path)
        self.raw_results: Dict[str, Any] = {}
        self.processed_data: pd.DataFrame = pd.DataFrame()
        
        print("LLMMultilingualDataProcessor initialized")
        print(f"   data path: {self.data_path}")
    
    def generate_entity_id(self, model_name: str, language: str) -> str:
        """Create the entity identifier used by downstream PCA code."""
        safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        return f"llm_{safe_model_name}_{language}"
    
    def load_raw_results(self, data_dir: Path = None) -> Dict[str, Any]:
        """Load cached multilingual interview outputs from disk."""
        if data_dir is None:
            data_dir = self.data_path / "llm_interviews" / "intrinsic" / "interview_raw"
        
        if not data_dir.exists():
            print(f"Data directory does not exist: {data_dir}")
            return {}
        
        results = {}
        loaded_count = 0
        skipped_count = 0
        
        print(f"\nLoading cached interview data from: {data_dir}")
        
        for pkl_file in sorted(data_dir.glob("*.pkl")):
            try:
                if pkl_file.name.startswith("llm_interview_raw_"):
                    skipped_count += 1
                    continue
                
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                model_name = data.get('model_name', '')
                language = data.get('language', '')
                
                if model_name and language:
                    key = f"{model_name}_{language}"
                    
                    if key in results:
                        existing_ts = results[key].get('timestamp', '')
                        new_ts = data.get('timestamp', '')
                        if new_ts > existing_ts:
                            results[key] = data
                            print(f"   Updated cached result: {key}")
                    else:
                        results[key] = data
                        loaded_count += 1
                        
            except Exception as e:
                print(f"   Could not load {pkl_file.name}: {e}")
        
        self.raw_results = results
        
        print("\nLoad summary:")
        print(f"   multilingual interview results: {loaded_count}")
        print(f"   skipped merged cache files: {skipped_count}")
        
        if results:
            models = set()
            languages = set()
            for key in results.keys():
                parts = key.rsplit('_', 1)
                if len(parts) == 2:
                    models.add(parts[0])
                    languages.add(parts[1])
            
            print(f"   models: {len(models)}")
            print(f"   languages: {len(languages)}")
            print(f"   language codes: {', '.join(sorted(languages))}")
        
        return results
    
    def convert_to_ivs_format(self, raw_results: Dict[str, Any] = None) -> pd.DataFrame:
        """Convert cached interview outputs into an IVS-compatible table."""
        if raw_results is None:
            raw_results = self.raw_results
        
        if not raw_results:
            print("No raw results available for conversion")
            return pd.DataFrame()
        
        print("\nConverting cached results to IVS format...")
        
        rows = []
        
        for key, data in raw_results.items():
            model_name = data.get('model_name', '')
            language = data.get('language', '')
            
            if not model_name or not language:
                continue
            
            row = {
                'entity_id': self.generate_entity_id(model_name, language),
                'model_name': model_name,
                'language': language,
                'language_name': UN_LANGUAGE_NAMES_ZH.get(language, language),
                'country_code': self.generate_entity_id(model_name, language),
                'year': 2025,
                'weight': 1.0,
                'data_source': 'LLM_Multilingual',
                'Cultural Region': 'AI Model'
            }
            
            responses = data.get('responses', [])
            response_dict = {}
            
            for resp in responses:
                if isinstance(resp, dict):
                    question_id = resp.get('question_id', '')
                    value = resp.get('processed_response') or resp.get('response')
                    is_valid = resp.get('is_valid', False)
                    
                    if is_valid and value is not None:
                        response_dict[question_id] = value
            
            ivs_questions = ['A008', 'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006']
            for q in ivs_questions:
                row[q] = response_dict.get(q)
            
            y002_value = response_dict.get('Y002')
            if isinstance(y002_value, (list, tuple)) and len(y002_value) == 2:
                row['Y002_first'] = y002_value[0]
                row['Y002_second'] = y002_value[1]
                row['Y002'] = IVSQuestionProcessor.process_y002(y002_value[0], y002_value[1])
            else:
                row['Y002_first'] = None
                row['Y002_second'] = None
                row['Y002'] = None
            
            y003_value = response_dict.get('Y003')
            if isinstance(y003_value, (list, tuple)) and len(y003_value) > 0:
                row['Y003_values'] = list(y003_value)
                y003_result = IVSQuestionProcessor.process_y003(list(y003_value))
                row['Y003'] = y003_result.get('y003_score')
            elif isinstance(y003_value, int):
                row['Y003_values'] = [y003_value]
                y003_result = IVSQuestionProcessor.process_y003([y003_value])
                row['Y003'] = y003_result.get('y003_score')
            else:
                row['Y003_values'] = None
                row['Y003'] = None
            
            rows.append(row)
        
        self.processed_data = pd.DataFrame(rows)
        
        print(f"Converted {len(self.processed_data)} records")
        
        if not self.processed_data.empty:
            valid_counts = {}
            for q in ivs_questions + ['Y002', 'Y003']:
                if q in self.processed_data.columns:
                    valid_counts[q] = self.processed_data[q].notna().sum()
            
            print("   valid response counts:")
            for q, count in valid_counts.items():
                print(f"     {q}: {count}/{len(self.processed_data)}")
        
        return self.processed_data

    def save_processed_results(self, output_dir: Path = None, 
                               prefix: str = "multilingual") -> Tuple[str, str]:
        """Save processed outputs in JSON and pickle formats."""
        if self.processed_data.empty:
            print("No processed data available to save")
            return ("", "")
        
        if output_dir is None:
            output_dir = self.data_path / "llm_interviews" / "intrinsic"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_df = self.processed_data.copy()
        
        if 'Y003_values' in save_df.columns:
            save_df['Y003_values_str'] = save_df['Y003_values'].apply(
                lambda x: str(x) if x is not None else None
            )
        
        json_file = output_dir / f"{prefix}_processed_{timestamp}.json"
        save_df.to_json(json_file, orient='records', indent=2, force_ascii=False)
        
        pkl_file = output_dir / f"{prefix}_processed_{timestamp}.pkl"
        self.processed_data.to_pickle(pkl_file)
        
        standard_json = output_dir / f"{prefix}_ivs_format.json"
        standard_pkl = output_dir / f"{prefix}_ivs_format.pkl"
        save_df.to_json(standard_json, orient='records', indent=2, force_ascii=False)
        self.processed_data.to_pickle(standard_pkl)
        
        print("\nSaved processed data:")
        print(f"   JSON: {json_file}")
        print(f"   Pickle: {pkl_file}")
        print(f"   Standard JSON: {standard_json}")
        print(f"   Standard pickle: {standard_pkl}")
        
        return (str(json_file), str(pkl_file))
    
    def run_pca_analysis(self, use_fixed_pca: bool = True) -> pd.DataFrame:
        """Run PCA projection for the processed multilingual table."""
        from src.llm_values.llm_pca_analysis import LLMPCAAnalyzer
        
        print("\nRunning PCA analysis...")
        
        if self.processed_data.empty:
            print("No processed data available; run convert_to_ivs_format() first")
            return pd.DataFrame()
        
        temp_dir = self.data_path / "llm_interviews" / "intrinsic"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_file = temp_dir / "llm_processed_responses_ivs_format_temp.pkl"
        self.processed_data.to_pickle(temp_file)
        
        try:
            analyzer = LLMPCAAnalyzer(data_path=str(self.data_path))
            
            analyzer.llm_data = self.processed_data
            
            if use_fixed_pca:
                entity_scores = analyzer.run_analysis_with_fixed_pca()
            else:
                entity_scores = analyzer.run_full_analysis()
            
            if not entity_scores.empty and 'entity_id' in entity_scores.columns:
                def extract_language(entity_id):
                    if not entity_id or not isinstance(entity_id, str):
                        return 'en'
                    for lang in ['zh-cn', 'en', 'fr', 'es', 'ru', 'ar']:
                        if entity_id.endswith(f'_{lang}'):
                            return lang
                    return 'en'
                
                def extract_model(entity_id):
                    if not entity_id or not isinstance(entity_id, str):
                        return ''
                    if entity_id.startswith('llm_'):
                        parts = entity_id[4:]
                        for lang in ['zh-cn', 'en', 'fr', 'es', 'ru', 'ar']:
                            if parts.endswith(f'_{lang}'):
                                return parts[:-len(f'_{lang}')]
                        return parts
                    return entity_id
                
                entity_scores['language'] = entity_scores['entity_id'].apply(extract_language)
                entity_scores['model_name'] = entity_scores['entity_id'].apply(extract_model)
                
                print(f"   language distribution: {entity_scores['language'].value_counts().to_dict()}")
            
            print(f"PCA analysis completed for {len(entity_scores)} entities")
            
            return entity_scores
            
        except Exception as e:
            print(f"PCA analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
        finally:
            if temp_file.exists():
                temp_file.unlink()
    
    def generate_summary_report(self, entity_scores: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate a cross-language summary report for each model."""
        print("\nGenerating cross-language summary report...")
        
        data = self.processed_data if entity_scores is None else entity_scores
        
        if data.empty:
            print("No data available for summary reporting")
            return {}
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_records': len(data),
            'models': {},
            'language_summary': {},
            'cross_language_comparison': []
        }
        
        if 'model_name' in data.columns:
            models = data['model_name'].unique()
            
            for model in models:
                model_data = data[data['model_name'] == model]
                languages = model_data['language'].unique() if 'language' in model_data.columns else []
                
                model_report = {
                    'model_name': model,
                    'languages_tested': list(languages),
                    'language_count': len(languages),
                    'responses_by_language': {}
                }
                
                for lang in languages:
                    lang_data = model_data[model_data['language'] == lang]
                    
                    lang_stats = {
                        'language': lang,
                        'language_name': UN_LANGUAGE_NAMES_ZH.get(lang, lang),
                        'record_count': len(lang_data)
                    }
                    
                    ivs_questions = ['A008', 'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002', 'Y003']
                    for q in ivs_questions:
                        if q in lang_data.columns:
                            values = lang_data[q].dropna()
                            if len(values) > 0:
                                lang_stats[q] = {
                                    'value': values.iloc[0] if len(values) == 1 else values.tolist(),
                                    'valid': True
                                }
                            else:
                                lang_stats[q] = {'value': None, 'valid': False}
                    
                    if 'PC1_rescaled' in lang_data.columns:
                        lang_stats['PC1'] = lang_data['PC1_rescaled'].iloc[0] if len(lang_data) > 0 else None
                    if 'PC2_rescaled' in lang_data.columns:
                        lang_stats['PC2'] = lang_data['PC2_rescaled'].iloc[0] if len(lang_data) > 0 else None
                    
                    model_report['responses_by_language'][lang] = lang_stats
                
                report['models'][model] = model_report
        
        if 'language' in data.columns:
            for lang in data['language'].unique():
                lang_data = data[data['language'] == lang]
                report['language_summary'][lang] = {
                    'language_name': UN_LANGUAGE_NAMES_ZH.get(lang, lang),
                    'model_count': len(lang_data['model_name'].unique()) if 'model_name' in lang_data.columns else 0,
                    'total_records': len(lang_data)
                }
        
        if 'model_name' in data.columns and 'language' in data.columns:
            for model in data['model_name'].unique():
                model_data = data[data['model_name'] == model]
                languages = model_data['language'].unique()
                
                if len(languages) > 1:
                    comparison = {
                        'model': model,
                        'languages': list(languages),
                        'value_differences': {}
                    }
                    
                    ivs_questions = ['A008', 'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002', 'Y003']
                    for q in ivs_questions:
                        if q in model_data.columns:
                            values = model_data.groupby('language')[q].first().dropna()
                            if len(values) > 1:
                                comparison['value_differences'][q] = {
                                    'values': values.to_dict(),
                                    'range': float(values.max() - values.min()) if values.dtype in ['int64', 'float64'] else None,
                                    'std': float(values.std()) if values.dtype in ['int64', 'float64'] else None
                                }
                    
                    report['cross_language_comparison'].append(comparison)
        
        print("Summary report completed")
        print(f"   models: {len(report['models'])}")
        print(f"   languages: {len(report['language_summary'])}")
        print(f"   cross-language comparison groups: {len(report['cross_language_comparison'])}")
        
        return report
    
    def save_summary_report(self, report: Dict[str, Any], output_dir: Path = None) -> str:
        """Save the summary report as JSON."""
        import numpy as np
        
        def convert_numpy_types(obj):
            """Recursively convert NumPy objects into JSON-serializable types."""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        if output_dir is None:
            output_dir = self.data_path / "llm_interviews" / "intrinsic"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"multilingual_summary_report_{timestamp}.json"
        
        report_converted = convert_numpy_types(report)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_converted, f, ensure_ascii=False, indent=2)
        
        print(f"Saved summary report: {report_file}")
        
        return str(report_file)
    
    def run_full_pipeline(self, data_dir: Path = None, 
                          run_pca: bool = True,
                          use_fixed_pca: bool = True) -> Dict[str, Any]:
        """Run the end-to-end multilingual processing pipeline."""
        print("\n" + "="*60)
        print("Running multilingual data-processing pipeline")
        print("="*60)
        
        results = {
            'raw_results': {},
            'processed_data': None,
            'entity_scores': None,
            'summary_report': None,
            'output_files': {}
        }
        
        results['raw_results'] = self.load_raw_results(data_dir)
        
        if not results['raw_results']:
            print("No raw results were found")
            return results
        
        results['processed_data'] = self.convert_to_ivs_format()
        
        if results['processed_data'].empty:
            print("IVS conversion failed")
            return results
        
        json_file, pkl_file = self.save_processed_results()
        results['output_files']['processed_json'] = json_file
        results['output_files']['processed_pkl'] = pkl_file
        
        if run_pca:
            results['entity_scores'] = self.run_pca_analysis(use_fixed_pca)
        
        results['summary_report'] = self.generate_summary_report(results['entity_scores'])
        report_file = self.save_summary_report(results['summary_report'])
        results['output_files']['summary_report'] = report_file
        
        print("\n" + "="*60)
        print("Multilingual data-processing pipeline completed")
        print("="*60)
        
        return results


def main():
    """Run a small smoke test for the multilingual data processor."""
    print("Running LLMMultilingualDataProcessor smoke test...")
    
    processor = LLMMultilingualDataProcessor(data_path="data")
    
    raw_results = processor.load_raw_results()
    
    if raw_results:
        processed_data = processor.convert_to_ivs_format()
        
        if not processed_data.empty:
            print("\nPreview of processed records:")
            print(processed_data[['entity_id', 'model_name', 'language', 'A008', 'Y002']].head(10))
            
            processor.save_processed_results()
            
            report = processor.generate_summary_report()
            processor.save_summary_report(report)
    else:
        print("No multilingual interview data found")
    
    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
