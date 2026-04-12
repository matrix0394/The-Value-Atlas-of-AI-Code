"""Process intrinsic LLM interview responses into IVS-aligned analysis tables."""

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from src.base.ivs_question_processor import IVSQuestionProcessor


@dataclass
class ProcessedResponse:
    """Container for the processed responses of a single model."""
    model_name: str
    model_region: str
    A008: Optional[int] = None
    A165: Optional[int] = None
    E018: Optional[int] = None
    E025: Optional[int] = None
    F063: Optional[int] = None
    F118: Optional[int] = None
    F120: Optional[int] = None
    G006: Optional[int] = None
    Y002_first: Optional[int] = None
    Y002_second: Optional[int] = None
    Y002_materialist: Optional[int] = None
    Y003_values: Optional[List[int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the processed response to a flat dictionary."""
        result = {
            'model_name': self.model_name,
            'model_region': self.model_region,
            'A008': self.A008,
            'A165': self.A165,
            'E018': self.E018,
            'E025': self.E025,
            'F063': self.F063,
            'F118': self.F118,
            'F120': self.F120,
            'G006': self.G006,
            'Y002_first': self.Y002_first,
            'Y002_second': self.Y002_second,
            'Y002_materialist': self.Y002_materialist
        }
        
        if self.Y003_values:
            for i in range(1, 12):
                result[f'Y003_{i}'] = 1 if i in self.Y003_values else 0
        else:
            for i in range(1, 12):
                result[f'Y003_{i}'] = 0
                
        return result


class LLMDataProcessor:
    """Processor for intrinsic LLM interview outputs."""
    
    def __init__(self, data_dir: str = None, config_path: str = None):
        """Initialize input and configuration paths."""
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'models', 'llm_models.json')
            
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        
        self.models_config = self._load_models_config()
        self.processed_data: List[ProcessedResponse] = []

    def _get_raw_response_dirs(self) -> List[Path]:
        """Return candidate raw-cache directories in priority order."""
        candidates = [
            self.data_dir / "llm_interviews" / "intrinsic" / "interview_raw",
            self.data_dir / "llm_values" / "interview_raw",
        ]
        unique_dirs = []
        seen = set()
        for path in candidates:
            if path not in seen:
                unique_dirs.append(path)
                seen.add(path)
        return unique_dirs
    
    def _load_models_config(self) -> Dict:
        """Load the model configuration used to map models to regions."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Model config not found: {self.config_path}")
            return {"models": {}}
        except json.JSONDecodeError as e:
            print(f"Model config could not be parsed: {e}")
            return {"models": {}}
    
    def _get_model_region(self, model_id: str) -> str:
        """Return the configured region label for a model."""
        models = self.models_config.get("models", {})
        if model_id in models:
            return models[model_id].get("region", "Unknown")
        return "Unknown"
    
    def load_all_responses(self) -> List[pd.DataFrame]:
        """Load per-model interview caches from the canonical raw-cache locations."""
        all_dataframes = []
        seen_models = set()

        for interview_raw_dir in self._get_raw_response_dirs():
            if not interview_raw_dir.exists():
                continue

            individual_files = [
                f for f in interview_raw_dir.glob("*.pkl")
                if not f.name.startswith("llm_interview_raw_")
            ]

            if not individual_files:
                continue

            print(f"Found {len(individual_files)} model files in {interview_raw_dir}")

            for file_path in individual_files:
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)

                    if isinstance(data, dict):
                        model_name = data.get('model_name', data.get('model', ''))
                        if not model_name:
                            print(f"{file_path.name}: could not determine model name")
                            continue

                        if model_name in seen_models:
                            continue

                        rows = []
                        for response in data.get('responses', []):
                            if isinstance(response, dict):
                                rows.append({
                                    'model_name': model_name,
                                    'question_id': response.get('question_id'),
                                    'response': (
                                        response.get('final_response')
                                        or response.get('processed_response')
                                        or response.get('response')
                                    ),
                                    'is_valid': response.get('is_valid', True),
                                    'source_file': file_path.name
                                })

                        if rows:
                            df = pd.DataFrame(rows)
                            all_dataframes.append(df)
                            seen_models.add(model_name)
                            print(f"   {model_name}: {len(rows)} responses")
                        else:
                            print(f"   {model_name}: no valid responses")

                except Exception as e:
                    print(f"   {file_path.name}: failed to load ({e})")
                    continue

        if all_dataframes:
            print(f"\nLoaded responses for {len(all_dataframes)} models")
        else:
            print("\nNo usable response files were found")
        
        return all_dataframes
    
    def process_single_model_responses(self, df: pd.DataFrame) -> Optional[ProcessedResponse]:
        """Process the responses for a single model."""
        if df.empty:
            return None
        
        model_name = df['model_name'].iloc[0] if 'model_name' in df.columns else "Unknown"
        model_region = self._get_model_region(model_name)
        
        result = ProcessedResponse(
            model_name=model_name,
            model_region=model_region
        )
        
        valid_count = 0
        
        for _, row in df.iterrows():
            if not row.get('is_valid', False):
                continue
                
            question_id = row['question_id']
            response = row['response']
            
            if question_id == 'A008':
                result.A008 = response
                valid_count += 1
            elif question_id == 'A165':
                result.A165 = response
                valid_count += 1
            elif question_id == 'E018':
                result.E018 = response
                valid_count += 1
            elif question_id == 'E025':
                result.E025 = response
                valid_count += 1
            elif question_id == 'F063':
                result.F063 = response
                valid_count += 1
            elif question_id == 'F118':
                result.F118 = response
                valid_count += 1
            elif question_id == 'F120':
                result.F120 = response
                valid_count += 1
            elif question_id == 'G006':
                result.G006 = response
                valid_count += 1
            elif question_id == 'Y002':
                if isinstance(response, (tuple, list)) and len(response) == 2:
                    result.Y002_first = response[0]
                    result.Y002_second = response[1]
                    result.Y002_materialist = IVSQuestionProcessor.process_y002(response[0], response[1])
                    valid_count += 1
            elif question_id == 'Y003':
                if isinstance(response, (tuple, list)):
                    result.Y003_values = list(response)
                    valid_count += 1
                elif isinstance(response, int):
                    result.Y003_values = [response]
                    valid_count += 1
        
        if valid_count < 6:
            print(f"{model_name}: filtered out because only {valid_count}/10 answers were valid")
            return None
        
        return result
    
    def process_all_models(self) -> pd.DataFrame:
        """Process all discovered model-response files."""
        self.processed_data = []
        
        all_dfs = self.load_all_responses()
        
        if not all_dfs:
            print("No response files were found")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        for model_name in combined_df['model_name'].unique():
            model_df = combined_df[combined_df['model_name'] == model_name]
            processed = self.process_single_model_responses(model_df)
            
            if processed:
                self.processed_data.append(processed)
        
        if self.processed_data:
            data_dicts = [item.to_dict() for item in self.processed_data]
            return pd.DataFrame(data_dicts)
        else:
            return pd.DataFrame()
    
    def create_ivs_compatible_dataframe(self) -> pd.DataFrame:
        """Convert processed responses into the IVS-compatible table used downstream."""
        processed_df = self.process_all_models()
        
        if processed_df.empty:
            return pd.DataFrame()
        
        ivs_compatible = pd.DataFrame()
        
        ivs_compatible['year'] = [2025] * len(processed_df)
        ivs_compatible['country_code'] = processed_df['model_name']
        ivs_compatible['weight'] = [1.0] * len(processed_df)
        ivs_compatible['model_region'] = processed_df['model_region']
        
        ivs_questions = ['A008', 'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006']
        for q in ivs_questions:
            ivs_compatible[q] = processed_df[q]
        
        ivs_compatible['Y002'] = processed_df['Y002_materialist']
        
        def get_y003_score(row):
            """Compute the Y003 score using the shared IVS recoder."""
            selected_values = []
            for i in range(1, 12):
                if row.get(f'Y003_{i}', 0) == 1:
                    selected_values.append(i)
            
            if not selected_values:
                return np.nan
            
            result = IVSQuestionProcessor.process_y003(selected_values)
            return result["y003_score"]
        
        ivs_compatible['Y003'] = processed_df.apply(get_y003_score, axis=1)
        
        return ivs_compatible
    
    def save_processed_data(self, output_path: str = None) -> Optional[str]:
        """Save the processed intrinsic-response tables in canonical and legacy locations."""
        canonical_dir = self.data_dir / 'llm_interviews' / 'intrinsic'
        canonical_dir.mkdir(parents=True, exist_ok=True)
        legacy_dir = self.data_dir / 'llm_values'
        legacy_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = canonical_dir / f'llm_processed_responses_ivs_format_{timestamp}.pkl'
        
        ivs_compatible = self.create_ivs_compatible_dataframe()
        
        if not ivs_compatible.empty:
            ivs_compatible.to_pickle(output_path)
            print(f"Saved IVS-compatible intrinsic data to: {output_path}")
            print(f"   - Models: {len(ivs_compatible)}")
            print("   - Format: aligned with the benchmark valid_data table")
            
            json_path = str(output_path).replace('.pkl', '.json')
            ivs_compatible.to_json(json_path, orient='records', indent=2)
            print(f"   - JSON copy: {json_path}")
            
            latest_path = canonical_dir / 'llm_processed_responses_ivs_format_latest.pkl'
            latest_json_path = canonical_dir / 'llm_processed_responses_ivs_format_latest.json'
            ivs_compatible.to_pickle(latest_path)
            ivs_compatible.to_json(latest_json_path, orient='records', indent=2)

            legacy_standard_path = legacy_dir / 'llm_values_ivs_format.pkl'
            legacy_standard_json = legacy_dir / 'llm_values_ivs_format.json'
            ivs_compatible.to_pickle(legacy_standard_path)
            ivs_compatible.to_json(legacy_standard_json, orient='records', indent=2)
            print(f"   - Canonical latest path: {latest_path}")
            print(f"   - Legacy compatibility path: {legacy_standard_path}")

            return str(latest_path)
        else:
            print("No processed data were available to save")
            return None
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Summarize the processed intrinsic-response dataset."""
        processed_df = self.process_all_models()
        
        if processed_df.empty:
            return {"message": "No processed data available"}
        
        stats = {
            "total_models": len(processed_df),
            "models_by_region": processed_df['model_region'].value_counts().to_dict(),
            "response_completeness": {},
            "y002_materialist_distribution": processed_df['Y002_materialist'].value_counts().to_dict() if 'Y002_materialist' in processed_df.columns else {}
        }
        
        ivs_questions = ['A008', 'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006']
        for q in ivs_questions:
            if q in processed_df.columns:
                valid_responses = processed_df[q].notna().sum()
                stats["response_completeness"][q] = f"{valid_responses}/{len(processed_df)}"
        
        return stats
    
    

if __name__ == "__main__":
    processor = LLMDataProcessor()
    print("Processing intrinsic LLM interview responses...")
    processed_df = processor.process_all_models()
    
    print(f"Processed data for {len(processed_df)} models")
    
    if not processed_df.empty:
        print("\nProcessed response preview:")
        print(processed_df[['model_name', 'model_region', 'A008', 'A165', 'Y002_materialist']].head())
        
        ivs_df = processor.create_ivs_compatible_dataframe()
        print("\nIVS-compatible table preview:")
        print(ivs_df[['country_code', 'model_region', 'A008', 'A165', 'Y002']].head())
        
        stats = processor.get_summary_statistics()
        print("\nSummary statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        processor.save_processed_data()
    else:
        print("No valid response files were found")
