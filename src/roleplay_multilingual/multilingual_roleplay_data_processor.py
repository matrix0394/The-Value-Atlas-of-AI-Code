"""Process multilingual roleplay responses into IVS-aligned analysis tables."""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import re
import sys

# Add the project root so local modules can be imported when the script is run directly.
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.base.ivs_question_processor import IVSQuestionProcessor
from src.utils.country_name_standardizer import CountryNameStandardizer


class MultilingualRoleplayDataProcessor:
    """Processor for multilingual roleplay interview outputs."""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.raw_data_dir = self.data_path / "llm_interviews" / "multilingual" / "interview_raw"
        self.processed_dir = self.data_path / "llm_interviews" / "multilingual" / "processed"
        
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.iv_qns = ['A008', 'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002', 'Y003']
        
        self.multilingual_config = self._load_multilingual_config()
        self.question_mapping = IVSQuestionProcessor.QUESTION_CONFIG
        self.cultural_mapping = self._load_cultural_mapping()
        self.country_standardizer = CountryNameStandardizer()
    
    def _load_multilingual_config(self) -> Dict:
        """Load the multilingual questionnaire configuration."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'questions' / 'multilingual' / 'multilingual_questions_complete.json'
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        print(f"Multilingual questionnaire config not found: {config_path}")
        return {}
    
    def _load_cultural_mapping(self) -> Dict:
        """Load the country-to-cultural-region mapping."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'country' / 'cultural_regions.json'
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('country_cultural_mapping', {})
        except Exception as e:
            print(f"Could not load cultural-region mapping: {e}")
            return {}
    
    def load_multilingual_results(self, results_file: str = None) -> Dict:
        """Load multilingual roleplay results from a summary file or individual caches."""
        if results_file is None:
            pkl_files = list(self.raw_data_dir.glob("roleplay_results_ml_*.pkl"))
            json_files = list(self.raw_data_dir.glob("roleplay_results_ml_*.json"))
            
            if pkl_files:
                results_file = max(pkl_files, key=lambda x: x.stat().st_mtime)
                print(f"Using latest PKL summary file: {results_file.name}")
            elif json_files:
                results_file = max(json_files, key=lambda x: x.stat().st_mtime)
                print(f"Using latest JSON summary file: {results_file.name}")
            else:
                print("No summary file found; scanning individual interview files instead.")
                return self._load_from_individual_files()
        else:
            results_file = Path(results_file)
        
        print(f"Loading results file: {results_file}")
        
        if results_file.suffix == '.pkl':
            with open(results_file, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        if isinstance(data, dict) and 'results' in data:
            print(f"Loaded {len(data.get('results', []))} interview results")
        else:
            raise ValueError("Unsupported data format; expected {results: [...]} structure")
        
        return data
    
    def _load_from_individual_files(self) -> Dict:
        """Load data from individual JSON/PKL files and deduplicate repeated runs."""
        json_files = []
        pkl_files = []

        def _collect(directory):
            for f in directory.glob("*_*_*.json"):
                if not f.name.startswith("roleplay_results_ml_"):
                    json_files.append(f)
            for f in directory.glob("*_*_*.pkl"):
                if not f.name.startswith("roleplay_results_ml_"):
                    pkl_files.append(f)

        _collect(self.raw_data_dir)
        for subdir in self.raw_data_dir.iterdir():
            if subdir.is_dir():
                _collect(subdir)

        # Prefer JSON; only fall back to PKL for stems without a JSON counterpart
        json_stems = {f.stem for f in json_files}
        individual_files = list(json_files)
        for f in pkl_files:
            if f.stem not in json_stems:
                individual_files.append(f)

        if not individual_files:
            raise FileNotFoundError(
                f"No interview files found under {self.raw_data_dir} (searched the root and subdirectories)"
            )

        print(
            f"Found {len(individual_files)} individual interview files "
            f"after removing JSON/PKL duplicates (from {len(json_files) + len(pkl_files)} raw files)"
        )

        unique_results: Dict[tuple, dict] = {}
        duplicates = 0

        for file_path in individual_files:
            try:
                if file_path.suffix == '.pkl':
                    import pickle
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                filename = file_path.stem
                parts = filename.split('_')

                timestamp_idx = -1
                for i, part in enumerate(parts):
                    if part.isdigit() and len(part) == 8:
                        timestamp_idx = i
                        break

                if timestamp_idx < 2:
                    continue

                model_country_lang = parts[:timestamp_idx]
                if len(model_country_lang) >= 3:
                    language = model_country_lang[-1]
                    country = model_country_lang[-2]
                    model = '_'.join(model_country_lang[:-2])
                elif len(model_country_lang) == 2:
                    country = model_country_lang[-1]
                    model = model_country_lang[0]
                    language = ''
                else:
                    continue

                ts_parts = parts[timestamp_idx:]
                file_timestamp = '_'.join(ts_parts)

                result = {
                    "model": model,
                    "country": country,
                    "language": language,
                    "timestamp": data.get("timestamp", file_timestamp),
                    "success_rate": data.get("success_rate", 1.0),
                    "responses": data.get("responses", []),
                    "_file_timestamp": file_timestamp,
                }

                key = (model, country, language)
                if key in unique_results:
                    duplicates += 1
                    if file_timestamp < unique_results[key].get("_file_timestamp", ""):
                        unique_results[key] = result
                else:
                    unique_results[key] = result

            except Exception as e:
                print(f"Skipping {file_path.name}: {e}")
                continue

        print(f"Removed {duplicates} duplicate runs for repeated model/country/language combinations")
        print(f"Retained {len(unique_results)} unique combinations")

        results = []
        for r in unique_results.values():
            r.pop("_file_timestamp", None)
            results.append(r)

        return {"results": results}
    
    def process_single_response(self, response: Dict) -> Dict:
        """Normalize a single raw response record from either supported format."""
        if "entity_id" in response:
            entity_id = response.get("entity_id", "")
            parts = entity_id.split("_", 1)
            country = parts[0] if len(parts) > 0 else ""
            language = parts[1] if len(parts) > 1 else ""
            model = response.get("model_name", "")
        else:
            model = response.get("model", "")
            country = response.get("country", "")
            language = response.get("language", "")
        
        processed = {
            "model": model,
            "country": country,
            "language": language,
            "timestamp": response.get("timestamp", ""),
            "success_rate": response.get("success_rate", 1.0),
            "processed_answers": {}
        }
        
        for resp in response.get("responses", []):
            question_id = resp.get("question_id", "")
            
            raw_response = str(resp.get("final_response") or resp.get("processed_response") or "")
            if not raw_response.strip():
                all_responses = resp.get("all_responses", [])
                if all_responses:
                    for attempt_resp in reversed(all_responses):
                        if isinstance(attempt_resp, str) and attempt_resp.strip():
                            raw_response = attempt_resp
                            break
                        elif isinstance(attempt_resp, dict) and attempt_resp.get("raw_response"):
                            raw_response = attempt_resp["raw_response"]
                            break
                else:
                    raw_response = resp.get("raw_response", "")
            
            if question_id in self.question_mapping:
                processed_answer = self._process_answer(
                    question_id, raw_response, raw_response
                )
                processed["processed_answers"][question_id] = processed_answer
        
        return processed
    
    def _process_answer(self, question_id: str, processed_response: str, raw_response: str) -> Dict:
        """Validate and recode one answer with the shared IVS processor."""
        return IVSQuestionProcessor.validate_and_process_response(raw_response, question_id)
    
    def process_all_results(self, results_data: Dict) -> pd.DataFrame:
        """Convert all loaded results into a flat tabular representation."""
        processed_data = []
        
        results = results_data.get("results", [])
        if isinstance(results, dict):
            results_list = list(results.values())
        else:
            results_list = results
        
        for result in results_list:
            processed = self.process_single_response(result)
            
            row = {
                "model": processed["model"],
                "country": processed["country"],
                "language": processed["language"],
                "timestamp": processed["timestamp"],
                "success_rate": processed["success_rate"]
            }
            
            for question_id in self.question_mapping.keys():
                if question_id in processed["processed_answers"]:
                    answer = processed["processed_answers"][question_id]
                    row[f"{question_id}_valid"] = answer["valid"]
                    row[f"{question_id}_numeric"] = answer["numeric_value"]
                    row[f"{question_id}_standardized"] = answer["standardized_value"]
                    row[f"{question_id}_raw"] = answer["raw_response"]
                else:
                    row[f"{question_id}_valid"] = False
                    row[f"{question_id}_numeric"] = None
                    row[f"{question_id}_standardized"] = None
                    row[f"{question_id}_raw"] = ""
            
            processed_data.append(row)
        
        return pd.DataFrame(processed_data)
    
    def create_ivs_format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the IVS-compatible table used for multilingual PCA projection."""
        ivs_data = []
        
        for _, row in df.iterrows():
            country_name = row["country"]
            numeric_code = self.country_standardizer.get_numeric_code(country_name)
            if numeric_code is None:
                print(f"Skipping {country_name}: no numeric country code was found")
                continue
            
            ivs_row = {
                "country_code": float(numeric_code),
                "Country": country_name,
                "year": 2025,
                "weight": 1.0,
                "data_source": "multilingual_roleplay",
                "model_name": row["model"],
                "model_region": self._get_model_region(row["model"]),
                "cultural_region": self._get_cultural_region(country_name),
                "language": row["language"],
                "entity_id": f"{row['model']}_{country_name}_{row['language']}"
            }
            
            for question_id in self.question_mapping.keys():
                valid_col = f"{question_id}_valid"
                
                is_valid = False
                if valid_col in row.index:
                    val = row[valid_col]
                    if isinstance(val, (list, np.ndarray)):
                        is_valid = bool(val[0]) if len(val) > 0 else False
                    else:
                        is_valid = bool(val)
                
                if question_id == 'Y002':
                    raw_col = f"{question_id}_raw"
                    if is_valid and raw_col in row.index and row[raw_col]:
                        try:
                            nums = [int(x) for x in str(row[raw_col]).split()]
                            if len(nums) >= 2:
                                ivs_row[question_id] = float(IVSQuestionProcessor.process_y002(nums[0], nums[1]))
                            else:
                                ivs_row[question_id] = np.nan
                        except (ValueError, TypeError):
                            ivs_row[question_id] = np.nan
                    else:
                        ivs_row[question_id] = np.nan
                elif question_id == 'Y003':
                    raw_col = f"{question_id}_raw"
                    if is_valid and raw_col in row.index and row[raw_col]:
                        try:
                            nums = [int(x) for x in str(row[raw_col]).split()]
                            result = IVSQuestionProcessor.process_y003(nums)
                            ivs_row[question_id] = float(result["y003_score"])
                        except (ValueError, TypeError):
                            ivs_row[question_id] = np.nan
                    else:
                        ivs_row[question_id] = np.nan
                else:
                    standardized_col = f"{question_id}_standardized"
                    if is_valid and standardized_col in row.index:
                        std_val = row[standardized_col]
                        if isinstance(std_val, (list, np.ndarray)):
                            std_val = std_val[0] if len(std_val) > 0 else np.nan
                        if pd.notna(std_val):
                            ivs_row[question_id] = std_val
                        else:
                            ivs_row[question_id] = np.nan
                    else:
                        ivs_row[question_id] = np.nan
            
            ivs_data.append(ivs_row)
        
        ivs_df = pd.DataFrame(ivs_data)
        
        iv_qns = ["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006", "Y002", "Y003"]
        original_count = len(ivs_df)
        ivs_df = ivs_df.dropna(subset=iv_qns, thresh=6)
        filtered_count = original_count - len(ivs_df)
        
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} rows with fewer than six valid IVS items")
        
        return ivs_df
    
    def _get_model_region(self, model_name: str) -> str:
        """Return a coarse region label for a model family."""
        if "openai" in model_name.lower():
            return "US"
        elif "google" in model_name.lower():
            return "US"
        elif "anthropic" in model_name.lower():
            return "US"
        elif "deepseek" in model_name.lower() or "qwen" in model_name.lower():
            return "CN"
        elif "llama" in model_name.lower():
            return "US"
        elif "mistral" in model_name.lower():
            return "EU"
        else:
            return "Unknown"
    
    def _get_cultural_region(self, country: str) -> str:
        """Return the mapped cultural region for a country."""
        return self.cultural_mapping.get(country, "Unknown")
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Summarize the processed multilingual roleplay dataset."""
        stats = {
            "total_responses": len(df),
            "languages": df["language"].unique().tolist(),
            "models": df["model"].unique().tolist(),
            "countries": df["country"].unique().tolist(),
            "overall_success_rate": df["success_rate"].mean(),
            "question_validity": {},
            "language_stats": {},
            "model_stats": {},
            "country_stats": {}
        }
        
        for question_id in self.question_mapping.keys():
            valid_col = f"{question_id}_valid"
            if valid_col in df.columns:
                stats["question_validity"][question_id] = {
                    "valid_count": df[valid_col].sum(),
                    "total_count": len(df),
                    "validity_rate": df[valid_col].mean()
                }
        
        for language in stats["languages"]:
            lang_df = df[df["language"] == language]
            stats["language_stats"][language] = {
                "count": len(lang_df),
                "success_rate": lang_df["success_rate"].mean(),
                "countries": lang_df["country"].unique().tolist()
            }
        
        for model in stats["models"]:
            model_df = df[df["model"] == model]
            stats["model_stats"][model] = {
                "count": len(model_df),
                "success_rate": model_df["success_rate"].mean(),
                "languages": model_df["language"].unique().tolist()
            }
        
        for country in stats["countries"]:
            country_df = df[df["country"] == country]
            stats["country_stats"][country] = {
                "count": len(country_df),
                "success_rate": country_df["success_rate"].mean(),
                "language": country_df["language"].iloc[0] if len(country_df) > 0 else ""
            }
        
        return stats
    
    def save_processed_data(self, df: pd.DataFrame, ivs_df: pd.DataFrame, stats: Dict, suffix: str = None):
        """Save processed multilingual tables in the canonical publication layout."""
        if suffix is None:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*60}")
        print("Saving processed multilingual outputs")
        print(f"{'='*60}")
        
        processed_file = self.processed_dir / f"llm_roleplay_ml_processed_responses_{suffix}.pkl"
        with open(processed_file, 'wb') as f:
            pickle.dump(df, f)
        print(f"Saved processed table: {processed_file.name}")
        
        ivs_file = self.processed_dir / f"llm_roleplay_ml_processed_responses_ivs_format_{suffix}.pkl"
        with open(ivs_file, 'wb') as f:
            pickle.dump(ivs_df, f)
        print(f"Saved IVS-format table: {ivs_file.name}")
        
        ivs_json_file = self.processed_dir / f"llm_roleplay_ml_processed_responses_ivs_format_{suffix}.json"
        ivs_df.to_json(ivs_json_file, orient='records', indent=2, force_ascii=False)
        print(f"Saved IVS-format JSON: {ivs_json_file.name}")
        
        csv_file = self.processed_dir / f"llm_roleplay_ml_processed_responses_{suffix}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Saved processed CSV: {csv_file.name}")
        
        ivs_csv_file = self.processed_dir / f"llm_roleplay_ml_processed_responses_ivs_format_{suffix}.csv"
        ivs_df.to_csv(ivs_csv_file, index=False, encoding='utf-8')
        print(f"Saved IVS-format CSV: {ivs_csv_file.name}")
        
        print(f"\nOutput directory: {self.processed_dir}")
        print(f"Row counts: processed={len(df)}, ivs_format={len(ivs_df)}")
        print(f"{'='*60}")
        
        stats_file = self.processed_dir / f"multilingual_roleplay_stats_{suffix}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        print("Saved files:")
        print(f"  processed table: {processed_file}")
        print(f"  IVS-format table: {ivs_file}")
        print(f"  processed CSV: {csv_file}")
        print(f"  statistics JSON: {stats_file}")
        
        return {
            "processed_file": processed_file,
            "ivs_file": ivs_file,
            "csv_file": csv_file,
            "stats_file": stats_file
        }
    
    def process_multilingual_data(self, results_file: str = None) -> Dict:
        """Run the full multilingual processing pipeline and return all outputs."""
        print("=== Multilingual roleplay data processing ===")
        
        print("1. Loading raw roleplay results...")
        results_data = self.load_multilingual_results(results_file)
        
        print("2. Processing interview responses...")
        df = self.process_all_results(results_data)
        print(f"   Processed {len(df)} rows")
        
        print("3. Building the IVS-format table...")
        ivs_df = self.create_ivs_format_data(df)
        print(f"   Created {len(ivs_df)} IVS-format rows")
        
        print("4. Computing summary statistics...")
        stats = self.calculate_statistics(df)
        
        print("5. Saving processed outputs...")
        file_paths = self.save_processed_data(df, ivs_df, stats)
        
        print("\\n=== Processing summary ===")
        print(f"Total responses: {stats['total_responses']}")
        print(f"Languages: {len(stats['languages'])}")
        print(f"Models: {len(stats['models'])}")
        print(f"Countries: {len(stats['countries'])}")
        print(f"Overall success rate: {stats['overall_success_rate']:.1f}%")
        
        print("\\nLanguage-level summary:")
        for lang, lang_stats in stats["language_stats"].items():
            print(f"  {lang}: {lang_stats['success_rate']:.1f}% ({lang_stats['count']} responses)")
        
        return {
            "processed_df": df,
            "ivs_df": ivs_df,
            "stats": stats,
            "file_paths": file_paths
        }
    
    def process_multilingual_data_to_ivs_format(self, results_file: str = None) -> pd.DataFrame:
        """Process multilingual roleplay data and return the IVS-format table directly."""
        print("Processing multilingual roleplay data into IVS format...")
        
        if results_file is None:
            print("Searching for the latest multilingual roleplay results...")
            results_data = self.load_multilingual_results()
        else:
            print(f"Loading the requested results file: {results_file}")
            with open(results_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
        
        if isinstance(results_data, dict) and 'results' in results_data:
            print("Detected the expected {results: [...]} format")
            return self._process_new_format_to_ivs(results_data)
        else:
            raise ValueError("Unsupported data format; expected {results: [...]} structure")
    
    def _process_new_format_to_ivs(self, results_data: Dict) -> pd.DataFrame:
        """Convert the {results: [...]} format into the IVS-format table."""
        results_list = results_data.get('results', [])
        print(f"Loaded {len(results_list)} interview results")
        
        unique_results = {}
        duplicates = 0
        
        for result in results_list:
            model_name = result.get('model_name', result.get('model'))
            country_raw = result.get('country')
            if isinstance(country_raw, dict):
                country = country_raw.get('name')
            else:
                country = country_raw
            language = result.get('language')
            
            key = (model_name, country, language)
            
            if key in unique_results:
                existing_timestamp = unique_results[key].get('timestamp', '')
                current_timestamp = result.get('timestamp', '')
                if current_timestamp > existing_timestamp:
                    unique_results[key] = result
                duplicates += 1
            else:
                unique_results[key] = result
        
        print(f"Removed {duplicates} duplicate rows")
        print(f"Retained {len(unique_results)} unique combinations")
        
        ivs_data = []
        
        for result in unique_results.values():
            # Extract the model field from either key used in cached results.
            model_name = result.get('model_name', result.get('model'))
            
            # Country may be stored either as a string or as a nested object.
            country_raw = result.get('country')
            if isinstance(country_raw, dict):
                country = country_raw.get('name')
            else:
                country = country_raw
            
            language = result.get('language')
            responses = result.get('responses', [])
            
            numeric_code = self.country_standardizer.get_numeric_code(country)
            if numeric_code is None:
                print(f"Skipping {country}: no numeric country code was found")
                continue
            
            ivs_row = {
                'country_code': float(numeric_code),
                'Country': country,
                'model_name': model_name,
                'language': language,
                'data_source': 'Multilingual',
                'year': 2025,
                'weight': 1.0,
                'Cultural Region': self._get_cultural_region(country),
                'entity_id': f"{model_name}_{country}_{language}"
            }
            
            if isinstance(responses, list):
                for item in responses:
                    if isinstance(item, dict) and 'question_id' in item:
                        question_id = item['question_id']
                        if question_id in self.iv_qns:
                            answer = item.get('final_response') or item.get('processed_response', '')
                            
                            result = IVSQuestionProcessor.validate_and_process_response(answer, question_id)
                            
                            if result["valid"]:
                                if question_id == "Y002" and "materialist_score" in result:
                                    ivs_row[question_id] = result["materialist_score"]
                                elif question_id == "Y003" and "y003_score" in result:
                                    ivs_row[question_id] = result["y003_score"]
                                else:
                                    ivs_row[question_id] = result["numeric_value"]
                            else:
                                ivs_row[question_id] = np.nan
            else:
                print(f"Unsupported responses format: {type(responses)}")
            
            ivs_data.append(ivs_row)
        
        print(f"Converted {len(ivs_data)} interview results")
        
        ivs_df = pd.DataFrame(ivs_data)
        print(f"Created {len(ivs_df)} IVS-format rows")
        
        if 'model_name' in ivs_df.columns:
            ivs_df['model'] = ivs_df['model_name']
        if 'country_code' in ivs_df.columns:
            ivs_df['country'] = ivs_df['country_code']
        
        return ivs_df


def main():
    """Entry point for multilingual roleplay data processing."""
    processor = MultilingualRoleplayDataProcessor()
    
    try:
        result = processor.process_multilingual_data()
        print("\\nMultilingual data processing completed.")
        print("\\nSuggested next step:")
        print("  python src/roleplay/multilingual_roleplay_pca_analysis.py")
        
    except Exception as e:
        print(f"Data processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
