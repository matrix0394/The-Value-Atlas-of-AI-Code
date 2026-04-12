#!/usr/bin/env python3
"""
Run the multilingual roleplay pipeline used in the paper.

This script supports the end-to-end workflow for multilingual roleplay data:
1. Collect or reuse roleplay interview output.
2. Process the responses into IVS-aligned tables.
3. Project the roleplay outputs into the benchmark coordinate system.
4. Generate comparison figures and summary visualizations.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, Any
from dataclasses import asdict, is_dataclass

# Add the project root so local modules can be imported when the script is run directly.
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.roleplay_multilingual.multilingual_roleplay_data_processor import MultilingualRoleplayDataProcessor
from src.roleplay_multilingual.multilingual_roleplay_pca_analysis import MultilingualRoleplayPCAAnalysis
from src.roleplay_multilingual.multilingual_roleplay_visualization import MultilingualRoleplayVisualizer


def convert_to_serializable(obj):
    """Convert nested objects into JSON-serializable structures."""
    if is_dataclass(obj):
        return asdict(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


class RoleplayMultilingualAnalysisRunner:
    """Runner for the multilingual roleplay pipeline."""
    
    def __init__(self, project_root=None):
        """Initialize project paths used by the roleplay pipeline."""
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.data_path = self.project_root / "data" / "llm_interviews" / "multilingual"
        self.results_path = self.project_root / "results" / "roleplay_multilingual"
        self.country_values_data_path = self.project_root / "data" / "country_values"
        self.roleplay_english_data_path = self.project_root / "data" / "roleplay_English"
        
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Project root: {self.project_root}")
        print(f"Multilingual data directory: {self.data_path}")
        print(f"Results directory: {self.results_path}")
        print(f"Country benchmark directory: {self.country_values_data_path}")
        print(f"English roleplay directory: {self.roleplay_english_data_path}")
    
    def _get_config_stats(self):
        """Read the configuration files and summarize the interview scope."""
        stats = {
            'total_models': 0,
            'total_countries': 0,
            'total_languages': 0,
            'language_country_pairs': 0,
            'english_only_countries': 0,
            'bilingual_countries': 0,
            'models_list': [],
            'consensus_count': 5
        }
        
        try:
            models_config_file = self.project_root / "config" / "models" / "llm_models.json"
            if models_config_file.exists():
                with open(models_config_file, 'r', encoding='utf-8') as f:
                    models_config = json.load(f)
                    stats['total_models'] = len(models_config.get('models', {}))
                    stats['models_list'] = list(models_config.get('models', {}).keys())

            multilingual_config_file = self.project_root / "config" / "questions" / "multilingual" / "multilingual_questions_complete.json"
            if multilingual_config_file.exists():
                with open(multilingual_config_file, 'r', encoding='utf-8') as f:
                    multilingual_config = json.load(f)
                    
                    languages = multilingual_config.get('languages', {})
                    stats['total_languages'] = len(languages)
                    
                    countries_set = set()
                    language_country_pairs = 0
                    
                    for lang_code, lang_info in languages.items():
                        lang_countries = lang_info.get('countries', [])
                        countries_set.update(lang_countries)
                        language_country_pairs += len(lang_countries)
                    
                    stats['total_countries'] = len(countries_set)
                    stats['language_country_pairs'] = language_country_pairs
                    
                    en_countries = set(languages.get('en', {}).get('countries', []))
                    other_countries = set()
                    for lang_code, lang_info in languages.items():
                        if lang_code != 'en':
                            other_countries.update(lang_info.get('countries', []))
                    
                    stats['english_only_countries'] = len(en_countries - other_countries)
                    stats['bilingual_countries'] = len(en_countries & other_countries)
            
            comprehensive_config_file = self.project_root / "config" / "questions" / "multilingual" / "comprehensive_multilingual_config.json"
            if comprehensive_config_file.exists():
                with open(comprehensive_config_file, 'r', encoding='utf-8') as f:
                    comprehensive_config = json.load(f)
                    stats['consensus_count'] = comprehensive_config.get('comprehensive_test_config', {}).get('repeat_count', 5)
        
        except Exception as e:
            print(f"Could not read multilingual configuration: {e}")
        
        return stats
    
    def step0_multilingual_interview(self):
        """Collect or reuse multilingual roleplay interview output."""
        print("\n" + "="*60)
        print("Step 0: Multilingual roleplay interviewing")
        print("="*60)
        
        try:
            from src.roleplay_multilingual.multilingual_roleplay_interview import MultilingualRoleplayInterview
            
            interviewer = MultilingualRoleplayInterview(consensus_count=2, data_path=str(self.data_path))
            
            existing_files = list(self.data_path.glob("interview_data_*.json"))
            if existing_files:
                latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
                print(f"Found existing interview output: {latest_file}")
                
                print("\nSelect an action:")
                print("1. Reuse the existing interview output")
                print("2. Re-run the interview stage")
                config_stats = self._get_config_stats()
                print(f"3. Run the full interview suite ({config_stats['total_countries']} countries x {config_stats['total_models']} models x {config_stats['consensus_count']} rounds)")
                
                choice = input("\nEnter a choice (1/2/3): ").strip()
                
                if choice == '1':
                    print("Reusing the existing interview output.")
                    return True
                elif choice == '3':
                    print("Starting the full interview suite...")
                    return self._run_comprehensive_test()
            
            config_file = self.project_root / "config" / "questions" / "multilingual" / "multilingual_questions_complete.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                recommended_countries = config.get('recommended_countries', [])
            else:
                recommended_countries = []
            
            print(f"\nRecommended country entries: {len(recommended_countries)}")
            
            models_config_file = self.project_root / "config" / "models" / "llm_models.json"
            with open(models_config_file, 'r', encoding='utf-8') as f:
                models_config = json.load(f)
            model_names = list(models_config.get('models', {}).keys())
            
            entities = []
            for country_info in recommended_countries:
                country = country_info.get('country')
                languages = country_info.get('languages', [])
                for lang in languages:
                    entities.append(f"{country}_{lang}")
            
            print(f"Interview scope: {len(model_names)} models x {len(entities)} language-country pairs")
            
            results = interviewer.batch_interview(
                model_names=model_names,
                entities=entities,
                max_workers=4
            )
            
            if not results or results.get('successful_tasks', 0) == 0:
                print("Parallel interviewing failed; retrying with a single worker.")
                results = interviewer.batch_interview(
                    model_names=model_names,
                    entities=entities,
                    max_workers=1
                )
            
            if results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.data_path / f"interview_data_{timestamp}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(convert_to_serializable(results), f, indent=2, ensure_ascii=False)
                print(f"Saved interview output to: {output_file}")
                return True
            else:
                print("Interview stage failed.")
                return False
                
        except Exception as e:
            print(f"Multilingual interview stage failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step1_data_processing(self, test_type: str = None):
        """Process multilingual roleplay responses into analysis-ready tables."""
        print("\n" + "="*60)
        print("Step 1: Multilingual response processing")
        print("="*60)
        
        try:
            data_root = str(self.project_root / "data")
            processor = MultilingualRoleplayDataProcessor(data_path=data_root)
            
            interview_files = list(self.data_path.glob("interview_data_*.json"))
            results_file = None
            
            if interview_files:
                latest_interview = max(interview_files, key=lambda x: x.stat().st_mtime)
                print(f"Using latest interview file: {latest_interview.name}")
                results_file = str(latest_interview)
            else:
                print("No recent interview JSON file found; falling back to existing processed inputs.")
            
            processed_dir = self.data_path / "processed"
            processed_files = list(processed_dir.glob("llm_roleplay_ml_processed_responses_ivs_format_*.pkl")) if processed_dir.exists() else []
            if processed_files:
                latest_file = max(processed_files, key=lambda x: x.stat().st_mtime)
                print(f"Found processed data file: {latest_file.name}")
                
                choice = input("\nReprocess the multilingual responses? (y/n): ").strip().lower()
                if choice != 'y':
                    print("Reusing the existing processed data.")
                    return True
            
            print("\nProcessing multilingual roleplay responses...")
            processor.process_multilingual_data(results_file=results_file)
            
            print("\nMultilingual response processing completed.")
            return True
            
        except Exception as e:
            print(f"Multilingual response processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step2_pca_analysis(self, test_type: str = None):
        """Project multilingual roleplay data into the benchmark coordinate system."""
        print("\n" + "="*60)
        print("Step 2: PCA projection")
        print("="*60)
        
        try:
            from pathlib import Path
            config_country_path = self.project_root / "config" / "country" / "country_codes.pkl"
            benchmark_candidates = [
                self.country_values_data_path / "ivs_df.pkl",
                self.country_values_data_path / "country_scores_pca.json",
                self.country_values_data_path / "country_scores_pca.pkl",
            ]
            required_files = [config_country_path]
            
            for file_path in required_files:
                if not file_path.exists():
                    print(f"Missing required file: {file_path}")
                    return False

            if not any(path.exists() for path in benchmark_candidates):
                print("Missing benchmark input. Expected one of:")
                for path in benchmark_candidates:
                    print(f"  - {path}")
                return False
            
            processed_dir = self.data_path / "processed"
            processed_files = list(processed_dir.glob("llm_roleplay_ml_processed_responses_ivs_format_*.pkl")) if processed_dir.exists() else []
            if not processed_files:
                print("No processed roleplay file found. Run Step 1 first.")
                return False
            
            latest_processed_file = max(processed_files, key=lambda x: x.stat().st_mtime)
            print(f"Using latest processed file: {latest_processed_file.name}")
            
            print("\nInitializing PCA analyzer...")
            analyzer = MultilingualRoleplayPCAAnalysis(
                data_path=str(self.project_root / "data")
            )
            
            print("\nRunning PCA projection...")
            entity_scores = analyzer.run_multilingual_analysis_for_runner()
            
            print(f"\nPCA projection completed for {len(entity_scores)} entities.")
            
            if 'PC1_rescaled' in entity_scores.columns:
                print(f"PC1 range: [{entity_scores['PC1_rescaled'].min():.2f}, {entity_scores['PC1_rescaled'].max():.2f}]")
            if 'PC2_rescaled' in entity_scores.columns:
                print(f"PC2 range: [{entity_scores['PC2_rescaled'].min():.2f}, {entity_scores['PC2_rescaled'].max():.2f}]")
            
            if 'data_source' in entity_scores.columns:
                print("\nData-source counts:")
                print(entity_scores['data_source'].value_counts())
            
            return True
            
        except Exception as e:
            print(f"PCA projection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step3_language_comparison_analysis(self, test_type: str = None):
        """Create language-comparison summaries from the projected coordinates."""
        print("\n" + "="*60)
        print("Step 3: Language comparison analysis")
        print("="*60)
        
        try:
            print("\nInitializing visualizer...")
            visualizer = MultilingualRoleplayVisualizer(
                data_path=str(self.project_root / "data"),
                results_path=str(self.results_path)
            )
            
            print("\nBuilding the comparison outputs...")
            result = visualizer.create_complete_visualization_suite()
            
            print("\nLanguage comparison analysis completed.")
            print("\nGenerated files:")
            for key, path in result.items():
                print(f"   - {key}: {Path(path).name}")
            
            if 'index_file' in result:
                print("\nIndex file:")
                print(f"   {result['index_file']}")
            
            return True
            
        except Exception as e:
            print(f"Language comparison analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step4_visualization(self, test_type: str = None):
        """Create the roleplay dashboard and related visual outputs."""
        print("\n" + "="*60)
        print("Step 4: Visualization")
        print("="*60)
        
        dashboard_path = self.results_path / "roleplay_ml_dashboard"
        dashboard_path.mkdir(parents=True, exist_ok=True)
        print(f"Dashboard directory: {dashboard_path}")
        
        try:
            pca_dir = self.project_root / "data" / "llm_pca" / "multilingual"
            latest_file = pca_dir / "roleplay_ml_pca_entity_scores_latest.pkl"
            if latest_file.exists():
                entity_scores_path = latest_file
                print(f"Using latest PCA file: {entity_scores_path.name}")
            else:
                pca_files = list(pca_dir.glob("roleplay_ml_pca_entity_scores_*.pkl")) if pca_dir.exists() else []
                if not pca_files:
                    print("No PCA output file found. Run Step 2 first.")
                    return False
                
                entity_scores_path = max(pca_files, key=lambda x: x.stat().st_mtime)
                print(f"Using latest PCA file: {entity_scores_path.name}")
            
            file_time = datetime.fromtimestamp(entity_scores_path.stat().st_mtime)
            print(f"   Modified: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            print("\nLoading projected entity scores...")
            entity_scores = pd.read_pickle(entity_scores_path)
            print(f"Loaded projected scores for {len(entity_scores)} entities.")
            
            print("\nBuilding the visualization suite...")
            visualizer = MultilingualRoleplayVisualizer(
                data_path=str(self.project_root / "data"),
                results_path=str(self.results_path)
            )
            
            viz_result = visualizer.create_complete_visualization_suite()
            
            print("\nVisualization suite completed.")
            print("Generated files:")
            for key, path in viz_result.items():
                print(f"   - {key}: {Path(path).name}")
            
            return True
            
        except Exception as e:
            print(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_complete_analysis(self, skip_interview=False):
        """Run the full multilingual roleplay pipeline."""
        print("Starting the multilingual roleplay pipeline...")
        print(f"Start time: {datetime.now()}")
        
        success_steps = []
        
        if not skip_interview:
            print("\nYou can skip the interview stage if suitable output already exists.")
            if self.step0_multilingual_interview():
                success_steps.append("multilingual roleplay interviewing")
            else:
                print("Interview stage failed, but the pipeline will continue in case reusable data already exists.")
        else:
            print("Skipping the interview stage and reusing existing data.")
        
        if self.step1_data_processing():
            success_steps.append("multilingual response processing")
        else:
            print("Response-processing stage failed. Stopping pipeline.")
            return False
        
        if self.step2_pca_analysis():
            success_steps.append("PCA projection")
        else:
            print("PCA projection failed. Stopping pipeline.")
            return False
        
        if self.step3_language_comparison_analysis():
            success_steps.append("language comparison analysis")
        else:
            print("Language comparison analysis failed, but the pipeline will continue.")
        
        if self.step4_visualization():
            success_steps.append("visualization")
        else:
            print("Visualization failed, but earlier stages completed.")
        
        print("\n" + "="*60)
        print("Multilingual roleplay pipeline finished.")
        print("="*60)
        print(f"Completed steps: {', '.join(success_steps)}")
        print(f"Data directory: {self.data_path}")
        print(f"Results directory: {self.results_path}")
        print(f"Finish time: {datetime.now()}")
        
        return len(success_steps) >= 3
    
    def run_interview_only(self):
        """Run the interview stage only."""
        print("Running the multilingual roleplay interview stage only...")
        print(f"Start time: {datetime.now()}")
        
        success = self.step0_multilingual_interview()
        
        print("\n" + "="*60)
        if success:
            print("Interview stage completed.")
        else:
            print("Interview stage failed.")
        print("="*60)
        print(f"Finish time: {datetime.now()}")
        
        return success
    
    def _run_comprehensive_test_with_full_pipeline(self):
        """Run the comprehensive interview suite and then the analysis pipeline."""
        try:
            config_file = self.project_root / "config" / "questions" / "multilingual" / "comprehensive_multilingual_config.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                test_config = config.get('comprehensive_test_config', {})
                task_calc = test_config.get('task_calculation', {})
                language_country_pairs = task_calc.get('total_language_country_pairs', 59)
                total_countries = task_calc.get('total_countries', 32)
                total_models = task_calc.get('total_models', 7)
                repeat_count = test_config.get('repeat_count', 5)
            else:
                # Fallback values used when the comprehensive config is unavailable.
                language_country_pairs = 59
                total_countries = 32
                total_models = 7
                repeat_count = 5
            
            config_stats = self._get_config_stats()
            print(f"Starting the full interview suite ({config_stats['total_countries']} countries x {config_stats['total_models']} models x {config_stats['consensus_count']} rounds).")
            print(f"   - Non-English paired entities: {config_stats['bilingual_countries']}")
            print(f"   - English-only entities: {config_stats['english_only_countries']}")
            
            questions_per_interview = 10
            total_questions = language_country_pairs * total_models * repeat_count * questions_per_interview
            
            print("\nInterview workload:")
            print(f"   - Language-country pairs: {language_country_pairs}")
            print(f"   - Models: {total_models}")
            print(f"   - Consensus count: {repeat_count}")
            print(f"   - Questions per interview: {questions_per_interview}")
            print(f"   - Total question calls: {total_questions:,}")
            
            if self._run_multilingual_interview_with_config('comprehensive'):
                print("\nComprehensive interviewing completed. Running the analysis pipeline...")
                return self._run_full_analysis_pipeline()
            else:
                return False
                
        except Exception as e:
            print(f"Comprehensive test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_small_scale_test_with_full_pipeline(self):
        """Run a small interview test and then execute the analysis pipeline."""
        try:
            print("Starting a small test run (3 models x 5 countries x language combinations x 3 repeats).")
            print("   - Models: GPT-4o-mini, DeepSeek, Gemini")
            print("   - Countries: Egypt, China, Mexico, Russia, USA")
            
            if self._run_multilingual_interview_with_config('small_scale'):
                print("\nSmall test interviews completed. Running the analysis pipeline...")
                return self._run_full_analysis_pipeline('small_scale')
            else:
                return False
                
        except Exception as e:
            print(f"Small test run failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_incremental_languages_with_full_pipeline(self):
        """Interview only missing language-country pairs, then rerun analysis."""
        try:
            print("Starting incremental interviewing for missing language-country pairs.")
            
            config_file = self.project_root / "config" / "questions" / "multilingual" / "multilingual_questions_complete.json"
            if not config_file.exists():
                print(f"Missing configuration file: {config_file}")
                return False
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            all_pairs = set()
            for country_info in config.get('recommended_countries', []):
                country = country_info.get('country')
                languages = country_info.get('languages', [])
                for lang in languages:
                    all_pairs.add((country, lang))
            
            print(f"Language-country pairs defined in config: {len(all_pairs)}")
            
            latest_file = self._get_latest_roleplay_results_file()
            if latest_file:
                completed_pairs = self._extract_completed_pairs_from_file(latest_file)
                print(f"Completed language-country pairs: {len(completed_pairs)}")
            else:
                completed_pairs = set()
                print("No existing interview file found.")
            
            missing_pairs = all_pairs - completed_pairs
            
            if not missing_pairs:
                print("All language-country pairs are already complete.")
                print("\nRunning the analysis pipeline directly...")
                return self._run_full_analysis_pipeline()
            
            print(f"\nMissing language-country pairs: {len(missing_pairs)}")
            
            from src.roleplay_multilingual.multilingual_roleplay_interview import MultilingualRoleplayInterview
            interviewer = MultilingualRoleplayInterview(consensus_count=5, data_path=str(self.data_path))
            
            incremental_countries = []
            for country, lang in missing_pairs:
                incremental_countries.append({
                    'country': country,
                    'languages': [lang]
                })
            
            models_config_file = self.project_root / "config" / "models" / "llm_models.json"
            with open(models_config_file, 'r', encoding='utf-8') as f:
                models_config = json.load(f)
            model_names = list(models_config.get('models', {}).keys())
            
            entities = []
            for country_info in incremental_countries:
                country = country_info.get('country')
                languages = country_info.get('languages', [])
                for lang in languages:
                    entities.append(f"{country}_{lang}")
            
            results = interviewer.batch_interview(
                model_names=model_names,
                entities=entities,
                max_workers=4
            )
            
            if results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.data_path / f"interview_data_incremental_{timestamp}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(convert_to_serializable(results), f, indent=2, ensure_ascii=False)
                print(f"Saved incremental interview output to: {output_file}")
                
                print("\nIncremental interviewing completed. Running the analysis pipeline...")
                return self._run_full_analysis_pipeline()
            else:
                print("Incremental interviewing failed.")
                return False
        
        except Exception as e:
            print(f"Incremental interviewing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_chinese_variant_test_with_full_pipeline(self):
        """Run the Chinese-variant comparison and then execute the analysis pipeline."""
        try:
            print("Starting the Chinese-variant comparison (traditional vs simplified Chinese).")
            print("   - Regions: Hong Kong (zh-hk), Taiwan (zh-tw), Macau (zh-hk)")
            print("   - Reference variant: zh-cn")
            print("   - Consensus count: 5")
            
            chinese_variants = [
                {'country': 'Hong Kong', 'languages': ['zh-hk']},
                {'country': 'Taiwan', 'languages': ['zh-tw']},
                {'country': 'Macau', 'languages': ['zh-hk']}
            ]
            
            from src.roleplay_multilingual.multilingual_roleplay_interview import MultilingualRoleplayInterview
            interviewer = MultilingualRoleplayInterview(consensus_count=5, data_path=str(self.data_path))
            
            models_config_file = self.project_root / "config" / "models" / "llm_models.json"
            with open(models_config_file, 'r', encoding='utf-8') as f:
                models_config = json.load(f)
            model_names = list(models_config.get('models', {}).keys())
            
            entities = []
            for country_info in chinese_variants:
                country = country_info.get('country')
                languages = country_info.get('languages', [])
                for lang in languages:
                    entities.append(f"{country}_{lang}")
            
            results = interviewer.batch_interview(
                model_names=model_names,
                entities=entities,
                max_workers=4
            )
            
            if results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.data_path / f"interview_data_chinese_variants_{timestamp}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(convert_to_serializable(results), f, indent=2, ensure_ascii=False)
                print(f"Saved Chinese-variant interview output to: {output_file}")
                
                print("\nChinese-variant interviews completed. Running the analysis pipeline...")
                return self._run_full_analysis_pipeline('chinese_variants')
            else:
                print("Chinese-variant interviewing failed.")
                return False
            
        except Exception as e:
            print(f"Chinese-variant test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_data_processing_pipeline(self):
        """Run the analysis pipeline using existing interview output only."""
        try:
            print("Starting the analysis pipeline using existing interview output...")
            return self._run_full_analysis_pipeline()
        except Exception as e:
            print(f"Pipeline run failed: {e}")
            return False
    
    def _run_multilingual_interview_with_config(self, config_type: str):
        """Run interviews using one of the predefined multilingual configs."""
        try:
            from src.roleplay_multilingual.multilingual_roleplay_interview import MultilingualRoleplayInterview
            
            if config_type == 'comprehensive':
                config_file = self.project_root / "config" / "questions" / "multilingual" / "comprehensive_multilingual_config.json"
                consensus_count = 5
            elif config_type == 'small_scale':
                config_file = self.project_root / "config" / "small_scale_test_config.json"
                consensus_count = 3
            else:
                print(f"Unknown config type: {config_type}")
                return False
            
            if not config_file.exists():
                print(f"Missing configuration file: {config_file}")
                return False
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if config_type == 'comprehensive':
                country_list = config.get('comprehensive_test_config', {}).get('countries', [])
            elif config_type == 'small_scale':
                country_list = config.get('small_scale_test', {}).get('countries', [])
            
            print(f"Config file: {config_file.name}")
            print(f"Countries in config: {len(country_list)}")
            print(f"Consensus count: {consensus_count}")
            
            interviewer = MultilingualRoleplayInterview(
                consensus_count=consensus_count,
                data_path=str(self.data_path)
            )
            
            if config_type == 'small_scale':
                model_names = config.get('small_scale_test', {}).get('models', [])
            else:
                models_config_file = self.project_root / "config" / "models" / "llm_models.json"
                with open(models_config_file, 'r', encoding='utf-8') as f:
                    models_config = json.load(f)
                model_names = list(models_config.get('models', {}).keys())
            
            entities = []
            for country_info in country_list:
                country = country_info.get('country')
                languages = country_info.get('languages', [])
                for lang in languages:
                    entities.append(f"{country}_{lang}")
            
            print(f"Interview scope: {len(model_names)} models x {len(entities)} language-country pairs = {len(model_names) * len(entities)} tasks")
            
            print("\nStarting batched interviews...")
            results = interviewer.batch_interview(
                model_names=model_names,
                entities=entities,
                max_workers=4
            )
            
            if not results or results.get('successful_tasks', 0) == 0:
                print("Parallel interviewing failed; retrying with a single worker.")
                results = interviewer.batch_interview(
                    model_names=model_names,
                    entities=entities,
                    max_workers=1
            )
            
            if results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.data_path / f"interview_data_{config_type}_{timestamp}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(convert_to_serializable(results), f, indent=2, ensure_ascii=False)
                print(f"Saved interview output to: {output_file}")
                return True
            else:
                print("Interview stage failed.")
                return False
                
        except Exception as e:
            print(f"Interview run failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_full_analysis_pipeline(self, test_type: str = None):
        """Run processing, PCA projection, comparison analysis, and visualization."""
        try:
            print("\n" + "="*60)
            print("Running the multilingual roleplay pipeline")
            print("="*60)
            
            if not self.step1_data_processing(test_type):
                return False
            
            if not self.step2_pca_analysis(test_type):
                return False
            
            self.step3_language_comparison_analysis(test_type)
            
            self.step4_visualization(test_type)
            
            print("\nMultilingual roleplay pipeline completed.")
            return True
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_latest_roleplay_results_file(self):
        """Return the latest saved roleplay interview file, if available."""
        interview_files = list(self.data_path.glob("interview_data_*.json"))
        if interview_files:
            return max(interview_files, key=lambda x: x.stat().st_mtime)
        return None
    
    def _extract_completed_pairs_from_file(self, file_path: Path):
        """Extract completed language-country pairs from a saved interview file."""
        completed_pairs = set()
        
        try:
            if file_path.suffix == '.pkl':
                import pickle
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:  # .json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
        except Exception as e:
            print(f"Could not read {file_path.name}: {e}")
            return completed_pairs
        
        if isinstance(data, dict):
            for country, country_data in data.items():
                if isinstance(country_data, dict):
                    for language in country_data.keys():
                        completed_pairs.add((country, language))
        
        return completed_pairs
    
    def _run_comprehensive_test(self) -> bool:
        """Run the legacy comprehensive interview suite for compatibility."""
        try:
            from src.roleplay_multilingual.multilingual_roleplay_interview import MultilingualRoleplayInterview
            
            config_file = self.project_root / "config" / "questions" / "multilingual" / "comprehensive_multilingual_config.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                country_list = config.get('comprehensive_test_config', {}).get('countries', [])
            else:
                print("Missing comprehensive_multilingual_config.json")
                return False
            
            models_config_file = self.project_root / "config" / "models" / "llm_models.json"
            with open(models_config_file, 'r', encoding='utf-8') as f:
                models_config = json.load(f)
            model_names = list(models_config.get('models', {}).keys())
            
            entities = []
            for country_info in country_list:
                country = country_info.get('country')
                languages = country_info.get('languages', [])
                for lang in languages:
                    entities.append(f"{country}_{lang}")
            
            interviewer = MultilingualRoleplayInterview(consensus_count=5, data_path=str(self.data_path))
            results = interviewer.batch_interview(
                model_names=model_names,
                entities=entities,
                max_workers=4
            )
            
            if results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.data_path / f"interview_data_comprehensive_{timestamp}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(convert_to_serializable(results), f, indent=2, ensure_ascii=False)
                print(f"Saved comprehensive interview output to: {output_file}")
                return True
            else:
                print("Comprehensive interview suite failed.")
                return False
                
        except Exception as e:
            print(f"Comprehensive interview suite failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Entry point for the multilingual roleplay pipeline."""
    
    print("\n" + "="*70)
    print("Multilingual roleplay analysis")
    print("="*70)
    
    temp_runner = RoleplayMultilingualAnalysisRunner()
    config_stats = temp_runner._get_config_stats()
    
    print("\nSelect an action:")
    print(f"\n1. Re-run the full interview suite ({config_stats['total_countries']} countries x {config_stats['total_models']} models x {config_stats['consensus_count']} rounds)")
    print(f"   - Non-English entities with paired local-language and English prompts: {config_stats['bilingual_countries']}")
    print(f"   - English-only entities: {config_stats['english_only_countries']}")
    print(f"   - Total language-country pairs: {config_stats['language_country_pairs']}")
    print("\n2. Run a small interview test (3 models x 5 countries)")
    print("   - Models: GPT-4o-mini, DeepSeek, Gemini")
    print("   - Countries: Egypt, China, Mexico, Russia, USA")
    print("   - Consensus count: 3")
    print("\n3. Reuse existing data and run processing, PCA, and visualization")
    print("\n4. Run the Chinese-variant comparison (traditional vs simplified Chinese)")
    print("   - Regions: Hong Kong, Taiwan, Macau")
    print(f"   - Consensus count: {config_stats['consensus_count']}")
    print("\n5. Incrementally fill in missing language-country pairs")
    print("   - Detect unfinished pairs from the multilingual configuration")
    print("   - Then rerun processing, PCA, and visualization")
    print("\n0. Exit")
    print("\n" + "="*70)
    
    try:
        runner = RoleplayMultilingualAnalysisRunner()
        
        while True:
            try:
                choice = input("\nEnter a choice (0-5): ").strip()
                
                if choice == '0':
                    print("Exiting.")
                    return 0
                elif choice in ['1', '2', '3', '4', '5']:
                    break
                else:
                    print("Invalid choice. Enter a value from 0 to 5.")
            except (KeyboardInterrupt, EOFError):
                print("\n\nCancelled by user.")
                return 0
        
        success = False
        
        if choice == '1':
            config_stats = runner._get_config_stats()
            print(f"\nOption 1: Re-run the full interview suite ({config_stats['total_countries']} countries x {config_stats['total_models']} models x {config_stats['consensus_count']} rounds)")
            print("\nEstimated workload:")
            print(f"   - Language-country pairs: {config_stats['language_country_pairs']}")
            print(f"   - Models: {config_stats['total_models']}")
            print(f"   - Consensus count: {config_stats['consensus_count']}")
            questions_per_interview = 10
            total_questions = config_stats['language_country_pairs'] * config_stats['total_models'] * config_stats['consensus_count'] * questions_per_interview
            print(f"   - Total question calls: {total_questions:,}")
            confirm = input("\nProceed? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return 0
            success = runner._run_comprehensive_test_with_full_pipeline()
        
        elif choice == '2':
            print("\nOption 2: Run a small interview test")
            confirm = input("\nProceed? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return 0
            success = runner._run_small_scale_test_with_full_pipeline()
        
        elif choice == '3':
            print("\nOption 3: Reuse existing data")
            confirm = input("\nProceed? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return 0
            success = runner._run_data_processing_pipeline()
        
        elif choice == '4':
            print("\nOption 4: Run the Chinese-variant comparison")
            confirm = input("\nProceed? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return 0
            success = runner._run_chinese_variant_test_with_full_pipeline()
        
        elif choice == '5':
            print("\nOption 5: Run incremental interviewing")
            confirm = input("\nProceed? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return 0
            success = runner._run_incremental_languages_with_full_pipeline()
        
        if success:
            print("\nPipeline completed.")
            return 0
        else:
            print("\nPipeline failed. See logs above.")
            return 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
