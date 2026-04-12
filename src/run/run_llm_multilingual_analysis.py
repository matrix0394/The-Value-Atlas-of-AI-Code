#!/usr/bin/env python3
"""
Run the multilingual intrinsic-values pipeline used in the paper.

This script supports three stages:
1. Interview the configured models in the six UN official languages.
2. Convert the responses into the IVS-aligned analysis format.
3. Project the processed responses into the benchmark coordinate system.
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.llm_values.llm_multilingual_interview import LLMMultilingualInterview, UN_LANGUAGE_NAMES_ZH
from src.llm_values.llm_multilingual_data_processor import LLMMultilingualDataProcessor


class LLMMultilingualAnalysisRunner:
    """Runner for the multilingual intrinsic-values pipeline."""

    UN_OFFICIAL_LANGUAGES = ['en', 'fr', 'es', 'ru', 'ar', 'zh-cn']

    def __init__(self, project_root=None):
        """Initialize the runner and create standard output directories."""
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.data_path = self.project_root / "data"
        self.results_path = self.project_root / "results"
        self.config_path = self.project_root / "config"

        self.data_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)

        print(f"Project root: {self.project_root}")
        print(f"Data directory: {self.data_path}")
        print(f"Results directory: {self.results_path}")

    def get_available_models(self) -> list:
        """Return the models currently available under the local configuration."""
        import json
        config_file = self.config_path / 'models' / 'llm_models.json'
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return list(config.get('models', {}).keys())
        except Exception as e:
            print(f"Could not load the model configuration: {e}")
            return []

    def step1_multilingual_interview(self,
                                     model_names: list = None,
                                     languages: list = None,
                                     skip_existing: bool = False,
                                     force_rerun: bool = False,
                                     consensus_count: int = 5) -> dict:
        """Interview the configured models in multiple languages."""
        print("\n" + "="*80)
        print("Step 1: Multilingual interviewing")
        print("="*80)

        if languages is None:
            languages = self.UN_OFFICIAL_LANGUAGES.copy()

        valid_languages = [lang for lang in languages if lang in self.UN_OFFICIAL_LANGUAGES]
        if not valid_languages:
            print(f"No valid languages were provided: {languages}")
            print(f"Supported languages: {self.UN_OFFICIAL_LANGUAGES}")
            return None

        print(f"\nInterview languages: {len(valid_languages)}")
        for lang in valid_languages:
            print(f"   - {lang}: {UN_LANGUAGE_NAMES_ZH.get(lang, lang)}")

        print(f"\nInitializing multilingual interviewer (consensus_count={consensus_count})...")
        interviewer = LLMMultilingualInterview(
            consensus_count=consensus_count,
            data_path=str(self.data_path)
        )

        available_models = [
            name for name in interviewer.model_configs.keys()
            if name in interviewer.api_keys
        ]

        if model_names:
            available_models = [m for m in model_names if m in available_models]
            if not available_models:
                print(f"None of the requested models are currently available: {model_names}")
                print("Check the model names and API-key configuration.")
                return None
            print(f"\nUsing a user-specified subset of {len(available_models)} models.")
        else:
            print(f"\nDiscovered {len(available_models)} available models.")

        if not available_models:
            print("No models are available. Check the API-key configuration.")
            return None

        for i, model in enumerate(available_models, 1):
            region = interviewer.model_configs[model].get('region', 'Unknown')
            print(f"   {i}. {model} ({region})")

        total_combinations = len(available_models) * len(valid_languages)
        print("\nInterview workload:")
        print(f"   - Models: {len(available_models)}")
        print(f"   - Languages: {len(valid_languages)}")
        print(f"   - Total combinations: {total_combinations}")
        print(f"   - Consensus count: {consensus_count}")
        print(f"   - Skip completed combinations: {'yes' if skip_existing else 'no'}")

        if not skip_existing:
            print(f"   - Estimated API calls: {total_combinations * 10 * consensus_count}")

        print("\nStarting batched multilingual interviews...")
        results = interviewer.batch_multilingual_interview(
            model_names=available_models,
            languages=valid_languages,
            skip_existing=skip_existing
        )

        if not results or not results.get('results'):
            print("Interviewing finished, but no new results were produced.")
            return results

        print("\nInterviewing completed.")
        print("Summary:")
        print(f"   - Success rate: {results['success_rate']:.1%}")
        print(f"   - Successful tasks: {results['successful_tasks']}/{results['total_tasks']}")

        return results

    def step2_data_processing(self) -> dict:
        """Convert multilingual responses into the IVS-aligned analysis format."""
        print("\n" + "="*80)
        print("Step 2: Response processing")
        print("="*80)

        print("Running LLMMultilingualDataProcessor...")
        processor = LLMMultilingualDataProcessor(data_path=str(self.data_path))

        raw_results = processor.load_raw_results()
        if not raw_results:
            print("No multilingual interview data was found.")
            return None

        processed_data = processor.convert_to_ivs_format()
        if processed_data.empty:
            print("Response conversion failed.")
            return None

        json_file, pkl_file = processor.save_processed_results()

        print("\nResponse processing completed.")
        print("Processed dataset summary:")
        print(f"   - Rows: {len(processed_data)}")
        print(f"   - Columns: {len(processed_data.columns)}")
        print("Saved outputs:")
        print(f"   - JSON: {json_file}")
        print(f"   - Pickle: {pkl_file}")

        return {
            'processed_data': processed_data,
            'json_file': json_file,
            'pkl_file': pkl_file
        }

    def step3_pca_analysis(self, use_fixed_pca: bool = True):
        """Project the processed multilingual responses into the benchmark space."""
        print("\n" + "="*80)
        print("Step 3: PCA projection")
        print("="*80)

        processor = LLMMultilingualDataProcessor(data_path=str(self.data_path))
        raw_results = processor.load_raw_results()
        if raw_results:
            processor.convert_to_ivs_format()
        else:
            candidates = [
                self.data_path / "llm_interviews" / "intrinsic" / "multilingual_ivs_format.pkl",
                self.data_path / "llm_interviews" / "intrinsic" / "multilingual_ivs_format.json",
            ]
            loaded = False
            for candidate in candidates:
                if not candidate.exists():
                    continue
                print(f"Loading released processed multilingual table: {candidate}")
                if candidate.suffix == ".pkl":
                    processor.processed_data = pd.read_pickle(candidate)
                else:
                    processor.processed_data = pd.DataFrame(json.loads(candidate.read_text()))
                loaded = True
                break

            if not loaded:
                print("No multilingual input data were found.")
                return None

        print(f"\nRunning PCA projection (use_fixed_pca={use_fixed_pca})...")
        entity_scores = processor.run_pca_analysis(use_fixed_pca=use_fixed_pca)

        if entity_scores is None or entity_scores.empty:
            print("PCA projection failed.")
            return None

        print("\nPCA projection completed.")
        print("Projection summary:")
        print(f"   - Total entities: {len(entity_scores)}")

        if 'PC1_rescaled' in entity_scores.columns:
            print(f"   - PC1 range: [{entity_scores['PC1_rescaled'].min():.2f}, {entity_scores['PC1_rescaled'].max():.2f}]")
            print(f"   - PC2 range: [{entity_scores['PC2_rescaled'].min():.2f}, {entity_scores['PC2_rescaled'].max():.2f}]")

        return entity_scores


def build_argparser():
    parser = argparse.ArgumentParser(description="Run multilingual intrinsic LLM analysis")
    parser.add_argument('--models', nargs='*', default=None)
    parser.add_argument('--languages', nargs='*', default=None)
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--consensus-count', type=int, default=5)
    parser.add_argument('--step', choices=['1', '2', '3', 'all'], default='all')
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    runner = LLMMultilingualAnalysisRunner()

    if args.step in ('1', 'all'):
        runner.step1_multilingual_interview(
            model_names=args.models,
            languages=args.languages,
            skip_existing=args.skip_existing,
            force_rerun=args.force,
            consensus_count=args.consensus_count,
        )

    if args.step in ('2', 'all'):
        runner.step2_data_processing()

    if args.step in ('3', 'all'):
        runner.step3_pca_analysis(use_fixed_pca=True)


if __name__ == "__main__":
    main()
