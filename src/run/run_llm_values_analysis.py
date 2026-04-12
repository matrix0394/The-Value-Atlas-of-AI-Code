#!/usr/bin/env python3
"""
Run the intrinsic LLM-values pipeline used in the paper.

This script supports four stages:
1. Collect or reuse interview responses from the configured models.
2. Convert responses into the IVS-aligned analysis format.
3. Project the processed responses into the benchmark coordinate system.
4. Generate the summary figures used for inspection and reporting.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add the project root so local modules can be imported when the script is run directly.
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.llm_values.llm_interview import LLMInterview
from src.llm_values.llm_data_processor import LLMDataProcessor
from src.llm_values.llm_pca_analysis import LLMPCAAnalyzer
from src.llm_values.llm_visualization import LLMCulturalMapVisualizer


class LLMValuesAnalysisRunner:
    """Runner for the intrinsic LLM-values pipeline."""
    
    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.data_path = self.project_root / "data"
        self.results_path = self.project_root / "results"
        
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Project root: {self.project_root}")
        print(f"Data directory: {self.data_path}")
        print(f"Results directory: {self.results_path}")
    
    def step1_llm_interview(self, force_rerun: bool = False, consensus_count: int = 5, 
                           model_names: list = None, skip_existing: bool = False):
        """Collect or reuse intrinsic interview responses."""
        print("\n" + "="*80)
        print("Step 1: LLM interviewing")
        print("="*80)
        
        interview_pattern = self.data_path / "llm_values" / "raw_interview_*.pkl"
        existing_files = list(self.data_path.glob("llm_values/raw_interview_*.pkl"))
        
        if existing_files and not force_rerun:
            latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
            print(f"Found existing interview output: {latest_file}")
            
            try:
                import json
                json_file = str(latest_file).replace('.pkl', '.json')
                if Path(json_file).exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        metadata = data.get('metadata', {})
                        results = data.get('results', [])
                        
                        print("Existing dataset summary:")
                        print(f"   - Models: {metadata.get('total_entities', 0)}")
                        print(f"   - Questions: {metadata.get('total_questions', 0)}")
                        print(f"   - Timestamp: {metadata.get('timestamp', 'unknown')}")
                        print(f"   - Consensus count: {metadata.get('consensus_count', 1)}")
                
                response = input("\nRerun the interview stage? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    print("Skipping interview stage and reusing existing data.")
                    return str(latest_file)
            except Exception as e:
                print(f"Could not inspect existing interview output: {e}")
                print("Re-running the interview stage.")
        
        print(f"\nInitializing interviewer (consensus_count={consensus_count})...")
        interviewer = LLMInterview(
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
                print("Check model names and API-key configuration.")
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
        
        print("\nStarting batched interviews...")
        print(f"   - Models: {len(available_models)}")
        print(f"   - Consensus count: {consensus_count}")
        print(f"   - Incremental mode: {'yes' if skip_existing else 'no'}")
        if not skip_existing:
            print(f"   - Estimated API calls: {len(available_models) * 10 * consensus_count}")
        
        results = interviewer.batch_interview(
            model_names=available_models,
            max_workers=1,
            skip_existing=skip_existing
        )
        
        if not results or not results.get('results'):
            print("Interview stage failed: no valid results were returned.")
            return None
        
        print("\nSaving interview results...")
        output_file = interviewer.save_results(results)
        
        print("\nInterview stage completed.")
        print("Summary:")
        print(f"   - Success rate: {results['success_rate']:.1%}")
        print(f"   - Successful tasks: {results['successful_tasks']}/{results['total_tasks']}")
        print(f"   - Output file: {output_file}")
        
        return output_file
    
    def step2_data_processing(self):
        """Convert interview output into the IVS-aligned analysis format."""
        print("\n" + "="*80)
        print("Step 2: Response processing")
        print("="*80)
        
        print("Running LLMDataProcessor...")
        processor = LLMDataProcessor(data_dir=str(self.data_path))
        
        print("Converting responses to IVS-aligned format...")
        output_file = processor.save_processed_data()
        
        if output_file and Path(output_file).exists():
            data = pd.read_pickle(output_file)
            print("\nResponse processing completed.")
            print("Processed dataset summary:")
            print(f"   - Models: {len(data)}")
            print(f"   - Columns: {len(data.columns)}")
            print("   - IVS items: A008-G006, Y002, Y003")
            print("   - Format: aligned with the benchmark valid_data table")
            print(f"   - Output file: {output_file}")
            return str(output_file)
        else:
            print("Response processing failed.")
            return None
    
    def step3_pca_analysis(self, use_fixed_pca: bool = True):
        """Project the processed responses into the benchmark coordinate system."""
        print("\n" + "="*80)
        print("Step 3: PCA projection")
        print("="*80)
        
        print("Initializing LLMPCAAnalyzer...")
        analyzer = LLMPCAAnalyzer(data_path=str(self.data_path))
        
        if use_fixed_pca:
            print("Projecting with the fixed benchmark PCA model.")
            entity_scores = analyzer.run_llm_analysis_for_runner(use_fixed_pca=True)
        else:
            print("Refitting PCA from scratch. This mode is not recommended for paper reproduction.")
            entity_scores = analyzer.run_full_analysis()
        
        if entity_scores is None or entity_scores.empty:
            print("PCA projection failed.")
            return None
        
        print("\nPCA projection completed.")
        print("Projection summary:")
        print(f"   - Total entities: {len(entity_scores)}")
        
        if 'is_llm' in entity_scores.columns:
            llm_count = entity_scores['is_llm'].sum()
            country_count = (~entity_scores['is_llm']).sum()
            print(f"   - LLM models: {llm_count}")
            print(f"   - Reference countries: {country_count}")
        
        if 'PC1_rescaled' in entity_scores.columns:
            print(f"   - PC1 range: [{entity_scores['PC1_rescaled'].min():.2f}, {entity_scores['PC1_rescaled'].max():.2f}]")
            print(f"   - PC2 range: [{entity_scores['PC2_rescaled'].min():.2f}, {entity_scores['PC2_rescaled'].max():.2f}]")
        
        print(f"   - Output file: {self.data_path}/llm_pca_entity_scores.pkl")
        
        return entity_scores
    
    def step4_visualization(self):
        """Generate the main intrinsic-values figures and dashboard outputs."""
        print("\n" + "="*80)
        print("Step 4: Visualization")
        print("="*80)
        
        print("Initializing visualizer...")
        visualizer = LLMCulturalMapVisualizer(
            data_path=str(self.data_path),
            results_path=str(self.results_path)
        )
        
        print("Generating dashboard outputs...")
        saved_files = visualizer.create_llm_dashboard()
        
        print("\nVisualization completed.")
        print(f"Generated {len(saved_files)} figure files.")
        for name, path in saved_files.items():
            print(f"   - {name}: {path}")
        
        return saved_files
    
    def run_full_analysis(self, force_interview: bool = False, 
                         consensus_count: int = 5,
                         skip_steps: list = None,
                         model_names: list = None,
                         skip_existing: bool = False):
        """Run the full intrinsic-values pipeline."""
        if skip_steps is None:
            skip_steps = []
        
        print("\n" + "="*80)
        print("Intrinsic LLM-values pipeline")
        print("="*80)
        print("Configuration:")
        print(f"   - consensus_count: {consensus_count}")
        print(f"   - force_interview: {force_interview}")
        print(f"   - skip_existing: {skip_existing}")
        print(f"   - model_names: {model_names if model_names else 'all configured models'}")
        print(f"   - skip_steps: {skip_steps if skip_steps else 'none'}")
        
        results = {}
        
        try:
            if 'interview' not in skip_steps:
                interview_result = self.step1_llm_interview(
                    force_rerun=force_interview,
                    consensus_count=consensus_count,
                    model_names=model_names,
                    skip_existing=skip_existing
                )
                
                if not interview_result:
                    print("Interview stage failed. Stopping pipeline.")
                    return None
                
                results['interview_result'] = interview_result
            else:
                print("\nSkipping Step 1: interviewing")
            
            if 'process' not in skip_steps:
                processed_file = self.step2_data_processing()
                if not processed_file:
                    print("Response-processing stage failed. Stopping pipeline.")
                    return None
                results['processed_file'] = processed_file
            else:
                print("\nSkipping Step 2: response processing")
            
            if 'pca' not in skip_steps:
                entity_scores = self.step3_pca_analysis()
                
                if entity_scores is None:
                    print("PCA projection failed. Stopping pipeline.")
                    return None
                
                results['entity_scores'] = entity_scores
            else:
                print("\nSkipping Step 3: PCA projection")
            
            if 'visualize' not in skip_steps:
                visualizations = self.step4_visualization()
                results['visualizations'] = visualizations
            else:
                print("\nSkipping Step 4: visualization")
            
            print("\n" + "="*80)
            print("Intrinsic LLM-values pipeline completed.")
            print("="*80)
            print("Outputs:")
            for key, value in results.items():
                if isinstance(value, dict):
                    print(f"   {key}: {len(value)} files")
                else:
                    print(f"   {key}: {value}")
            
            return results
            
        except Exception as e:
            print(f"\nPipeline error: {e}")
            import traceback
            traceback.print_exc()
            return None


def interactive_menu():
    """Display the interactive menu and return the selected action."""
    print("\n" + "="*70)
    print("Intrinsic LLM-values analysis")
    print("="*70)
    print("\nSelect an action:")
    print("\n1. Re-run all interviews (all configured models)")
    print("2. Interview new models and merge with existing data")
    print("3. Run a small test interview (2 models, 3 consensus rounds)")
    print("4. Reuse existing data and run processing, PCA, and figures")
    print("0. Exit")
    print("\n" + "="*70)
    
    while True:
        try:
            choice = input("\nEnter a choice (0-4): ").strip()
            
            if choice == '0':
                print("Exiting.")
                return
            elif choice in ['1', '2', '3', '4']:
                return choice
            else:
                print("Invalid choice. Enter a value from 0 to 4.")
        except (KeyboardInterrupt, EOFError):
            print("\n\nCancelled by user.")
            return None


def main():
    """Entry point for the intrinsic-values pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Intrinsic LLM-values analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run_llm_values_analysis.py
  
  # Force a fresh interview run
  python run_llm_values_analysis.py --force-interview
  
  # Run a single stage
  python run_llm_values_analysis.py --step pca
  
  # Skip selected stages
  python run_llm_values_analysis.py --skip interview process
        """
    )
    
    parser.add_argument('--force-interview', action='store_true', 
                       help='force a fresh interview run')
    parser.add_argument('--consensus-count', type=int, default=5,
                       help='number of repeated runs per question (default: 5)')
    parser.add_argument('--step', 
                       choices=['interview', 'process', 'pca', 'visualize', 'all'],
                       help='run a single stage')
    parser.add_argument('--skip', nargs='+',
                       choices=['interview', 'process', 'pca', 'visualize'],
                       help='skip one or more stages')
    parser.add_argument('--interactive', action='store_true',
                       help='use the interactive menu')
    
    args = parser.parse_args()
    
    runner = LLMValuesAnalysisRunner()

    use_interactive = args.interactive or (not args.step and not args.force_interview and not args.skip)
    
    if use_interactive:
        choice = interactive_menu()
        
        if choice == '1':
            print("\nOption 1: Re-run all interviews")
            print("  - Interview all configured models")
            print("  - Consensus count: 5")
            print("  - Then run processing, PCA projection, and visualization")
            
            confirm = input("\nProceed? (y/n): ").strip().lower()
            if confirm == 'y':
                runner.run_full_analysis(
                    force_interview=True,
                    consensus_count=5,
                    skip_steps=[]
                )
            else:
                print("Cancelled.")
        
        elif choice == '2':
            print("\nOption 2: Interview new models and merge with existing data")
            print("  - Detect models that already have completed output")
            print("  - Interview only the newly requested models")
            print("  - Merge results before processing, PCA, and visualization")
            
            import json
            config_path = runner.project_root / 'config' / 'models' / 'llm_models.json'
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                available_models_list = list(config.get('models', {}).keys())
            except Exception as e:
                print(f"Could not load model configuration: {e}")
                return
            
            print("\nAvailable models:")
            model_map = {}
            for i, model in enumerate(available_models_list, 1):
                print(f"  {i}. {model}")
                model_map[str(i)] = model
            
            print("\nEnter one or more model numbers separated by commas, or press Enter to process all unfinished models.")
            
            model_choice = input("\nSelect models (Enter = all unfinished models): ").strip()
            
            try:
                if model_choice:
                    model_choice = model_choice.replace('，', ',')
                    selected = [model_map[c.strip()] for c in model_choice.split(',') if c.strip() in model_map]
                    if not selected:
                        print("No valid models were selected.")
                        return
                    
                    print("\nSelected models:")
                    for m in selected:
                        print(f"  - {m}")
                    
                    confirm = input("\nProceed? (y/n): ").strip().lower()
                    if confirm == 'y':
                        runner.run_full_analysis(
                            force_interview=False,
                            consensus_count=5,
                            model_names=selected,
                            skip_existing=True,
                            skip_steps=[]
                        )
                    else:
                        print("Cancelled.")
                else:
                    print("\nAll unfinished models will be processed automatically.")
                    confirm = input("\nProceed? (y/n): ").strip().lower()
                    if confirm == 'y':
                        runner.run_full_analysis(
                            force_interview=False,
                            consensus_count=5,
                            model_names=None,
                            skip_existing=True,
                            skip_steps=[]
                        )
                    else:
                        print("Cancelled.")
            except Exception as e:
                print(f"Input error: {e}")
        
        elif choice == '3':
            print("\nOption 3: Run a small interview test")
            print("  - Interview 2 models only")
            print("  - Consensus count: 3")
            
            import json
            config_path = runner.project_root / 'config' / 'models' / 'llm_models.json'
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                available_models_list = list(config.get('models', {}).keys())
            except Exception as e:
                print(f"Could not load model configuration: {e}")
                return
            
            print("\nAvailable models:")
            model_map = {}
            for i, model in enumerate(available_models_list, 1):
                print(f"  {i}. {model}")
                model_map[str(i)] = model
            
            print("\nEnter exactly two model numbers separated by commas (for example: 1,5).")
            
            model_choice = input("\nSelect models: ").strip()
            
            try:
                model_choice = model_choice.replace('，', ',')
                selected = [model_map[c.strip()] for c in model_choice.split(',') if c.strip() in model_map]
                if len(selected) != 2:
                    print("Select exactly two models.")
                    return
                
                print("\nSelected models:")
                for m in selected:
                    print(f"  - {m}")
                
                confirm = input("\nProceed? (y/n): ").strip().lower()
                if confirm == 'y':
                    runner.run_full_analysis(
                        force_interview=True,
                        consensus_count=3,
                        model_names=selected,
                        skip_steps=[]
                    )
                else:
                    print("Cancelled.")
            except Exception as e:
                print(f"Input error: {e}")
        
        elif choice == '4':
            print("\nOption 4: Reuse existing data")
            print("  - Skip the interview stage")
            print("  - Run response processing, PCA projection, and visualization")
            
            confirm = input("\nProceed? (y/n): ").strip().lower()
            if confirm == 'y':
                runner.run_full_analysis(
                    force_interview=False,
                    skip_steps=['interview']
                )
            else:
                print("Cancelled.")
        
        elif choice is None:
            return
    
    else:
        if args.step:
            if args.step == 'interview':
                runner.step1_llm_interview(
                    force_rerun=args.force_interview,
                    consensus_count=args.consensus_count
                )
            elif args.step == 'process':
                runner.step2_data_processing()
            elif args.step == 'pca':
                runner.step3_pca_analysis()
            elif args.step == 'visualize':
                runner.step4_visualization()
        else:
            runner.run_full_analysis(
                force_interview=args.force_interview,
                consensus_count=args.consensus_count,
                skip_steps=args.skip if args.skip else []
            )


if __name__ == "__main__":
    main()
