"""Intrinsic LLM interviewing built on the shared BaseInterview workflow."""

import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.base.base_interview import BaseInterview
from src.base.ivs_questionnaire import LLMResponse


class LLMInterview(BaseInterview):
    """Interview helper for the intrinsic-value stage."""

    def _get_canonical_interview_dir(self) -> Path:
        """Canonical cache directory used by the cleaned reproducibility repo."""
        return self.data_path / "llm_interviews" / "intrinsic" / "interview_raw"

    def _get_legacy_interview_dir(self) -> Path:
        """Legacy cache directory kept for backwards compatibility."""
        return self.data_path / "llm_values" / "interview_raw"

    def _get_interview_dirs_for_loading(self) -> List[Path]:
        """Return candidate directories in overwrite order (legacy first)."""
        dirs = [self._get_legacy_interview_dir(), self._get_canonical_interview_dir()]
        unique_dirs = []
        seen = set()
        for path in dirs:
            if path not in seen:
                unique_dirs.append(path)
                seen.add(path)
        return unique_dirs
    
    def __init__(self, max_retry: int = 3, consensus_count: int = 5, data_path: str = "data"):
        """Initialize the intrinsic interview runner."""
        super().__init__(max_retry=max_retry, data_path=data_path, 
                        consensus_count=consensus_count)

        self.iv_qns = ["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006", "Y002", "Y003"]

        self.system_prompt = """CRITICAL: You are a survey response system. Your ONLY task is to output numbers.

STRICT OUTPUT FORMAT:
1. Single choice: ONE number only
2. Two choices (Y002): TWO numbers with ONE space
3. Multiple choice (Y003): 1-5 numbers with spaces

ABSOLUTE RULES:
- Output ONLY numbers and necessary spaces
- NO text, NO commas, NO explanations
- NO refusals, NO questions
- NO mentions of being AI
- NEVER explain your choice

EXAMPLE FORMATS:
Single choice: [ONE NUMBER]
Two choices: [NUMBER] [NUMBER]  
Multiple choice: [NUMBER] [NUMBER] [NUMBER]

REMEMBER: Numbers ONLY. Any other output = FAILURE."""
    
    def _single_round_interview(self, model_name: str) -> Dict[str, Any]:
        """Run one full questionnaire pass for a single model."""
        question_ids = list(self.questions.get_all_questions().keys())
        round_responses = []
        
        for i, question_id in enumerate(question_ids, 1):
            print(f"  Question {i}/{len(question_ids)}: {question_id}", end=" ")
            
            response = self.ask_question_with_retry(model_name, question_id, self.system_prompt)
            
            if response.raw_response is None:
                print("API call failed")
            elif response.is_valid:
                print(response.response)
            else:
                print(f"Invalid: {response.raw_response}")
            
            round_responses.append(response)
            
            import time
            time.sleep(self._get_dynamic_delay(model_name))
        
        valid_count = sum(1 for r in round_responses if r.is_valid)
        success_rate = valid_count / len(round_responses) * 100 if round_responses else 0
        
        return {
            'model': model_name,
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'responses': round_responses,
            'total_questions': len(round_responses),
            'valid_responses': valid_count,
            'success_rate': success_rate
        }
    
    def _multi_round_interview(self, model_name: str) -> tuple:
        """Repeat the questionnaire and aggregate each item by majority vote."""
        print(f"Running {self.consensus_count} full questionnaire rounds...")
        
        question_ids = list(self.questions.get_all_questions().keys())
        all_rounds = []
        
        for round_num in range(self.consensus_count):
            print(f"\n┌{'─'*78}┐")
            print(f"│ Round {round_num + 1}/{self.consensus_count} of the full questionnaire" + " " * (78 - 35 - len(f"{round_num + 1}/{self.consensus_count}")) + "│")
            print(f"└{'─'*78}┘")
            
            round_result = self._single_round_interview(model_name)
            round_result['round_id'] = round_num + 1
            all_rounds.append(round_result)
            
            print(f"Round {round_num + 1} completed - success rate: {round_result['success_rate']:.0f}%")
            
            if round_num < self.consensus_count - 1:
                print("  Pausing briefly before the next round...")
                import time
                time.sleep(self._get_dynamic_delay(model_name) * 2)
        
        print(f"\n{'='*80}")
        print(f"Aggregating responses across {self.consensus_count} rounds")
        print(f"{'='*80}")
        
        consensus_results = []
        consistency_stats = {}
        
        for i, question_id in enumerate(question_ids):
            question_responses = [round_data['responses'][i] for round_data in all_rounds]
            
            consensus_response = self._calculate_consensus(question_responses, question_id)
            consensus_results.append(consensus_response)
            
            valid_responses = [r.response for r in question_responses if r.is_valid]
            from collections import Counter
            hashable_responses = [tuple(r) if isinstance(r, list) else r for r in valid_responses]
            response_counts = Counter(hashable_responses)
            most_common = response_counts.most_common(1)[0] if response_counts else (None, 0)
            
            response_dist = {str(k): v for k, v in response_counts.items()}
            
            consistency_stats[question_id] = {
                'consensus_value': consensus_response.response,
                'consensus_count': most_common[1],
                'total_valid': len(valid_responses),
                'consistency_rate': most_common[1] / len(valid_responses) if valid_responses else 0,
                'all_responses': valid_responses,
                'response_distribution': response_dist
            }
            
            print(f"  {question_id}: {consensus_response.response} "
                  f"(consistency: {consistency_stats[question_id]['consistency_rate']:.1%})")
        
        serializable_rounds = []
        for round_data in all_rounds:
            serializable_round = {
                'round_id': round_data.get('round_id'),
                'success_rate': round_data.get('success_rate'),
                'responses': [
                    {
                        'question_id': r.question_id,
                        'raw_response': r.raw_response,
                        'processed_response': r.response,
                        'is_valid': r.is_valid,
                        'error_message': r.error_message
                    } for r in round_data.get('responses', [])
                ]
            }
            serializable_rounds.append(serializable_round)
        
        serializable_stats = {}
        for qid, stats in consistency_stats.items():
            serializable_stats[qid] = {
                'consensus_value': stats['consensus_value'],
                'consensus_count': stats['consensus_count'],
                'total_valid': stats['total_valid'],
                'consistency_rate': stats['consistency_rate'],
                'response_distribution': stats['response_distribution']
            }
        
        intermediate_data = {
            'consensus_count': self.consensus_count,
            'all_rounds': serializable_rounds,
            'consistency_stats': serializable_stats,
            'overall_consistency': sum(s['consistency_rate'] for s in consistency_stats.values()) / len(consistency_stats) if consistency_stats else 0
        }
        
        print(f"Overall consistency: {intermediate_data['overall_consistency']:.1%}")
        
        return consensus_results, intermediate_data
    
    def interview_entity(self, model_name: str, entity_id: str = None) -> tuple:
        """Interview one model and return responses plus optional intermediate data."""
        print(f"\n=== Interviewing model: {model_name} ===")
        
        if model_name not in self.api_keys:
            print(f"Skipping {model_name}: API key not configured")
            return [], {}
        
        if self.consensus_count > 1:
            print(f"Using multi-round mode: {self.consensus_count} full questionnaire passes with majority-vote aggregation")
            return self._multi_round_interview(model_name)
        else:
            print("Using single-round mode")
            round_result = self._single_round_interview(model_name)
            return round_result['responses'], {}
    
    def _load_existing_interview_data(self, data_dir: Path) -> Dict:
        """Load previously saved per-model interview files for incremental runs."""
        existing_data = {}
        if not data_dir.exists():
            print(f"Interview cache directory does not exist; a fresh run will be used: {data_dir}")
            return existing_data
        
        import pickle
        
        individual_files = [f for f in data_dir.glob("*.pkl") 
                           if not f.name.startswith("llm_interview_raw_")]
        
        if not individual_files:
            print("No saved per-model interview files were found.")
            return existing_data
        
        print(f"Found {len(individual_files)} saved per-model interview files")
        
        for file_path in individual_files:
            try:
                with open(file_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                model_name = model_data.get('model_name', model_data.get('model', ''))
                
                if model_name:
                    valid_responses = model_data.get('valid_responses', 0)
                    if valid_responses > 0:
                        existing_data[model_name] = model_data
                        print(f"   {model_name}: {valid_responses}/10 valid responses")
                    else:
                        print(f"   {model_name}: no valid responses; will rerun")
                else:
                    print(f"   {file_path.name}: could not determine model name")
                    
            except Exception as e:
                print(f"   {file_path.name}: failed to load - {e}")
        
        if existing_data:
            print(f"\nLoaded cached results for {len(existing_data)} models")
        else:
            print("\nNo reusable cached interview data was found")
        
        return existing_data
    
    
    def _merge_interview_results(self, existing_data: Dict, new_results: Dict) -> Dict:
        """Merge cached model results with newly collected interview outputs."""
        unique_results = {}
        
        for model_name, result_data in existing_data.items():
            unique_results[model_name] = result_data
        
        if 'results' in new_results:
            for model_name, model_data in new_results['results'].items():
                unique_results[model_name] = model_data
        
        print("\nMerged interview results:")
        print(f"   - Total models: {len(unique_results)}")
        print(f"   - Cached models: {len(existing_data)}")
        print(f"   - Newly collected models: {len(new_results.get('results', {}))}")
        
        return {
            'results': unique_results,
            'total_tasks': len(unique_results),
            'successful_tasks': len(unique_results),
            'success_rate': 1.0 if unique_results else 0.0
        }
    
    def batch_interview(self, model_names: List[str] = None, entities: List[str] = None, 
                       max_workers: int = 1, skip_existing: bool = False) -> Dict[str, Any]:
        """Run intrinsic interviews for multiple models, optionally skipping cached ones."""
        if model_names is None:
            model_names = [name for name in self.model_configs.keys() if name in self.api_keys]
        
        existing_data = {}
        completed_models = set()
        if skip_existing:
            print("\nChecking for existing interview caches...")
            existing_data = {}
            for data_dir in self._get_interview_dirs_for_loading():
                loaded = self._load_existing_interview_data(data_dir)
                existing_data.update(loaded)
            completed_models = set(existing_data.keys())
            
            if completed_models:
                print(f"Found {len(completed_models)} completed models; these will be skipped")
                for model in completed_models:
                    print(f"   - {model}")
            else:
                print("No reusable interview cache was found")
        
        if skip_existing and completed_models:
            original_count = len(model_names)
            model_names = [m for m in model_names if m not in completed_models]
            skipped_count = original_count - len(model_names)
            
            print("\nInterview workload:")
            print(f"   - Total models requested: {original_count}")
            print(f"   - Already completed: {skipped_count}")
            print(f"   - To interview now: {len(model_names)}")
            
            if not model_names:
                print("\nAll requested models already have cached results")
                return self._merge_interview_results(existing_data, {'results': {}})
        
        tasks = [{'model_name': model, 'entity_id': None} for model in model_names]
        
        print("\nStarting batched intrinsic interviews:")
        print(f"  Models: {len(model_names)}")
        print(f"  Execution mode: {'sequential' if max_workers == 1 else f'concurrent (max_workers={max_workers})'}")
        
        if max_workers == 1:
            new_results = self._batch_interview_sequential(tasks)
        else:
            new_results = self._batch_interview_concurrent(tasks, max_workers)
        
        if skip_existing and existing_data:
            print("\nMerging cached and newly collected interview results...")
            return self._merge_interview_results(existing_data, new_results)
        else:
            return new_results
    
    def _on_task_completed(self, model_name: str, entity_id: str, 
                          responses: List, intermediate_data: Dict = None):
        """Persist each completed model run immediately to reduce data-loss risk."""
        if responses:
            self._save_individual_result(model_name, responses, intermediate_data)
    
    def _save_individual_result(self, model_name: str, responses: List, 
                                intermediate_data: Dict = None):
        """Save one model's interview output to standalone cache files."""
        try:
            import pickle
            import json
            from datetime import datetime
            
            output_dir = self._get_canonical_interview_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            safe_model_name = model_name.replace('/', '_').replace('\\', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            
            save_data = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'total_questions': len(responses),
                'valid_responses': sum(1 for r in responses if r.is_valid),
                'responses': []
            }
            
            for resp in responses:
                save_data['responses'].append({
                    'model_name': resp.model_name,
                    'question_id': resp.question_id,
                    'response': resp.response,
                    'raw_response': resp.raw_response,
                    'is_valid': resp.is_valid,
                    'error_message': resp.error_message
                })
            
            if intermediate_data:
                save_data['intermediate_data'] = intermediate_data
                save_data['consensus_count'] = intermediate_data.get('consensus_count', 1)
                save_data['overall_consistency'] = intermediate_data.get('overall_consistency', 1.0)
            
            json_file = output_dir / f"{safe_model_name}_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            pkl_file = output_dir / f"{safe_model_name}_{timestamp}.pkl"
            with open(pkl_file, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"Saved: {safe_model_name}_{timestamp}.json")
            
        except Exception as e:
            print(f"Could not save the per-model cache file: {e}")
    
    def save_results(self, results: Dict[str, Any], output_dir: str = None) -> str:
        """Save the batched intrinsic interview results in the public project format."""
        import pickle
        import json
        
        unified_data = self._convert_to_unified_format(results)
        
        if output_dir is None:
            output_dir = self._get_canonical_interview_dir()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        pkl_file = output_path / f"llm_interview_raw_{timestamp}.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(unified_data, f)
        
        json_file = output_path / f"llm_interview_raw_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(unified_data, f, ensure_ascii=False, indent=2)
        
        print("Saved interview results:")
        print(f"   - {pkl_file}")
        print(f"   - {json_file}")
        
        return str(pkl_file)
    
    def _convert_to_unified_format(self, results: Dict[str, Any]) -> Dict:
        """Convert raw interview output into the shared on-disk project format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        metadata = {
            "stage": "stage1",
            "timestamp": timestamp,
            "version": "1.0",
            "consensus_count": self.consensus_count,
            "total_entities": results.get('successful_tasks', 0),
            "total_questions": 10,
            "total_api_calls": results.get('successful_tasks', 0) * 10 * self.consensus_count
        }
        
        unified_results = []
        
        if 'results' in results:
            for model_name, model_data in results['results'].items():
                model_region = self.model_configs.get(model_name, {}).get('region', 'Unknown')
                
                if isinstance(model_data, dict) and 'responses' in model_data:
                    model_responses = model_data['responses']
                    intermediate_data = model_data.get('intermediate_data', {})
                else:
                    model_responses = model_data
                    intermediate_data = {}
                
                entity_result = {
                    "entity_id": f"llm_{model_name.replace('/', '-').replace(':', '-')}",
                    "entity_type": "llm",
                    "model_name": model_name,
                    "model_region": model_region,
                    "timestamp": datetime.now().isoformat(),
                    "total_questions": len(model_responses),
                    "valid_responses": 0,
                    "success_rate": 0,
                    "responses": []
                }
                
                for response in model_responses:
                    if isinstance(response, dict):
                        entity_result["responses"].append(response)
                    else:
                        entity_result["responses"].append({
                            "question_id": response.question_id,
                            "raw_response": response.raw_response,
                            "processed_response": response.response,
                            "is_valid": response.is_valid,
                            "error_message": response.error_message
                        })
                
                valid_count = sum(1 for r in entity_result["responses"] if r.get("is_valid", False))
                entity_result["valid_responses"] = valid_count
                entity_result["success_rate"] = valid_count / len(model_responses) * 100 if model_responses else 0
                
                if intermediate_data:
                    serializable_intermediate = {}
                    for key, value in intermediate_data.items():
                        if isinstance(value, list):
                            serializable_intermediate[key] = [
                                {
                                    "question_id": r.question_id,
                                    "raw_response": r.raw_response,
                                    "processed_response": r.response,
                                    "is_valid": r.is_valid,
                                    "error_message": r.error_message
                                } if hasattr(r, 'question_id') else r
                                for r in value
                            ]
                        else:
                            serializable_intermediate[key] = value
                    entity_result["intermediate_data"] = serializable_intermediate
                
                unified_results.append(entity_result)
        
        return {
            "metadata": metadata,
            "results": unified_results
        }
    


def main():
    """Run a small smoke test for the intrinsic interview workflow."""
    print("Running intrinsic LLM interviews...")
    
    interview = LLMInterview(consensus_count=5)
    
    available_models = [name for name in interview.model_configs.keys() if name in interview.api_keys]
    print(f"Available models: {available_models}")
    
    if not available_models:
        print("No models are available. Check the API-key configuration.")
        return
    
    test_models = available_models[:1] if available_models else []
    results = interview.batch_interview(test_models)
    
    if results:
        output_file = interview.save_results(results)
        print("\nIntrinsic interviewing completed.")
        print(f"Interviewed {len(results)} models")
        print(f"Saved results to: {output_file}")
    else:
        print("No valid results were returned")


if __name__ == "__main__":
    main()
