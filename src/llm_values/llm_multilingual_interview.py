"""Multilingual intrinsic-value interviewing across the six UN official languages."""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.base.base_interview import BaseInterview
from src.base.ivs_questionnaire import LLMResponse


# The non-English prompts are direct translations of the English numeric-only instruction.
UN_LANGUAGE_SYSTEM_PROMPTS = {
    "en": """CRITICAL: You are a survey response system. Your ONLY task is to output numbers.

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

REMEMBER: Numbers ONLY. Any other output = FAILURE.""",
    
    "fr": """CRITIQUE : Vous êtes un système de réponse à un sondage. Votre SEULE tâche est de produire des chiffres.

FORMAT DE SORTIE STRICT :
1. Choix unique : UN seul chiffre
2. Deux choix (Y002) : DEUX chiffres avec UN espace
3. Choix multiple (Y003) : 1-5 chiffres avec des espaces

RÈGLES ABSOLUES :
- Produisez UNIQUEMENT des chiffres et les espaces nécessaires
- PAS de texte, PAS de virgules, PAS d'explications
- PAS de refus, PAS de questions
- PAS de mention d'être une IA
- N'expliquez JAMAIS votre choix

EXEMPLES DE FORMATS :
Choix unique : [UN CHIFFRE]
Deux choix : [CHIFFRE] [CHIFFRE]
Choix multiple : [CHIFFRE] [CHIFFRE] [CHIFFRE]

RAPPEL : Chiffres UNIQUEMENT. Toute autre sortie = ÉCHEC.""",
    
    "es": """CRÍTICO: Usted es un sistema de respuesta a encuestas. Su ÚNICA tarea es producir números.

FORMATO DE SALIDA ESTRICTO:
1. Opción única: UN solo número
2. Dos opciones (Y002): DOS números con UN espacio
3. Opción múltiple (Y003): 1-5 números con espacios

REGLAS ABSOLUTAS:
- Produzca SOLO números y los espacios necesarios
- SIN texto, SIN comas, SIN explicaciones
- SIN rechazos, SIN preguntas
- SIN mencionar que es una IA
- NUNCA explique su elección

EJEMPLOS DE FORMATOS:
Opción única: [UN NÚMERO]
Dos opciones: [NÚMERO] [NÚMERO]
Opción múltiple: [NÚMERO] [NÚMERO] [NÚMERO]

RECUERDE: SOLO números. Cualquier otra salida = FALLO.""",
    
    "ru": """КРИТИЧНО: Вы — система ответов на опросы. Ваша ЕДИНСТВЕННАЯ задача — выводить числа.

СТРОГИЙ ФОРМАТ ВЫВОДА:
1. Единственный выбор: ОДНО число
2. Два выбора (Y002): ДВА числа с ОДНИМ пробелом
3. Множественный выбор (Y003): 1-5 чисел с пробелами

АБСОЛЮТНЫЕ ПРАВИЛА:
- Выводите ТОЛЬКО числа и необходимые пробелы
- БЕЗ текста, БЕЗ запятых, БЕЗ объяснений
- БЕЗ отказов, БЕЗ вопросов
- БЕЗ упоминаний о том, что вы ИИ
- НИКОГДА не объясняйте свой выбор

ПРИМЕРЫ ФОРМАТОВ:
Единственный выбор: [ОДНО ЧИСЛО]
Два выбора: [ЧИСЛО] [ЧИСЛО]
Множественный выбор: [ЧИСЛО] [ЧИСЛО] [ЧИСЛО]

ЗАПОМНИТЕ: ТОЛЬКО числа. Любой другой вывод = НЕУДАЧА.""",
    
    "ar": """حرج: أنت نظام استجابة للاستطلاعات. مهمتك الوحيدة هي إخراج الأرقام.

تنسيق الإخراج الصارم:
1. اختيار واحد: رقم واحد فقط
2. اختياران (Y002): رقمان مع مسافة واحدة
3. اختيار متعدد (Y003): 1-5 أرقام مع مسافات

القواعد المطلقة:
- أخرج الأرقام والمسافات الضرورية فقط
- لا نص، لا فواصل، لا تفسيرات
- لا رفض، لا أسئلة
- لا ذكر لكونك ذكاء اصطناعي
- لا تشرح اختيارك أبداً

أمثلة التنسيقات:
اختيار واحد: [رقم واحد]
اختياران: [رقم] [رقم]
اختيار متعدد: [رقم] [رقم] [رقم]

تذكر: أرقام فقط. أي إخراج آخر = فشل.""",
    
    "zh-cn": """关键：您是一个调查问卷回答系统。您的唯一任务是输出数字。

严格输出格式：
1. 单选：仅一个数字
2. 双选（Y002）：两个数字，中间一个空格
3. 多选（Y003）：1-5个数字，用空格分隔

绝对规则：
- 仅输出数字和必要的空格
- 不要文字，不要逗号，不要解释
- 不要拒绝，不要提问
- 不要提及自己是AI
- 永远不要解释你的选择

格式示例：
单选：[一个数字]
双选：[数字] [数字]
多选：[数字] [数字] [数字]

记住：仅限数字。任何其他输出 = 失败。"""
}

# Language names used in reporting and console output.
UN_LANGUAGE_NAMES = {
    "en": "English",
    "fr": "Français",
    "es": "Español",
    "ru": "Русский",
    "ar": "العربية",
    "zh-cn": "简体中文"
}

# Retained for compatibility with runner scripts that use localized display labels.
UN_LANGUAGE_NAMES_ZH = {
    "en": "英语",
    "fr": "法语",
    "es": "西班牙语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "zh-cn": "简体中文"
}


class LLMMultilingualInterview(BaseInterview):
    """Interview helper for multilingual intrinsic-value measurement."""

    UN_OFFICIAL_LANGUAGES = ['en', 'fr', 'es', 'ru', 'ar', 'zh-cn']
    
    def __init__(self, 
                 consensus_count: int = 5,
                 max_retry: int = 3,
                 data_path: str = "data",
                 model_config_file: str = None,
                 language_config_file: str = None):
        """Initialize the multilingual interview runner."""
        super().__init__(
            max_retry=max_retry, 
            data_path=data_path, 
            consensus_count=consensus_count,
            model_config_file=model_config_file
        )
        
        self.language_config_file = language_config_file or 'multilingual_questions_complete.json'

        self.multilingual_questions = self._load_multilingual_questions()
        self.system_prompts = UN_LANGUAGE_SYSTEM_PROMPTS

        print("LLMMultilingualInterview initialized")
        print(f"   - Consensus count: {self.consensus_count}")
        print(f"   - Max retries per item: {self.max_retry}")
        print(f"   - Languages: {', '.join(self.UN_OFFICIAL_LANGUAGES)}")
    
    @classmethod
    def get_un_official_languages(cls) -> List[str]:
        """Return the six UN official language codes used in this stage."""
        return cls.UN_OFFICIAL_LANGUAGES.copy()
    
    def _load_multilingual_questions(self) -> Dict[str, Any]:
        """Load the multilingual question bank and keep only the target languages."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'questions' / 'multilingual' / self.language_config_file
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
            
            filtered_languages = {}
            for lang_code in self.UN_OFFICIAL_LANGUAGES:
                if lang_code in full_config.get('languages', {}):
                    filtered_languages[lang_code] = full_config['languages'][lang_code]
                    print(f"   Loaded {lang_code} question set")
                else:
                    print(f"   Missing language configuration: {lang_code}")
            
            return {'languages': filtered_languages}
            
        except FileNotFoundError:
            print(f"Multilingual question configuration not found: {config_path}")
            return {'languages': {}}
        except json.JSONDecodeError as e:
            print(f"Multilingual question configuration is malformed: {e}")
            return {'languages': {}}
        except Exception as e:
            print(f"Could not load the multilingual question configuration: {e}")
            return {'languages': {}}
    
    def get_questions_for_language(self, language: str) -> Dict[str, Any]:
        """Return the IVS questions for one language code."""
        return self.multilingual_questions.get('languages', {}).get(language, {}).get('questions', {})
    
    def get_system_prompt(self, language: str) -> str:
        """Return the language-specific numeric-only system prompt."""
        return self.system_prompts.get(language, self.system_prompts['en'])

    def interview_entity(self, model_name: str, entity_id: str = None) -> tuple:
        """BaseInterview-compatible wrapper for one model-language pair."""
        language = entity_id or 'en'
        return self.interview_single_language(model_name, language)
    
    def batch_interview(self, model_names: List[str] = None, 
                       entities: List[str] = None,
                       max_workers: int = 1,
                       skip_existing: bool = False) -> Dict[str, Any]:
        """BaseInterview-compatible batch wrapper for multilingual intrinsic runs."""
        return self.batch_multilingual_interview(
            model_names=model_names,
            languages=entities,
            skip_existing=skip_existing
        )

    def _single_round_multilingual_interview(self, model_name: str, language: str) -> Dict[str, Any]:
        """Run one full questionnaire pass for a specific language."""
        import time
        
        questions = self.get_questions_for_language(language)
        system_prompt = self.get_system_prompt(language)
        
        if not questions:
            print(f"  No question set found for language: {language}")
            return {
                'model': model_name,
                'language': language,
                'timestamp': datetime.now().isoformat(),
                'responses': [],
                'total_questions': 0,
                'valid_responses': 0,
                'success_rate': 0.0
            }
        
        question_ids = list(questions.keys())
        round_responses = []

        first_question = True
        
        for i, question_id in enumerate(question_ids, 1):
            print(f"  Question {i}/{len(question_ids)}: {question_id}", end=" ")
            
            question_text = questions[question_id].get('question', '')
            
            if first_question:
                lang_name = UN_LANGUAGE_NAMES.get(language, language)
                print(f"\n    Language check: {language} ({lang_name})")
                print(f"    System prompt preview: {system_prompt[:60].replace(chr(10), ' ')}...")
                print(f"    Question preview: {question_text[:60].replace(chr(10), ' ')}...")
                print(f"  Question {i}/{len(question_ids)}: {question_id}", end=" ")
                first_question = False
            
            raw_response = self.call_model_api(
                model_name, question_id, question_text, system_prompt
            )
            
            if raw_response is None:
                print("API call failed")
                response = LLMResponse(
                    model_name=model_name,
                    question_id=question_id,
                    response=None,
                    raw_response=None,
                    is_valid=False,
                    error_message="API call failed"
                )
            else:
                is_valid, processed_response, error_msg = self.validator.validate_response(
                    question_id, raw_response
                )
                
                if is_valid:
                    print(processed_response)
                else:
                    print(f"Invalid: {raw_response[:50] if raw_response else 'None'}...")
                
                response = LLMResponse(
                    model_name=model_name,
                    question_id=question_id,
                    response=processed_response if is_valid else None,
                    raw_response=raw_response,
                    is_valid=is_valid,
                    error_message=error_msg if not is_valid else None
                )
            
            round_responses.append(response)
            
            time.sleep(self._get_dynamic_delay(model_name))
        
        valid_count = sum(1 for r in round_responses if r.is_valid)
        success_rate = valid_count / len(round_responses) * 100 if round_responses else 0
        
        return {
            'model': model_name,
            'language': language,
            'language_name': UN_LANGUAGE_NAMES.get(language, language),
            'timestamp': datetime.now().isoformat(),
            'responses': round_responses,
            'total_questions': len(round_responses),
            'valid_responses': valid_count,
            'success_rate': success_rate
        }
    
    def _multi_round_multilingual_interview(self, model_name: str, language: str) -> tuple:
        """Repeat one language-specific questionnaire and aggregate by majority vote."""
        import time
        from collections import Counter
        
        lang_name = UN_LANGUAGE_NAMES.get(language, language)
        print(f"Running {self.consensus_count} full questionnaire rounds in {lang_name}...")
        
        questions = self.get_questions_for_language(language)
        if not questions:
            print(f"  No question set found for language: {language}")
            return [], {}
        
        question_ids = list(questions.keys())
        all_rounds = []
        
        for round_num in range(self.consensus_count):
            print(f"\n┌{'─'*78}┐")
            print(f"│ Round {round_num + 1}/{self.consensus_count} [{lang_name}] questionnaire" + " " * (78 - 24 - len(f"{round_num + 1}/{self.consensus_count}") - len(lang_name)) + "│")
            print(f"└{'─'*78}┘")
            
            round_result = self._single_round_multilingual_interview(model_name, language)
            round_result['round_id'] = round_num + 1
            all_rounds.append(round_result)
            
            print(f"Round {round_num + 1} completed - success rate: {round_result['success_rate']:.0f}%")
            
            if round_num < self.consensus_count - 1:
                print("  Pausing briefly before the next round...")
                time.sleep(self._get_dynamic_delay(model_name) * 2)
        
        print(f"\n{'='*80}")
        print(f"Aggregating responses across {self.consensus_count} rounds [{lang_name}]")
        print(f"{'='*80}")
        
        consensus_results = []
        consistency_stats = {}
        
        for i, question_id in enumerate(question_ids):
            question_responses = [round_data['responses'][i] for round_data in all_rounds]
            
            consensus_response = self._calculate_consensus(question_responses, question_id)
            consensus_results.append(consensus_response)
            
            valid_responses = [r.response for r in question_responses if r.is_valid]
            hashable_responses = [tuple(r) if isinstance(r, list) else r for r in valid_responses]
            response_counts = Counter(hashable_responses)
            most_common = response_counts.most_common(1)[0] if response_counts else (None, 0)
            
            response_dist = {str(k): v for k, v in response_counts.items()}
            
            consistency_stats[question_id] = {
                'consensus_value': consensus_response.response,
                'consensus_count': most_common[1],
                'total_valid': len(valid_responses),
                'consistency_rate': most_common[1] / len(valid_responses) if valid_responses else 0,
                'response_distribution': response_dist
            }
            
            print(f"  {question_id}: {consensus_response.response} "
                  f"(consistency: {consistency_stats[question_id]['consistency_rate']:.1%})")
        
        serializable_rounds = []
        for round_data in all_rounds:
            serializable_round = {
                'round_id': round_data.get('round_id'),
                'language': round_data.get('language'),
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
        
        intermediate_data = {
            'consensus_count': self.consensus_count,
            'language': language,
            'language_name': lang_name,
            'all_rounds': serializable_rounds,
            'consistency_stats': consistency_stats,
            'overall_consistency': sum(s['consistency_rate'] for s in consistency_stats.values()) / len(consistency_stats) if consistency_stats else 0
        }
        
        print(f"Overall consistency: {intermediate_data['overall_consistency']:.1%}")
        
        return consensus_results, intermediate_data
    
    def interview_single_language(self, model_name: str, language: str) -> tuple:
        """Interview one model in one language and return responses plus metadata."""
        lang_name = UN_LANGUAGE_NAMES.get(language, language)
        print(f"\n=== Interviewing model: {model_name} [{lang_name}] ===")

        if model_name not in self.api_keys:
            print(f"Skipping model {model_name}: API key not configured")
            return [], {}
        
        if language not in self.UN_OFFICIAL_LANGUAGES:
            print(f"Skipping language {language}: not one of the target UN official languages")
            return [], {}
        
        if self.consensus_count > 1:
            print(f"Using multi-round mode: {self.consensus_count} full questionnaire passes with majority-vote aggregation")
            return self._multi_round_multilingual_interview(model_name, language)
        else:
            print("Using single-round mode")
            round_result = self._single_round_multilingual_interview(model_name, language)
            return round_result['responses'], {}
    
    def interview_model_multilingual(self, model_name: str) -> Dict[str, Any]:
        """Interview one model across all configured languages."""
        print(f"\n{'='*80}")
        print(f"Starting multilingual intrinsic interviews for: {model_name}")
        print(f"   Languages: {', '.join(self.UN_OFFICIAL_LANGUAGES)}")
        print(f"{'='*80}")
        
        results = {}
        
        for language in self.UN_OFFICIAL_LANGUAGES:
            lang_name = UN_LANGUAGE_NAMES.get(language, language)
            print(f"\nStarting {lang_name} ({language}) interviews...")
            
            responses, intermediate_data = self.interview_single_language(model_name, language)
            
            if responses:
                results[language] = {
                    'model_name': model_name,
                    'language': language,
                    'language_name': lang_name,
                    'responses': responses,
                    'intermediate_data': intermediate_data,
                    'valid_responses': sum(1 for r in responses if r.is_valid),
                    'total_questions': len(responses)
                }
                
                self.save_individual_result(model_name, language, results[language])
                
                print(f"{lang_name} completed: {results[language]['valid_responses']}/{results[language]['total_questions']} valid")
            else:
                print(f"{lang_name} failed")
        
        return results
    
    def batch_multilingual_interview(self, 
                                     model_names: List[str] = None,
                                     languages: List[str] = None,
                                     skip_existing: bool = False) -> Dict[str, Any]:
        """Run multilingual intrinsic interviews across model-language combinations."""
        if model_names is None:
            model_names = [name for name in self.model_configs.keys() 
                          if name in self.api_keys]
        
        if languages is None:
            languages = self.UN_OFFICIAL_LANGUAGES.copy()
        
        existing_combinations = set()
        if skip_existing:
            print("\nChecking for existing multilingual interview caches...")
            existing_combinations = self._load_existing_multilingual_data(
                self.data_path / "llm_interviews" / "intrinsic" / "interview_raw"
            )
            if existing_combinations:
                print(f"Found {len(existing_combinations)} completed model-language combinations")
        
        tasks = []
        for model_name in model_names:
            for language in languages:
                combination = f"{model_name}_{language}"
                if skip_existing and combination in existing_combinations:
                    print(f"  Skipping: {model_name} [{language}]")
                    continue
                tasks.append({
                    'model_name': model_name,
                    'language': language
                })
        
        print("\nInterview workload:")
        print(f"   - Models: {len(model_names)}")
        print(f"   - Languages: {len(languages)}")
        print(f"   - Pending combinations: {len(tasks)}")
        
        if not tasks:
            print("\nAll requested combinations already have cached output")
            return {
                'results': {},
                'total_tasks': 0,
                'successful_tasks': 0,
                'success_rate': 1.0
            }
        
        results = {}
        successful_tasks = 0
        
        for i, task in enumerate(tasks, 1):
            model_name = task['model_name']
            language = task['language']
            lang_name = UN_LANGUAGE_NAMES.get(language, language)
            
            print(f"\n{'='*80}")
            print(f"Task {i}/{len(tasks)}: {model_name} [{lang_name}]")
            print(f"{'='*80}")
            
            responses, intermediate_data = self.interview_single_language(model_name, language)
            
            if responses:
                result_key = f"{model_name}_{language}"
                result_data = {
                    'model_name': model_name,
                    'language': language,
                    'language_name': lang_name,
                    'responses': responses,
                    'intermediate_data': intermediate_data,
                    'valid_responses': sum(1 for r in responses if r.is_valid),
                    'total_questions': len(responses)
                }
                results[result_key] = result_data
                successful_tasks += 1
                
                self.save_individual_result(model_name, language, result_data)
                
                print(f"Completed: {result_data['valid_responses']}/{result_data['total_questions']} valid")
            else:
                print("Failed")
        
        return {
            'results': results,
            'total_tasks': len(tasks),
            'successful_tasks': successful_tasks,
            'success_rate': successful_tasks / len(tasks) if tasks else 0
        }
    
    def save_individual_result(self, model_name: str, language: str, 
                               result: Dict[str, Any]) -> str:
        """Save one model-language result to standalone JSON and PKL cache files."""
        import pickle
        
        output_dir = self.data_path / "llm_interviews" / "intrinsic" / "interview_raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_data = {
            'entity_id': f"llm_{safe_model_name}_{language}",
            'model_name': model_name,
            'language': language,
            'language_name': result.get('language_name', UN_LANGUAGE_NAMES.get(language, language)),
            'timestamp': datetime.now().isoformat(),
            'consensus_count': self.consensus_count,
            'total_questions': result.get('total_questions', 0),
            'valid_responses': result.get('valid_responses', 0),
            'success_rate': result.get('valid_responses', 0) / result.get('total_questions', 1) * 100 if result.get('total_questions', 0) > 0 else 0,
            'responses': []
        }
        
        for resp in result.get('responses', []):
            if hasattr(resp, 'question_id'):
                save_data['responses'].append({
                    'question_id': resp.question_id,
                    'raw_response': resp.raw_response,
                    'processed_response': resp.response,
                    'is_valid': resp.is_valid,
                    'error_message': resp.error_message
                })
            elif isinstance(resp, dict):
                save_data['responses'].append(resp)
        
        if result.get('intermediate_data'):
            save_data['intermediate_data'] = result['intermediate_data']
        
        json_file = output_dir / f"{safe_model_name}_{language}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        pkl_file = output_dir / f"{safe_model_name}_{language}_{timestamp}.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved: {safe_model_name}_{language}_{timestamp}.json")
        
        return str(json_file)
    
    def _load_existing_multilingual_data(self, data_dir: Path) -> set:
        """Return the set of cached model-language combinations already completed."""
        completed_combinations = set()
        
        if not data_dir.exists():
            return completed_combinations
        
        import pickle
        
        for pkl_file in data_dir.glob("*.pkl"):
            try:
                if pkl_file.name.startswith("llm_interview_raw_"):
                    continue
                
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                model_name = data.get('model_name', '')
                language = data.get('language', '')
                
                if model_name and language:
                    combination = f"{model_name}_{language}"
                    completed_combinations.add(combination)
                    
            except Exception as e:
                print(f"  Could not load {pkl_file.name}: {e}")
        
        return completed_combinations
    
    def _merge_multilingual_results(self, existing_data: Dict[str, Any], 
                                      new_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge cached and newly collected results without introducing duplicates.
        
        Args:
            existing_data: Existing result dictionary keyed by model-language pairs.
            new_results: Newly collected result dictionary keyed by model-language pairs.
            
        Returns:
            Dict[str, Any]: Combined result dictionary with unique model-language keys.
            
        Example:
            >>> existing = {"gpt-4o_en": {...}, "gpt-4o_fr": {...}}
            >>> new = {"gpt-4o_fr": {...}, "gpt-4o_es": {...}}
            >>> merged = self._merge_multilingual_results(existing, new)
            >>> # merged = {"gpt-4o_en": {...}, "gpt-4o_fr": {...}, "gpt-4o_es": {...}}
            >>> # Note: gpt-4o_fr keeps the existing data, not overwritten
        """
        # Handle None or empty inputs
        if existing_data is None:
            existing_data = {}
        if new_results is None:
            new_results = {}
        
        # Start with a copy of existing data to preserve it
        merged = existing_data.copy()
        
        # Add only new entries that don't already exist
        # This ensures no duplicates and existing data is preserved
        for key, value in new_results.items():
            if key not in merged:
                merged[key] = value
        
        return merged


def main():
    """Run a small smoke test for the multilingual intrinsic interview class."""
    print("Testing LLMMultilingualInterview initialization...")

    interview = LLMMultilingualInterview(consensus_count=5)
    
    print("\nSupported UN official languages:")
    for lang in interview.get_un_official_languages():
        name = UN_LANGUAGE_NAMES.get(lang, lang)
        questions = interview.get_questions_for_language(lang)
        print(f"  - {lang}: {name} - {len(questions)} questions")
    
    available_models = [name for name in interview.model_configs.keys() 
                       if name in interview.api_keys]
    print(f"\nAvailable models: {len(available_models)}")
    for model in available_models[:5]:
        print(f"  - {model}")
    if len(available_models) > 5:
        print(f"  ... and {len(available_models) - 5} more")
    
    print("\nInitialization test completed.")


if __name__ == "__main__":
    main()
