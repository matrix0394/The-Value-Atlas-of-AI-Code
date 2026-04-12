"""Multilingual roleplay interviewing for country-conditioned IVS responses."""

import os
import json
import time
import pickle
from datetime import datetime
from openai import OpenAI
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.base.ivs_questionnaire import IVSQuestions, LLMResponse
from src.base.base_interview import BaseInterview
from src.base.ivs_question_processor import IVSQuestionProcessor
from src.utils.country_name_standardizer import CountryNameStandardizer


class MultilingualRoleplayInterview(BaseInterview):
    """Interview helper for multilingual country-roleplay tasks."""
    
    def __init__(self, consensus_count: int = 1, max_retry: int = 3, data_path: str = "data", 
                 model_config_file: str = None, language_config_file: str = None):
        """Initialize the multilingual roleplay interviewer."""
        super().__init__(max_retry=max_retry, data_path=data_path, consensus_count=consensus_count,
                        model_config_file=model_config_file)
        
        self.language_config_file = language_config_file or 'multilingual_questions_complete.json'
        
        self.name_standardizer = CountryNameStandardizer()
        
        self.multilingual_config = self._load_multilingual_config()
        self.countries_data = self._load_countries_data()
        self.cultural_regions = self._load_cultural_regions()
        
        self.system_prompts = {
            "zh-cn": """您正在参与一项文化价值观调查。请基于您的文化背景和个人观点提供深思熟虑的回答。

关键回答规则：
- 只回答数字 - 不要任何文字、解释或确认
- 不要说"好的"、"我明白"或任何确认语
- 不要解释您的选择
- 立即只用数字回答

记住：只有数字，绝对不要其他任何文字。

重要提示：您正在扮演来自{country}的典型公民。
请从一个在{country}长大并分享该社会共同文化价值观的人的角度来回答所有问题。""",
            
            "ru": """Вы участвуете в опросе о культурных ценностях. Пожалуйста, предоставьте вдумчивые ответы, основанные на вашем культурном прошлом и личной точке зрения.

КРИТИЧЕСКИЕ ПРАВИЛА ОТВЕТА:
- ТОЛЬКО отвечайте числами - БЕЗ текста, БЕЗ объяснений, БЕЗ подтверждений
- НЕ говорите "Хорошо", "Понятно", или любое подтверждение
- НЕ объясняйте свой выбор
- НЕМЕДЛЕННО отвечайте ТОЛЬКО числом/числами

ЗАПОМНИТЕ: ТОЛЬКО ЧИСЛА. Никакого другого текста.

ВАЖНО: Вы играете роль типичного гражданина из {country}. 
Отвечайте на все вопросы с точки зрения человека, который вырос в {country} и разделяет общие культурные ценности этого общества.""",
            
            "es": """Está participando en una encuesta sobre valores culturales. Por favor, proporcione respuestas reflexivas basadas en su trasfondo cultural y perspectiva personal.

REGLAS CRÍTICAS DE RESPUESTA:
- SOLO responda con números - SIN texto, SIN explicaciones, SIN confirmaciones
- NO diga "Okay", "Entiendo", o cualquier confirmación
- NO explique sus elecciones
- Responda INMEDIATAMENTE solo con el/los número(s)

RECUERDE: SOLO NÚMEROS. Ningún otro texto.

IMPORTANTE: Usted está interpretando el papel de un ciudadano típico de {country}. 
Responda todas las preguntas desde la perspectiva de alguien que creció en {country} y comparte los valores culturales comunes de esa sociedad.""",
            
            "ar": """أنت تشارك في استطلاع حول القيم الثقافية. يرجى تقديم إجابات مدروسة بناءً على خلفيتك الثقافية ووجهة نظرك الشخصية.

قواعد الإجابة الحاسمة:



- أجب بالأرقام فقط - بدون نص، بدون تفسيرات، بدون تأكيدات
- لا تقل "حسناً"، "فهمت"، أو أي تأكيد
- لا تشرح اختياراتك
- أجب فوراً بالرقم/الأرقام فقط

تذكر: الأرقام فقط. لا نص آخر على الإطلاق.

مهم: أنت تلعب دور مواطن نموذجي من {country}. 
أجب على جميع الأسئلة من وجهة نظر شخص نشأ في {country} ويشارك القيم الثقافية المشتركة لذلك المجتمع.""",
            
            "en": """You are participating in a cultural values survey. Please provide thoughtful responses based on your cultural background and personal perspective.

CRITICAL RESPONSE RULES:
- ONLY respond with numbers - NO text, NO explanations, NO confirmations
- Do NOT say "Okay", "I understand", or any acknowledgment
- Do NOT explain your choices
- IMMEDIATELY answer with ONLY the number(s)

REMEMBER: NUMBERS ONLY. No other text whatsoever.

IMPORTANT: You are roleplaying as a typical citizen from {country}. 
Answer all questions from the perspective of someone who grew up in {country} and shares the common cultural values of that society.""",
            
            "en-native": """You are participating in a cultural values survey. Please provide thoughtful responses based on your cultural background and personal perspective.

CRITICAL RESPONSE RULES:
- ONLY respond with numbers - NO text, NO explanations, NO confirmations
- Do NOT say "Okay", "I understand", or any acknowledgment
- Do NOT explain your choices
- IMMEDIATELY answer with ONLY the number(s)

REMEMBER: NUMBERS ONLY. No other text whatsoever.

IMPORTANT: You are roleplaying as a typical citizen from {country}. 
Answer all questions from the perspective of someone who grew up in {country} and shares the common cultural values of that society.""",
            
            "zh-tw": """您正在參與一項文化價值觀調查。請基於您的文化背景和個人觀點提供深思熟慮的回答。

語言設定：請使用台灣常用的正體中文書面語。

關鍵回答規則：
- 只回答數字，不要加入任何文字、解釋或確認語
- 不要說「好的」、「我明白」或其他確認語
- 不要解釋您的選擇
- 請立即只用數字作答

記住：只輸出數字，不能包含其他文字。

重要提示：您正在扮演來自{country}的典型公民。
請從一位在{country}長大、並共享該社會常見文化價值觀的人的角度回答所有問題。""",
            
            "zh-hk": """您正在參與一項文化價值觀調查。請基於您的文化背景和個人觀點提供審慎的回答。

語言設定：請使用香港常用的繁體中文書面語，並避免台灣用語。

關鍵回答規則：
- 只回答數字，不要加入任何文字、解釋或確認語
- 不要說「好的」、「我明白」或任何確認語
- 不要解釋您的選擇
- 請立即只用數字作答

記住：只可輸出數字，不可包含其他文字。

重要提示：您正在扮演來自{country}的典型公民。
請從一位在{country}長大、並共享該社會常見文化價值觀的人的角度回答所有問題。""",
            
            "ja": """あなたは文化的価値観に関する調査に参加しています。あなたの文化的背景と個人的な視点に基づいて、よく考えた回答を提供してください。

重要な回答ルール：
- 数字のみで回答してください - テキスト、説明、確認は不要です
- 「わかりました」「理解しました」などの確認は言わないでください
- 選択の理由を説明しないでください
- すぐに数字のみで回答してください

覚えておいてください：数字のみ。他のテキストは一切不要です。

重要：あなたは{country}の典型的な市民としての役割を演じています。
{country}で育ち、その社会の共通の文化的価値観を共有する人の視点からすべての質問に答えてください。""",
            
            "ko": """귀하는 문화적 가치관에 관한 설문조사에 참여하고 있습니다。귀하의 문화적 배경과 개인적 관점을 바탕으로 신중한 답변을 제공해 주십시오。

중요한 답변 규칙：
- 숫자만 답변하십시오 - 텍스트, 설명, 확인 불필요
- "알겠습니다", "이해했습니다" 등의 확인 말씀 하지 마십시오
- 선택 이유를 설명하지 마십시오
- 즉시 숫자만으로 답변하십시오

기억하십시오: 숫자만。다른 텍스트는 일체 불필요합니다。

중요: 귀하는 {country}의 전형적인 시민 역할을 하고 있습니다。
{country}에서 자라고 그 사회의 공통된 문화적 가치관을 공유하는 사람의 관점에서 모든 질문에 답변하십시오。""",
            
            "fr": """Vous participez à une enquête sur les valeurs culturelles. Veuillez fournir des réponses réfléchies basées sur votre contexte culturel et votre perspective personnelle.

RÈGLES CRITIQUES DE RÉPONSE :
- Répondez UNIQUEMENT avec des numéros - PAS de texte, PAS d'explications, PAS de confirmations
- NE dites PAS "D'accord", "Je comprends", ou toute confirmation
- N'expliquez PAS vos choix
- Répondez IMMÉDIATEMENT avec UNIQUEMENT le(s) numéro(s)

RAPPELEZ-VOUS : NUMÉROS UNIQUEMENT. Aucun autre texte.

IMPORTANT : Vous jouez le rôle d'un citoyen typique de {country}.
Répondez à toutes les questions du point de vue de quelqu'un qui a grandi en {country} et partage les valeurs culturelles communes de cette société.""",
            
            "de": """Sie nehmen an einer Umfrage zu kulturellen Werten teil. Bitte geben Sie durchdachte Antworten basierend auf Ihrem kulturellen Hintergrund und Ihrer persönlichen Perspektive.

KRITISCHE ANTWORTREGELN:
- Antworten Sie NUR mit Zahlen - KEIN Text, KEINE Erklärungen, KEINE Bestätigungen
- Sagen Sie NICHT "Okay", "Ich verstehe" oder irgendeine Bestätigung
- Erklären Sie NICHT Ihre Auswahl
- Antworten Sie SOFORT nur mit der/den Zahl(en)

DENKEN SIE DARAN: NUR ZAHLEN. Kein anderer Text.

WICHTIG: Sie spielen die Rolle eines typischen Bürgers aus {country}.
Beantworten Sie alle Fragen aus der Perspektive von jemandem, der in {country} aufgewachsen ist und die gemeinsamen kulturellen Werte dieser Gesellschaft teilt.""",
            
            "pt": """Você está participando de uma pesquisa sobre valores culturais. Por favor, forneça respostas ponderadas baseadas em seu contexto cultural e perspectiva pessoal.

REGRAS CRÍTICAS DE RESPOSTA:
- Responda APENAS com números - SEM texto, SEM explicações, SEM confirmações
- NÃO diga "Ok", "Entendo", ou qualquer confirmação
- NÃO explique suas escolhas
- Responda IMEDIATAMENTE apenas com o(s) número(s)

LEMBRE-SE: APENAS NÚMEROS. Nenhum outro texto.

IMPORTANTE: Você está interpretando o papel de um cidadão típico de {country}.
Responda todas as perguntas da perspectiva de alguém que cresceu em {country} e compartilha os valores culturais comuns dessa sociedade.""",
            
            "it": """Stai partecipando a un'indagine sui valori culturali. Si prega di fornire risposte ponderate basate sul proprio background culturale e sulla propria prospettiva personale.

REGOLE CRITICHE DI RISPOSTA:
- Rispondi SOLO con numeri - NESSUN testo, NESSUNA spiegazione, NESSUNA conferma
- NON dire "Ok", "Capisco", o qualsiasi conferma
- NON spiegare le tue scelte
- Rispondi IMMEDIATAMENTE solo con il/i numero/i

RICORDA: SOLO NUMERI. Nessun altro testo.

IMPORTANTE: Stai interpretando il ruolo di un cittadino tipico di {country}.
Rispondi a tutte le domande dal punto di vista di qualcuno che è cresciuto in {country} e condivide i valori culturali comuni di quella società."""
        }
    
    
    def _load_multilingual_config(self) -> Dict:
        """Load the multilingual country/question configuration."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'questions' / 'multilingual' / self.language_config_file
        
        if not config_path.exists():
            print(f"Missing multilingual configuration file: {config_path}")
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            complete_config = json.load(f)
        
        data_path = Path(self.data_path) if isinstance(self.data_path, str) else self.data_path
        recommended_config_path = data_path / "recommended_countries_for_multilingual.json"
        
        if recommended_config_path.exists():
            print(f"Using recommended-country subset: {recommended_config_path}")
            with open(recommended_config_path, 'r', encoding='utf-8') as f:
                recommended_data = json.load(f)
            
            selected_countries_by_lang = {}
            for lang_code, country_list in recommended_data.items():
                if lang_code != 'metadata' and isinstance(country_list, list):
                    selected_countries_by_lang[lang_code] = [c.get('name') for c in country_list]
            
            result = {"languages": {}}
            for lang_code in selected_countries_by_lang:
                if lang_code in complete_config.get("languages", {}):
                    result["languages"][lang_code] = {
                        "questions": complete_config["languages"][lang_code]["questions"],
                        "countries": selected_countries_by_lang[lang_code]
                    }
                    if "name" in complete_config["languages"][lang_code]:
                        result["languages"][lang_code]["name"] = complete_config["languages"][lang_code]["name"]
            
            print(f"Languages included after filtering: {list(result['languages'].keys())}")
            for lang, data in result["languages"].items():
                print(f"   {lang}: {len(data['countries'])} countries")
            
            return result
        else:
            print("Using the full multilingual configuration")
            return complete_config
    
    def _load_countries_data(self) -> Dict:
        """Load the country metadata table used throughout the pipeline."""
        config_country_path = Path(__file__).parent.parent.parent / "config" / "country" / "country_codes.pkl"
        
        if config_country_path.exists():
            with open(config_country_path, 'rb') as f:
                return pickle.load(f)
        
        print(f"Missing country_codes.pkl: {config_country_path}")
        return {}
    
    def _load_cultural_regions(self) -> Dict:
        """Load the cultural-region configuration."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'country' / 'cultural_regions.json'
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        print(f"Missing cultural_regions.json: {config_path}")
        return {}
    
    def _create_roleplay_prompt(self, country: str, language: str) -> str:
        """Create the language-specific roleplay system prompt."""
        system_prompt = self.system_prompts.get(language, self.system_prompts["zh-cn"])
        
        if language in self.system_prompts:
            print(f"  Using {language} prompt template")
        else:
            print(f"  Language {language} not found; falling back to zh-cn")
        
        return system_prompt.format(country=country)
    
    
    
    
    def interview_country_multilingual_with_repeats(self, model_name: str, country: str, language: str) -> Dict[str, Any]:
        """Repeat the full roleplay questionnaire and aggregate by majority vote."""
        print(f"Running {self.consensus_count} full questionnaire rounds...")
        
        all_rounds = []
        
        for round_idx in range(self.consensus_count):
            print(f"\n┌{'─'*78}┐")
            print(f"│ Round {round_idx + 1}/{self.consensus_count} of the full questionnaire" + " " * (78 - 35 - len(f"{round_idx + 1}/{self.consensus_count}")) + "│")
            print(f"└{'─'*78}┘")
            
            round_result = self.interview_country_multilingual(model_name, country, language)
            round_result['round_id'] = round_idx + 1
            all_rounds.append(round_result)
            
            print(f"Round {round_idx + 1} completed - success rate: {round_result['success_rate']:.0f}%")
            
            if round_idx < self.consensus_count - 1:
                time.sleep(0.5)
        
        final_responses = self._aggregate_repeated_interviews(all_rounds)
        
        valid_count = sum(1 for r in final_responses if r['final_response'] is not None)
        avg_confidence = sum(r['confidence'] for r in final_responses) / len(final_responses) if final_responses else 0
        
        intermediate_data = {
            'consensus_count': self.consensus_count,
            'all_rounds': all_rounds,
            'overall_consistency': avg_confidence,
            'consistency_stats': {}
        }
        
        result = {
            "model": model_name,
            "country": country,
            "language": language,
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(final_responses),
            "responses": final_responses,
            "intermediate_data": intermediate_data,
            "valid_responses": valid_count,
            "success_rate": valid_count / len(final_responses) * 100 if final_responses else 0
        }
        
        return result
    
    def _aggregate_repeated_interviews(self, all_rounds: List[Dict]) -> List[Dict]:
        """Aggregate repeated questionnaire rounds question by question."""
        from collections import defaultdict, Counter
        
        question_responses = defaultdict(lambda: {
            'question_id': None,
            'question': None,
            'scale': None,
            'dimension': None,
            'all_responses': [],
            'all_raw_responses': []
        })
        
        for round_data in all_rounds:
            for response in round_data['responses']:
                qid = response['question_id']
                question_responses[qid]['question_id'] = qid
                question_responses[qid]['question'] = response['question']
                question_responses[qid]['scale'] = response['scale']
                question_responses[qid]['dimension'] = response['dimension']
                question_responses[qid]['all_responses'].append(response['processed_response'])
                question_responses[qid]['all_raw_responses'].append(response['raw_response'])
        
        final_responses = []
        for qid, data in question_responses.items():
            final_response, confidence = self._calculate_mode(data['all_responses'])
            
            valid_responses = [r for r in data['all_responses'] if r is not None]
            response_distribution = dict(Counter(valid_responses)) if valid_responses else {}
            
            confidence_label = "high" if confidence >= 0.8 else "medium" if confidence >= 0.6 else "low"
            print(f"  {qid}: {data['all_responses']} -> [{final_response}] (confidence: {confidence*100:.0f}%, {confidence_label})")
            
            final_responses.append({
                'question_id': qid,
                'question': data['question'],
                'scale': data['scale'],
                'dimension': data['dimension'],
                'final_response': final_response,
                'confidence': confidence,
                'all_responses': data['all_responses'],
                'all_raw_responses': data['all_raw_responses'],
                'response_distribution': response_distribution,
            })
        
        return final_responses
    
    def interview_country_multilingual(self, model_name: str, country: str, language: str) -> Dict[str, Any]:
        """Run one full multilingual roleplay questionnaire for a country."""
        questions = self.multilingual_config["languages"][language]["questions"]
        
        system_prompt = self._create_roleplay_prompt(country, language)
        
        responses = []
        valid_responses = 0
        
        question_items = list(questions.items())
        for idx, (question_id, question_data) in enumerate(question_items):
            try:
                response_text = self.call_model_api(
                    model_name, 
                    question_id,
                    question_data["question"],
                    system_prompt=system_prompt,
                    max_tokens=100
                )
                
                output_line = f"  Question {idx+1}/{len(question_items)}: {question_id} "
                
                if response_text:
                    processed_response = self._process_response(response_text, question_id)
                    
                    if processed_response:
                        valid_responses += 1
                        output_line += f"{processed_response}"
                    else:
                        output_line += f"Invalid: {response_text.strip()}"
                    
                    print(output_line)
                    
                    responses.append({
                        "question_id": question_id,
                        "question": question_data["question"],
                        "raw_response": response_text,
                        "processed_response": processed_response,
                        "scale": question_data["scale"],
                        "dimension": question_data["dimension"]
                    })
                else:
                    output_line += "API call failed (see earlier console output for details)"
                    print(output_line)
                    
                    failure_info = {
                        "question_id": question_id,
                        "question": question_data["question"],
                        "raw_response": None,
                        "processed_response": None,
                        "scale": question_data["scale"],
                        "dimension": question_data["dimension"],
                        "failure_reason": "API call returned None - see console for details"
                    }
                    responses.append(failure_info)
                    
                    import json
                    from pathlib import Path
                    temp_dir = Path(self.data_path) / "temp_failures"
                    temp_dir.mkdir(exist_ok=True)
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_file = temp_dir / f"{model_name.replace('/', '_')}_{country}_{language}_{question_id}_{timestamp_str}.json"
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "model": model_name,
                            "country": country,
                            "language": language,
                            "question_id": question_id,
                            "question": question_data["question"],
                            "timestamp": datetime.now().isoformat(),
                            "note": "Check console output above for detailed error message from base_interview.py"
                        }, f, indent=2, ensure_ascii=False)
                    print(f"   Failure note saved to: {temp_file.name}")
                
                if any(x in model_name.lower() for x in ['gpt', 'claude']):
                    time.sleep(0.2)
                else:
                    time.sleep(0.3)
                
            except Exception as e:
                print(f"  {question_id}: {e}")
                responses.append({
                    "question_id": question_id,
                    "question": question_data.get("question", ""),
                    "raw_response": None,
                    "processed_response": None,
                    "scale": question_data.get("scale", ""),
                    "dimension": question_data.get("dimension", ""),
                    "error": str(e)
                })
        
        result = {
            "model": model_name,
            "country": country,
            "language": language,
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(questions),
            "valid_responses": valid_responses,
            "success_rate": valid_responses / len(questions) * 100 if questions else 0,
            "responses": responses
        }
        
        print(f"  Completed: {valid_responses}/{len(questions)} valid responses")
        return result
    
    def _calculate_mode(self, responses: List) -> Tuple[Any, float]:
        """Calculate the modal response and its confidence, ignoring list order when needed."""
        from collections import Counter
        
        valid_responses = [r for r in responses if r is not None]
        
        if not valid_responses:
            return None, 0.0
        
        if valid_responses and isinstance(valid_responses[0], (list, tuple)):
            normalized_responses = [tuple(sorted(r)) if isinstance(r, (list, tuple)) else r 
                                   for r in valid_responses]
            
            counter = Counter(normalized_responses)
            most_common_normalized, mode_count = counter.most_common(1)[0]
            
            mode_value = None
            for i, norm_resp in enumerate(normalized_responses):
                if norm_resp == most_common_normalized:
                    mode_value = valid_responses[i]
                    break
        else:
            counter = Counter(valid_responses)
            most_common = counter.most_common(1)[0]
            mode_value = most_common[0]
            mode_count = most_common[1]
        
        confidence = mode_count / len(valid_responses)
        
        return mode_value, confidence
    
    def _process_response(self, response_text: str, question_id: str) -> Optional[str]:
        """Parse and validate one response using the shared IVS response processor."""
        return IVSQuestionProcessor.parse_response_text(response_text, question_id)
    
    def _load_test_specific_config(self, test_type: str) -> Dict:
        """Load a test-specific multilingual configuration when requested."""
        project_root = Path(__file__).parent.parent.parent
        
        if test_type == "small_scale":
            config_path = project_root / "config" / "questions" / "multilingual" / "small_scale_test_config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    test_config = config_data.get('small_scale_test_config', {})
                    print(f"Loaded small-scale test configuration: {config_path.name}")
                    return test_config
        elif test_type == "comprehensive":
            config_path = project_root / "config" / "questions" / "multilingual" / "comprehensive_multilingual_config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    test_config = config_data.get('comprehensive_test_config', {})
                    print(f"Loaded comprehensive test configuration: {config_path.name}")
                    return test_config
        
        print("Using the default multilingual configuration")
        return self.multilingual_config
    
    
    def _extract_country_name(self, country_data: Any) -> str:
        """Extract a country name from either a string or a structured mapping."""
        if isinstance(country_data, dict):
            return country_data.get('name', str(country_data))
        elif isinstance(country_data, str):
            return country_data
        else:
            return str(country_data)
    
    def _load_existing_interview_data(self, data_dir: Path = None) -> Dict[str, List[Dict[str, Any]]]:
        """Load cached per-task roleplay interview files for incremental runs."""
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data" / "roleplay_multilingual" / "llm_responses_roleplay_ml"
        
        if not data_dir.exists():
            print(f"Roleplay cache directory does not exist: {data_dir}")
            return {}
        
        import pickle
        
        individual_pkl_files = []
        individual_json_files = []
        
        for subdir in data_dir.iterdir():
            if subdir.is_dir():
                individual_pkl_files.extend([f for f in subdir.glob("*.pkl") 
                                            if not f.name.startswith("roleplay_results_ml_")])
                individual_json_files.extend([f for f in subdir.glob("*.json") 
                                             if not f.name.startswith("roleplay_results_ml_")])
        
        if individual_pkl_files:
            result_files = individual_pkl_files
            print(f"Found {len(individual_pkl_files)} standalone PKL files")
        elif individual_json_files:
            result_files = individual_json_files
            print(f"Found {len(individual_json_files)} standalone JSON files")
        else:
            print("No cached roleplay interview files were found")
            return {}
        
        existing_data = {}
        loaded_count = 0
        
        for result_file in result_files:
            try:
                if result_file.suffix == '.pkl':
                    with open(result_file, 'rb') as f:
                        result = pickle.load(f)
                else:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                
                if not isinstance(result, dict):
                    continue
                
                model = result.get('model') or result.get('model_name', '')
                if not model:
                    print(f"   {result_file.name}: missing model information")
                    continue
                
                country_data = result.get('country')
                if not country_data:
                    print(f"   {result_file.name}: missing country information")
                    continue
                
                language = result.get('language', '')
                if not language:
                    print(f"   {result_file.name}: missing language information")
                    continue
                
                country = self._extract_country_name(country_data)
                
                valid_responses = result.get('valid_responses', 0)
                if valid_responses == 0:
                    responses = result.get('responses', [])
                    if isinstance(responses, list) and len(responses) > 0:
                        valid_count = sum(1 for r in responses 
                                        if isinstance(r, dict) and 
                                        (r.get('processed_response') or r.get('final_response')))
                        if valid_count == 0:
                            print(f"   {model} | {country} | {language}: no valid responses")
                            continue
                    else:
                        print(f"   {model} | {country} | {language}: no valid responses")
                        continue
                
                key = (model, country, language)
                
                if key not in existing_data:
                    existing_data[key] = []
                
                existing_data[key].append(result)
                loaded_count += 1
                print(f"   {model} | {country} | {language}: {valid_responses}/10 valid responses")
                    
            except Exception as e:
                print(f"   {result_file.name}: failed to load - {e}")
                continue
        
        print(f"\nLoaded {loaded_count} cached results across {len(existing_data)} task combinations")
        return existing_data
    
    
    def _merge_interview_results(self, existing_data: Dict, new_results: List[Dict]) -> Dict:
        """Merge cached task results with newly collected roleplay outputs."""
        unique_results = {}
        
        for (model, country, language), results in existing_data.items():
            key = (model, country, language)
            
            if isinstance(results, list):
                latest_result = None
                latest_timestamp = ''
                
                for result in results:
                    current_timestamp = result.get('timestamp', '')
                    if current_timestamp > latest_timestamp:
                        latest_timestamp = current_timestamp
                        latest_result = result
                
                if latest_result:
                    unique_results[key] = latest_result
            else:
                unique_results[key] = results
        
        for result in new_results:
            if result and result.get('valid_responses', 0) > 0:
                model = result.get('model_name', result.get('model'))
                country_raw = result.get('country')
                if isinstance(country_raw, dict):
                    country = country_raw.get('name')
                else:
                    country = country_raw
                language = result.get('language')
                
                key = (model, country, language)
                
                if key in unique_results:
                    existing_timestamp = unique_results[key].get('timestamp', '')
                    current_timestamp = result.get('timestamp', '')
                    if current_timestamp > existing_timestamp:
                        unique_results[key] = result
                        print(f"   Updated: {model.split('/')[-1]} | {country} | {language}")
                else:
                    unique_results[key] = result
                    print(f"   Added: {model.split('/')[-1]} | {country} | {language}")
        
        all_results = list(unique_results.values())
        
        print("\nMerged roleplay results:")
        print(f"   - Unique task combinations: {len(unique_results)}")
        print(f"   - Total records: {len(all_results)}")
        
        return {
            "experiment_type": "multilingual_roleplay",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_tasks": len(all_results),
            "successful_tasks": len([r for r in all_results if r.get('valid_responses', 0) > 0]),
            "results": all_results
        }
    
    def run_multilingual_experiment(self, models: List[str] = None, max_workers: int = 4, repeat_count: int = 1, test_type: str = "standard", skip_existing: bool = True) -> Dict[str, Any]:
        """Run the multilingual roleplay interview stage, with optional incremental reuse."""
        if models is None:
            models = list(self.model_configs.keys())
            if not models:
                print("No configured models were found; falling back to a small default set")
                models = ["openai/gpt-4o-mini", "anthropic/claude-3.7-sonnet"]
            print(f"Using all available models: {len(models)}")
            for model in models:
                print(f"   - {model}")
        else:
            print(f"Using a user-specified subset of {len(models)} models")
        
        config_to_use = self._load_test_specific_config(test_type)
        
        existing_data = {}
        completed_tasks = set()
        if skip_existing:
            print("\nChecking for cached roleplay interviews...")
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data" / "roleplay_multilingual" / "llm_responses_roleplay_ml"
            existing_data = self._load_existing_interview_data(data_dir)
            completed_tasks = set(existing_data.keys())
            
            if completed_tasks:
                print(f"Found {len(completed_tasks)} completed task combinations; these will be skipped")
                sample_tasks = list(completed_tasks)[:10]
                for model, country, language in sample_tasks:
                    print(f"   - {model} | {country} | {language}")
                if len(completed_tasks) > 10:
                    print(f"   ... and {len(completed_tasks) - 10} more")
            else:
                print("No reusable cached roleplay data was found")
        
        all_tasks = []
        for language, config in config_to_use["languages"].items():
            countries = config.get("countries", [])
            for country_info in countries:
                if isinstance(country_info, dict):
                    country = country_info.get("name", str(country_info))
                elif isinstance(country_info, str):
                    country = country_info
                else:
                    country = str(country_info)
                
                for model in models:
                    all_tasks.append((model, country, language))
        
        if skip_existing and completed_tasks:
            tasks = []
            skipped_count = 0
            for task in all_tasks:
                model, country, language = task
                if (model, country, language) in completed_tasks:
                    skipped_count += 1
                else:
                    tasks.append(task)
            
            print("\nInterview workload:")
            print(f"   - Total task combinations: {len(all_tasks)}")
            print(f"   - Already completed: {skipped_count}")
            print(f"   - To run now: {len(tasks)}")
        else:
            tasks = all_tasks
            print(f"\nTotal task combinations: {len(tasks)}")
        
        if not tasks:
            print("\nAll requested task combinations already have cached results")
            return self._merge_interview_results(existing_data, [])

        print(f"\nPreparing to run {len(tasks)} multilingual roleplay tasks")
        print(f"Concurrency: {max_workers}")
        if max_workers >= 10:
            print("High concurrency may interleave console output, but improves throughput")
        if self.consensus_count > 1:
            print(f"Each task will repeat the questionnaire {self.consensus_count} times")
            print(f"Estimated total API calls: {len(tasks)} × 10 questions × {self.consensus_count} rounds = {len(tasks) * 10 * self.consensus_count}\n")
        
        results = []
        completed = 0
        
        import threading
        progress_lock = threading.Lock()
        
        def run_single_interview(task):
            model, country, language = task
            try:
                print(f"\n{'='*60}")
                print(f"Task: {model} roleplaying {country} ({language})")
                print(f"{'='*60}")
                
                if self.consensus_count > 1:
                    result = self.interview_country_multilingual_with_repeats(model, country, language)
                else:
                    result = self.interview_country_multilingual(model, country, language)
                
                if result and result.get('valid_responses', 0) > 0:
                    print(f"{model} roleplaying {country} ({language}) completed: {result['valid_responses']} valid responses")
                    if result.get('intermediate_data'):
                        consistency = result['intermediate_data'].get('overall_consistency', 0)
                        print(f"   Overall consistency: {consistency:.1%}")
                    
                    try:
                        self._save_individual_result(model, country, language, result)
                    except Exception as save_error:
                        print(f"Could not save the per-task result: {save_error}")
                else:
                    print(f"{model} roleplaying {country} ({language}) failed")
                
                return result
            except Exception as e:
                print(f"\nTask failed: {model} - {country} ({language}): {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(run_single_interview, task): task for task in tasks}
            
            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result()
                if result:
                    results.append(result)
                
                with progress_lock:
                    completed += 1
                    task = future_to_task[future]
                    model, country, language = task
                    progress_pct = completed / len(tasks) * 100
                    print(f"\n{'='*60}")
                    print(f"Overall progress: [{completed}/{len(tasks)}] ({progress_pct:.1f}%)")
                    print(f"{'='*60}")
        
        print("\nMerging cached and newly collected roleplay results...")
        merged_data = self._merge_interview_results(existing_data, results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data" / "roleplay_multilingual" / "llm_responses_roleplay_ml"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_file = output_dir / f"roleplay_results_ml_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        pkl_file = output_dir / f"roleplay_results_ml_{timestamp}.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(merged_data, f)
        
        print(f"\n{'='*60}")
        print("All roleplay results saved:")
        print(f"   - Newly completed tasks: {len([r for r in results if r])}")
        print(f"   - Cached tasks: {len(existing_data)}")
        print(f"   - Total tasks: {merged_data['total_tasks']}")
        print(f"   - Successful tasks: {merged_data['successful_tasks']}")
        print(f"   - Directory: {output_dir}")
        print(f"   - JSON: roleplay_results_ml_{timestamp}.json")
        print(f"   - Pickle: roleplay_results_ml_{timestamp}.pkl")
        print(f"{'='*60}")
        
        return merged_data
    
    def interview_entity(self, model_name: str, entity_id: str) -> Tuple[List[LLMResponse], Dict]:
        """Interview one roleplay entity and return BaseInterview-compatible outputs."""
        if '_' in entity_id:
            country, language = entity_id.rsplit('_', 1)
        else:
            country, language = entity_id, 'zh-cn'
        
        if self.consensus_count > 1:
            result = self.interview_country_multilingual_with_repeats(model_name, country, language)
        else:
            result = self.interview_country_multilingual(model_name, country, language)
        
        responses = []
        for response_data in result.get('responses', []):
            llm_response = LLMResponse(
                model_name=model_name,
                question_id=response_data['question_id'],
                response=response_data.get('processed_response'),
                raw_response=response_data.get('raw_response', ''),
                is_valid=response_data.get('processed_response') is not None
            )
            responses.append(llm_response)
        
        intermediate_data = result.get('intermediate_data', {})
        
        return responses, intermediate_data
    
    def _save_individual_result(self, model_name: str, country: str, language: str, result: Dict) -> None:
        """Save one task's roleplay output to standalone cache files."""
        try:
            project_root = Path(__file__).parent.parent.parent
            save_dir = project_root / "data" / "roleplay_multilingual" / "llm_responses_roleplay_ml"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            model_short = model_name.split('/')[-1]
            filename = f"{model_short}_{country}_{language}_{timestamp}"
            
            json_path = save_dir / f"{filename}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            pkl_path = save_dir / f"{filename}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(result, f)
            
            print(f"Saved: {filename}.json")
            
        except Exception as e:
            print(f"Could not save the per-task cache file: {e}")
    
    def _on_task_completed(self, model_name: str, entity_id: str, 
                          responses: List, intermediate_data: Dict = None):
        """Persist each completed roleplay task immediately."""
        if '_' in entity_id:
            country, language = entity_id.rsplit('_', 1)
        else:
            country, language = entity_id, 'unknown'
        
        if not responses:
            print(f"Skipping save because no valid responses were produced: {model_name} - {country} ({language})")
            return
        
        country_standardized = self.name_standardizer.standardize(country)
        if not country_standardized:
            country_standardized = country
        
        result = {
            "model": model_name,
            "country": country_standardized,
            "country_original": country,
            "language": language,
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(responses) if responses else 0,
            "valid_responses": sum(1 for r in responses if r and hasattr(r, 'is_valid') and r.is_valid),
            "responses": []
        }
        
        for r in responses:
            if r and hasattr(r, 'question_id'):
                result["responses"].append({
                    "question_id": r.question_id,
                    "raw_response": r.raw_response,
                    "processed_response": r.response,
                    "is_valid": r.is_valid
                })
        
        if intermediate_data:
            result["intermediate_data"] = intermediate_data
            
            if "country" in result["intermediate_data"]:
                del result["intermediate_data"]["country"]
        
        result["country"] = country_standardized
        result["country_original"] = country
        
        self._save_individual_result(model_name, country, language, result)
    
    def batch_interview(self, model_names: List[str], entities: List[str] = None, max_workers: int = 4) -> Dict[str, Any]:
        """Run batch roleplay interviews through the shared BaseInterview helpers."""
        if entities is None:
            entities = []
            for language, config in self.multilingual_config["languages"].items():
                for country in config["countries"]:
                    entities.append(f"{country}_{language}")
        
        print("Starting batched multilingual roleplay interviews:")
        print(f"  Models: {len(model_names)}")
        print(f"  Language-country pairs: {len(entities)}")
        print(f"  Total tasks: {len(model_names) * len(entities)}")
        print(f"  Execution mode: {'sequential' if max_workers == 1 else f'concurrent (max_workers={max_workers})'}")
        if self.consensus_count > 1:
            print(f"  Consensus mode: each task repeats the questionnaire {self.consensus_count} times")
        
        tasks = [
            {'model_name': model, 'entity_id': entity}
            for model in model_names
            for entity in entities
        ]
        
        if max_workers == 1:
            return super()._batch_interview_sequential(tasks)
        else:
            return super()._batch_interview_concurrent(tasks, max_workers)
    


def main():
    """Run a small smoke test for multilingual roleplay interviewing."""
    print("=== Multilingual roleplay interview system ===")
    
    interviewer = MultilingualRoleplayInterview()
    
    models = ["openai/gpt-4o-mini"]
    results = interviewer.run_multilingual_experiment(models=models, max_workers=2)
    
    print("\nRun completed")
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Successful tasks: {results['successful_tasks']}")
    print(f"Success rate: {results['successful_tasks']/results['total_tasks']*100:.1f}%")


if __name__ == "__main__":
    main()
