"""Shared interview utilities used by the LLM-response pipelines."""

import os
import json
import time
import pickle
from datetime import datetime
from openai import OpenAI
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod

# Load API credentials from the project-local .env file only.
_ENV_VARS = {}
try:
    from dotenv import dotenv_values
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        _ENV_VARS = dotenv_values(env_path)
        loaded_keys = ", ".join([k for k in _ENV_VARS.keys() if 'KEY' in k])
        if loaded_keys:
            print(f"Loaded API credentials from .env: {loaded_keys}")
except ImportError:
    pass

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.base.ivs_questionnaire import IVSQuestions, LLMResponse, ResponseValidator


class BaseInterview(ABC):
    """Base class for model interviews and response collection."""
    
    def __init__(self, max_retry: int = 3, data_path: str = "data", 
                 consensus_count: int = 1, model_config_file: str = None):
        """Initialize shared interview settings and model configuration."""
        self.questions = IVSQuestions()
        self.validator = ResponseValidator()
        self.model_config_file = model_config_file or 'llm_models.json'
        self.model_configs = self._load_model_configs()
        self.api_keys = self._load_api_keys()
        self.max_retry = max_retry
        self.consensus_count = consensus_count
        self.data_path = Path(data_path)
    
    def _load_model_configs(self) -> Dict[str, Dict[str, str]]:
        """Load model definitions from the configured JSON file."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'models' / self.model_config_file
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('models', {})
        except Exception as e:
            print(f"Could not load model configuration: {e}")
            return {}
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from the project-local .env file."""
        api_keys = {}
        env_loaded = bool(_ENV_VARS)
        for model_name, config in self.model_configs.items():
            api_key_name = config.get('api_key', 'OPENROUTER_API_KEY')
            api_key = _ENV_VARS.get(api_key_name)
            if api_key:
                api_keys[model_name] = api_key
            elif env_loaded:
                print(f"Missing API key for {model_name} in .env ({api_key_name}).")
        return api_keys
    
    def get_client(self, model_name: str) -> OpenAI:
        """Create an OpenAI-compatible client for a configured model."""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        api_key = self.api_keys.get(model_name)
        
        if not api_key:
            raise ValueError(f"API key is not configured for model {model_name}")
        
        # Smaller hosted models sometimes need a longer timeout budget.
        timeout = 120.0 if any(x in model_name.lower() for x in ['llama-3.2', 'qwen3-1.7b', 'qwen-1.5', 'phi']) else 60.0
        
        return OpenAI(
            api_key=api_key,
            base_url=config.get('base_url', 'https://api.openai.com/v1'),
            timeout=timeout
        )
    
    def _get_format_hint(self, question_id: str, attempt: int = 0, language: str = "en") -> str:
        """Return a language-specific response-format hint."""
        format_hints = {
            "zh-cn": {
                "Y002": "\n\n格式：回答两个不同的数字，用空格分隔（例如：'1 3'）。两个数字必须不同。",
                "Y003": "\n\n格式：回答1-5个数字（从1-11中选），用空格分隔（例如：'2 4 6 8 10'）。",
                "single": "\n\n格式：只回答一个数字（例如：'3'）。",
                "retry": "\n\n第{attempt}次尝试：请只提供数字，不要文字。"
            },
            "en": {
                "Y002": "\n\nFormat: Respond with exactly 2 DIFFERENT numbers separated by space (e.g., '1 3'). The two numbers must be different.",
                "Y003": "\n\nFormat: Respond with 1-5 numbers (from 1-11) separated by spaces (e.g., '2 4 6 8 10').",
                "single": "\n\nFormat: Respond with ONE number only (e.g., '3').",
                "retry": "\n\nAttempt {attempt}: Please provide ONLY numbers, no text."
            },
            "ru": {
                "Y002": "\n\nФормат: Ответьте ДВУМЯ РАЗНЫМИ числами, разделенными пробелом (например: '1 3'). Два числа должны быть разными.",
                "Y003": "\n\nФормат: Ответьте 1-5 числами (от 1 до 11), разделенными пробелами (например: '2 4 6 8 10').",
                "single": "\n\nФормат: Ответьте ОДНИМ числом (например: '3').",
                "retry": "\n\nПопытка {attempt}: Пожалуйста, предоставьте ТОЛЬКО числа, без текста."
            },
            "es": {
                "Y002": "\n\nFormato: Responda con exactamente 2 números DIFERENTES separados por espacio (ej: '1 3'). Los dos números deben ser diferentes.",
                "Y003": "\n\nFormato: Responda con 1-5 números (del 1 al 11) separados por espacios (ej: '2 4 6 8 10').",
                "single": "\n\nFormato: Responda con UN solo número (ej: '3').",
                "retry": "\n\nIntento {attempt}: Por favor, proporcione SOLO números, sin texto."
            },
            "ar": {
                "Y002": "\n\nالتنسيق: أجب برقمين مختلفين مفصولين بمسافة (مثال: '1 3'). يجب أن يكون الرقمان مختلفين.",
                "Y003": "\n\nالتنسيق: أجب بـ 1-5 أرقام (من 1 إلى 11) مفصولة بمسافات (مثال: '2 4 6 8 10').",
                "single": "\n\nالتنسيق: أجب برقم واحد فقط (مثال: '3').",
                "retry": "\n\nالمحاولة {attempt}: يرجى تقديم الأرقام فقط، بدون نص."
            },
            "fr": {
                "Y002": "\n\nFormat: Répondez avec exactement 2 numéros DIFFÉRENTS séparés par un espace (ex: '1 3'). Les deux numéros doivent être différents.",
                "Y003": "\n\nFormat: Répondez avec 1-5 numéros (de 1 à 11) séparés par des espaces (ex: '2 4 6 8 10').",
                "single": "\n\nFormat: Répondez avec UN seul numéro (ex: '3').",
                "retry": "\n\nTentative {attempt}: Veuillez fournir UNIQUEMENT des numéros, sans texte."
            },
            "de": {
                "Y002": "\n\nFormat: Antworten Sie mit genau 2 VERSCHIEDENEN Zahlen, getrennt durch Leerzeichen (z.B.: '1 3'). Die zwei Zahlen müssen unterschiedlich sein.",
                "Y003": "\n\nFormat: Antworten Sie mit 1-5 Zahlen (von 1 bis 11), getrennt durch Leerzeichen (z.B.: '2 4 6 8 10').",
                "single": "\n\nFormat: Antworten Sie mit NUR EINER Zahl (z.B.: '3').",
                "retry": "\n\nVersuch {attempt}: Bitte geben Sie NUR Zahlen an, keinen Text."
            },
            "pt": {
                "Y002": "\n\nFormato: Responda com exatamente 2 números DIFERENTES separados por espaço (ex: '1 3'). Os dois números devem ser diferentes.",
                "Y003": "\n\nFormato: Responda com 1-5 números (de 1 a 11) separados por espaços (ex: '2 4 6 8 10').",
                "single": "\n\nFormato: Responda com APENAS UM número (ex: '3').",
                "retry": "\n\nTentativa {attempt}: Por favor, forneça APENAS números, sem texto."
            },
            "it": {
                "Y002": "\n\nFormato: Rispondi con esattamente 2 numeri DIVERSI separati da uno spazio (es: '1 3'). I due numeri devono essere diversi.",
                "Y003": "\n\nFormato: Rispondi con 1-5 numeri (da 1 a 11) separati da spazi (es: '2 4 6 8 10').",
                "single": "\n\nFormato: Rispondi con UN solo numero (es: '3').",
                "retry": "\n\nTentativo {attempt}: Si prega di fornire SOLO numeri, nessun testo."
            },
            "ja": {
                "Y002": "\n\n形式：2つの異なる数字をスペースで区切って回答してください（例：「1 3」）。2つの数字は異なる必要があります。",
                "Y003": "\n\n形式：1〜5個の数字（1〜11から選択）をスペースで区切って回答してください（例：「2 4 6 8 10」）。",
                "single": "\n\n形式：1つの数字のみで回答してください（例：「3」）。",
                "retry": "\n\n試行{attempt}回目：数字のみを提供してください。テキストは不要です。"
            },
            "ko": {
                "Y002": "\n\n형식: 공백으로 구분된 2개의 다른 숫자로 답변하십시오 (예: '1 3'). 두 숫자는 달라야 합니다.",
                "Y003": "\n\n형식: 공백으로 구분된 1-5개의 숫자 (1-11에서 선택)로 답변하십시오 (예: '2 4 6 8 10').",
                "single": "\n\n형식: 하나의 숫자만 답변하십시오 (예: '3').",
                "retry": "\n\n시도 {attempt}회: 숫자만 제공하십시오. 텍스트는 불필요합니다."
            },
            "zh-tw": {
                "Y002": "\n\n格式：請回答兩個不同的數字，並以空格分隔（例如：'1 3'）。",
                "Y003": "\n\n格式：請回答 1 到 5 個數字（從 1 到 11 中選），並以空格分隔（例如：'2 4 6 8 10'）。",
                "single": "\n\n格式：請只回答一個數字（例如：'3'）。",
                "retry": "\n\n第{attempt}次嘗試：請只提供數字，不需任何文字。"
            },
            "zh-hk": {
                "Y002": "\n\n格式：請回答兩個不同的數字，並以空格分隔（例如：'1 3'）。",
                "Y003": "\n\n格式：請回答 1 至 5 個數字（由 1 至 11 中選），並以空格分隔（例如：'2 4 6 8 10'）。",
                "single": "\n\n格式：請只回答一個數字（例如：'3'）。",
                "retry": "\n\n第{attempt}次嘗試：請只提供數字，無須任何文字。"
            },
            "en-native": {
                "Y002": "\n\nFormat: Respond with exactly 2 DIFFERENT numbers separated by space (e.g., '1 3'). The two numbers must be different.",
                "Y003": "\n\nFormat: Respond with 1-5 numbers (from 1-11) separated by spaces (e.g., '2 4 6 8 10').",
                "single": "\n\nFormat: Respond with ONE number only (e.g., '3').",
                "retry": "\n\nAttempt {attempt}: Please provide ONLY numbers, no text."
            }
        }
        
        lang_hints = format_hints.get(language, format_hints["en"])
        
        if question_id == "Y002":
            format_hint = lang_hints["Y002"]
        elif question_id == "Y003":
            format_hint = lang_hints["Y003"]
        else:
            format_hint = lang_hints["single"]
        
        if attempt > 0:
            format_hint += lang_hints["retry"].format(attempt=attempt + 1)
        
        return format_hint
    
    def _get_dynamic_delay(self, model_name: str) -> float:
        """Return an inter-request delay based on model family."""
        if any(x in model_name.lower() for x in ['gpt', 'claude']):
            return 0.2
        elif "gemini" in model_name.lower():
            return 2.0
        else:
            return 0.3
    
    def call_model_api(self, model_name: str, question_id: str, question_text: str, 
                      system_prompt: str, max_tokens: int = 50) -> Optional[str]:
        """Call a chat-completions endpoint with retries and response validation hints."""
        max_retries = 5 if "gemini" in model_name.lower() else 3
        
        language = "en"
        if "您正在参与" in system_prompt or "请基于您的文化背景" in system_prompt:
            language = "zh-cn"
        elif "您正在參與" in system_prompt:
            if "香港常用的繁體中文書面語" in system_prompt or "避免台灣用語" in system_prompt:
                language = "zh-hk"
            elif "台灣常用的正體中文書面語" in system_prompt:
                language = "zh-tw"
            else:
                language = "zh-tw"
        elif "Вы участвуете" in system_prompt:
            language = "ru"
        elif "Está participando" in system_prompt:
            language = "es"
        elif "أنت تشارك" in system_prompt:
            language = "ar"
        elif "Vous participez" in system_prompt:
            language = "fr"
        elif "Sie nehmen" in system_prompt:
            language = "de"
        elif "Você está participando" in system_prompt:
            language = "pt"
        elif "Stai partecipando" in system_prompt:
            language = "it"
        elif "あなたは文化的価值観" in system_prompt:
            language = "ja"
        elif "귀하는 문화적 가치관" in system_prompt:
            language = "ko"
        elif "You are participating" in system_prompt and "CRITICAL RESPONSE RULES" in system_prompt:
            language = "en"
        
        current_max_tokens = max_tokens
        
        for attempt in range(max_retries):
            try:
                format_hint = self._get_format_hint(question_id, attempt, language)
                
                enhanced_system_prompt = system_prompt
                thinking_models = ['gemini-3-pro', 'qwq', 'gpt-5', 'glm']
                if any(keyword in model_name.lower() for keyword in thinking_models):
                    no_reasoning_instructions = {
                        "en": "\n\n🚫 CRITICAL: Your response MUST be ONLY the number(s). DO NOT include ANY reasoning, thinking, or explanation. NO text except the number(s). This is mandatory.",
                        "en-native": "\n\n🚫 CRITICAL: Your response MUST be ONLY the number(s). DO NOT include ANY reasoning, thinking, or explanation. NO text except the number(s). This is mandatory.",
                        "zh-cn": "\n\n🚫 关键：你的回答必须只有数字。绝对不要包含任何推理、思考或解释。除了数字不要任何文字。这是强制要求。",
                        "zh-tw": "\n\n🚫 關鍵：你的回答必須只有數字。絕對不要包含任何推理、思考或解釋。除了數字之外，不需任何文字。這是強制要求。",
                        "zh-hk": "\n\n🚫 關鍵：你的回答必須只有數字。絕對不要包含任何推理、思考或解釋。除數字外，無須任何文字。這是強制要求。",
                        "ar": "\n\n🚫 حاسم: يجب أن تكون إجابتك أرقامًا فقط. لا تُضمّن أي تفكير أو استدلال أو شرح. لا نص إلا الأرقام. هذا إلزامي.",
                        "es": "\n\n🚫 CRÍTICO: Tu respuesta DEBE ser SOLO el/los número(s). NO incluyas NINGÚN razonamiento, pensamiento o explicación. NINGÚN texto excepto el/los número(s). Esto es obligatorio.",
                        "ru": "\n\n🚫 КРИТИЧЕСКИ ВАЖНО: Ваш ответ ДОЛЖЕН быть ТОЛЬКО числом/числами. НЕ включайте рассуждения, мышление или объяснения. Никакого текста, кроме чисел. Это обязательно.",
                        "fr": "\n\n🚫 CRITIQUE : Votre réponse DOIT être UNIQUEMENT le(s) numéro(s). N'incluez AUCUN raisonnement, réflexion ou explication. AUCUN texte sauf le(s) numéro(s). C'est obligatoire.",
                        "de": "\n\n🚫 KRITISCH: Ihre Antwort MUSS NUR die Zahl(en) sein. Fügen Sie KEINE Überlegungen, Gedanken oder Erklärungen hinzu. KEIN Text außer der Zahl(en). Dies ist zwingend erforderlich.",
                        "pt": "\n\n🚫 CRÍTICO: Sua resposta DEVE ser APENAS o(s) número(s). NÃO inclua NENHUM raciocínio, pensamento ou explicação. NENHUM texto exceto o(s) número(s). Isto é obrigatório.",
                        "it": "\n\n🚫 CRITICO: La tua risposta DEVE essere SOLO il/i numero/i. NON includere ALCUN ragionamento, pensiero o spiegazione. NESSUN testo tranne il/i numero/i. Questo è obbligatorio.",
                        "ja": "\n\n🚫 重要：あなたの回答は数字のみでなければなりません。推論、思考、説明を一切含めないでください。数字以外のテキストは禁止です。これは必須です。",
                        "ko": "\n\n🚫 중요: 당신의 답변은 반드시 숫자만이어야 합니다. 추론, 사고, 설명을 포함하지 마세요. 숫자 외의 텍스트는 금지입니다. 이것은 필수입니다.",
                    }
                    instruction = no_reasoning_instructions.get(language, no_reasoning_instructions["en"])
                    enhanced_system_prompt = system_prompt + instruction
                messages = [
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": question_text + format_hint}
                ]
                
                client = self.get_client(model_name)
                
                if attempt == 0:
                    if "deepseek" in model_name.lower():
                        current_max_tokens = 200
                    elif "gemini" in model_name.lower():
                        if "pro" in model_name.lower():
                            current_max_tokens = 1000
                        else:
                            current_max_tokens = 500
                    elif "qwq" in model_name.lower():
                        current_max_tokens = 1500
                    elif "gpt-5" in model_name.lower():
                        current_max_tokens = 800
                    elif "glm" in model_name.lower():
                        current_max_tokens = 2000
                
                params = {
                    'model': model_name,
                    'messages': messages,
                    'temperature': 0.1,
                    'max_tokens': current_max_tokens
                }
                
                if "qwen" in model_name.lower() and "qwq" not in model_name.lower():
                    params['extra_body'] = {'enable_thinking': False}
                elif any(keyword in model_name.lower() for keyword in ['qwq', 'glm-4']):
                    params['extra_body'] = {
                        'reasoning': {
                            'exclude': True
                        }
                    }
                elif 'gemini-3-pro' in model_name.lower():
                    params['extra_body'] = {
                        'reasoning': {
                            'effort': 'low',
                            'exclude': True
                        }
                    }
                
                response = client.chat.completions.create(**params)
                
                is_thinking_model = any(keyword in model_name.lower() for keyword in ['gemini-3-pro', 'qwq', 'gpt-5', 'glm'])
                if is_thinking_model and hasattr(response, 'choices') and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    has_reasoning = hasattr(response.choices[0].message, 'reasoning') and response.choices[0].message.reasoning
                    
                    if has_reasoning:
                        print(f"   reasoning field present; length={len(response.choices[0].message.reasoning)}")
                    
                    if hasattr(response, 'usage') and response.usage:
                        usage = response.usage
                        total = getattr(usage, 'total_tokens', 0)
                        prompt = getattr(usage, 'prompt_tokens', 0)
                        completion = getattr(usage, 'completion_tokens', 0)
                        print(f"   tokens: prompt={prompt}, completion={completion}, total={total}")
                    
                    if content:
                        content_preview = content[:50] if len(content) > 50 else content
                        print(f"   content preview: '{content_preview}{'...' if len(content) > 50 else ''}'")
                    else:
                        print("   content is empty")
                
                if isinstance(response, str):
                    print(f"API returned a plain string: {response[:100]}")
                    return response.strip()
                elif hasattr(response, 'choices'):
                    content = response.choices[0].message.content
                    finish_reason = response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else 'unknown'
                    
                    if content is None or (isinstance(content, str) and not content.strip()):
                        print(f"\nEmpty API response for question {question_id} (attempt {attempt+1}/{max_retries})")
                        print(f"   finish_reason: {finish_reason}")
                        print(f"   model: {model_name}")
                        print(f"   max_tokens: {params.get('max_tokens', 'unknown')}")
                        
                        if hasattr(response.choices[0], '__dict__'):
                            print(f"   response.choices[0]: {response.choices[0].__dict__}")
                        
                        if "gemini" in model_name.lower():
                            if finish_reason == 'SAFETY':
                                print("   Gemini safety filter blocked the response.")
                                print(f"   Question preview: {question_text[:100]}...")
                            elif finish_reason == 'RECITATION':
                                print("   Gemini recitation filter blocked the response.")
                            elif finish_reason == 'length':
                                print("   Response was truncated by the length limit; increasing max_tokens.")
                            else:
                                print(f"   Unrecognized Gemini finish_reason: {finish_reason}")
                        
                        if finish_reason == 'length' and "gemini" in model_name.lower():
                            if attempt < max_retries - 1:
                                max_limit = 8192 if "pro" in model_name.lower() else 2048
                                new_tokens = min(current_max_tokens * 2, max_limit)
                                
                                if new_tokens == current_max_tokens:
                                    print(f"   Reached max_tokens upper bound ({max_limit}); stopping retries.")
                                    raise ValueError(f"Gemini would require more than {max_limit} tokens")
                                
                                print(f"   Increasing max_tokens from {current_max_tokens} to {new_tokens}.")
                                current_max_tokens = new_tokens
                                continue
                        
                        raise ValueError(f"API returned empty content (finish_reason: {finish_reason})")
                    
                    return content.strip()
                else:
                    print(f"Unknown response format: {type(response)}, {str(response)[:100]}")
                    return str(response).strip()
                    
            except Exception as e:
                error_msg = str(e)
                print(f"\nAPI call failed ({model_name}, question {question_id}, attempt {attempt+1}/{max_retries})")
                print(f"   error type: {type(e).__name__}")
                print(f"   error message: {error_msg[:800]}")
                
                if "gemini" in model_name.lower():
                    if "safety" in error_msg.lower() or "block" in error_msg.lower():
                        print("   Gemini safety filter likely blocked the request.")
                    elif "RECITATION" in error_msg:
                        print("   Gemini recitation filter likely blocked the request.")
                
                if attempt < max_retries - 1:
                    print(f"   preparing retry {attempt+2}...")
                    if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                        if any(x in model_name.lower() for x in ['llama-3.2', 'qwen3-1.7b', 'qwen-1.5', 'phi']):
                            wait_time = 5.0 * (attempt + 1)
                        else:
                            wait_time = 3.0 * (attempt + 1)
                        print(f"   timeout detected; waiting {wait_time} seconds before retrying.")
                        time.sleep(wait_time)
                    elif "500" in error_msg or "Internal Server Error" in error_msg:
                        if "gemini" in model_name.lower():
                            wait_time = 3.0 * (attempt + 1)
                        else:
                            wait_time = 2.0 * (attempt + 1)
                        print(f"   server error detected; waiting {wait_time} seconds before retrying.")
                        time.sleep(wait_time)
                    else:
                        time.sleep(0.5 * (attempt + 1))
        
        print(f"{model_name} question {question_id}: all {max_retries} retries failed; returning None.")
        return None
    
    def ask_question_with_retry(self, model_name: str, question_id: str, 
                               system_prompt: str) -> LLMResponse:
        """Ask a single question with retry and validation logic."""
        question_data = self.questions.get_question(question_id)
        if not question_data:
            return LLMResponse(
                model_name=model_name,
                question_id=question_id,
                response=None,
                raw_response=None,
                is_valid=False,
                error_message=f"Question not found: {question_id}"
            )
        
        question_text = question_data['question']
        last_error = None
        
        for attempt in range(self.max_retry):
            raw_response = self.call_model_api(
                model_name, question_id, question_text, system_prompt
            )
            
            if raw_response is None:
                continue
            
            is_valid, processed_response, error_msg = self.validator.validate_response(
                question_id, raw_response
            )
            
            if is_valid:
                return LLMResponse(
                    model_name=model_name,
                    question_id=question_id,
                    response=processed_response,
                    raw_response=raw_response,
                    is_valid=True
            )
            
            last_error = error_msg
            time.sleep(0.2)
        
        return LLMResponse(
            model_name=model_name,
            question_id=question_id,
            response=None,
            raw_response=None,
            is_valid=False,
            error_message=last_error or "All retry attempts failed"
        )
    
    def _calculate_consensus(self, responses: List[LLMResponse], 
                            question_id: str) -> LLMResponse:
        """Compute a consensus response across repeated runs."""
        from collections import Counter
        
        valid_responses = [r.response for r in responses 
                          if r.is_valid and r.response is not None]
        
        if not valid_responses:
            print(f"  {question_id}: all {len(responses)} attempts failed")
            return responses[0]
        
        if question_id in ['Y002', 'Y003']:
            normalized_responses = []
            for resp in valid_responses:
                if isinstance(resp, (list, tuple)):
                    normalized_responses.append(tuple(sorted(resp)))
                else:
                    normalized_responses.append(resp)
            
            most_common_normalized, count = Counter(normalized_responses).most_common(1)[0]
            
            consensus_value = None
            for i, norm_resp in enumerate(normalized_responses):
                if norm_resp == most_common_normalized:
                    consensus_value = valid_responses[i]
                    break
        else:
            most_common_value, count = Counter(valid_responses).most_common(1)[0]
            consensus_value = most_common_value
        
        confidence = count / len(valid_responses) if valid_responses else 0
        
        base_response = responses[0]
        return LLMResponse(
            model_name=base_response.model_name,
            question_id=question_id,
            response=consensus_value,
            raw_response=f"Consensus from {len(valid_responses)} responses (confidence={confidence:.1%}): {consensus_value}",
            is_valid=True,
            error_message=None
        )
    
    @abstractmethod
    def interview_entity(self, model_name: str, entity_id: str) -> List[LLMResponse]:
        """Interview a single entity. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def batch_interview(self, model_names: List[str], entities: List[str]) -> Dict[str, Any]:
        """Run batched interviews. Must be implemented by subclasses."""
        pass
    
    def save_results(self, results: Dict[str, Any], output_dir: str = None) -> str:
        """Save interview results to pickle and, when possible, JSON."""
        if output_dir is None:
            output_dir = self.data_path / "interview_results"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        pkl_file = output_path / f"interview_results_{timestamp}.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(results, f)
        
        try:
            json_file = output_path / f"interview_results_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"Could not write JSON output: {e}")
        
        print(f"Saved results to: {pkl_file}")
        return str(pkl_file)
    
    def _on_task_completed(self, model_name: str, entity_id: str, 
                          responses: List, intermediate_data: Dict = None):
        """Optional hook that subclasses can use for incremental saving."""
        pass
    
    def _batch_interview_sequential(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the interview tasks sequentially."""
        results = {}
        successful_tasks = 0
        total_tasks = len(tasks)
        
        for i, task in enumerate(tasks, 1):
            model_name = task['model_name']
            entity_id = task.get('entity_id', None)
            
            print(f"\n{'='*50}")
            print(f"Task {i}/{total_tasks}: {model_name}", end="")
            if entity_id:
                print(f" → {entity_id}")
            else:
                print()
            print(f"{'='*50}")
            
            responses, intermediate_data = self.interview_entity(model_name, entity_id)
            
            if responses:
                if entity_id:
                    result_key = f"{model_name}_{entity_id}"
                else:
                    result_key = model_name
                
                results[result_key] = {
                    'model_name': model_name,
                    'entity_id': entity_id,
                    'responses': responses,
                    'intermediate_data': intermediate_data
                }
                successful_tasks += 1
                
                consistency_info = ""
                if intermediate_data and 'overall_consistency' in intermediate_data:
                    consistency_info = f" (consistency: {intermediate_data['overall_consistency']:.1%})"
                print(f"Completed: {len(responses)} responses{consistency_info}")
                
                try:
                    self._on_task_completed(model_name, entity_id, responses, intermediate_data)
                except Exception as e:
                    print(f"Task-completion hook failed: {e}")
            else:
                print("Failed")
        
        return {
            'results': results,
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0
        }
    
    def _batch_interview_concurrent(self, tasks: List[Dict[str, Any]], 
                                   max_workers: int = 4) -> Dict[str, Any]:
        """Run the interview tasks concurrently."""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        results = {}
        successful_tasks = 0
        total_tasks = len(tasks)
        results_lock = threading.Lock()
        
        def interview_task(task: Dict[str, Any]):
            """Run a single interview task."""
            nonlocal successful_tasks
            
            model_name = task['model_name']
            entity_id = task.get('entity_id', None)
            
            print(f"\nStarting task: {model_name}", end="")
            if entity_id:
                print(f" → {entity_id}")
            else:
                print()
            
            responses, intermediate_data = self.interview_entity(model_name, entity_id)
            
            with results_lock:
                if responses:
                    if entity_id:
                        result_key = f"{model_name}_{entity_id}"
                    else:
                        result_key = model_name
                    
                    results[result_key] = {
                        'model_name': model_name,
                        'entity_id': entity_id,
                        'responses': responses,
                        'intermediate_data': intermediate_data
                    }
                    successful_tasks += 1
                    
                    consistency_info = ""
                    if intermediate_data and 'overall_consistency' in intermediate_data:
                        consistency_info = f" (consistency: {intermediate_data['overall_consistency']:.1%})"
                    
                    entity_info = f" → {entity_id}" if entity_id else ""
                    print(f"Completed: {model_name}{entity_info} ({len(responses)} responses{consistency_info})")
                    
                    try:
                        self._on_task_completed(model_name, entity_id, responses, intermediate_data)
                    except Exception as e:
                        print(f"Task-completion hook failed: {e}")
                else:
                    entity_info = f" → {entity_id}" if entity_id else ""
                    print(f"Failed: {model_name}{entity_info}")
        
        print(f"\nStarting concurrent interviewing: {total_tasks} tasks, {max_workers} workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(interview_task, task) for task in tasks]
            for future in futures:
                future.result()
        
        return {
            'results': results,
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0
        }
