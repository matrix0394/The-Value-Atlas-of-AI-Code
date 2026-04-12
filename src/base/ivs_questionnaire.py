"""Question definitions and response validation for the IVS item set."""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd


@dataclass
class LLMResponse:
    """Container for a single model response."""
    model_name: str
    question_id: str
    response: Any
    raw_response: str
    is_valid: bool
    error_message: Optional[str] = None
    failure_type: Optional[str] = None


class IVSQuestions:
    """Load IVS question definitions from the project configuration."""
    _questions_config: Dict = {}
    _config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'questions', 'ivs_questions.json')

    @classmethod
    def _load_config(cls):
        if not cls._questions_config:
            try:
                with open(cls._config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    cls._questions_config = config.get('ivs_questions', {})
                    print(f"IVSQuestions: loaded {len(cls._questions_config)} questions from configuration")
            except FileNotFoundError:
                print(f"IVSQuestions: configuration file not found: {cls._config_path}")
            except json.JSONDecodeError as e:
                print(f"IVSQuestions: invalid configuration format: {e}")

    @classmethod
    def get_question(cls, question_id: str) -> Dict[str, Any]:
        """Return the metadata for one question."""
        cls._load_config()
        return cls._questions_config.get(question_id, {})

    @classmethod
    def get_all_questions(cls) -> Dict[str, Dict[str, Any]]:
        """Return all question metadata."""
        cls._load_config()
        return cls._questions_config.copy()
    
    @classmethod
    def get_question_ids(cls) -> List[str]:
        """Return the ordered list of question identifiers."""
        cls._load_config()
        return list(cls._questions_config.keys())
    
    @classmethod
    def get_question_text(cls, question_id: str) -> str:
        """Return the text of one question."""
        cls._load_config()
        question_data = cls._questions_config.get(question_id, {})
        return question_data.get('question', '')


class ResponseValidator:
    """Validate response format and item-specific answer ranges."""
    
    @staticmethod
    def validate_response(question_id: str, response: str) -> tuple[bool, Any, str]:
        """Validate the format and content of a response."""
        if not response or not isinstance(response, str):
            return False, None, "Response is empty or malformed"
        
        response = response.strip()
        
        question_config = IVSQuestions.get_question(question_id)
        if not question_config:
            return False, None, f"Unknown question ID: {question_id}"
        
        # Item-specific validation is defined here because the JSON config
        # stores question text but not a standardized response schema.
        try:
            if question_id in ['A008', 'G006']:
                value = int(response)
                if 1 <= value <= 4:
                    return True, value, ""
                else:
                    return False, None, f"Expected a single choice between 1 and 4, got: {value}"
            
            elif question_id in ['A165']:
                value = int(response)
                if 1 <= value <= 2:
                    return True, value, ""
                else:
                    return False, None, f"Expected a single choice between 1 and 2, got: {value}"
                    
            elif question_id in ['E018', 'E025']:
                value = int(response)
                if 1 <= value <= 3:
                    return True, value, ""
                else:
                    return False, None, f"Expected a single choice between 1 and 3, got: {value}"
                    
            elif question_id in ['F063', 'F118', 'F120']:
                value = int(response)
                if 1 <= value <= 10:
                    return True, value, ""
                else:
                    return False, None, f"Expected a single choice between 1 and 10, got: {value}"
                    
            elif question_id == 'Y002':
                parts = response.split()
                if len(parts) != 2:
                    return False, None, f"Y002 expects exactly 2 selections, got {len(parts)}"
                
                values = [int(part) for part in parts]
                for value in values:
                    if not (1 <= value <= 4):
                        return False, None, f"Y002 selections must be between 1 and 4, got: {value}"
                
                return True, values, ""
                
            elif question_id == 'Y003':
                parts = response.split()
                if len(parts) < 1 or len(parts) > 5:
                    return False, None, f"Y003 expects between 1 and 5 selections, got {len(parts)}"
                
                values = [int(part) for part in parts]
                for value in values:
                    if not (1 <= value <= 11):
                        return False, None, f"Y003 selections must be between 1 and 11, got: {value}"
                
                return True, values, ""
                
            else:
                return False, None, f"Unknown question ID: {question_id}"
                
        except ValueError as e:
            return False, None, f"Invalid numeric format: {e}"
        except Exception as e:
            return False, None, f"Validation failed: {e}"
