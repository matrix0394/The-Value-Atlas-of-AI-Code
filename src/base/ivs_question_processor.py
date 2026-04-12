"""Helpers for parsing and recoding IVS questionnaire responses."""

import re
from typing import Tuple, List, Dict, Any, Optional


class IVSQuestionProcessor:
    """Parse and transform IVS responses into analysis-ready values."""
    
    # Configuration for supported IVS items.
    QUESTION_CONFIG = {
        "A008": {"type": "single", "scale": (1, 4), "reverse": False},
        "A165": {"type": "single", "scale": (1, 2), "reverse": False},
        "E018": {"type": "single", "scale": (1, 3), "reverse": False},
        "E025": {"type": "single", "scale": (1, 3), "reverse": False},
        "F063": {"type": "single", "scale": (1, 10), "reverse": False},
        "F118": {"type": "single", "scale": (1, 10), "reverse": False},
        "F120": {"type": "single", "scale": (1, 10), "reverse": False},
        "G006": {"type": "single", "scale": (1, 4), "reverse": False},
        "Y002": {"type": "multi", "options": 4, "choices": 2},
        "Y003": {"type": "multi", "options": 11, "choices": 5}
    }
    
    # Y002 Postmaterialism index (matches reference project culture_map.py Y002_transform)
    # Options: 1=Maintaining order, 2=Giving people more say, 3=Fighting rising prices, 4=Protecting freedom of speech
    Y002_MATERIALIST_MAPPING = {
        # 1 = Materialist (chose order+prices: options 1&3)
        (1, 3): 1, (3, 1): 1,
        # 3 = Postmaterialist (chose participation+freedom: options 2&4)
        (2, 4): 3, (4, 2): 3,
        # 2 = Mixed
        (1, 2): 2, (2, 1): 2,
        (1, 4): 2, (4, 1): 2,
        (2, 3): 2, (3, 2): 2,
        (3, 4): 2, (4, 3): 2,
    }
    
    @classmethod
    def process_y002(cls, first_choice: int, second_choice: int) -> int:
        """Process Y002 response, returns postmaterialism index value.
        Matches reference project culture_map.py Y002_transform.
        
        Args:
            first_choice: first choice (1-4)
            second_choice: second choice (1-4)
            
        Returns:
            1: Materialist (order+prices)
            2: Mixed
            3: Postmaterialist (participation+freedom)
        """
        if first_choice < 1 or first_choice > 4 or second_choice < 1 or second_choice > 4:
            return 2
            
        return cls.Y002_MATERIALIST_MAPPING.get((first_choice, second_choice), 2)
    
    @classmethod
    def process_y003(cls, selected_values: List[int]) -> Dict[str, Any]:
        """Process Y003 and return derived traditional-versus-secular measures.
        
        Args:
            selected_values: Selected child qualities (1-11).
            
        Returns:
            Dictionary containing binary encodings and derived scores.
        """
        bool_list = [i in selected_values for i in range(1, 12)]
        scores = [1 if selected else 2 for selected in bool_list]
        
        value_mapping = {
            "q7_good_manners": scores[0],
            "q8_independence": scores[1],
            "q9_hard_work": scores[2],
            "q10_responsibility": scores[3],
            "q11_imagination": scores[4],
            "q12_tolerance": scores[5],
            "q13_thrift": scores[6],
            "q14_determination": scores[7],
            "q15_religious_faith": scores[8],
            "q16_unselfishness": scores[9],
            "q17_obedience": scores[10],
        }
        
        traditional_score = value_mapping["q15_religious_faith"] + value_mapping["q17_obedience"]
        secular_rational_score = value_mapping["q8_independence"] + value_mapping["q14_determination"]
        
        y003_score = traditional_score - secular_rational_score
        
        return {
            "selected_values": selected_values,
            "value_mapping": value_mapping,
            "traditional_score": traditional_score,
            "secular_rational_score": secular_rational_score,
            "y003_score": y003_score,
            "binary_encoding": {f"Y003_{i}": (1 if i in selected_values else 0) for i in range(1, 12)}
        }
    
    @classmethod
    def process_single_choice(cls, question_id: str, value: int) -> Dict[str, Any]:
        """Process a single-choice IVS item.
        
        Args:
            question_id: Question identifier such as ``A008`` or ``F063``.
            value: Selected numeric response.
            
        Returns:
            Dictionary with the processed response.
        """
        return {
            "question_id": question_id,
            "raw_value": value,
            "processed_value": value,
            "valid": True
        }
    
    @classmethod
    def parse_response_text(cls, response_text: str, question_id: str) -> Optional[str]:
        """Extract a normalized numeric response string from raw text.
        
        Args:
            response_text: Raw model response.
            question_id: IVS question identifier.
            
        Returns:
            Parsed numeric string, or ``None`` if validation fails.
        """
        if not response_text:
            return None
        
        response = response_text.strip().lower()
        
        numbers = re.findall(r'\d+', response)
        
        if not numbers:
            return None
        
        if question_id == "Y002":
            if len(numbers) >= 2:
                return f"{numbers[0]} {numbers[1]}"
        elif question_id == "Y003":
            if len(numbers) <= 5:
                return " ".join(numbers[:5])
        else:
            return numbers[0]
        
        return None
    
    @classmethod
    def validate_and_process_response(cls, response_text: str, question_id: str) -> Dict[str, Any]:
        """Validate a response and return standardized outputs.
        
        Args:
            response_text: Raw model response.
            question_id: IVS question identifier.
            
        Returns:
            Dictionary with validation flags and derived values.
        """
        result = {
            "question_id": question_id,
            "raw_response": response_text,
            "processed_response": None,
            "numeric_value": None,
            "standardized_value": None,
            "valid": False
        }
        
        processed_response = cls.parse_response_text(response_text, question_id)
        if not processed_response:
            return result
        
        result["processed_response"] = processed_response
        
        try:
            if question_id == "Y002":
                values = [int(x.strip()) for x in processed_response.split()]
                if len(values) == 2 and all(1 <= v <= 4 for v in values):
                    result["valid"] = True
                    result["numeric_value"] = values
                    result["standardized_value"] = (values[0] - 1) / 3
                    result["materialist_score"] = cls.process_y002(values[0], values[1])
            
            elif question_id == "Y003":
                values = [int(x.strip()) for x in processed_response.split()]
                if 1 <= len(values) <= 5 and all(1 <= v <= 11 for v in values):
                    result["valid"] = True
                    result["numeric_value"] = values
                    result["standardized_value"] = (len(values) - 1) / 4
                    y003_analysis = cls.process_y003(values)
                    result.update(y003_analysis)
            
            else:
                value = int(processed_response.strip())
                result["valid"] = True
                result["numeric_value"] = value
                result["standardized_value"] = value
        
        except (ValueError, AttributeError) as e:
            print(f"Error while processing response for {question_id}: {e}")
        
        return result
