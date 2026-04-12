from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class CountryNameStandardizer:
    """
    Minimal country-name standardizer used by the PCA and multilingual
    processing pipelines.

    Responsibilities:
    1. Normalize country names to the canonical names used in the project.
    2. Map canonical country names to numeric country codes when available.
    """

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.project_root = project_root

        self.name_mapping = self._load_name_mapping()
        self.numeric_mapping = self._load_numeric_mapping()

    def _load_name_mapping(self) -> Dict[str, str]:
        """
        Optional name mapping file.

        Expected JSON format:
        {
            "South Korea": "Korea, Republic of",
            "Taiwan": "Taiwan, Province of China"
        }
        """
        candidates = [
            self.project_root / "config" / "country" / "country_name_mapping.json",
        ]

        for path in candidates:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data

        # Minimal fallback mapping
        return {
            "South Korea": "Korea, Republic of",
            "Korea": "Korea, Republic of",
            "Taiwan": "Taiwan, Province of China",
            "Hong Kong SAR": "Hong Kong",
            "Macao SAR": "Macao",
            "United States": "United States of America",
        }

    def _load_numeric_mapping(self) -> Dict[str, Any]:
        """
        Load country numeric codes from config/country/country_codes.pkl if present.
        Falls back to an empty dict if unavailable.
        """
        path = self.project_root / "config" / "country" / "country_codes.pkl"
        if not path.exists():
            return {}

        try:
            df = pd.read_pickle(path)
        except Exception:
            return {}

        mapping: Dict[str, Any] = {}

        if "Country" in df.columns and "Numeric" in df.columns:
            for _, row in df.iterrows():
                country = row.get("Country")
                numeric = row.get("Numeric")
                if pd.notna(country) and pd.notna(numeric):
                    mapping[str(country)] = numeric

        return mapping

    def standardize(self, name: Optional[Any]) -> Optional[str]:
        """
        Return the canonical project name for a country.
        """
        if name is None or pd.isna(name):
            return name

        name_str = str(name).strip()
        if not name_str:
            return name_str

        return self.name_mapping.get(name_str, name_str)

    def get_numeric_code(self, name: Optional[Any]) -> Optional[Any]:
        """
        Return the numeric country code for a country name if available.
        """
        standardized = self.standardize(name)
        if standardized is None or pd.isna(standardized):
            return None

        return self.numeric_mapping.get(str(standardized))
