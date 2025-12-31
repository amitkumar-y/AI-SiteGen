"""
Data Loader Module
Loads and validates the design component knowledge base.
"""

import pandas as pd
import os
from typing import List, Dict, Any


def load_knowledge_base(filepath: str) -> pd.DataFrame:
    """
    Load and validate the knowledge base from CSV.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame containing validated design components
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Knowledge base file not found at {filepath}")

    df = pd.read_csv(filepath)
    required_cols = ['ID', 'Feature', 'Keywords', 'Tone', 'HTML_Snippet']

    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV missing required columns. Expected: {required_cols}")

    return df


def dataframe_to_components(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to list of component dictionaries."""
    return [
        {
            "id": row.get("ID", ""),
            "feature": row.get("Feature", ""),
            "keywords": row.get("Keywords", ""),
            "tone": row.get("Tone", ""),
            "html": row.get("HTML_Snippet", ""),
        }
        for _, row in df.iterrows()
    ]
