"""
Data Loader Module
-----------------
This module is responsible for loading and validating the design component knowledge base.
It handles all data ingestion tasks, including:
- Loading the design components from a CSV file
- Validating the file structure and required columns
- Providing clean data to other parts of the application
- Converting DataFrame to list of dictionaries for vector store

Key Functions:
- load_knowledge_base(): Loads and validates the component database
- dataframe_to_components(): Converts DataFrame to component dictionaries
"""

import pandas as pd
import os
from typing import List, Dict, Any


def load_knowledge_base(filepath: str) -> pd.DataFrame:
    """
    Load and validate the knowledge base from a CSV file.
    
    This function:
    1. Checks if the file exists at the given path
    2. Loads the CSV data into a pandas DataFrame
    3. Validates that all required columns are present
    4. Returns the cleaned DataFrame
    
    Args:
        filepath (str): Path to the CSV file containing design components
        
    Returns:
        pd.DataFrame: DataFrame containing the validated design components
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If required columns are missing from the CSV
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Knowledge base file not found at {filepath}")
        
        df = pd.read_csv(filepath)
        required_cols = ['ID', 'Feature', 'Keywords', 'Tone', 'HTML_Snippet']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV missing required columns. Expected: {required_cols}")
        
        return df
    
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        raise


def dataframe_to_components(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert DataFrame to a list of component dictionaries.

    This function transforms the DataFrame into a format suitable for
    the vector store and retrieval system.

    Args:
        df (pd.DataFrame): DataFrame containing design components

    Returns:
        List[Dict[str, Any]]: List of component dictionaries with standardized keys
    """
    components = []

    for _, row in df.iterrows():
        component = {
            "id": row.get("ID", ""),
            "feature": row.get("Feature", ""),
            "keywords": row.get("Keywords", ""),
            "tone": row.get("Tone", ""),
            "html": row.get("HTML_Snippet", ""),
        }
        components.append(component)

    return components