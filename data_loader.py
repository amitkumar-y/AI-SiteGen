"""
Data Loader Module
-----------------
This module is responsible for loading and validating the design component knowledge base.
It handles all data ingestion tasks, including:
- Loading the design components from a CSV file
- Validating the file structure and required columns
- Providing clean data to other parts of the application

Key Functions:
- load_knowledge_base(): Loads and validates the component database
"""

import pandas as pd  # Fixed typo: changed 'panda' to 'pandas'
import os


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