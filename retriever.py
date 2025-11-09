"""
Component Retriever Module
--------------------------
Handles the retrieval of relevant components using fuzzy matching.
Uses rapidfuzz for efficient string matching.
"""

from rapidfuzz import process, fuzz
from typing import List, Dict
import pandas as pd


class ComponentRetriever:
    """
    Handles the retrieval of relevant components based on user queries.
    Uses fuzzy string matching to find the most relevant components.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Combine Keywords, Feature, and Tone for better matching
        self.search_targets = (
            df["Keywords"].astype(str)
            + " "
            + df["Feature"].astype(str)
            + " "
            + df["Tone"].astype(str)
        ).tolist()

    def get_relevant_components(self, query: str, top_n: int = 5) -> List[Dict]:
        """
        Retrieve top N relevant components using fuzzy matching.

        Args:
            query: User's search query
            top_n: Number of top results to return

        Returns:
            List of dictionaries containing component information
        """
        # Perform fuzzy matching
        matches = process.extract(
            query=query,
            choices=self.search_targets,
            scorer=fuzz.token_set_ratio,
            limit=top_n,
        )

        # Process and return matches
        results = []
        for _, score, idx in matches:
            results.append(
                {
                    "id": self.df.iloc[idx]["ID"],
                    "feature": self.df.iloc[idx]["Feature"],
                    "keywords": self.df.iloc[idx]["Keywords"],
                    "tone": self.df.iloc[idx]["Tone"],
                    "score": score,
                    "html": self.df.iloc[idx]["HTML_Snippet"],
                }
            )

        return results
