"""
Component Retriever Module
Handles retrieval using vector search (primary) and fuzzy matching (fallback).
"""

from rapidfuzz import process, fuzz
from typing import List, Dict, Optional
import pandas as pd
from vector_store import VectorStore
from data_loader import dataframe_to_components


class ComponentRetriever:
    """Retrieves relevant components using vector search or fuzzy matching."""

    def __init__(self, df: pd.DataFrame, use_vector_search: bool = True):
        """Initialize the retriever."""
        self.df = df
        self.use_vector_search = use_vector_search
        self.vector_store: Optional[VectorStore] = None

        # Initialize vector store
        if use_vector_search:
            try:
                components = dataframe_to_components(df)
                self.vector_store = VectorStore()
                self.vector_store.add_components(components, force_reload=False)
                print("[OK] Vector search enabled!")
            except Exception as e:
                print(f"[WARN] Vector store failed: {e}. Using fuzzy matching...")
                self.use_vector_search = False

        # Prepare fuzzy matching targets
        self.search_targets = (
            df["Keywords"].astype(str) + " " +
            df["Feature"].astype(str) + " " +
            df["Tone"].astype(str)
        ).tolist()

    def get_relevant_components(self, query: str, top_n: int = 5) -> List[Dict]:
        """Retrieve top N relevant components."""
        if self.use_vector_search and self.vector_store:
            try:
                results = self._vector_search(query, top_n)
                if results:
                    return results
            except Exception as e:
                print(f"[WARN] Vector search failed: {e}. Falling back...")

        return self._fuzzy_search(query, top_n)

    def _vector_search(self, query: str, top_n: int) -> List[Dict]:
        """Perform vector-based semantic search."""
        vector_results = self.vector_store.search(query, top_n=top_n)

        enriched_results = []
        for result in vector_results:
            matching_row = self.df[self.df["ID"] == result["id"]]
            if not matching_row.empty:
                enriched_results.append({
                    "id": result["id"],
                    "feature": result["feature"],
                    "keywords": result["keywords"],
                    "tone": result["tone"],
                    "score": result.get("score", 0),
                    "html": matching_row.iloc[0]["HTML_Snippet"],
                })

        return enriched_results

    def _fuzzy_search(self, query: str, top_n: int) -> List[Dict]:
        """Perform fuzzy string matching search."""
        matches = process.extract(
            query=query,
            choices=self.search_targets,
            scorer=fuzz.token_set_ratio,
            limit=top_n,
        )

        return [
            {
                "id": self.df.iloc[idx]["ID"],
                "feature": self.df.iloc[idx]["Feature"],
                "keywords": self.df.iloc[idx]["Keywords"],
                "tone": self.df.iloc[idx]["Tone"],
                "score": score,
                "html": self.df.iloc[idx]["HTML_Snippet"],
            }
            for _, score, idx in matches
        ]
