"""
Component Retriever Module
--------------------------
Handles the retrieval of relevant components using both:
1. Vector-based semantic search (primary method using ChromaDB)
2. Fuzzy matching (fallback method using rapidfuzz)

The vector search provides better semantic understanding of queries.
"""

from rapidfuzz import process, fuzz
from typing import List, Dict, Optional
import pandas as pd
from vector_store import VectorStore
from data_loader import dataframe_to_components


class ComponentRetriever:
    """
    Handles the retrieval of relevant components based on user queries.
    Uses vector-based semantic search as primary method with fuzzy matching as fallback.
    """

    def __init__(self, df: pd.DataFrame, use_vector_search: bool = True):
        """
        Initialize the retriever.

        Args:
            df: DataFrame containing component data
            use_vector_search: Whether to use vector search (default: True)
        """
        self.df = df
        self.use_vector_search = use_vector_search
        self.vector_store: Optional[VectorStore] = None

        # Initialize vector store if enabled
        if self.use_vector_search:
            try:
                # Convert DataFrame to components
                components = dataframe_to_components(df)

                # Initialize vector store
                self.vector_store = VectorStore()

                # Add components (will skip if already loaded)
                self.vector_store.add_components(components, force_reload=False)

                print("[OK] Vector search enabled and ready!")

            except Exception as e:
                print(f"[WARN] Vector store initialization failed: {e}")
                print("       Falling back to fuzzy matching...")
                self.use_vector_search = False
                self.vector_store = None

        # Prepare fuzzy matching targets (used as fallback or primary if vector disabled)
        self.search_targets = (
            df["Keywords"].astype(str)
            + " "
            + df["Feature"].astype(str)
            + " "
            + df["Tone"].astype(str)
        ).tolist()

    def get_relevant_components(self, query: str, top_n: int = 5) -> List[Dict]:
        """
        Retrieve top N relevant components using vector search or fuzzy matching.

        Args:
            query: User's search query
            top_n: Number of top results to return

        Returns:
            List of dictionaries containing component information
        """
        # Try vector search first if enabled
        if self.use_vector_search and self.vector_store:
            try:
                results = self._vector_search(query, top_n)
                if results:
                    return results
            except Exception as e:
                print(f"[WARN] Vector search failed: {e}. Falling back to fuzzy matching...")

        # Fallback to fuzzy matching
        return self._fuzzy_search(query, top_n)

    def _vector_search(self, query: str, top_n: int) -> List[Dict]:
        """
        Perform vector-based semantic search.

        Args:
            query: User's search query
            top_n: Number of results to return

        Returns:
            List of component dictionaries with HTML from DataFrame
        """
        # Get vector search results
        vector_results = self.vector_store.search(query, top_n=top_n)

        # Enrich with HTML from DataFrame
        enriched_results = []
        for result in vector_results:
            component_id = result["id"]

            # Find matching row in DataFrame
            matching_row = self.df[self.df["ID"] == component_id]

            if not matching_row.empty:
                enriched_result = {
                    "id": result["id"],
                    "feature": result["feature"],
                    "keywords": result["keywords"],
                    "tone": result["tone"],
                    "score": result.get("score", 0),
                    "html": matching_row.iloc[0]["HTML_Snippet"],
                }
                enriched_results.append(enriched_result)

        return enriched_results

    def _fuzzy_search(self, query: str, top_n: int) -> List[Dict]:
        """
        Perform fuzzy string matching search (fallback method).

        Args:
            query: User's search query
            top_n: Number of results to return

        Returns:
            List of component dictionaries
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
