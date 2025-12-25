"""
Vector Store Module using ChromaDB
Provides semantic search capabilities for design components.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd


class VectorStore:
    """
    Manages vector embeddings and semantic search using ChromaDB.
    """

    def __init__(
        self,
        collection_name: str = "law_firm_designs",
        persist_directory: str = "./chroma_db",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model_name: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name

        # Initialize sentence transformer model (free, local)
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print(f"[OK] Embedding model loaded successfully!")

        # Initialize ChromaDB client
        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False,
            )
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"[OK] Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Law firm website design components"},
            )
            print(f"[OK] Created new collection: {collection_name}")

    def _create_searchable_text(self, component: Dict[str, Any]) -> str:
        """
        Create a searchable text representation of a component.

        Args:
            component: Component dictionary with keys like id, feature, keywords, tone

        Returns:
            Combined text string for embedding
        """
        parts = []

        # Add feature type
        if "feature" in component:
            parts.append(f"Type: {component['feature']}")

        # Add keywords
        if "keywords" in component:
            parts.append(f"Keywords: {component['keywords']}")

        # Add tone
        if "tone" in component:
            parts.append(f"Tone: {component['tone']}")

        # Add ID for context
        if "id" in component:
            parts.append(f"ID: {component['id']}")

        return " | ".join(parts)

    def add_components(self, components: List[Dict[str, Any]], force_reload: bool = False):
        """
        Add components to the vector store.

        Args:
            components: List of component dictionaries
            force_reload: If True, clear existing collection and reload
        """
        if force_reload:
            # Delete and recreate collection
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Law firm website design components"},
            )
            print(f"[RELOAD] Cleared and recreated collection: {self.collection_name}")

        # Check if collection already has items
        existing_count = self.collection.count()
        if existing_count > 0 and not force_reload:
            print(f"[INFO] Collection already has {existing_count} items. Skipping reload.")
            print(f"       Use force_reload=True to reload components.")
            return

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        print(f"Generating embeddings for {len(components)} components...")

        for component in components:
            # Create ID
            component_id = component.get("id", f"comp_{len(ids)}")
            ids.append(component_id)

            # Create searchable document
            doc_text = self._create_searchable_text(component)
            documents.append(doc_text)

            # Create metadata (store original data)
            metadata = {
                "id": component.get("id", ""),
                "feature": component.get("feature", ""),
                "keywords": component.get("keywords", ""),
                "tone": component.get("tone", ""),
                # Note: ChromaDB doesn't support storing large HTML in metadata well
                # We'll retrieve full details from original DataFrame if needed
            }
            metadatas.append(metadata)

            # Generate embedding
            embedding = self.embedding_model.encode(doc_text).tolist()
            embeddings.append(embedding)

        # Add to ChromaDB
        self.collection.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

        print(f"[OK] Added {len(components)} components to vector store!")

    def search(
        self, query: str, top_n: int = 6, include_distances: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar components using semantic search.

        Args:
            query: User's search query
            top_n: Number of results to return
            include_distances: Whether to include similarity distances

        Returns:
            List of matching components with metadata and scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_n, include=["metadatas", "documents", "distances"]
        )

        # Format results
        formatted_results = []

        if results and results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                result_dict = {
                    "id": results["metadatas"][0][i].get("id", ""),
                    "feature": results["metadatas"][0][i].get("feature", ""),
                    "keywords": results["metadatas"][0][i].get("keywords", ""),
                    "tone": results["metadatas"][0][i].get("tone", ""),
                    "document": results["documents"][0][i],
                }

                if include_distances and "distances" in results:
                    # Convert distance to similarity score (0-100)
                    distance = results["distances"][0][i]
                    # ChromaDB uses L2 distance, convert to similarity score
                    similarity = max(0, 100 - (distance * 10))
                    result_dict["score"] = round(similarity, 2)

                formatted_results.append(result_dict)

        return formatted_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with stats
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_components": count,
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
            "persist_directory": self.persist_directory,
        }


def create_vector_store(
    components: List[Dict[str, Any]] = None, force_reload: bool = False
) -> VectorStore:
    """
    Factory function to create and initialize a vector store.

    Args:
        components: Optional list of components to add
        force_reload: Whether to force reload the collection

    Returns:
        Initialized VectorStore instance
    """
    store = VectorStore()

    if components:
        store.add_components(components, force_reload=force_reload)

    return store


if __name__ == "__main__":
    # Test the vector store
    print("=" * 60)
    print("Testing Vector Store")
    print("=" * 60)

    # Sample components for testing
    test_components = [
        {
            "id": "H-01",
            "feature": "Hero",
            "keywords": "trust, estate planning, traditional",
            "tone": "Dark Blue, Conservative",
        },
        {
            "id": "H-02",
            "feature": "Hero",
            "keywords": "criminal defense, aggressive, urgent",
            "tone": "Black & Red, Bold",
        },
        {
            "id": "S-01",
            "feature": "Services",
            "keywords": "grid, minimalist, fast",
            "tone": "Clean, Modern",
        },
    ]

    # Create store
    store = create_vector_store(test_components, force_reload=True)

    # Get stats
    stats = store.get_collection_stats()
    print(f"\nVector Store Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test searches
    print(f"\n" + "=" * 60)
    print("Test Search: 'aggressive criminal defense website'")
    print("=" * 60)
    results = store.search("aggressive criminal defense website", top_n=3)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  ID: {result['id']}")
        print(f"  Feature: {result['feature']}")
        print(f"  Keywords: {result['keywords']}")
        print(f"  Score: {result.get('score', 'N/A')}")

    print(f"\n" + "=" * 60)
    print("Test Search: 'traditional estate planning firm'")
    print("=" * 60)
    results = store.search("traditional estate planning firm", top_n=3)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  ID: {result['id']}")
        print(f"  Feature: {result['feature']}")
        print(f"  Keywords: {result['keywords']}")
        print(f"  Score: {result.get('score', 'N/A')}")

    print(f"\n[OK] Vector store testing complete!")
