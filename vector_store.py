"""
Vector Store Module using ChromaDB
Provides semantic search capabilities for design components.
"""

from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Manages vector embeddings and semantic search using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "law_firm_designs",
        persist_directory: str = "./chroma_db",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        """Initialize the vector store."""
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name

        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("[OK] Embedding model loaded!")

        self.client = chromadb.Client(
            Settings(persist_directory=persist_directory, anonymized_telemetry=False)
        )

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
        """Create searchable text representation of a component."""
        parts = []
        if "feature" in component:
            parts.append(f"Type: {component['feature']}")
        if "keywords" in component:
            parts.append(f"Keywords: {component['keywords']}")
        if "tone" in component:
            parts.append(f"Tone: {component['tone']}")
        if "id" in component:
            parts.append(f"ID: {component['id']}")
        return " | ".join(parts)

    def add_components(self, components: List[Dict[str, Any]], force_reload: bool = False):
        """Add components to the vector store."""
        if force_reload:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Law firm website design components"},
            )
            print(f"[RELOAD] Cleared and recreated collection")

        existing_count = self.collection.count()
        if existing_count > 0 and not force_reload:
            print(f"[INFO] Collection has {existing_count} items. Skipping reload.")
            return

        ids, documents, metadatas, embeddings = [], [], [], []

        print(f"Generating embeddings for {len(components)} components...")

        for component in components:
            ids.append(component.get("id", f"comp_{len(ids)}"))
            documents.append(self._create_searchable_text(component))
            metadatas.append({
                "id": component.get("id", ""),
                "feature": component.get("feature", ""),
                "keywords": component.get("keywords", ""),
                "tone": component.get("tone", ""),
            })
            embeddings.append(self.embedding_model.encode(documents[-1]).tolist())

        self.collection.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )
        print(f"[OK] Added {len(components)} components to vector store!")

    def search(self, query: str, top_n: int = 6) -> List[Dict[str, Any]]:
        """Search for similar components using semantic search."""
        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            include=["metadatas", "documents", "distances"]
        )

        formatted_results = []
        if results and results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                similarity = max(0, 100 - (distance * 10))
                formatted_results.append({
                    "id": results["metadatas"][0][i].get("id", ""),
                    "feature": results["metadatas"][0][i].get("feature", ""),
                    "keywords": results["metadatas"][0][i].get("keywords", ""),
                    "tone": results["metadatas"][0][i].get("tone", ""),
                    "document": results["documents"][0][i],
                    "score": round(similarity, 2),
                })

        return formatted_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "collection_name": self.collection_name,
            "total_components": self.collection.count(),
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
        }
