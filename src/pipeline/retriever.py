"""
RAG Retriever — Semantic search over historical AmazonHelp responses.

Embeds customer messages using sentence-transformers (local) and
retrieves the most similar historical customer→brand pairs from ChromaDB.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DB_PATH, PROCESSED_DATA_PATH, RETRIEVER_TOP_K


@dataclass
class RetrievedThread:
    customer_text: str
    brand_reply: str
    similarity_score: float
    thread_id: str


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "amazon_threads"


class Retriever:
    """Semantic retriever using sentence-transformers + ChromaDB."""

    def __init__(
        self,
        chroma_path: str = None,
        data_path: str = None,
        top_k: int = None,
    ):
        self.chroma_path = chroma_path or CHROMA_DB_PATH
        self.data_path = data_path or PROCESSED_DATA_PATH
        self.top_k = top_k or RETRIEVER_TOP_K

        # Load embedding model
        print(f"[Retriever] Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Init ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or build it from processed data."""
        collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Check if already populated
        if collection.count() > 0:
            print(f"[Retriever] Collection exists with {collection.count():,} documents.")
            return collection

        # Build the collection from processed data
        print(f"[Retriever] Building collection from {self.data_path}...")
        self._index_threads(collection)
        return collection

    def _index_threads(self, collection: chromadb.Collection):
        """Index all processed threads into ChromaDB."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            threads = json.load(f)

        # Use a subsample for indexing (full 151K would be slow)
        # We keep a diverse 10K sample for retrieval quality vs speed
        import random
        random.seed(42)
        if len(threads) > 10000:
            threads = random.sample(threads, 10000)
            print(f"[Retriever] Subsampled to {len(threads):,} threads for indexing.")

        batch_size = 500
        for i in range(0, len(threads), batch_size):
            batch = threads[i : i + batch_size]

            texts = [t["customer_text"] for t in batch]
            embeddings = self.embed_model.encode(texts, show_progress_bar=False).tolist()
            ids = [t["thread_id"] for t in batch]
            metadatas = [
                {
                    "customer_text": t["customer_text"][:500],
                    "brand_reply": t["brand_reply"][:500],
                }
                for t in batch
            ]

            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )

            if (i // batch_size) % 4 == 0:
                print(f"  Indexed {min(i + batch_size, len(threads)):,}/{len(threads):,}...")

        print(f"[Retriever] Indexed {collection.count():,} documents total.")

    def retrieve(
        self,
        customer_text: str,
        top_k: int = None,
        intent: str = None,
    ) -> list[RetrievedThread]:
        """Retrieve the most similar historical threads for a customer message.

        Args:
            customer_text: The incoming customer message.
            top_k: Number of results to return (default: self.top_k).
            intent: Optional intent filter (not used currently, reserved for future).

        Returns:
            List of RetrievedThread objects sorted by similarity.
        """
        k = top_k or self.top_k

        # Embed the query
        query_embedding = self.embed_model.encode([customer_text]).tolist()

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["metadatas", "distances"],
        )

        retrieved = []
        if results and results["metadatas"]:
            for metadata, distance, doc_id in zip(
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],
            ):
                # ChromaDB cosine distance → similarity = 1 - distance
                similarity = 1.0 - distance

                retrieved.append(RetrievedThread(
                    customer_text=metadata.get("customer_text", ""),
                    brand_reply=metadata.get("brand_reply", ""),
                    similarity_score=similarity,
                    thread_id=doc_id,
                ))

        return retrieved
