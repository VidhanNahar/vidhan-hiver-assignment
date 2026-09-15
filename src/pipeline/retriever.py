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

from src.config import (
    CHROMA_DB_PATH,
    PROCESSED_DATA_PATH,
    SAMPLE_FIXTURE_PATH,
    GOLDEN_SET_PATH,
    RETRIEVER_TOP_K,
    RETRIEVER_INDEX_SIZE,
)


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
        golden_set_path: str = None,
        top_k: int = None,
        index_size: int = None,
    ):
        self.chroma_path = chroma_path or CHROMA_DB_PATH
        self.data_path = data_path or PROCESSED_DATA_PATH
        self.golden_set_path = golden_set_path or GOLDEN_SET_PATH
        self.top_k = top_k or RETRIEVER_TOP_K
        self.index_size = index_size or RETRIEVER_INDEX_SIZE

        # Load embedding model
        print(f"[Retriever] Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Init ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self._get_or_create_collection()

    def _resolve_data_path(self) -> Path:
        """Resolve dataset path, falling back to bundled sample fixture if processed data is missing."""
        path = Path(self.data_path)
        if path.exists():
            return path
        fixture_path = Path(SAMPLE_FIXTURE_PATH)
        if fixture_path.exists():
            print(f"[Retriever] Notice: {path} not found. Falling back to bundled fixture at {fixture_path} for retrieval index.")
            return fixture_path
        raise FileNotFoundError(
            f"Neither processed data ({path}) nor sample fixture ({fixture_path}) was found. "
            "Please run 'make preprocess' or provide a dataset fixture."
        )

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or build it from processed data."""
        meta_file = Path(self.chroma_path) / ".index_meta.json"
        actual_path = self._resolve_data_path()

        collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Check if already populated and verified complete
        if collection.count() > 0 and meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("status") == "complete" and meta.get("count") == collection.count():
                    print(f"[Retriever] Verified complete collection with {collection.count():,} documents.")
                    return collection
            except Exception:
                pass
            print("[Retriever] Detected incomplete or stale index. Re-indexing...")
            try:
                self.client.delete_collection(name=COLLECTION_NAME)
            except Exception:
                pass
            collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        # Build the collection from processed data
        print(f"[Retriever] Building collection from {actual_path}...")
        self._index_threads(collection, actual_path)

        # Persist completion marker
        try:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump({
                    "status": "complete",
                    "count": collection.count(),
                    "data_source": str(actual_path),
                }, f)
        except Exception as e:
            print(f"[Retriever] Warning: Could not save index metadata: {e}")

        return collection

    def _index_threads(self, collection: chromadb.Collection, data_path: Path):
        """Index processed threads into ChromaDB while strictly excluding golden evaluation threads."""
        with open(data_path, "r", encoding="utf-8") as f:
            threads = json.load(f)

        # Disjoint evaluation split: Exclude golden set threads to prevent data leakage
        golden_ids = set()
        golden_path = Path(self.golden_set_path)
        if golden_path.exists():
            try:
                with open(golden_path, "r", encoding="utf-8") as gf:
                    golden_data = json.load(gf)
                golden_ids = {g["thread_id"] for g in golden_data if "thread_id" in g}
            except Exception as e:
                print(f"[Retriever] Warning: Could not read golden set to filter exclusions: {e}")

        initial_len = len(threads)
        threads = [t for t in threads if t.get("thread_id") not in golden_ids]
        excluded = initial_len - len(threads)
        if excluded > 0:
            print(f"[Retriever] Excluded {excluded} golden evaluation threads to prevent test-set data leakage.")

        # Subsample for indexing (e.g. 10K sample for retrieval quality vs speed)
        import random
        random.seed(42)
        if len(threads) > self.index_size:
            threads = random.sample(threads, self.index_size)
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
