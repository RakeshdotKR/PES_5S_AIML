import os
import json
import pickle
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Abstract Base Class (the "protocol" / unified interface)
# ══════════════════════════════════════════════════════════════════════════════

class VectorStore(ABC):
    """
    Abstract base class defining the unified interface all three backends implement.
    
    By programming to this interface instead of to Chroma/FAISS/Pinecone directly,
    the rest of the codebase (ingest.py, rag_engine.py, agents) never needs to change
    when you swap backends. This is the Dependency Inversion principle in practice.
    """

    @abstractmethod
    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """Add or update documents. Implementations must be idempotent (upsert semantics)."""

    @abstractmethod
    def query(
        self,
        embedding: List[float],
        k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return top-k results for the given query embedding.
        Each result dict must contain: 'id', 'document', 'metadata', 'score'.
        score is cosine similarity (higher = more relevant).
        filter is a backend-specific metadata filter dict (may be ignored by FAISS).
        """

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Remove documents by ID."""

    @abstractmethod
    def persist(self) -> None:
        """Save state to disk (no-op for backends that persist automatically)."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored documents."""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FAISS Backend
# ══════════════════════════════════════════════════════════════════════════════

import faiss  # faiss-cpu package — pure Python import, no server needed


class FaissStore(VectorStore):
    """
    FAISS-backed VectorStore.
    
    What FAISS provides: blazing-fast vector similarity search.
    What FAISS does NOT provide: metadata, document text, persistence, or IDs.
    
    We compensate with:
    - A Python list (self._id_map) mapping FAISS's sequential integer index → our string ID
    - A Python dict (self._sidecar) mapping string ID → {document, metadata}
    - pickle for disk persistence of both sidecar structures
    - Post-query metadata filtering (filter applied AFTER retrieval — less efficient but necessary)
    """

    def __init__(self, dim: int = 768, index_path: str = "./faiss_index"):
        self.dim = dim
        self.index_path = index_path
        self._sidecar_path = index_path + "_sidecar.pkl"

        # IndexFlatIP: exact inner-product search.
        # For cosine similarity, vectors must be L2-normalised before adding.
        # After normalisation, inner product equals cosine similarity.
        self._index = faiss.IndexFlatIP(dim)

        # _id_map[faiss_int_idx] = our string ID — grows as we add documents.
        # FAISS assigns sequential integers 0, 1, 2, ... internally.
        self._id_map: List[str] = []

        # _sidecar maps our string ID → dict with 'document' and 'metadata'.
        self._sidecar: Dict[str, Dict] = {}

        # Load from disk if an existing index is found.
        self._load_if_exists()

    def _load_if_exists(self):
        """Restore index + sidecar from disk if they exist from a prior run."""
        if os.path.exists(self.index_path + ".index"):
            self._index = faiss.read_index(self.index_path + ".index")
            with open(self._sidecar_path, "rb") as f:
                saved = pickle.load(f)
                self._id_map = saved["id_map"]
                self._sidecar = saved["sidecar"]
            print(f"[FAISS] Loaded {self._index.ntotal} vectors from {self.index_path}")

    def _normalise(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalise rows so that inner product = cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)   # avoid division by zero for zero vectors
        return vectors / norms

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """
        Add documents. FAISS does not support true upserts — we simulate it by
        skipping IDs that already exist (idempotent but not update-aware).
        For a true update, delete first then re-add.
        """
        new_ids, new_docs, new_metas, new_embs = [], [], [], []
        for id_, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
            if id_ not in self._sidecar:   # skip existing IDs
                new_ids.append(id_)
                new_docs.append(doc)
                new_metas.append(meta)
                new_embs.append(emb)

        if not new_ids:
            return   # nothing to add

        vectors = np.array(new_embs, dtype=np.float32)
        vectors = self._normalise(vectors)
        self._index.add(vectors)    # FAISS assigns sequential int IDs internally

        for id_, doc, meta in zip(new_ids, new_docs, new_metas):
            self._id_map.append(id_)
            self._sidecar[id_] = {"document": doc, "metadata": meta}

    def query(
        self,
        embedding: List[float],
        k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for top-k nearest neighbours.
        filter is applied POST-retrieval (FAISS has no native metadata filtering).
        We over-fetch (k*10) to ensure enough results survive filtering.
        """
        if self._index.ntotal == 0:
            return []

        vec = np.array([embedding], dtype=np.float32)
        vec = self._normalise(vec)

        # Over-fetch to have headroom for post-filter elimination.
        fetch_k = min(k * 10, self._index.ntotal)
        scores, faiss_indices = self._index.search(vec, fetch_k)
        # scores[0] are inner-product scores (= cosine similarity after normalisation)
        # faiss_indices[0] are the internal sequential integer IDs

        results = []
        for score, idx in zip(scores[0], faiss_indices[0]):
            if idx == -1:               # FAISS returns -1 for "no result found"
                continue
            if idx >= len(self._id_map):
                continue                # guard against stale index

            str_id = self._id_map[idx]
            if str_id not in self._sidecar:
                continue

            meta = self._sidecar[str_id]["metadata"]

            # Apply metadata filter if provided (simple equality check)
            if filter:
                match = all(meta.get(key) == val for key, val in filter.items())
                if not match:
                    continue

            results.append({
                "id":       str_id,
                "document": self._sidecar[str_id]["document"],
                "metadata": meta,
                "score":    float(score),
            })

            if len(results) == k:
                break

        return results

    def delete(self, ids: List[str]) -> None:
        """
        Mark IDs as deleted in the sidecar — FAISS IndexFlatIP does not support
        in-place deletion. Deleted entries are excluded from future query results
        but remain in the FAISS index (memory leak for large churn workloads;
        rebuild the index periodically in production).
        """
        for id_ in ids:
            if id_ in self._sidecar:
                del self._sidecar[id_]

    def persist(self) -> None:
        """Save the FAISS index and sidecar structures to disk."""
        faiss.write_index(self._index, self.index_path + ".index")
        with open(self._sidecar_path, "wb") as f:
            pickle.dump({"id_map": self._id_map, "sidecar": self._sidecar}, f)
        print(f"[FAISS] Persisted {self._index.ntotal} vectors → {self.index_path}")

    def count(self) -> int:
        """Return number of live (non-deleted) documents."""
        return len(self._sidecar)