"""
FAISS Vector Index Management

Provides GPU-accelerated vector similarity search for moral framework
context retrieval.
"""

import logging
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# FAISS import with GPU fallback
try:
    import faiss
    FAISS_AVAILABLE = True
    try:
        # Check for GPU support
        faiss.get_num_gpus()
        FAISS_GPU = faiss.get_num_gpus() > 0
    except:
        FAISS_GPU = False
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU = False
    logger.warning("FAISS not available, using numpy fallback")


@dataclass
class SearchResult:
    """Result from vector similarity search."""

    indices: NDArray[np.int64]
    distances: NDArray[np.float32]
    texts: Optional[list[str]] = None


class FAISSIndex:
    """
    GPU-accelerated FAISS index for efficient similarity search.

    Supports multiple index types optimized for different use cases:
    - Flat: Exact search, best for small datasets
    - IVF: Approximate search with inverted file index
    - HNSW: Hierarchical navigable small world graphs
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        use_gpu: bool = True,
        nlist: int = 100,  # For IVF index
        m: int = 32,  # For HNSW index
    ):
        """
        Initialize FAISS index.

        Args:
            dimension: Vector dimension
            index_type: Type of index ('flat', 'ivf', 'hnsw')
            use_gpu: Whether to use GPU if available
            nlist: Number of clusters for IVF index
            m: Number of connections per layer for HNSW
        """
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu and FAISS_GPU
        self.nlist = nlist
        self.m = m

        self._index: Optional["faiss.Index"] = None
        self._texts: list[str] = []
        self._is_trained = False

        if FAISS_AVAILABLE:
            self._create_index()
        else:
            self._vectors: Optional[NDArray[np.float32]] = None

        logger.info(
            f"FAISSIndex initialized: type={index_type}, "
            f"dim={dimension}, gpu={self.use_gpu}"
        )

    def _create_index(self) -> None:
        """Create the FAISS index based on configuration."""
        if self.index_type == "flat":
            # Exact L2 search
            index = faiss.IndexFlatL2(self.dimension)
            self._is_trained = True

        elif self.index_type == "ivf":
            # IVF with flat quantizer
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                self.nlist,
                faiss.METRIC_L2,
            )

        elif self.index_type == "hnsw":
            # HNSW index
            index = faiss.IndexHNSWFlat(self.dimension, self.m)
            self._is_trained = True

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Move to GPU if available
        if self.use_gpu and FAISS_GPU:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("Index moved to GPU")

        self._index = index

    def add(
        self,
        vectors: NDArray[np.float32],
        texts: Optional[list[str]] = None,
    ) -> None:
        """
        Add vectors to the index.

        Args:
            vectors: Array of shape (n, dimension)
            texts: Optional associated text for each vector
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} != index dimension {self.dimension}"
            )

        if FAISS_AVAILABLE:
            # Train if needed
            if not self._is_trained:
                logger.info(f"Training index on {len(vectors)} vectors")
                self._index.train(vectors)
                self._is_trained = True

            self._index.add(vectors)
        else:
            # Numpy fallback
            if self._vectors is None:
                self._vectors = vectors
            else:
                self._vectors = np.vstack([self._vectors, vectors])

        # Store texts
        if texts:
            self._texts.extend(texts)

        logger.debug(f"Added {len(vectors)} vectors to index")

    def search(
        self,
        queries: NDArray[np.float32],
        k: int = 5,
        nprobe: int = 10,  # For IVF
    ) -> SearchResult:
        """
        Search for nearest neighbors.

        Args:
            queries: Query vectors of shape (n, dimension)
            k: Number of neighbors to return
            nprobe: Number of clusters to search (IVF only)

        Returns:
            SearchResult with indices and distances
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)

        if FAISS_AVAILABLE:
            # Set IVF search parameters
            if self.index_type == "ivf":
                self._index.nprobe = nprobe

            distances, indices = self._index.search(queries, k)
        else:
            # Numpy fallback: brute force L2
            if self._vectors is None:
                return SearchResult(
                    indices=np.array([[-1] * k] * len(queries)),
                    distances=np.array([[float('inf')] * k] * len(queries)),
                )

            # Compute L2 distances
            diff = self._vectors[np.newaxis, :, :] - queries[:, np.newaxis, :]
            distances = np.sum(diff ** 2, axis=2)

            # Get top-k
            indices = np.argsort(distances, axis=1)[:, :k]
            distances = np.take_along_axis(distances, indices, axis=1)

        # Retrieve texts if available
        texts = None
        if self._texts:
            texts = []
            for idx_row in indices:
                row_texts = []
                for idx in idx_row:
                    if 0 <= idx < len(self._texts):
                        row_texts.append(self._texts[idx])
                    else:
                        row_texts.append("")
                texts.append(row_texts)

        return SearchResult(
            indices=indices,
            distances=distances,
            texts=texts,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if FAISS_AVAILABLE:
            # Move to CPU for saving if on GPU
            if self.use_gpu and FAISS_GPU:
                cpu_index = faiss.index_gpu_to_cpu(self._index)
                faiss.write_index(cpu_index, str(path.with_suffix(".faiss")))
            else:
                faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        else:
            if self._vectors is not None:
                np.save(path.with_suffix(".npy"), self._vectors)

        # Save texts separately
        if self._texts:
            import json
            with open(path.with_suffix(".texts.json"), "w") as f:
                json.dump(self._texts, f)

        logger.info(f"Index saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load index from disk."""
        path = Path(path)

        if FAISS_AVAILABLE:
            faiss_path = path.with_suffix(".faiss")
            if faiss_path.exists():
                self._index = faiss.read_index(str(faiss_path))
                self._is_trained = True

                # Move to GPU if needed
                if self.use_gpu and FAISS_GPU:
                    res = faiss.StandardGpuResources()
                    self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
        else:
            npy_path = path.with_suffix(".npy")
            if npy_path.exists():
                self._vectors = np.load(npy_path)

        # Load texts
        texts_path = path.with_suffix(".texts.json")
        if texts_path.exists():
            import json
            with open(texts_path) as f:
                self._texts = json.load(f)

        logger.info(f"Index loaded from {path}")

    @property
    def ntotal(self) -> int:
        """Total number of vectors in the index."""
        if FAISS_AVAILABLE and self._index is not None:
            return self._index.ntotal
        elif self._vectors is not None:
            return len(self._vectors)
        return 0

    def reset(self) -> None:
        """Clear all vectors from the index."""
        if FAISS_AVAILABLE:
            self._create_index()
        else:
            self._vectors = None
        self._texts = []
        logger.info("Index reset")
