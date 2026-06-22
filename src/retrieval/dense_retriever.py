# src/retrieval/dense_retriever.py

"""
Dense retriever using multilingual sentence embeddings + FAISS index.
Model: paraphrase-multilingual-MiniLM-L12-v2 (supports 50+ languages).
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """
    FAISS-backed dense retriever with multilingual sentence encoder.

    Architecture:
        Query → SentenceTransformer → L2-normalized embedding
               → FAISS IVF flat index → top-k nearest neighbors
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        embedding_dim: int = 384,
        nlist: int = 256,
        nprobe: int = 32,
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.nprobe = nprobe

        self.encoder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.corpus: List[Dict] = []

    def _load_encoder(self):
        if self.encoder is None:
            logger.info(f"Loading sentence encoder: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        self._load_encoder()
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
        )
        return embeddings.astype(np.float32)

    def build_index(self, corpus: List[Dict], text_field: str = "text", batch_size: int = 64):
        """
        Build FAISS IVF index from corpus.
        Uses IVFFlat for large corpora, Flat for small (<10K passages).
        """
        logger.info(f"Building dense index over {len(corpus)} passages...")
        self.corpus = corpus
        texts = [doc[text_field] for doc in corpus]

        embeddings = self.encode(texts, batch_size=batch_size)

        if len(corpus) < 10_000:
            # Small corpus: use exact flat index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info("Using Flat index (small corpus)")
        else:
            # Large corpus: use IVF for speed
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            logger.info(f"Training IVF index with nlist={self.nlist}...")
            self.index.train(embeddings)
            self.index.nprobe = self.nprobe

        self.index.add(embeddings)
        logger.info(f"Dense index built: {self.index.ntotal} vectors")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Returns top-k passages with cosine similarity scores."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_emb = self.encode([query])
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            results.append({
                **self.corpus[idx],
                "dense_score": float(score),
                "rank": rank + 1,
            })
        return results

    def batch_retrieve(
        self, queries: List[str], top_k: int = 10
    ) -> List[List[Dict]]:
        """Retrieve for multiple queries in one FAISS call."""
        if self.index is None:
            raise RuntimeError("Index not built.")

        query_embs = self.encode(queries)
        all_scores, all_indices = self.index.search(query_embs, top_k)

        all_results = []
        for q_scores, q_indices in zip(all_scores, all_indices):
            results = []
            for rank, (score, idx) in enumerate(zip(q_scores, q_indices)):
                if idx == -1:
                    continue
                results.append({
                    **self.corpus[idx],
                    "dense_score": float(score),
                    "rank": rank + 1,
                })
            all_results.append(results)
        return all_results

    def save(self, index_path: str, corpus_path: str):
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(corpus_path, "w", encoding="utf-8") as f:
            for doc in self.corpus:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        logger.info(f"Dense index saved: {index_path}")

    def load(self, index_path: str, corpus_path: str):
        self.index = faiss.read_index(index_path)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

        self.corpus = []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                self.corpus.append(json.loads(line))

        logger.info(
            f"Dense index loaded: {self.index.ntotal} vectors, {len(self.corpus)} passages"
        )
