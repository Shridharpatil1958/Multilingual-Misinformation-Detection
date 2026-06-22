# src/retrieval/bm25_retriever.py

"""
BM25 sparse retriever over a multilingual corpus.
Uses rank_bm25 with language-aware tokenization.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from src.utils.logger import get_logger

logger = get_logger(__name__)


def simple_tokenize(text: str) -> List[str]:
    """Whitespace tokenizer — works for all scripts."""
    return text.lower().split()


class BM25Retriever:
    """
    BM25 retriever with:
    - Lazy corpus loading
    - Serializable index (pickle)
    - Per-language sub-indices (optional)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []

    def build(self, corpus: List[Dict], text_field: str = "text"):
        """
        corpus: list of dicts with at least {text_field, chunk_id, doc_id}
        """
        logger.info(f"Building BM25 index over {len(corpus)} passages...")
        self.corpus = corpus
        self.tokenized_corpus = [
            simple_tokenize(doc[text_field]) for doc in corpus
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        logger.info("BM25 index built.")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Returns top_k passages with BM25 score."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokenized_query = simple_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    **self.corpus[idx],
                    "bm25_score": float(scores[idx]),
                    "rank": len(results) + 1,
                })

        return results

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"bm25": self.bm25, "corpus": self.corpus},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info(f"BM25 index saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BM25Retriever":
        with open(path, "rb") as f:
            data = pickle.load(f)
        retriever = cls()
        retriever.bm25 = data["bm25"]
        retriever.corpus = data["corpus"]
        logger.info(f"BM25 index loaded from {path} ({len(retriever.corpus)} passages)")
        return retriever


class MultilingualBM25:
    """
    Maintains separate BM25 indices per language for better retrieval.
    Falls back to combined index if language-specific index is missing.
    """

    def __init__(self):
        self.indices: Dict[str, BM25Retriever] = {}
        self.combined: Optional[BM25Retriever] = None

    def build(self, corpus: List[Dict], lang_field: str = "language"):
        # Combined index
        self.combined = BM25Retriever()
        self.combined.build(corpus)

        # Per-language indices
        lang_groups: Dict[str, List[Dict]] = {}
        for doc in corpus:
            lang = doc.get(lang_field, "en")
            lang_groups.setdefault(lang, []).append(doc)

        for lang, docs in lang_groups.items():
            logger.info(f"Building BM25 for language '{lang}': {len(docs)} passages")
            retriever = BM25Retriever()
            retriever.build(docs)
            self.indices[lang] = retriever

    def retrieve(
        self, query: str, query_lang: str = "en", top_k: int = 10
    ) -> List[Dict]:
        # Use language-specific index if available
        if query_lang in self.indices:
            return self.indices[query_lang].retrieve(query, top_k)
        elif self.combined:
            return self.combined.retrieve(query, top_k)
        return []
