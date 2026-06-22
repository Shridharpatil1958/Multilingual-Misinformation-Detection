# src/retrieval/hybrid_retriever.py

"""
Hybrid retriever: BM25 (sparse) + Dense FAISS (dense) fused with
Reciprocal Rank Fusion (RRF).

RRF formula: score(d) = Σ 1 / (k + rank_i(d))
where k=60 dampens the effect of very high ranks.
"""

from typing import List, Dict, Optional
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    id_field: str = "chunk_id",
    k: int = 60,
) -> List[Dict]:
    """
    Merge multiple ranked lists using RRF.
    Higher RRF score = better combined rank.
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc[id_field]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = {k: v for k, v in doc.items()
                                   if k not in ("rank", "bm25_score", "dense_score")}

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    results = []
    for rank, doc_id in enumerate(sorted_ids, start=1):
        doc = doc_map[doc_id]
        doc["rrf_score"] = rrf_scores[doc_id]
        doc["rank"] = rank
        results.append(doc)

    return results


class HybridRetriever:
    """
    Two-stage retriever:
      Stage 1: BM25 + Dense retrieval (top-k each)
      Stage 2: RRF fusion → reranked list
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        final_top_k: int = 10,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.final_top_k = final_top_k

    def retrieve(
        self,
        query: str,
        query_lang: str = "en",
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve and fuse results from BM25 and dense retriever.

        Returns:
            List of passages sorted by RRF score, each with:
            - text, doc_id, chunk_id, rrf_score, rank
        """
        top_k = top_k or self.final_top_k

        # Stage 1: Retrieve from both systems
        bm25_results = self.bm25.retrieve(query, top_k=self.bm25_top_k)
        dense_results = self.dense.retrieve(query, top_k=self.dense_top_k)

        if not bm25_results and not dense_results:
            logger.warning(f"No results for query: {query[:80]}")
            return []

        # Stage 2: RRF fusion
        fused = reciprocal_rank_fusion(
            [bm25_results, dense_results],
            k=self.rrf_k,
        )

        return fused[:top_k]

    def retrieve_with_scores(self, query: str, top_k: int = 10) -> List[Dict]:
        """Same as retrieve() but includes raw BM25 and dense scores."""
        results = self.retrieve(query, top_k=top_k * 2)

        # Enrich with original scores
        bm25_map = {r["chunk_id"]: r.get("bm25_score", 0.0)
                    for r in self.bm25.retrieve(query, top_k=top_k * 2)}
        dense_map = {r["chunk_id"]: r.get("dense_score", 0.0)
                     for r in self.dense.retrieve(query, top_k=top_k * 2)}

        for doc in results:
            cid = doc.get("chunk_id", "")
            doc["bm25_score"] = bm25_map.get(cid, 0.0)
            doc["dense_score"] = dense_map.get(cid, 0.0)

        return results[:top_k]


class PipelineRetriever:
    """
    Full retrieval pipeline:
    1. Detect query language
    2. Expand Hinglish if needed
    3. Retrieve with hybrid retriever
    4. Return top passages with metadata
    """

    def __init__(self, hybrid: HybridRetriever, top_k: int = 5):
        from src.data.preprocessor import TextPreprocessor
        from src.data.hinglish_handler import HinglishPipeline

        self.hybrid = hybrid
        self.top_k = top_k
        self.preprocessor = TextPreprocessor()
        self.hinglish = HinglishPipeline()

    def retrieve(self, claim: str) -> Dict:
        """
        Full pipeline: clean → detect language → retrieve evidence.

        Returns:
            {
              "query": str,
              "language": str,
              "is_hinglish": bool,
              "passages": List[Dict],
              "retrieval_scores": List[float],
            }
        """
        # 1. Clean
        clean_claim = self.preprocessor.clean(claim)

        # 2. Language detection + Hinglish handling
        hinglish_result = self.hinglish.process(clean_claim)
        query = hinglish_result["normalized"] if hinglish_result["is_hinglish"] else clean_claim
        lang = hinglish_result["dominant_language"]

        # 3. Retrieve
        passages = self.hybrid.retrieve_with_scores(query, top_k=self.top_k)

        return {
            "query": clean_claim,
            "language": lang,
            "is_hinglish": hinglish_result["is_hinglish"],
            "passages": passages,
            "retrieval_scores": [p.get("rrf_score", 0.0) for p in passages],
        }
