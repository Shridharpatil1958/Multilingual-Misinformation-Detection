# scripts/build_index.py

"""
Build BM25 + FAISS retrieval indices from Wikipedia + news corpus.
Run: python scripts/build_index.py --config configs/retrieval_config.yaml

Steps:
  1. Load corpus JSONL files
  2. Chunk documents
  3. Build BM25 index
  4. Build dense FAISS index
"""

import argparse
import json
import yaml
from pathlib import Path
from tqdm import tqdm

from src.data.preprocessor import CorpusChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_corpus(sources: list) -> list:
    """Load and merge all corpus JSONL files."""
    documents = []
    for source in sources:
        path = Path(source["path"])
        if not path.exists():
            logger.warning(f"Corpus file not found: {path}. Skipping.")
            continue
        logger.info(f"Loading corpus: {source['name']} from {path}")
        with open(path, encoding="utf-8") as f:
            for line in tqdm(f, desc=source["name"]):
                doc = json.loads(line)
                doc["source"] = source["name"]
                documents.append(doc)
        logger.info(f"Loaded {len(documents)} total documents so far")
    return documents


def main(args):
    config = load_config(args.config)
    model_config = load_config("configs/model_config.yaml")

    # 1. Load corpus
    corpus_raw = load_corpus(config["corpus"]["sources"])
    if not corpus_raw:
        logger.error("No corpus documents found. Create data/processed/*.jsonl files first.")
        return

    # 2. Chunk documents
    chunker = CorpusChunker(
        chunk_size=config["corpus"]["chunk_size"],
        overlap=config["corpus"]["chunk_overlap"],
    )
    chunks = chunker.chunk_corpus(corpus_raw)
    logger.info(f"Total chunks: {len(chunks)}")

    # 3. Build BM25 index
    bm25_path = model_config["retriever"]["index_path"].replace(".faiss", "_bm25.pkl")
    if not args.skip_bm25:
        logger.info("Building BM25 index...")
        bm25 = BM25Retriever(k1=config["bm25"]["k1"], b=config["bm25"]["b"])
        bm25.build(chunks)
        bm25.save(bm25_path)
        logger.info(f"BM25 index saved: {bm25_path}")

    # 4. Build dense FAISS index
    index_path = model_config["retriever"]["index_path"]
    corpus_path = model_config["retriever"]["corpus_path"]
    if not args.skip_dense:
        logger.info("Building dense FAISS index (this may take a while)...")
        dense = DenseRetriever(
            model_name=config["dense"]["model"],
            nlist=config["index"]["nlist"],
            nprobe=config["index"]["nprobe"],
        )
        dense.build_index(chunks, batch_size=config["dense"]["batch_size"])
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        dense.save(index_path, corpus_path)
        logger.info(f"Dense index saved: {index_path}")

    logger.info("✅ Index building complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retrieval_config.yaml")
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()
    main(args)
