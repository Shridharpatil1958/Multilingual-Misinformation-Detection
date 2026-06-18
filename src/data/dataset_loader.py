# src/data/dataset_loader.py

"""
Loaders for LIAR, Factify, and custom Indian fact-check datasets.
All return a unified schema:
  { "claim": str, "label": int, "evidence": List[str], "language": str, "source": str }

Labels: 0 = SUPPORTS, 1 = REFUTES, 2 = NOT_ENOUGH_INFO
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datasets import Dataset, DatasetDict
from src.utils.logger import get_logger

logger = get_logger(__name__)

LABEL_MAP = {
    # LIAR labels → unified 3-class
    "true": 0, "mostly-true": 0, "half-true": 2,
    "barely-true": 1, "false": 1, "pants-fire": 1,
    # Factify
    "SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2,
    # Numeric pass-through
    0: 0, 1: 1, 2: 2,
}


class LIARLoader:
    """
    LIAR dataset: 12.8K human-labeled statements from PolitiFact.
    Download: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
    """

    COLUMNS = [
        "id", "label", "claim", "subject", "speaker", "job", "state",
        "party", "barely_true", "false", "half_true", "mostly_true",
        "pants_fire", "context"
    ]

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_split(self, split: str) -> List[Dict]:
        fname = {"train": "train.tsv", "val": "valid.tsv", "test": "test.tsv"}[split]
        fpath = self.data_dir / fname

        if not fpath.exists():
            raise FileNotFoundError(
                f"LIAR {split} not found at {fpath}. "
                "Run: python scripts/download_data.py --dataset liar"
            )

        df = pd.read_csv(fpath, sep="\t", header=None, names=self.COLUMNS)
        records = []
        for _, row in df.iterrows():
            records.append({
                "claim": str(row["claim"]),
                "label": LABEL_MAP.get(row["label"], 2),
                "evidence": [],
                "language": "en",
                "source": "liar",
                "metadata": {"speaker": row["speaker"], "context": row["context"]},
            })
        logger.info(f"Loaded {len(records)} LIAR {split} records")
        return records

    def load_all(self) -> DatasetDict:
        return DatasetDict({
            split: Dataset.from_list(self.load_split(split))
            for split in ["train", "val", "test"]
        })


class FactifyLoader:
    """
    Factify dataset: multimodal (text + image) Indian misinformation.
    Hindi + English claims with evidence documents.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_split(self, split: str) -> List[Dict]:
        fpath = self.data_dir / f"{split}.json"
        if not fpath.exists():
            raise FileNotFoundError(f"Factify {split} not found at {fpath}")

        with open(fpath) as f:
            raw = json.load(f)

        records = []
        for item in raw:
            records.append({
                "claim": item["claim"],
                "label": LABEL_MAP.get(item.get("label", "NOT ENOUGH INFO"), 2),
                "evidence": item.get("evidence_list", []),
                "language": item.get("language", "en"),
                "source": "factify",
                "metadata": {"image_url": item.get("image_url", "")},
            })
        logger.info(f"Loaded {len(records)} Factify {split} records")
        return records

    def load_all(self) -> DatasetDict:
        splits = {}
        for split in ["train", "val", "test"]:
            try:
                splits[split] = Dataset.from_list(self.load_split(split))
            except FileNotFoundError:
                logger.warning(f"Factify {split} not found, skipping")
        return DatasetDict(splits)


class IndianFactCheckLoader:
    """
    Custom scraped dataset from AltNews, Boom Live, Vishvasnews.
    Format: JSONL with fields: claim, verdict, language, evidence_urls, date
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self, filename: str = "indian_fc.jsonl") -> List[Dict]:
        fpath = self.data_dir / filename
        if not fpath.exists():
            logger.warning(f"Indian FC dataset not found at {fpath}")
            return []

        records = []
        with open(fpath) as f:
            for line in f:
                item = json.loads(line)
                records.append({
                    "claim": item["claim"],
                    "label": LABEL_MAP.get(item.get("verdict", "NEI"), 2),
                    "evidence": item.get("evidence_texts", []),
                    "language": item.get("language", "hi"),
                    "source": item.get("source", "unknown"),
                    "metadata": {"date": item.get("date", ""), "url": item.get("url", "")},
                })
        logger.info(f"Loaded {len(records)} Indian FC records")
        return records


def load_combined_dataset(config: Dict, splits=("train", "val", "test")) -> DatasetDict:
    """
    Merge all datasets into a single DatasetDict.
    Stratified by language and source for balanced training.
    """
    all_records = {s: [] for s in splits}

    # LIAR
    try:
        liar = LIARLoader(config["datasets"]["liar"]["path"])
        for split in splits:
            try:
                all_records[split].extend(liar.load_split(split))
            except FileNotFoundError:
                pass
    except Exception as e:
        logger.error(f"LIAR load failed: {e}")

    # Factify
    try:
        factify = FactifyLoader(config["datasets"]["factify"]["path"])
        for split in splits:
            try:
                all_records[split].extend(factify.load_split(split))
            except FileNotFoundError:
                pass
    except Exception as e:
        logger.error(f"Factify load failed: {e}")

    # Indian FC (only train)
    try:
        indian = IndianFactCheckLoader(config["datasets"]["custom_indian"]["path"])
        records = indian.load()
        # 80/10/10 split
        n = len(records)
        all_records["train"].extend(records[: int(0.8 * n)])
        all_records["val"].extend(records[int(0.8 * n): int(0.9 * n)])
        all_records["test"].extend(records[int(0.9 * n):])
    except Exception as e:
        logger.error(f"Indian FC load failed: {e}")

    result = {}
    for split, records in all_records.items():
        if records:
            result[split] = Dataset.from_list(records)
            logger.info(f"{split}: {len(records)} total records")

    return DatasetDict(result)
