# scripts/download_data.py

"""
Download and prepare datasets for training.
Usage:
  python scripts/download_data.py --dataset all
  python scripts/download_data.py --dataset liar
  python scripts/download_data.py --dataset wikipedia
"""

import argparse
import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

DATASETS = {
    "liar": {
        "url": "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
        "dest": DATA_DIR / "liar",
        "description": "LIAR: 12.8K labeled political statements",
    },
}


def download_file(url: str, dest: Path, chunk_size: int = 8192):
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))


def download_liar():
    info = DATASETS["liar"]
    zip_path = DATA_DIR / "liar.zip"
    info["dest"].mkdir(parents=True, exist_ok=True)

    print(f"Downloading LIAR dataset...")
    try:
        download_file(info["url"], zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(info["dest"])
        zip_path.unlink()
        print(f"✅ LIAR saved to {info['dest']}")
    except Exception as e:
        print(f"❌ LIAR download failed: {e}")
        print("   Manual download: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")


def download_wikipedia_dumps():
    """
    Download Hindi and English Wikipedia article dumps for the retrieval corpus.
    Uses the Hugging Face datasets library (much easier than raw dumps).
    """
    print("Downloading Wikipedia corpora via HuggingFace datasets...")
    try:
        from datasets import load_dataset
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        for lang, name in [("en", "wikipedia_en"), ("hi", "wikipedia_hi")]:
            print(f"  Downloading {lang} Wikipedia (20220301 subset)...")
            ds = load_dataset(
                "wikipedia",
                f"20220301.{lang}",
                split="train",
                trust_remote_code=True,
            )
            out_path = PROCESSED_DIR / f"wiki_{lang}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for i, article in enumerate(tqdm(ds, desc=f"wiki_{lang}")):
                    if i >= 500_000:  # limit for storage
                        break
                    record = {
                        "id": f"wiki_{lang}_{article['id']}",
                        "text": article["text"][:2000],  # first 2K chars per article
                        "title": article["title"],
                        "language": lang,
                        "source": f"wikipedia_{lang}",
                    }
                    f.write(__import__("json").dumps(record, ensure_ascii=False) + "\n")
            print(f"  ✅ Saved {out_path}")
    except Exception as e:
        print(f"❌ Wikipedia download failed: {e}")


def create_sample_indian_fc():
    """Create a small sample Indian fact-check dataset for testing."""
    import json
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "indian_fc").mkdir(parents=True, exist_ok=True)

    samples = [
        {"claim": "COVID vaccine mein microchip hai", "verdict": "REFUTES",
         "language": "hi", "source": "altnews",
         "evidence_texts": ["COVID vaccines do not contain microchips. They contain mRNA."]},
        {"claim": "Onion juice cures COVID-19", "verdict": "REFUTES",
         "language": "en", "source": "boomlive",
         "evidence_texts": ["No scientific evidence supports onion juice as a COVID cure."]},
        {"claim": "India launched Chandrayaan-3 in 2023", "verdict": "SUPPORTS",
         "language": "en", "source": "vishvasnews",
         "evidence_texts": ["India successfully launched Chandrayaan-3 on July 14, 2023."]},
    ]

    out = DATA_DIR / "indian_fc" / "indian_fc.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"✅ Sample Indian FC dataset created: {out}")


def main(args):
    datasets = args.dataset.split(",") if args.dataset != "all" else list(DATASETS.keys()) + ["wikipedia", "indian_fc"]

    for ds in datasets:
        ds = ds.strip()
        if ds == "liar":
            download_liar()
        elif ds == "wikipedia":
            download_wikipedia_dumps()
        elif ds == "indian_fc":
            create_sample_indian_fc()
        else:
            print(f"Unknown dataset: {ds}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        help="Dataset(s) to download: all, liar, wikipedia, indian_fc")
    args = parser.parse_args()
    main(args)
