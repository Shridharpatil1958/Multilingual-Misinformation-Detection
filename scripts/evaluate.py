# scripts/evaluate.py

"""
End-to-end pipeline evaluation on the test set.
Reports: Macro F1, FEVER score, Recall@K, per-language breakdown.
Run: python scripts/evaluate.py --config configs/model_config.yaml
"""

import argparse
import yaml
import json
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from src.data.dataset_loader import load_combined_dataset
from src.data.preprocessor import TextPreprocessor
from src.models.classifier import ClaimClassifier
from src.models.nli_model import NLIModel, MultiEvidenceAggregator
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever, PipelineRetriever
from src.utils.metrics import full_evaluation, retrieval_recall_at_k, fever_score
from src.utils.logger import get_logger
from transformers import AutoTokenizer

logger = get_logger(__name__)
LABEL_NAMES = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_pipeline(config, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Evaluating on {device}")

    # Load test data
    raw_data = load_combined_dataset(config)
    if "test" not in raw_data:
        logger.error("No test split found")
        return

    test_data = raw_data["test"]
    logger.info(f"Test set: {len(test_data)} examples")

    # Load models
    cls_path = config["classifier"]["output_dir"]
    nli_path = config["nli"]["output_dir"]
    index_path = config["retriever"]["index_path"]
    corpus_path = config["retriever"]["corpus_path"]

    # Classifier
    classifier = None
    if Path(cls_path).exists():
        classifier = ClaimClassifier.load(cls_path).to(device).eval()

    # NLI
    nli_aggregator = None
    if Path(nli_path).exists():
        nli_model = NLIModel.load(nli_path).to(device).eval()
        nli_tokenizer = AutoTokenizer.from_pretrained(nli_path)
        nli_aggregator = MultiEvidenceAggregator(nli_model, nli_tokenizer)

    # Retriever
    pipeline_retriever = None
    if Path(index_path).exists():
        dense = DenseRetriever(config["retriever"]["encoder_model"])
        dense.load(index_path, corpus_path)
        bm25_path = index_path.replace(".faiss", "_bm25.pkl")
        bm25 = BM25Retriever.load(bm25_path) if Path(bm25_path).exists() else BM25Retriever()
        hybrid = HybridRetriever(bm25, dense, final_top_k=config["retriever"]["top_k"])
        pipeline_retriever = PipelineRetriever(hybrid)

    # Evaluate
    preds, refs, langs = [], [], []
    evidence_found = []
    retrieval_results, retrieval_relevant = [], []

    preprocessor = TextPreprocessor(config["classifier"]["model_name"])

    for example in tqdm(test_data, desc="Evaluating"):
        claim = preprocessor.clean(example["claim"])
        label = example["label"]
        lang = example.get("language", "en")
        gold_evidence = example.get("evidence", [])

        refs.append(label)
        langs.append(lang)

        # Retrieve evidence
        passages, scores, found = [], [], False
        if pipeline_retriever:
            ret_result = pipeline_retriever.retrieve(claim)
            passages = [p.get("text", "") for p in ret_result["passages"]]
            scores = ret_result["retrieval_scores"]

            retrieved_ids = [p.get("chunk_id", "") for p in ret_result["passages"]]
            retrieval_results.append(retrieved_ids)
            retrieval_relevant.append([str(i) for i in range(len(gold_evidence))])

            # Check if any gold evidence retrieved (simplified: text overlap)
            found = any(
                any(ge[:50] in p for p in passages)
                for ge in gold_evidence if ge
            )
        evidence_found.append(found)

        # Predict
        pred = 2  # default: NEI
        if nli_aggregator and passages:
            result = nli_aggregator.predict_with_evidence(claim, passages, scores, device)
            label_str = result["final_label"]
            pred = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}[label_str]
        elif classifier:
            enc = preprocessor.tokenize([claim])
            enc = {k: v.to(device) for k, v in enc.items()}
            preds_tensor, _ = classifier.predict(enc["input_ids"], enc["attention_mask"])
            pred = preds_tensor[0].item()

        preds.append(pred)

    # Compute metrics
    eval_result = full_evaluation(preds, refs, return_confusion=True)
    fever = fever_score(preds, refs, evidence_found)

    if retrieval_results:
        recall_10 = retrieval_recall_at_k(retrieval_results, retrieval_relevant, k=10)
        eval_result["retrieval_recall_at_10"] = recall_10

    eval_result["fever_score"] = fever

    # Per-language breakdown
    lang_groups = defaultdict(lambda: {"preds": [], "refs": []})
    for pred, ref, lang in zip(preds, refs, langs):
        lang_groups[lang]["preds"].append(pred)
        lang_groups[lang]["refs"].append(ref)

    eval_result["per_language"] = {}
    for lang, group in lang_groups.items():
        from sklearn.metrics import f1_score
        f1 = f1_score(group["refs"], group["preds"], average="macro", zero_division=0)
        eval_result["per_language"][lang] = {"macro_f1": round(f1, 4), "n": len(group["refs"])}

    # Print and save
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Macro F1:       {eval_result['macro_f1']:.4f}")
    print(f"Accuracy:       {eval_result['accuracy']:.4f}")
    print(f"FEVER Score:    {eval_result.get('fever_score', 'N/A'):.4f}")
    if "retrieval_recall_at_10" in eval_result:
        print(f"Retrieval R@10: {eval_result['retrieval_recall_at_10']:.4f}")
    print("\nPer-class F1:")
    for name, metrics in eval_result["per_class"].items():
        print(f"  {name:20s}: F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")
    print("\nPer-language Macro F1:")
    for lang, m in eval_result["per_language"].items():
        print(f"  {lang}: {m['macro_f1']:.4f}  (n={m['n']})")

    out_path = Path("models/checkpoints/eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate_pipeline(config, args)
