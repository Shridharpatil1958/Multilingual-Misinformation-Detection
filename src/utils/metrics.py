# src/utils/metrics.py

"""
Evaluation metrics for classification and retrieval tasks.
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import f1_score, classification_report, confusion_matrix


LABEL_NAMES = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]


def compute_metrics(eval_pred) -> Dict[str, float]:
    """For HuggingFace Trainer. Returns macro F1 + per-class F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class = f1_score(labels, preds, average=None, zero_division=0, labels=[0, 1, 2])

    metrics = {"macro_f1": macro_f1}
    for i, name in enumerate(LABEL_NAMES):
        metrics[f"f1_{name.lower()}"] = per_class[i] if i < len(per_class) else 0.0
    return metrics


def full_evaluation(
    predictions: List[int],
    references: List[int],
    return_confusion: bool = False,
) -> Dict:
    """Full evaluation report including per-class metrics."""
    report = classification_report(
        references,
        predictions,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "macro_f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in LABEL_NAMES
        },
    }

    if return_confusion:
        result["confusion_matrix"] = confusion_matrix(references, predictions).tolist()

    return result


def retrieval_recall_at_k(
    retrieved_ids: List[List[str]],
    relevant_ids: List[List[str]],
    k: int = 10,
) -> float:
    """Recall@K for evidence retrieval evaluation."""
    recalls = []
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        if not relevant:
            continue
        top_k = set(retrieved[:k])
        rel_set = set(relevant)
        recall = len(top_k & rel_set) / len(rel_set)
        recalls.append(recall)
    return float(np.mean(recalls)) if recalls else 0.0


def fever_score(
    predictions: List[int],
    references: List[int],
    evidence_found: List[bool],
) -> float:
    """
    FEVER score: correct label AND at least one correct evidence passage found.
    Standard metric for fact verification pipelines.
    """
    correct = sum(
        1 for pred, ref, ev in zip(predictions, references, evidence_found)
        if pred == ref and (ref == 2 or ev)  # NEI doesn't require evidence
    )
    return correct / len(predictions) if predictions else 0.0
