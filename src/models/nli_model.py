# src/models/nli_model.py

"""
Cross-lingual Natural Language Inference (NLI) model using mDeBERTa-v3.
Input: (claim, evidence_passage) → SUPPORTS / REFUTES / NOT_ENOUGH_INFO

Handles cross-lingual pairs: Hindi claim + English evidence, etc.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import List, Dict, Tuple, Optional

LABEL2ID = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class NLIModel(nn.Module):
    """
    mDeBERTa-v3 with NLI head.
    mDeBERTa outperforms XLM-R on NLI due to disentangled attention.
    """

    def __init__(
        self,
        model_name: str = "microsoft/mdeberta-v3-base",
        num_labels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size

        self.nli_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.nli_head(cls_repr)

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)
        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            out = self.forward(input_ids, attention_mask)
            probs = torch.softmax(out["logits"], dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs

    def save(self, path: str):
        self.encoder.save_pretrained(path)
        torch.save(self.nli_head.state_dict(), f"{path}/nli_head.pt")

    @classmethod
    def load(cls, path: str, model_name: str = "microsoft/mdeberta-v3-base") -> "NLIModel":
        model = cls(model_name=model_name)
        from transformers import AutoModel
        model.encoder = AutoModel.from_pretrained(path)
        model.nli_head.load_state_dict(
            torch.load(f"{path}/nli_head.pt", map_location="cpu")
        )
        return model


class MultiEvidenceAggregator:
    """
    Aggregate NLI predictions across multiple retrieved evidence passages.
    Strategy: weighted vote by retrieval score.
    """

    def __init__(self, model: NLIModel, tokenizer: AutoTokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def predict_with_evidence(
        self,
        claim: str,
        evidence_passages: List[str],
        retrieval_scores: Optional[List[float]] = None,
        device: str = "cpu",
    ) -> Dict:
        """
        Returns:
          - final_label: str
          - final_probs: List[float]
          - per_evidence: List[Dict] with individual predictions
          - best_evidence: str (most supporting/refuting passage)
        """
        if not evidence_passages:
            return {
                "final_label": "NOT_ENOUGH_INFO",
                "final_probs": [0.0, 0.0, 1.0],
                "per_evidence": [],
                "best_evidence": "",
            }

        if retrieval_scores is None:
            retrieval_scores = [1.0] * len(evidence_passages)

        # Normalize retrieval scores to weights
        total = sum(retrieval_scores)
        weights = [s / total for s in retrieval_scores]

        per_evidence = []
        weighted_probs = torch.zeros(3)

        for passage, weight in zip(evidence_passages, weights):
            enc = self.tokenizer(
                claim,
                passage,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)

            preds, probs = self.model.predict(
                enc["input_ids"], enc["attention_mask"]
            )
            label_id = preds[0].item()
            per_evidence.append({
                "evidence": passage[:200] + "..." if len(passage) > 200 else passage,
                "label": ID2LABEL[label_id],
                "probs": probs[0].tolist(),
                "weight": weight,
            })
            weighted_probs += weight * probs[0].cpu()

        final_label_id = weighted_probs.argmax().item()

        # Best evidence = highest confidence non-NEI prediction
        sorted_ev = sorted(
            per_evidence,
            key=lambda x: max(x["probs"][0], x["probs"][1]),
            reverse=True,
        )

        return {
            "final_label": ID2LABEL[final_label_id],
            "final_probs": weighted_probs.tolist(),
            "per_evidence": per_evidence,
            "best_evidence": sorted_ev[0]["evidence"] if sorted_ev else "",
        }
