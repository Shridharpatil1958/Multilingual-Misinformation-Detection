# src/models/classifier.py

"""
Cross-lingual claim classifier using XLM-RoBERTa.
Classifies claims into: SUPPORTS (0) / REFUTES (1) / NOT_ENOUGH_INFO (2)
without evidence (Phase 1 baseline).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict, Tuple

LABEL2ID = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class ClaimClassifier(nn.Module):
    """
    XLM-RoBERTa with a classification head.
    Uses [CLS] token representation + dropout for regularization.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-large",
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

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
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
        # [CLS] representation
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_repr)

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)

        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted_label_ids, probabilities)."""
        self.eval()
        with torch.no_grad():
            out = self.forward(input_ids, attention_mask)
            probs = torch.softmax(out["logits"], dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs

    def save(self, path: str):
        self.encoder.save_pretrained(path)
        torch.save(self.classifier.state_dict(), f"{path}/classifier_head.pt")

    @classmethod
    def load(cls, path: str, model_name: str = "xlm-roberta-large") -> "ClaimClassifier":
        model = cls(model_name=model_name)
        from transformers import AutoModel
        model.encoder = AutoModel.from_pretrained(path)
        model.classifier.load_state_dict(
            torch.load(f"{path}/classifier_head.pt", map_location="cpu")
        )
        return model


class CalibratedClassifier:
    """
    Wraps ClaimClassifier with Platt scaling for calibrated probabilities.
    Train the scaler after main model training on val set.
    """

    def __init__(self, model: ClaimClassifier):
        self.model = model
        self.scaler = None   # sklearn CalibratedClassifierCV-style

    def fit_calibration(self, val_logits: torch.Tensor, val_labels: torch.Tensor):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        logits_np = val_logits.numpy()
        labels_np = val_labels.numpy()

        # One-vs-rest Platt scaling per class
        self.scaler = LogisticRegression(multi_class="ovr", C=1.0, max_iter=1000)
        self.scaler.fit(logits_np, labels_np)

    def predict_calibrated(self, logits: torch.Tensor):
        import numpy as np
        if self.scaler is None:
            raise RuntimeError("Call fit_calibration() first")
        probs = self.scaler.predict_proba(logits.numpy())
        return torch.tensor(probs, dtype=torch.float32)
