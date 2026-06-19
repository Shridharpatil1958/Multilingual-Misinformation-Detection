# src/models/trainer.py

"""
Training loop for both classifier and NLI model.
Uses HuggingFace Trainer API for fp16, gradient accumulation, and checkpointing.
Run: python -m src.models.trainer --config configs/model_config.yaml --task classify
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, classification_report

from src.data.dataset_loader import load_combined_dataset
from src.data.preprocessor import TextPreprocessor
from src.models.classifier import ClaimClassifier
from src.models.nli_model import NLIModel
from src.utils.logger import get_logger
from src.utils.metrics import compute_metrics

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_classify_dataset(raw: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    """Tokenize claims for classification task."""
    preprocessor = TextPreprocessor(tokenizer.name_or_path, max_length)

    def tokenize_fn(batch):
        cleaned = [preprocessor.clean(c) for c in batch["claim"]]
        enc = tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        enc["labels"] = batch["label"]
        return enc

    return raw.map(tokenize_fn, batched=True, remove_columns=raw["train"].column_names)


def prepare_nli_dataset(raw: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    """Tokenize (claim, evidence) pairs for NLI task."""
    preprocessor = TextPreprocessor(tokenizer.name_or_path, max_length)

    def tokenize_fn(batch):
        claims = [preprocessor.clean(c) for c in batch["claim"]]
        # Use first evidence if available, else empty string
        evidences = [
            e[0] if isinstance(e, list) and e else ""
            for e in batch["evidence"]
        ]
        enc = tokenizer(
            claims,
            evidences,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        enc["labels"] = batch["label"]
        return enc

    return raw.map(tokenize_fn, batched=True, remove_columns=raw["train"].column_names)


def train_classifier(config: dict):
    cfg = config["classifier"]
    logger.info(f"Training classifier: {cfg['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    raw_data = load_combined_dataset(config)
    tokenized = prepare_classify_dataset(raw_data, tokenizer, cfg["max_length"])

    model = ClaimClassifier(
        model_name=cfg["model_name"],
        num_labels=cfg["num_labels"],
    )

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        fp16=cfg["fp16"] and torch.cuda.is_available(),
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        evaluation_strategy=config["training"]["eval_strategy"],
        save_strategy=config["training"]["save_strategy"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        logging_steps=config["training"]["logging_steps"],
        seed=config["training"]["seed"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("val"),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    # Evaluate on test set
    if "test" in tokenized:
        results = trainer.evaluate(tokenized["test"])
        logger.info(f"Test results: {results}")

    logger.info(f"Classifier saved to {cfg['output_dir']}")


def train_nli(config: dict):
    cfg = config["nli"]
    logger.info(f"Training NLI model: {cfg['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    raw_data = load_combined_dataset(config)

    # Filter to only records that have evidence
    for split in raw_data:
        raw_data[split] = raw_data[split].filter(
            lambda x: isinstance(x["evidence"], list) and len(x["evidence"]) > 0
        )
        logger.info(f"NLI {split}: {len(raw_data[split])} records with evidence")

    tokenized = prepare_nli_dataset(raw_data, tokenizer, cfg["max_length"])

    model = NLIModel(
        model_name=cfg["model_name"],
        num_labels=cfg["num_labels"],
    )

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        fp16=cfg["fp16"] and torch.cuda.is_available(),
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        evaluation_strategy=config["training"]["eval_strategy"],
        save_strategy=config["training"]["save_strategy"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        logging_steps=config["training"]["logging_steps"],
        seed=config["training"]["seed"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("val"),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting NLI training...")
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    logger.info(f"NLI model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--task", choices=["classify", "nli"], required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(config["training"]["seed"])

    if args.task == "classify":
        train_classifier(config)
    elif args.task == "nli":
        train_nli(config)
