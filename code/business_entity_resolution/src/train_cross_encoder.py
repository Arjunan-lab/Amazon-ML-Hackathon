"""Deep Multilingual Cross-Encoder Pipeline for AWS GPU Training (Stage 4).

Architecture:
- Model: microsoft/mdeberta-v3-base (or FacebookAI/xlm-roberta-base)
- License: MIT License (fully compliant with competition rules)
- Parameter Count: ~278 Million (strictly below 8 Billion ceiling)

Designed for AWS EC2 GPU instances (e.g., g5.2xlarge with NVIDIA A10G or p3.2xlarge with V100):
- Mixed Precision (FP16 / AMP)
- AdamW with Linear Warmup
- Pairwise Binary Cross-Entropy Loss
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Note: torch and transformers are required on the AWS GPU machine
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class EntityPairDataset:
    """Pairwise dataset for Transformer Cross-Encoder."""

    def __init__(self, texts_s1: List[str], texts_cand: List[str], labels: Optional[List[int]] = None, tokenizer=None, max_length: int = 128):
        self.texts_s1 = texts_s1
        self.texts_cand = texts_cand
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts_s1)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts_s1[idx],
            self.texts_cand[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # Squeeze batch dimension
        item = {key: val.squeeze(0) for key, val in item.items()}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def format_entity_text(name: str, address: str) -> str:
    """Formats entity fields into clean textual sequence for transformer input."""
    return f"{name} [ADDR] {address}".strip()


def train_cross_encoder_on_gpu(
    train_pairs: List[Tuple[str, str, int]],
    val_pairs: Optional[List[Tuple[str, str, int]]] = None,
    model_name: str = "microsoft/mdeberta-v3-base",
    output_dir: str = "models/cross_encoder",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    fp16: bool = True,
):
    """Fine-tunes a multilingual cross-encoder on AWS GPU."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch and Transformers are required for GPU training. "
            "Please install them on your AWS instance via: pip install torch transformers accelerate"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using compute device: {device}")
    if torch.cuda.is_available():
        print(f"--> GPU Device Name: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"--> Loading base tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(device)

    # Prepare datasets
    s1_train = [p[0] for p in train_pairs]
    cand_train = [p[1] for p in train_pairs]
    y_train = [p[2] for p in train_pairs]

    train_dataset = EntityPairDataset(s1_train, cand_train, y_train, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=fp16 and torch.cuda.is_available())

    print(f"--> Training {epochs} epochs across {len(train_pairs):,} candidate pairs...")
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=fp16 and torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 500 == 0:
                print(f"    Epoch {epoch}/{epochs} | Step {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"--> [Epoch {epoch}/{epochs}] Finished in {time.time()-t0:.1f}s | Average Loss: {avg_loss:.4f}")

    # Save fine-tuned weights
    print(f"--> Saving fine-tuned model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("--> Model successfully saved!")


def predict_cross_encoder_logits(
    model_dir: str,
    pairs: List[Tuple[str, str]],
    batch_size: int = 64,
) -> np.ndarray:
    """Computes cross-encoder probability scores for inference."""
    if not HAS_TORCH:
        raise ImportError("PyTorch and Transformers are required for inference.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    s1_texts = [p[0] for p in pairs]
    cand_texts = [p[1] for p in pairs]
    dataset = EntityPairDataset(s1_texts, cand_texts, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    scores = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits.squeeze(-1)).cpu().numpy()
            scores.extend(probs.tolist())

    return np.array(scores, dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deep Multilingual Cross-Encoder on AWS GPU")
    parser.add_argument("--model-name", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--output-dir", default="models/cross_encoder")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    print("PyTorch GPU Available:", torch.cuda.is_available() if HAS_TORCH else "Torch Not Installed")
