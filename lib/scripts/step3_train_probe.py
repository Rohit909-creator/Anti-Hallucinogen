"""
step3_train_probe.py — Train a linear hallucination detector on CETT features.

Reads:  data/activations/activations.pt  (from step 2)
Writes: models/detector.pt

Run:
  python scripts/step3_train_probe.py
"""

import gc
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

import config
from src.probe import (
    OnlineStandardScaler,
    HallucinationProbe,
    train_probe,
    evaluate,
    inspect_h_neurons,
    save_probe,
)


# ──────────────────────────────────────────────────────────────────────────────
# Balanced train/test split
# ──────────────────────────────────────────────────────────────────────────────

def build_split(
    jsonl_path: str = config.ANSWER_TOKENS_PATH,
    output_path: str = config.TRAIN_QIDS_PATH,
    num_per_class: int = config.NUM_SAMPLES_PER_CLASS,
    seed: int = 42,
) -> dict:
    """
    Group qids by label, sample `num_per_class` from each, and produce
    an 80/20 train/test split.  Balancing prevents a trivial baseline that
    just predicts the majority class.
    """
    random.seed(seed)
    buckets: dict = defaultdict(set)

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            label, qid = sample.get("label", -1), sample.get("qid", "")
            if label in (0, 1) and qid:
                buckets[label].add(qid)

    faithful     = list(buckets[0])
    hallucinated = list(buckets[1])
    print(f"Available — faithful: {len(faithful)},  hallucinated: {len(hallucinated)}")

    n = min(num_per_class, len(faithful), len(hallucinated))
    if n < num_per_class:
        print(f"  ⚠  Only {n} samples available per class (requested {num_per_class}).")

    train_n = int(n * 0.8)
    test_n  = n - train_n

    random.shuffle(faithful)
    random.shuffle(hallucinated)

    train_ids = faithful[:train_n]    + hallucinated[:train_n]
    test_ids  = faithful[train_n:n]   + hallucinated[train_n:n]
    random.shuffle(train_ids)
    random.shuffle(test_ids)

    result = {"train": train_ids, "test": test_ids}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved split → {output_path}")
    print(f"  Train: {len(train_ids)} ({train_n} per class)")
    print(f"  Test:  {len(test_ids)} ({test_n} per class)")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Load + split activations
# ──────────────────────────────────────────────────────────────────────────────

def load_and_split(activations_path: str, split_path: str):
    print("Loading activations ...")
    ckpt = torch.load(activations_path, map_location="cpu")

    # Keep as float16 on CPU — only move batches to GPU during training
    all_features = ckpt["features"].to(torch.float16)
    all_labels   = ckpt["labels"].long()
    all_qids     = ckpt["qids"]

    print(f"  Records     : {len(all_qids)}")
    print(f"  Feature dim : {all_features.shape[1]}")
    print(f"  Hallucinated: {all_labels.sum().item()}  "
          f"  Faithful: {(all_labels == 0).sum().item()}")

    with open(split_path) as f:
        split = json.load(f)

    train_set, test_set = set(split["train"]), set(split["test"])
    train_mask = torch.tensor([q in train_set for q in all_qids])
    test_mask  = torch.tensor([q in test_set  for q in all_qids])

    X_train, y_train = all_features[train_mask], all_labels[train_mask]
    X_test,  y_test  = all_features[test_mask],  all_labels[test_mask]

    print(f"\n  Train : {len(X_train)}  ({y_train.sum().item()} hallucinated)")
    print(f"  Test  : {len(X_test)}   ({y_test.sum().item()} hallucinated)")
    del ckpt
    gc.collect()
    return X_train, y_train, X_test, y_test


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Build balanced split
    build_split()

    # 2. Load data
    X_train, y_train, X_test, y_test = load_and_split(
        f"{config.ACTIVATIONS_DIR}/activations.pt",
        config.TRAIN_QIDS_PATH,
    )

    input_dim = X_train.shape[1]

    # 3. Fit scaler on training data only
    scaler = OnlineStandardScaler(input_dim)
    scaler.fit(X_train)
    scaler.to(config.DEVICE)

    # 4. Build and train probe
    probe = HallucinationProbe(input_dim, scaler)
    probe = train_probe(
        probe, X_train, y_train,
        device=config.DEVICE,
        penalty=config.PENALTY,
        lam=config.LAMBDA,
        lr=config.LR,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        patience=config.PATIENCE,
    )

    # 5. Evaluate on held-out test set
    evaluate(probe, X_test, y_test, device=config.DEVICE, split_name="Test Set")

    # 6. Print H-Neurons (most diagnostic neurons)
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(config.MODEL_PATH)
        intermediate_size = getattr(cfg, "intermediate_size", input_dim // 32)
    except Exception:
        intermediate_size = input_dim // 32

    inspect_h_neurons(probe, intermediate_size, config.PENALTY, top_n=20)

    # 7. Save
    save_probe(probe, config.DETECTOR_PATH, config.PENALTY, config.LAMBDA)
