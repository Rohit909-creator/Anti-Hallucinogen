"""
probe.py — Hallucination probe: scaler, model, training loop, evaluation.

Classes / Functions:
  OnlineStandardScaler  — memory-efficient z-score normaliser (GPU-friendly)
  HallucinationProbe    — linear probe (logistic regression in PyTorch)
  train_probe           — full training loop with early stopping
  evaluate              — metrics on a held-out split
  inspect_h_neurons     — decode flat weight indices → (layer, neuron)
  load_probe            — load a saved probe from disk
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# ──────────────────────────────────────────────────────────────────────────────

class OnlineStandardScaler(nn.Module):
    """
    Computes feature-wise mean and std from the training data in float32 chunks
    (avoids loading the full float32 matrix into RAM at once), then applies
    z-score normalisation on the GPU during forward passes.

    Stored as non-trainable buffers so they travel with `model.state_dict()`.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.register_buffer("mean_", torch.zeros(num_features))
        self.register_buffer("std_",  torch.ones(num_features))
        self.fitted = False

    def fit(self, X_fp16: torch.Tensor, chunk_size: int = 1000) -> None:
        """Compute mean / std from a float16 tensor without materialising float32 at once."""
        n, d    = X_fp16.shape
        mean    = torch.zeros(d, dtype=torch.float32)
        mean_sq = torch.zeros(d, dtype=torch.float32)

        for start in range(0, n, chunk_size):
            chunk    = X_fp16[start: start + chunk_size].float()
            mean    += chunk.sum(0)
            mean_sq += chunk.pow(2).sum(0)
            del chunk

        mean    /= n
        mean_sq /= n
        std      = (mean_sq - mean.pow(2)).clamp(min=1e-8).sqrt()

        self.mean_.copy_(mean)
        self.std_.copy_(std)
        self.fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast fp16 → fp32, then normalise
        return (x.float() - self.mean_) / self.std_


# ──────────────────────────────────────────────────────────────────────────────

class HallucinationProbe(nn.Module):
    """
    Single linear layer equivalent to logistic regression.

    Keeping the probe linear means the weights ARE the H-Neuron scores,
    interpretable exactly as sklearn's `coef_`.  Each weight tells you
    how strongly that neuron's activation predicts hallucination.

    The scaler is baked into the forward pass so a single
    `probe(features)` call handles both normalisation and scoring.
    """

    def __init__(self, input_dim: int, scaler: OnlineStandardScaler):
        super().__init__()
        self.scaler = scaler
        self.linear = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scaler(x)                        # fp16 → fp32 + normalise
        return self.linear(x).squeeze(-1)          # [batch] raw logits


# ──────────────────────────────────────────────────────────────────────────────

def _reg_loss(
    model: HallucinationProbe,
    penalty: str,
    lam: float,
) -> torch.Tensor:
    w = model.linear.weight
    if penalty == "l1":
        return lam * w.abs().sum()
    return lam * w.pow(2).sum()


def train_probe(
    probe: HallucinationProbe,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    *,
    device,
    penalty: str = "l2",
    lam: float = 1e-5,
    lr: float = 1e-4,
    epochs: int = 30,
    batch_size: int = 512,
    patience: int = 10,
    val_fraction: float = 0.2,
) -> HallucinationProbe:
    """
    Train the probe with early stopping.

    If `X_val` / `y_val` are None, an 80/20 split of `X_train` is used.

    Returns the probe loaded with the best-validation-loss weights.
    """
    if X_val is None:
        val_n    = int(val_fraction * len(X_train))
        train_n  = len(X_train) - val_n
        train_ds, val_ds = random_split(
            TensorDataset(X_train, y_train), [train_n, val_n],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_ds = TensorDataset(X_train, y_train)
        val_ds   = TensorDataset(X_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)

    probe.to(device)

    # Class-balanced positive weight
    all_y    = torch.cat([y for _, y in train_loader])
    n_pos    = all_y.sum().float()
    n_neg    = (all_y == 0).sum().float()
    pos_w    = (n_neg / n_pos).to(device)
    print(f"  Class weight (neg/pos): {pos_w.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    print(f"\n{'Epoch':>6}  {'Train':>10}  {'Val':>8}  {'Acc':>7}  {'AUC':>7}")
    print("─" * 48)

    for epoch in range(1, epochs + 1):
        # ---- train ----
        probe.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.float().to(device)
            optimizer.zero_grad()
            loss = criterion(probe(Xb), yb) + _reg_loss(probe, penalty, lam)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # ---- validate ----
        probe.eval()
        val_loss, all_logits, all_labels = 0.0, [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.float().to(device)
                logits = probe(Xb)
                val_loss += criterion(logits, yb).item() * len(yb)
                all_logits.append(logits.cpu())
                all_labels.append(yb.cpu())

        val_loss  /= len(val_loader.dataset)
        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        preds      = (torch.sigmoid(logits_cat) > 0.5).long()
        acc        = (preds == labels_cat.long()).float().mean().item()

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels_cat.numpy(), torch.sigmoid(logits_cat).numpy())
        except Exception:
            auc = float("nan")

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>8.4f}  "
              f"{acc:>7.4f}  {auc:>7.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in probe.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stop at epoch {epoch}.")
                break

    probe.load_state_dict(best_state)
    print(f"  Best val loss: {best_val_loss:.4f}")
    return probe


def evaluate(
    probe: HallucinationProbe,
    X: torch.Tensor,
    y: torch.Tensor,
    device,
    batch_size: int = 512,
    split_name: str = "Test",
) -> None:
    """Print classification report and AUROC for a dataset split."""
    from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

    probe.eval()
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    all_logits, all_labels = [], []

    with torch.no_grad():
        for Xb, yb in loader:
            all_logits.append(probe(Xb.to(device)).cpu())
            all_labels.append(yb)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs  = torch.sigmoid(logits).numpy()
    preds  = (probs > 0.5).astype(int)
    y_np   = labels.numpy()

    print(f"\n{'='*50}")
    print(f"  {split_name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {accuracy_score(y_np, preds):.4f}")
    print(f"  AUROC    : {roc_auc_score(y_np, probs):.4f}")
    print(classification_report(y_np, preds, target_names=["faithful", "hallucinated"]))


def inspect_h_neurons(
    probe: HallucinationProbe,
    intermediate_size: int,
    penalty: str = "l2",
    top_n: int = 20,
) -> None:
    """
    Decode flat weight indices back to (layer, neuron) pairs and print a table.

    For L1-regularised probes many weights will be exactly zero — only
    non-zero weights are true H-Neurons.
    For L2-regularised probes all weights are non-zero; we print the top_n
    by absolute magnitude.
    """
    coef = probe.linear.weight.detach().cpu().float().numpy()[0]

    if penalty == "l1":
        indices = np.where(np.abs(coef) > 1e-6)[0]
        print(f"\nH-Neurons (non-zero L1): {len(indices)} / {len(coef)} "
              f"({100*len(indices)/len(coef):.3f}%)")
    else:
        indices = np.argsort(np.abs(coef))[::-1][:top_n]
        print(f"\nTop {top_n} neurons by |weight|:")

    sorted_idx = indices[np.argsort(np.abs(coef[indices]))[::-1]]

    print(f"  {'rank':<5} {'layer':<7} {'neuron':<8} {'weight':>10}")
    print(f"  {'─'*35}")
    for rank, flat_idx in enumerate(sorted_idx[:top_n]):
        layer  = int(flat_idx) // intermediate_size
        neuron = int(flat_idx) %  intermediate_size
        print(f"  {rank+1:<5} {layer:<7} {neuron:<8} {coef[flat_idx]:>+10.4f}")


# ──────────────────────────────────────────────────────────────────────────────

def save_probe(
    probe: HallucinationProbe,
    path: str,
    penalty: str,
    lam: float,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": probe.state_dict(),
        "input_dim":   probe.linear.in_features,
        "penalty":     penalty,
        "lambda":      lam,
    }, path)
    print(f"Probe saved → {path}")


def load_probe(path: str, device) -> HallucinationProbe:
    """Reconstruct a HallucinationProbe from a saved checkpoint."""
    ckpt      = torch.load(path, map_location="cpu")
    input_dim = ckpt["input_dim"]
    scaler    = OnlineStandardScaler(input_dim)
    probe     = HallucinationProbe(input_dim, scaler)
    probe.load_state_dict(ckpt["model_state"])
    probe.eval()
    probe.to(device)
    print(f"Probe loaded  (input_dim={input_dim}, penalty={ckpt.get('penalty','?')})")
    return probe
