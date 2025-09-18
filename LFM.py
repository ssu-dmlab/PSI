# mf_rating.py
# Latent Factor (Matrix Factorization with biases) for rating prediction on user–item graphs.
# Author: you + ChatGPT
# Usage:
#   python mf_rating.py --csv path/to/data.csv
#   (optional) Convert raw JSON first:
#   python mf_rating.py --from_yelp_json review.json --csv yelp.csv
#   python mf_rating.py --from_amazon_json reviews_Musical_Instruments_5.json --csv amazon.csv

import argparse
import math
import os
import random
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from tqdm import tqdm

# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def factorize_series(s: pd.Series) -> Tuple[np.ndarray, Dict]:
    """Return integer codes and mapping dict {original_id -> code}."""
    codes, uniques = pd.factorize(s, sort=True)
    mapping = {k: int(v) for v, k in enumerate(uniques.tolist())}
    return codes.astype(np.int64), mapping


def detect_header(csv_path: str) -> bool:
    # Tries to detect whether the first row looks like header
    peek = pd.read_csv(csv_path, nrows=5)
    cols = [c.lower() for c in peek.columns]
    header_like = any(k in cols for k in ["user", "user_id"]) and any(
        k in cols for k in ["item", "item_id", "business_id", "product_id"]
    )
    return header_like



# -------------------------
# Dataset
# -------------------------
class RatingsDataset(Dataset):
    def __init__(self, u: np.ndarray, i: np.ndarray, r: np.ndarray):
        self.u = torch.from_numpy(u).long()
        self.i = torch.from_numpy(i).long()
        self.r = torch.from_numpy(r).float()

    def __len__(self):
        return self.r.shape[0]

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx]


# -------------------------
# Model
# -------------------------
class MFWithBias(nn.Module):
    """
    Rating_hat = mu + b_u + b_i + <P_u, Q_i>
    where P: user embedding, Q: item embedding
    """

    def __init__(self, n_users: int, n_items: int, dim: int = 64, init_std: float = 0.01):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, dim)
        self.item_factors = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        # init
        nn.init.normal_(self.user_factors.weight, std=init_std)
        nn.init.normal_(self.item_factors.weight, std=init_std)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, u: torch.LongTensor, i: torch.LongTensor):
        p_u = self.user_factors(u)          # (B, d)
        q_i = self.item_factors(i)          # (B, d)
        bu = self.user_bias(u).squeeze(-1)  # (B,)
        bi = self.item_bias(i).squeeze(-1)  # (B,)
        dot = (p_u * q_i).sum(dim=-1)       # (B,)
        return self.global_bias + bu + bi + dot


# -------------------------
# Training / Eval
# -------------------------
def rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - true) ** 2)).item())


def mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - true)).item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    l2_reg: float = 1e-5,
):
    model.train()
    total_loss = 0.0
    bce_loss = nn.BCEWithLogitsLoss()
    for u, i, r in loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        logits = model(u, i)
        bce = bce_loss(logits, r)
        # L2 on embeddings only (common choice)
        l2 = (
            model.user_factors(u).pow(2).sum()
            + model.item_factors(i).pow(2).sum()
        ) / u.size(0)
        loss = bce + l2_reg * l2
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * u.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_logits, all_labels = [], []
    for u, i, r in loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        logits = model(u, i)
        all_logits.append(logits.detach().cpu())
        all_labels.append(r.detach().cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    # Probabilities for metrics
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int32)

    metrics: Dict[str, float] = {}
    # Some metrics require both classes present
    has_both = (np.unique(labels).shape[0] == 2)
    if has_both:
        try:
            metrics["ROC_AUC"] = float(roc_auc_score(labels, probs))
        except Exception:
            metrics["ROC_AUC"] = float("nan")
        try:
            metrics["PR_AUC"] = float(average_precision_score(labels, probs))
        except Exception:
            metrics["PR_AUC"] = float("nan")
    else:
        metrics["ROC_AUC"] = float("nan")
        metrics["PR_AUC"] = float("nan")

    metrics["ACC"] = float(accuracy_score(labels, preds))
    # Optional: report BCE as well
    eps = 1e-12
    bce = -(
        labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps)
    ).mean()
    metrics["BCE"] = float(bce)
    return metrics


def prepare_tensors(
    df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    # Fit mappings on TRAIN ONLY, then map valid/test; unseen ids in valid/test are dropped
    u_codes, u_map = factorize_series(df_train["user"])
    i_codes, i_map = factorize_series(df_train["item"])

    def map_or_drop(df: pd.DataFrame) -> pd.DataFrame:
        mask = df["user"].isin(u_map.keys()) & df["item"].isin(i_map.keys())
        d = df.loc[mask].copy()
        d["u"] = d["user"].map(u_map).astype(np.int64)
        d["i"] = d["item"].map(i_map).astype(np.int64)
        return d

    train_m = map_or_drop(df_train)
    valid_m = map_or_drop(df_valid)
    test_m = map_or_drop(df_test)

    # Binary prior for bias initialization: use train positive rate
    pos_rate = float(train_m["label"].mean()) if len(train_m) else 0.5
    pos_rate = min(max(pos_rate, 1e-6), 1.0 - 1e-6)
    global_logit = float(math.log(pos_rate / (1.0 - pos_rate)))

    u_train = train_m["u"].to_numpy(np.int64)
    i_train = train_m["i"].to_numpy(np.int64)
    r_train = train_m["label"].to_numpy(np.float32)

    u_valid = valid_m["u"].to_numpy(np.int64)
    i_valid = valid_m["i"].to_numpy(np.int64)
    r_valid = valid_m["label"].to_numpy(np.float32)

    u_test = test_m["u"].to_numpy(np.int64)
    i_test = test_m["i"].to_numpy(np.int64)
    r_test = test_m["label"].to_numpy(np.float32)

    n_users = int(u_train.max()) + 1
    n_items = int(i_train.max()) + 1

    return u_train, i_train, r_train, u_valid, i_valid, r_valid, u_test, i_test, r_test, n_users, n_items, global_logit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False, help="Dataset name")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5, help="early stop patience (epochs)")
    parser.add_argument("--regression", type=int, default=0, help="whether to use regression or not")
    args = parser.parse_args()

    set_seed(args.seed)


    train_df = pd.read_csv(f"data/processed/{args.dataset}_10core_train.csv")
    test_df = pd.read_csv(f"data/processed/{args.dataset}_10core_test.csv")

    train_df, valid_df = train_test_split(train_df, test_size=args.valid_ratio, random_state=args.seed)

    # Prepare tensors/mappings
    (
        u_tr, i_tr, r_tr,
        u_va, i_va, r_va,
        u_te, i_te, r_te,
        n_users, n_items, global_logit
    ) = prepare_tensors(train_df, valid_df, test_df)

    print("Load datasets...")
    # Datasets / loaders
    train_ds = RatingsDataset(u_tr, i_tr, r_tr)
    valid_ds = RatingsDataset(u_va, i_va, r_va) if len(r_va) else None
    test_ds  = RatingsDataset(u_te, i_te, r_te) if len(r_te) else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False) if valid_ds else None
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False) if test_ds else None

    # Model / optim
    device = torch.device(args.device)
    model = MFWithBias(n_users, n_items, dim=args.dim).to(device)
    model.global_bias.data.fill_(global_logit)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train loop with early stopping on valid ROC-AUC (maximize)
    best_score = -float("inf")
    best_state = None
    patience = args.patience
    no_improve = 0
    print("Training...")
    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss = train_one_epoch(model, train_loader, opt, device, l2_reg=args.l2)
        if valid_loader:
            metrics = evaluate(model, valid_loader, device)
            val_auc = metrics.get("ROC_AUC", float("nan"))
            val_pr = metrics.get("PR_AUC", float("nan"))
            val_acc = metrics.get("ACC", float("nan"))
            val_bce = metrics.get("BCE", float("nan"))
            print(f"[Epoch {epoch:02d}] train_BCE={train_loss:.5f}  val_ROC_AUC={val_auc:.5f}  val_PR_AUC={val_pr:.5f}  val_ACC={val_acc:.4f}  val_BCE={val_bce:.5f}")
            # Choose early stop metric: prefer ROC_AUC if available, else ACC
            cur_score = val_auc if not math.isnan(val_auc) else val_acc
            if cur_score > best_score + 1e-6:
                best_score = cur_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[EarlyStop] No improvement for {patience} epochs.")
                    break
        else:
            print(f"[Epoch {epoch:02d}] train_BCE={train_loss:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    if test_loader:
        test_metrics = evaluate(model, test_loader, device)
        msg = (
            f"[Test] ROC_AUC={test_metrics['ROC_AUC']:.5f}  "
            f"PR_AUC={test_metrics['PR_AUC']:.5f}  "
            f"ACC={test_metrics['ACC']:.4f}  BCE={test_metrics['BCE']:.5f}"
        )
        print(msg)
    else:
        print("[Warn] No test set generated (too few interactions per user).")

    # Save model
    out_path = os.path.splitext(os.path.basename(args.dataset))[0] + f".mf_dim{args.dim}.pt"
    torch.save({"state_dict": model.state_dict(), "n_users": n_users, "n_items": n_items, "dim": args.dim}, out_path)
    print(f"[OK] Saved model to {out_path}")


if __name__ == "__main__":
    main()
