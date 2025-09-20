import argparse
import os
from typing import Optional

import numpy as np
import torch

from .model import MFWithBias


def _load_allpos_from_train(dataset_name: str, n_users: int, n_items: int):
    """Read data/<dataset>/train.txt and build allPos per user without importing world/dataloader."""
    train_file = os.path.join("data", dataset_name, "train.txt")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"train file not found: {train_file}")
    lists = [[] for _ in range(n_users)]
    max_u = -1
    max_i = -1
    with open(train_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            uid = int(parts[0])
            items = [int(x) for x in parts[1:]] if len(parts) > 1 else []
            if 0 <= uid < n_users:
                lists[uid].extend(items)
            max_u = max(max_u, uid)
            if items:
                max_i = max(max_i, max(items))
    if (max_u + 1) > n_users or (max_i + 1) > n_items:
        raise ValueError(
            f"Dataset exceeds checkpoint dims: train=({max_u+1},{max_i+1}) vs ckpt=({n_users},{n_items})"
        )
    return [np.asarray(x, dtype=np.int64) if len(x) else np.asarray([], dtype=np.int64) for x in lists]


@torch.no_grad()
def predict_full_matrix_memmap(
    ckpt_path: str,
    output_path: str,
    device: Optional[str] = None,
    users_per_batch: int = 1024,
    apply_sigmoid: bool = False,
    dataset_name: Optional[str] = None,
    exclude_interactions: bool = False,
):
    """
    Load best MFWithBias checkpoint and predict ratings for all (user,item) pairs.
    Save predictions to disk as float16 .npy memmap (memory efficient, row-wise streaming).

    Args:
        ckpt_path: Path to checkpoint produced by LFM/LFM.py.
        output_path: Target .npy file. Load later via numpy.load(..., mmap_mode='r').
        device: Torch device string. Defaults to 'cuda' if available else 'cpu'.
        users_per_batch: Number of users per forward block.
        apply_sigmoid: If True, apply sigmoid to raw logits (for implicit models).
        dataset_name: Optional dataset name (e.g., 'amazon-book') to load allPos.
        exclude_interactions: If True, set existing interactions to float16 min.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device)
    n_users = int(ckpt["n_users"])  # type: ignore[index]
    n_items = int(ckpt["n_items"])  # type: ignore[index]
    dim = int(ckpt.get("dim", 64))

    model = MFWithBias(
        n_users=n_users,
        n_items=n_items,
        dim=dim,
        clip_output=bool(ckpt.get("config", {}).get("clip", False)),
        rating_min=float(ckpt.get("config", {}).get("rating_min", 1.0)),
        rating_max=float(ckpt.get("config", {}).get("rating_max", 5.0)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
    model.eval()

    allPos = None
    if exclude_interactions:
        if dataset_name is None:
            raise ValueError("dataset_name must be provided when exclude_interactions=True")
        allPos = _load_allpos_from_train(dataset_name, n_users, n_items)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    from numpy.lib.format import open_memmap

    mm = open_memmap(
        filename=output_path,
        mode="w+",
        dtype=np.float16,
        shape=(n_users, n_items),
    )

    item_factors = model.item_factors.weight.to(device)  # (I, D)
    item_bias = model.item_bias.weight.squeeze(1).to(device)  # (I,)
    global_bias = model.global_bias.to(device)  # (1,)

    f16_min = np.finfo(np.float16).min

    for u_start in range(0, n_users, users_per_batch):
        u_end = min(u_start + users_per_batch, n_users)
        user_idx = torch.arange(u_start, u_end, device=device)
        u_factors = model.user_factors(user_idx)  # (B, D)
        u_bias = model.user_bias(user_idx).squeeze(1)  # (B,)

        scores = torch.matmul(u_factors, item_factors.T)  # (B, I)
        scores = scores + u_bias.unsqueeze(1) + item_bias.unsqueeze(0) + global_bias

        if model.clip_output:
            span = model.rating_max - model.rating_min
            scores = model.rating_min + span * torch.sigmoid(scores)
        elif apply_sigmoid:
            scores = torch.sigmoid(scores)

        block = scores.detach().cpu().numpy().astype(np.float16, copy=False)

        if allPos is not None:
            for local_row, user in enumerate(range(u_start, u_end)):
                pos_items = allPos[user]
                if len(pos_items) > 0:
                    block[local_row, np.asarray(pos_items, dtype=np.int64)] = f16_min

        mm[u_start:u_end, :] = block

    mm.flush()
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best model checkpoint (.pt)")
    parser.add_argument("--output", type=str, required=True, help="Output .npy path for memmap storage")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--users_per_batch", type=int, default=1024)
    parser.add_argument("--apply_sigmoid", action="store_true", help="Apply sigmoid to logits")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (e.g., amazon-book)")
    parser.add_argument("--exclude_interactions", action="store_true", help="Mask seen interactions to float16 min")
    args = parser.parse_args()

    predict_full_matrix_memmap(
        ckpt_path=args.ckpt,
        output_path=args.output,
        device=args.device,
        users_per_batch=args.users_per_batch,
        apply_sigmoid=args.apply_sigmoid,
        dataset_name=args.dataset,
        exclude_interactions=args.exclude_interactions,
    )


if __name__ == "__main__":
    main()


