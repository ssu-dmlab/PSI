import torch
from torch import nn



# -------------------------
# Model
# -------------------------
class MFWithBias(nn.Module):
    """
    Rating_hat = mu + b_u + b_i + <P_u, Q_i>
    where P: user embedding, Q: item embedding
    """

    def __init__(self, n_users: int, n_items: int, dim: int = 64, init_std: float = 0.01, dropout: float = 0.0, clip_output: bool = False, rating_min: float = 1.0, rating_max: float = 5.0):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, dim)
        self.item_factors = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.clip_output = clip_output
        self.rating_min = rating_min
        self.rating_max = rating_max
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
        p_u = self.dropout(p_u)
        q_i = self.dropout(q_i)
        dot = (p_u * q_i).sum(dim=-1)       # (B,)
        raw = self.global_bias + bu + bi + dot
        if self.clip_output:
            span = self.rating_max - self.rating_min
            return self.rating_min + span * torch.sigmoid(raw)
        return raw