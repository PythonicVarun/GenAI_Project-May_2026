import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SelfAttention(nn.Module):
    """Additive attention: scores each BiLSTM timestep, returns weighted sum."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h    : (B, T, H)
        # mask : (B, T)  1=real 0=pad
        energy = self.v(torch.tanh(self.proj(h))).squeeze(-1)
        energy = energy.masked_fill(mask == 0, -1e9)
        weights = F.softmax(energy, dim=-1)
        return (weights.unsqueeze(-1) * h).sum(dim=1)


class BiLSTMScorer(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        H = hidden_dim * 2

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_drop = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = SelfAttention(H)

        self.head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed_drop(self.embed(ids))
        packed = pack_padded_sequence(
            emb, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )

        out, _ = self.bilstm(packed)
        h, _ = pad_packed_sequence(out, batch_first=True)
        mask = (ids[:, : h.size(1)] != self.pad_idx).float()
        ctx = self.attn(h, mask)
        return self.head(ctx)

    def forward(self, triples: torch.Tensor, lengths: torch.Tensor):
        # triples: (B, 5, T)   lengths: (B, 5)
        B, C, T = triples.shape
        scores = self._encode(
            triples.view(B * C, T),
            lengths.view(B * C),
        )
        return scores.view(B, C)
