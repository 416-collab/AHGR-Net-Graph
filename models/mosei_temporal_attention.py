import torch
import torch.nn as nn


class MOSEITemporalAttention(nn.Module):
    """
    BiGRU + attention pooling for MOSEI sequence features.
    Input: (B, T, 64)
    """

    def __init__(self, input_dim=64, hidden_dim=128, num_layers=1, dropout=0.3, num_classes=2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, seq, lengths, return_attention=False):
        # Sort for packing
        lengths_cpu = lengths.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            seq,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.gru(packed)

        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
        )

        B, T, H = out.shape
        mask = torch.arange(T, device=seq.device).unsqueeze(0) < lengths.unsqueeze(1)

        attn_scores = self.attn(out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)

        pooled = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)
        logits = self.classifier(pooled)

        if return_attention:
            return {
                "logits": logits,
                "attention": attn_weights,
                "pooled": pooled,
            }

        return logits
