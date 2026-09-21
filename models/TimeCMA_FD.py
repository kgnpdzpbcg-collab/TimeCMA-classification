"""Version 2 patch-level signal-text aligned TimeCMA fault diagnosis model."""

from __future__ import annotations

import torch
from torch import nn

from layers.Cross_Modal_Align import CrossModal
from layers.StandardNorm import Normalize


class TimeCMAFaultDiagnosis(nn.Module):
    def __init__(
        self,
        num_nodes: int = 1,
        seq_len: int = 1024,
        num_classes: int = 4,
        channel: int = 64,
        d_llm: int = 768,
        patch_len: int = 64,
        e_layer: int = 2,
        head: int = 8,
        d_ff: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError("seq_len must be divisible by patch_len")
        self.num_patches = seq_len // patch_len
        self.patch_len = patch_len
        self.normalize = Normalize(num_nodes, affine=False)

        self.patch_embedding = nn.Linear(patch_len, channel)
        self.ts_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(channel, head, batch_first=True, norm_first=True, dropout=dropout),
            num_layers=e_layer,
        )
        self.prompt_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_llm, head, batch_first=True, norm_first=True, dropout=dropout),
            num_layers=e_layer,
        )
        self.cross = CrossModal(
            d_model=self.num_patches,
            n_heads=1,
            d_ff=d_ff,
            norm="LayerNorm",
            attn_dropout=dropout,
            dropout=dropout,
            pre_norm=True,
            activation="gelu",
            res_attention=True,
            n_layers=1,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Dropout(dropout),
            nn.Linear(channel, num_classes),
        )

    def forward(self, signal: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        signal = self.normalize(signal.float(), "norm").squeeze(-1)
        patches = signal.reshape(signal.shape[0], self.num_patches, self.patch_len)
        ts_tokens = self.ts_encoder(self.patch_embedding(patches))

        # Version 2: embeddings are patch-level [B,E,P,1]
        prompt_tokens = embeddings.float().squeeze(-1).permute(0, 2, 1)
        prompt_tokens = self.prompt_encoder(prompt_tokens)

        aligned = self.cross(
            ts_tokens.permute(0, 2, 1),
            prompt_tokens.permute(0, 2, 1),
            prompt_tokens.permute(0, 2, 1),
        )
        aligned = aligned.permute(0, 2, 1)
        return self.classifier(aligned.mean(dim=1))
