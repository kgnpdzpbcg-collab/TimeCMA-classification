"""Patch-token TimeCMA based CWRU fault diagnosis model (Version 1).

Replace the original iTransformer-style sensor token with local vibration patches.
The CMA module is retained, while signal tokens are changed from sensor-level
representation to temporal patch-level representation.
"""

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
            raise ValueError("seq_len 必须能够被 patch_len 整除")
        if channel % head != 0:
            raise ValueError("channel 必须能够被 head 整除")

        self.num_patches = seq_len // patch_len
        self.patch_len = patch_len

        self.normalize = Normalize(num_nodes, affine=False)

        # [B,L,1] -> [B,num_patches,patch_len]
        # Each local vibration segment is treated as a token.
        self.patch_embedding = nn.Linear(patch_len, channel)

        self.ts_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                channel,
                head,
                batch_first=True,
                norm_first=True,
                dropout=dropout,
            ),
            num_layers=e_layer,
        )

        self.prompt_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_llm,
                head,
                batch_first=True,
                norm_first=True,
                dropout=dropout,
            ),
            num_layers=e_layer,
        )

        # CMA feature dimension now corresponds to patch tokens instead of
        # physical sensor number. This keeps the original CrossModal design.
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
        if signal.ndim != 3:
            raise ValueError(f"signal 应为 [B,L,N]，实际为 {tuple(signal.shape)}")
        if embeddings.ndim != 4:
            raise ValueError(f"embeddings 应为 [B,E,N,1]，实际为 {tuple(embeddings.shape)}")

        # Signal branch
        signal = self.normalize(signal.float(), "norm")
        signal = signal.squeeze(-1)

        # [B,L] -> [B,num_patches,patch_len]
        patches = signal.reshape(signal.shape[0], self.num_patches, self.patch_len)
        ts_tokens = self.patch_embedding(patches)
        ts_tokens = self.ts_encoder(ts_tokens)  # [B,P,C]

        # Prompt branch. Version1 uses global prompt embedding shared by all patches.
        prompt = embeddings.float().squeeze(-1).squeeze(-1)  # [B,E]
        prompt_tokens = prompt.unsqueeze(1).repeat(1, self.num_patches, 1)
        prompt_tokens = self.prompt_encoder(prompt_tokens)  # [B,P,E]

        # CrossModal expects [B,feature_dim,token_num]
        aligned = self.cross(
            ts_tokens.permute(0, 2, 1),
            prompt_tokens.permute(0, 2, 1),
            prompt_tokens.permute(0, 2, 1),
        )

        aligned = aligned.permute(0, 2, 1)  # [B,P,C]
        pooled = aligned.mean(dim=1)
        return self.classifier(pooled)
