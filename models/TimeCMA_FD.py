"""Version 3：重叠 patch、位置编码、CLS 聚合与 token-level CMA 的故障诊断模型。"""

from __future__ import annotations

import torch
from torch import nn

from layers.StandardNorm import Normalize


class TokenCrossAttention(nn.Module):
    """以信号 token 查询文本 token 的真正 token-level 跨模态注意力。

    输入的序列维始终是 ``[CLS, patch_1, ..., patch_P]``。先把 64 维信号和
    768 维文本投影到同一 ``align_dim``，再在序列维计算注意力；因此 attention
    权重的形状为 ``[B, heads, P+1, P+1]``，不再把 patch 数错误地作为特征维。
    """

    def __init__(self, signal_dim: int, prompt_dim: int, align_dim: int, heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        if align_dim % heads != 0:
            raise ValueError("align_dim must be divisible by cross-attention heads")
        self.signal_projection = nn.Linear(signal_dim, align_dim)
        self.prompt_projection = nn.Sequential(nn.LayerNorm(prompt_dim), nn.Linear(prompt_dim, align_dim))
        self.attention = nn.MultiheadAttention(align_dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm_attention = nn.LayerNorm(align_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(align_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, align_dim),
        )
        self.norm_feed_forward = nn.LayerNorm(align_dim)

    def forward(self, signal_tokens: torch.Tensor, prompt_tokens: torch.Tensor) -> torch.Tensor:
        """返回与信号 token 一一对应的融合 token，形状为 ``[B, P+1, align_dim]``。"""
        query = self.signal_projection(signal_tokens)
        key_value = self.prompt_projection(prompt_tokens)
        attended, _ = self.attention(query, key_value, key_value, need_weights=False)
        fused = self.norm_attention(query + self.dropout(attended))
        return self.norm_feed_forward(fused + self.dropout(self.feed_forward(fused)))


class TimeCMAFaultDiagnosis(nn.Module):
    def __init__(
        self,
        num_nodes: int = 1,
        seq_len: int = 1024,
        num_classes: int = 4,
        channel: int = 64,
        d_llm: int = 768,
        patch_len: int = 256,
        patch_stride: int = 128,
        e_layer: int = 2,
        head: int = 8,
        align_dim: int = 128,
        d_ff: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if patch_len <= 0 or patch_stride <= 0:
            raise ValueError("patch_len and patch_stride must be positive")
        if seq_len < patch_len:
            raise ValueError("seq_len must be no smaller than patch_len")
        if (seq_len - patch_len) % patch_stride != 0:
            raise ValueError("the configured patch stride leaves an uncovered signal tail")

        # V3-A 使用长度 256、步长 128 的重叠片段：1024 点窗口恰好得到 7 个 token。
        # 明确保存 token 数，避免模型、embedding 缓存与 CMA 的维度各自推导而发生错配。
        self.num_patches = 1 + (seq_len - patch_len) // patch_stride
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.normalize = Normalize(num_nodes, affine=False)

        # 每个局部 token 同时保留 DE、FE 的同步采样点，故输入维为 patch_len × 传感器数。
        self.num_nodes = num_nodes
        self.patch_embedding = nn.Linear(patch_len * num_nodes, channel)
        # TransformerEncoder 本身不添加时序位置。两个模态分别加入可学习位置编码，
        # 使“冲击出现在哪个局部片段”成为模型可利用的信息。
        self.ts_cls_token = nn.Parameter(torch.empty(1, 1, channel))
        self.ts_position = nn.Parameter(torch.empty(1, self.num_patches + 1, channel))
        self.prompt_cls_token = nn.Parameter(torch.empty(1, 1, d_llm))
        self.prompt_position = nn.Parameter(torch.empty(1, self.num_patches + 1, d_llm))
        self.ts_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(channel, head, batch_first=True, norm_first=True, dropout=dropout),
            num_layers=e_layer,
        )
        self.prompt_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_llm, head, batch_first=True, norm_first=True, dropout=dropout),
            num_layers=e_layer,
        )
        self.cross = TokenCrossAttention(
            signal_dim=channel,
            prompt_dim=d_llm,
            align_dim=align_dim,
            heads=head,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(align_dim),
            nn.Dropout(dropout),
            nn.Linear(align_dim, num_classes),
        )
        self._init_token_parameters()

    def _init_token_parameters(self) -> None:
        """以小方差初始化新增的 CLS 与位置参数，避免训练开始时压过信号内容。"""
        for parameter in (self.ts_cls_token, self.ts_position, self.prompt_cls_token, self.prompt_position):
            nn.init.normal_(parameter, mean=0.0, std=0.02)

    def forward(self, signal: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        expected_length = self.num_patches * self.patch_stride + self.patch_len - self.patch_stride
        if signal.ndim != 3 or signal.shape[1] != expected_length or signal.shape[2] != self.num_nodes:
            raise ValueError(f"signal must have shape [B, {expected_length}, {self.num_nodes}]")
        if embeddings.ndim != 4 or embeddings.shape[1] != self.prompt_position.shape[-1] or embeddings.shape[2] != self.num_patches or embeddings.shape[3] != 1:
            expected = f"[B, {self.prompt_position.shape[-1]}, {self.num_patches}, 1]"
            raise ValueError(f"V3 embedding shape mismatch: expected {expected}, got {tuple(embeddings.shape)}")

        signal = self.normalize(signal.float(), "norm").squeeze(-1)
        # unfold 后为 [B, 7, 2, 256]。调整并拼接 DE、FE，得到每个 patch 一个 token 的 [B, 7, 512]。
        patches = signal.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        patches = patches.permute(0, 1, 3, 2).reshape(signal.shape[0], self.num_patches, -1)
        ts_tokens = self.patch_embedding(patches)
        ts_cls = self.ts_cls_token.expand(signal.shape[0], -1, -1)
        ts_tokens = torch.cat((ts_cls, ts_tokens), dim=1) + self.ts_position
        ts_tokens = self.ts_encoder(ts_tokens)

        # V3 embedding 仍按每个 patch 一条 prompt 存储为 [B, E, P, 1]。
        prompt_tokens = embeddings.float().squeeze(-1).permute(0, 2, 1)
        prompt_cls = self.prompt_cls_token.expand(signal.shape[0], -1, -1)
        prompt_tokens = torch.cat((prompt_cls, prompt_tokens), dim=1) + self.prompt_position
        prompt_tokens = self.prompt_encoder(prompt_tokens)

        # V3-B：Q、K、V 的序列长度都是 CLS+7 个 token；不再进行 [B,D,P] 的 permute。
        aligned = self.cross(ts_tokens, prompt_tokens)
        # CLS 已通过 cross-attention 汇集所有局部文本 token，替代 V2 的全 patch 平均池化。
        return self.classifier(aligned[:, 0, :])
