"""V8：固定故障知识原型、样本证据跨注意力及两者组合的三种诊断模式。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from layers.StandardNorm import Normalize


V8_MODES = ("prototype_only", "evidence_only", "prototype_evidence")


class EvidenceCrossAttention(nn.Module):
    """信号 token 查询实例证据 token；输出仍在信号语义投影后的维度。"""

    def __init__(self, signal_dim: int, evidence_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.evidence_projection = nn.Sequential(nn.LayerNorm(evidence_dim), nn.Linear(evidence_dim, signal_dim))
        self.attention = nn.MultiheadAttention(signal_dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm_attention = nn.LayerNorm(signal_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(signal_dim, signal_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(signal_dim * 2, signal_dim),
        )
        self.norm_feed_forward = nn.LayerNorm(signal_dim)

    def forward(self, signal_tokens: torch.Tensor, evidence_tokens: torch.Tensor) -> torch.Tensor:
        """返回 `[B, 8, signal_dim]`；信号 CLS 可读取全部七个 Evidence patch。"""
        key_value = self.evidence_projection(evidence_tokens)
        attended, _ = self.attention(signal_tokens, key_value, key_value, need_weights=False)
        fused = self.norm_attention(signal_tokens + self.dropout(attended))
        return self.norm_feed_forward(fused + self.dropout(self.feed_forward(fused)))


class TimeCMAFaultDiagnosisV8(nn.Module):
    """P、E、P+E 共用信号编码器与余弦分类形式。

    `prototype_only` 和 `prototype_evidence` 使用冻结的四类文本原型；
    `evidence_only` 使用形状相同的可训练类别向量，明确不输入类别机理文本。
    P+E 的融合前信号得分单独返回，供训练器加辅助对齐损失。
    """

    def __init__(
        self, *, mode: str, prototypes: torch.Tensor | None, num_nodes: int = 2,
        seq_len: int = 1024, num_classes: int = 4, channel: int = 64, d_llm: int = 768,
        patch_len: int = 256, patch_stride: int = 128, encoder_layers: int = 2,
        heads: int = 8, align_dim: int = 128, dropout: float = 0.2,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in V8_MODES:
            raise ValueError(f"未知 V8 模式: {mode}")
        if patch_len <= 0 or patch_stride <= 0 or seq_len < patch_len:
            raise ValueError("窗口及 patch 配置不合法")
        if (seq_len - patch_len) % patch_stride:
            raise ValueError("patch_stride 会使窗口尾部不能被完整覆盖")
        if align_dim % heads or channel % heads or d_llm % heads:
            raise ValueError("信号、证据及对齐维度必须均能被注意力头数整除")
        if temperature <= 0:
            raise ValueError("temperature 必须为正数")
        self.mode = mode
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.d_llm = d_llm
        self.num_patches = 1 + (seq_len - patch_len) // patch_stride
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.temperature = temperature

        # 三种模式使用完全相同的 DE/FE 标准化、局部 patch 与 Signal Transformer。
        self.normalize = Normalize(num_nodes, affine=False)
        self.patch_embedding = nn.Linear(patch_len * num_nodes, channel)
        self.signal_cls = nn.Parameter(torch.empty(1, 1, channel))
        self.signal_position = nn.Parameter(torch.empty(1, self.num_patches + 1, channel))
        self.signal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(channel, heads, batch_first=True, norm_first=True, dropout=dropout),
            num_layers=encoder_layers,
        )
        self.signal_projection = nn.Sequential(nn.LayerNorm(channel), nn.Linear(channel, align_dim))
        # 融合前和融合后共用此投影；不允许 P+E 用另一套映射绕过信号对齐目标。
        self.semantic_projection = nn.Sequential(nn.LayerNorm(align_dim), nn.Linear(align_dim, d_llm))

        if mode in {"evidence_only", "prototype_evidence"}:
            self.evidence_cls = nn.Parameter(torch.empty(1, 1, d_llm))
            self.evidence_position = nn.Parameter(torch.empty(1, self.num_patches + 1, d_llm))
            self.evidence_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_llm, heads, batch_first=True, norm_first=True, dropout=dropout),
                num_layers=encoder_layers,
            )
            self.cross = EvidenceCrossAttention(align_dim, d_llm, heads, dropout)

        if mode == "evidence_only":
            # E 没有故障文本原型，四个向量只是受监督学习的分类参数。
            self.class_vectors = nn.Parameter(torch.empty(num_classes, d_llm))
            nn.init.normal_(self.class_vectors, mean=0.0, std=0.02)
        else:
            if prototypes is None or tuple(prototypes.shape) != (num_classes, d_llm):
                raise ValueError(f"固定 Prototype 必须为 [{num_classes}, {d_llm}]")
            if not torch.isfinite(prototypes).all():
                raise ValueError("固定 Prototype 含非有限数值")
            # buffer 随 checkpoint 保存和设备移动，但不参与梯度更新。
            self.register_buffer("class_vectors", F.normalize(prototypes.float().clone(), dim=-1))

        nn.init.normal_(self.signal_cls, mean=0.0, std=0.02)
        nn.init.normal_(self.signal_position, mean=0.0, std=0.02)
        if mode in {"evidence_only", "prototype_evidence"}:
            nn.init.normal_(self.evidence_cls, mean=0.0, std=0.02)
            nn.init.normal_(self.evidence_position, mean=0.0, std=0.02)

    def _encode_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """将 `[B, 1024, 2]` 转为 `[B, 8, align_dim]`，首位为全窗 CLS。"""
        if signal.ndim != 3 or tuple(signal.shape[1:]) != (self.seq_len, self.num_nodes):
            raise ValueError(f"Signal 应为 [B, {self.seq_len}, {self.num_nodes}]")
        normalized = self.normalize(signal.float(), "norm").squeeze(-1)
        patches = normalized.unfold(1, self.patch_len, self.patch_stride)
        patches = patches.permute(0, 1, 3, 2).reshape(signal.shape[0], self.num_patches, -1)
        tokens = self.patch_embedding(patches)
        cls = self.signal_cls.expand(signal.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1) + self.signal_position
        return self.signal_projection(self.signal_encoder(tokens))

    def _encode_evidence(self, embeddings: torch.Tensor | None) -> torch.Tensor:
        """将七个冻结 GPT-2 Evidence embedding 编码为带 CLS 的 token 序列。"""
        expected = (self.d_llm, self.num_patches, 1)
        if embeddings is None or embeddings.ndim != 4 or tuple(embeddings.shape[1:]) != expected:
            raise ValueError(f"Evidence 应为 [B, {expected[0]}, {expected[1]}, 1]")
        tokens = embeddings.float().squeeze(-1).permute(0, 2, 1)
        cls = self.evidence_cls.expand(tokens.shape[0], -1, -1)
        return self.evidence_encoder(torch.cat((cls, tokens), dim=1) + self.evidence_position)

    def _scores(self, cls_token: torch.Tensor) -> torch.Tensor:
        """同一余弦分类公式用于固定文本原型或 E 模式的可训练类别向量。"""
        signal_semantic = F.normalize(self.semantic_projection(cls_token), dim=-1)
        class_vectors = F.normalize(self.class_vectors, dim=-1)
        return signal_semantic @ class_vectors.T / self.temperature

    def forward(
        self, signal: torch.Tensor, embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """返回最终四类 logits，以及 P+E 模式融合前的 Signal→Prototype logits。"""
        signal_tokens = self._encode_signal(signal)
        if self.mode == "prototype_only":
            return self._scores(signal_tokens[:, 0, :]), None
        evidence_tokens = self._encode_evidence(embeddings)
        fused_tokens = self.cross(signal_tokens, evidence_tokens)
        final_logits = self._scores(fused_tokens[:, 0, :])
        signal_logits = self._scores(signal_tokens[:, 0, :]) if self.mode == "prototype_evidence" else None
        return final_logits, signal_logits
