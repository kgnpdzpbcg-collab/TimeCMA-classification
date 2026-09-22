"""TimeCMA-FD：双模态模型及 Signal-only / Prompt-only 消融模型。"""

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
    """CWRU 故障诊断模型。

    ``ablation`` 只改变最终可见的模态，所有模式共用相同的窗口、patch 与训练流程：

    - ``dual``：原始 V4 的信号编码器、Prompt 编码器和跨模态注意力；
    - ``signal_only``：仅保留信号 token 与信号 Transformer，直接用信号 CLS 分类；
    - ``prompt_only``：仅保留 Prompt token 与 Prompt Transformer，直接用文本 CLS 分类。

    这样可以在不改数据划分、优化器或训练预算的前提下，分别测量两个输入源的独立能力。
    """

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
        ablation: str = "dual",
    ) -> None:
        super().__init__()
        if ablation not in {"dual", "signal_only", "prompt_only"}:
            raise ValueError(f"unsupported ablation mode: {ablation}")
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
        self.num_nodes = num_nodes
        self.d_llm = d_llm
        self.ablation = ablation

        # Signal-only 与 dual 共用同一套信号 token 化和 Transformer，确保信号分支的
        # 数据处理定义不因消融而变化。
        if ablation in {"dual", "signal_only"}:
            self.normalize = Normalize(num_nodes, affine=False)
            # 每个局部 token 同时保留 DE、FE 的同步采样点，故输入维为 patch_len × 传感器数。
            self.patch_embedding = nn.Linear(patch_len * num_nodes, channel)
            self.ts_cls_token = nn.Parameter(torch.empty(1, 1, channel))
            self.ts_position = nn.Parameter(torch.empty(1, self.num_patches + 1, channel))
            self.ts_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(channel, head, batch_first=True, norm_first=True, dropout=dropout),
                num_layers=e_layer,
            )

        # Prompt-only 与 dual 共用冻结 GPT-2 输出后的 Prompt Transformer；消融不改变
        # embedding 的来源和形状，只移除另一个模态。
        if ablation in {"dual", "prompt_only"}:
            self.prompt_cls_token = nn.Parameter(torch.empty(1, 1, d_llm))
            self.prompt_position = nn.Parameter(torch.empty(1, self.num_patches + 1, d_llm))
            self.prompt_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_llm, head, batch_first=True, norm_first=True, dropout=dropout),
                num_layers=e_layer,
            )

        # 原始 V4 的双模态路径保持结构和参数名不变，历史 V4 checkpoint 可以严格加载。
        if ablation == "dual":
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
        elif ablation == "signal_only":
            # 只对信号 CLS 分类，不保留未使用的 Prompt 参数，避免将无效容量算入基线。
            self.classifier = nn.Sequential(
                nn.LayerNorm(channel),
                nn.Dropout(dropout),
                nn.Linear(channel, num_classes),
            )
        else:
            # Prompt-only 直接检验英文局部统计描述的可分类信息，不借助任何波形 token。
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_llm),
                nn.Dropout(dropout),
                nn.Linear(d_llm, num_classes),
            )

        self._init_token_parameters()

    def _init_token_parameters(self) -> None:
        """以小方差初始化当前模式实际存在的 CLS 与位置参数。"""
        token_parameters = []
        for name in ("ts_cls_token", "ts_position", "prompt_cls_token", "prompt_position"):
            parameter = getattr(self, name, None)
            if parameter is not None:
                token_parameters.append(parameter)
        for parameter in token_parameters:
            nn.init.normal_(parameter, mean=0.0, std=0.02)

    def _encode_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """将原始 DE/FE 窗口编码为含 CLS 的信号 token。"""
        expected_length = self.num_patches * self.patch_stride + self.patch_len - self.patch_stride
        if signal.ndim != 3 or signal.shape[1] != expected_length or signal.shape[2] != self.num_nodes:
            raise ValueError(f"signal must have shape [B, {expected_length}, {self.num_nodes}]")
        signal = self.normalize(signal.float(), "norm").squeeze(-1)
        patches = signal.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        patches = patches.permute(0, 1, 3, 2).reshape(signal.shape[0], self.num_patches, -1)
        ts_tokens = self.patch_embedding(patches)
        ts_cls = self.ts_cls_token.expand(signal.shape[0], -1, -1)
        ts_tokens = torch.cat((ts_cls, ts_tokens), dim=1) + self.ts_position
        return self.ts_encoder(ts_tokens)

    def _encode_prompt(self, embeddings: torch.Tensor | None) -> torch.Tensor:
        """将缓存的 patch prompt embedding 编码为含 CLS 的文本 token。"""
        if embeddings is None:
            raise ValueError("prompt_only and dual modes require prompt embeddings")
        if embeddings.ndim != 4 or embeddings.shape[1] != self.d_llm or embeddings.shape[2] != self.num_patches or embeddings.shape[3] != 1:
            expected = f"[B, {self.d_llm}, {self.num_patches}, 1]"
            raise ValueError(f"V3 embedding shape mismatch: expected {expected}, got {tuple(embeddings.shape)}")
        prompt_tokens = embeddings.float().squeeze(-1).permute(0, 2, 1)
        prompt_cls = self.prompt_cls_token.expand(prompt_tokens.shape[0], -1, -1)
        prompt_tokens = torch.cat((prompt_cls, prompt_tokens), dim=1) + self.prompt_position
        return self.prompt_encoder(prompt_tokens)

    def forward(self, signal: torch.Tensor, embeddings: torch.Tensor | None = None) -> torch.Tensor:
        """根据消融模式输出四类故障 logits。"""
        if self.ablation == "signal_only":
            ts_tokens = self._encode_signal(signal)
            return self.classifier(ts_tokens[:, 0, :])
        if self.ablation == "prompt_only":
            prompt_tokens = self._encode_prompt(embeddings)
            return self.classifier(prompt_tokens[:, 0, :])

        ts_tokens = self._encode_signal(signal)
        prompt_tokens = self._encode_prompt(embeddings)
        # Q、K、V 的序列长度都是 CLS+7 个 token；CLS 通过 cross-attention 汇集局部文本 token。
        aligned = self.cross(ts_tokens, prompt_tokens)
        return self.classifier(aligned[:, 0, :])
