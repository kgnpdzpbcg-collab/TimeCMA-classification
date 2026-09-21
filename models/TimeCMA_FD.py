"""基于 TimeCMA Cross-Modality Alignment 的 CWRU 故障分类模型。"""

from __future__ import annotations

import torch
from torch import nn

from layers.Cross_Modal_Align import CrossModal
from layers.StandardNorm import Normalize


class TimeCMAFaultDiagnosis(nn.Module):
    """保留 TimeCMA 双分支与 CMA，用分类头替换预测 decoder。"""

    def __init__(
        self,
        num_nodes: int = 1,
        seq_len: int = 1024,
        num_classes: int = 4,
        channel: int = 64,
        d_llm: int = 768,
        e_layer: int = 2,
        head: int = 8,
        d_ff: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if channel % head != 0:
            raise ValueError(f"channel={channel} 必须能被 head={head} 整除")
        self.normalize = Normalize(num_nodes, affine=False)
        # iTransformer 风格：每个传感器完整窗口作为一个 token。
        self.length_to_feature = nn.Linear(seq_len, channel)
        self.ts_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(channel, head, batch_first=True, norm_first=True, dropout=dropout),
            num_layers=e_layer,
        )
        self.prompt_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_llm, head, batch_first=True, norm_first=True, dropout=dropout),
            num_layers=e_layer,
        )
        # 与官方实现一致：传感器维度 N 是 CMA 的特征维度。
        self.cross = CrossModal(
            d_model=num_nodes,
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
        # 对传感器 token 做均值池化后分类；没有未来序列输出，因此不做 RevIN 反归一化。
        self.classifier = nn.Sequential(nn.LayerNorm(channel), nn.Dropout(dropout), nn.Linear(channel, num_classes))

    def forward(self, signal: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """返回分类 logits。

        ``signal`` 为 [B,L,N]，``embeddings`` 为缓存的 [B,E,N,1]。embedding 中不包含标签。
        """
        if signal.ndim != 3:
            raise ValueError(f"signal 应为 [B,L,N]，实际为 {tuple(signal.shape)}")
        if embeddings.ndim != 4:
            raise ValueError(f"embeddings 应为 [B,E,N,1]，实际为 {tuple(embeddings.shape)}")

        ts_tokens = self.normalize(signal.float(), "norm").permute(0, 2, 1)
        ts_tokens = self.ts_encoder(self.length_to_feature(ts_tokens))  # [B,N,C]

        prompt_tokens = embeddings.float().squeeze(-1).permute(0, 2, 1)  # [B,N,E]
        prompt_tokens = self.prompt_encoder(prompt_tokens)

        # CrossModal 以最后一维作为 d_model，故需要交换到 [B,C,N] 与 [B,E,N]。
        aligned = self.cross(ts_tokens.permute(0, 2, 1), prompt_tokens.permute(0, 2, 1), prompt_tokens.permute(0, 2, 1))
        aligned = aligned.permute(0, 2, 1)  # [B,N,C]
        pooled = aligned.mean(dim=1)
        return self.classifier(pooled)
