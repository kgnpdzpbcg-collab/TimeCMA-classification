"""V9：DE/FE CNN 信号表示与四个冻结故障机理 Prototype 的余弦匹配。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.CNN_FD import CWRUCNN


class PrototypeCNNFaultDiagnosisV9(nn.Module):
    """仅保留信号与类别知识两条分支，不含样本 Evidence 或 Cross Attention。

    ``prototypes`` 来自冻结 GPT-2，形状为 ``[4, 768]``，作为 buffer 保存；
    每个窗口的 DE/FE 波形经 CNN 形成一个全局向量，再投影并与四类向量比较。
    最终得分必须经过 Prototype，不设置可绕过文本知识的独立分类头。
    """

    def __init__(
        self, prototypes: torch.Tensor, num_nodes: int = 2, seq_len: int = 1024,
        base_channels: int = 64, dropout: float = 0.2, temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if tuple(prototypes.shape) != (4, 768) or not torch.isfinite(prototypes).all():
            raise ValueError("四类固定 Prototype 应为有限的 [4, 768] 张量")
        if temperature <= 0:
            raise ValueError("余弦温度 temperature 必须大于 0")

        # 复用 CNN 基线的全部卷积和池化，仅去掉其普通四类分类头。
        self.signal_encoder = CWRUCNN(
            num_nodes=num_nodes, seq_len=seq_len, num_classes=4,
            base_channels=base_channels, dropout=dropout, feature_only=True,
        )
        self.signal_projection = nn.Sequential(
            nn.LayerNorm(self.signal_encoder.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.signal_encoder.feature_dim, 768),
        )
        self.register_buffer("prototypes", F.normalize(prototypes.float().clone(), dim=-1))
        self.temperature = temperature

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """返回 ``[B, 4]`` logits；四列顺序为 Normal、Ball、Inner、Outer。"""
        signal_vector = self.signal_projection(self.signal_encoder(signal))
        normalized_signal = F.normalize(signal_vector, dim=-1)
        return normalized_signal @ self.prototypes.T / self.temperature
