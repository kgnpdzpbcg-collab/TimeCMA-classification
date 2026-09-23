"""CWRU 故障诊断的纯信号 1D CNN 基线，不读取文本或故障原型。"""

from __future__ import annotations

import torch
from torch import nn

from layers.StandardNorm import Normalize


class CWRUCNN(nn.Module):
    """对同步 DE/FE 振动窗口编码，默认再进行四类分类。

    输入为 ``[B, 1024, 2]``。默认输出 ``[B, 4]`` 的分类 logits；
    ``feature_only=True`` 时输出全局信号特征，供 V9 的语义投影使用。
    与 V8 信号分支一样，先在每个窗口内分别标准化 DE、FE；后续卷积、
    时序池化和分类头都只接收波形，不接收负载、文件名或 GPT-2 缓存。
    """

    def __init__(
        self, num_nodes: int = 2, seq_len: int = 1024, num_classes: int = 4,
        base_channels: int = 64, dropout: float = 0.2, feature_only: bool = False,
    ) -> None:
        super().__init__()
        if num_nodes <= 0 or seq_len < 16 or num_classes <= 1 or base_channels <= 0:
            raise ValueError("传感器数、窗口长度、类别数或卷积通道数不合法")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须位于 [0, 1) 区间")

        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.feature_only = feature_only
        self.normalize = Normalize(num_nodes, affine=False)

        # 四层卷积分别提取局部冲击及其更长时间尺度的组合；每层只下采样一半，
        # 对 1024 点窗口最终保留 64 个时序位置，再由全局池化汇总。
        widths = (base_channels, base_channels * 2, base_channels * 4, base_channels * 4)
        kernels = (7, 5, 5, 3)
        blocks: list[nn.Module] = []
        in_channels = num_nodes
        for out_channels, kernel_size in zip(widths, kernels):
            blocks.extend((
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.maximum_pool = nn.AdaptiveMaxPool1d(1)
        self.feature_dim = widths[-1] * 2
        # 默认分类路径的参数名称保持不变，已有 CNN checkpoint 仍可直接加载。
        self.classifier = (
            nn.Identity() if feature_only
            else nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, num_classes))
        )

    def encode(self, signal: torch.Tensor) -> torch.Tensor:
        """提取全局特征；平均池化概括持续振动，最大池化保留强局部响应。"""
        if signal.ndim != 3 or tuple(signal.shape[1:]) != (self.seq_len, self.num_nodes):
            raise ValueError(f"Signal 应为 [B, {self.seq_len}, {self.num_nodes}]")
        normalized = self.normalize(signal.float(), "norm")
        features = self.features(normalized.transpose(1, 2))
        summary = torch.cat((self.average_pool(features), self.maximum_pool(features)), dim=1)
        return summary.squeeze(-1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """默认返回分类 logits；特征模式直接返回全局信号向量。"""
        return self.classifier(self.encode(signal))
