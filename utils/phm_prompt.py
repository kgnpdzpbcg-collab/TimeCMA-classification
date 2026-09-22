"""冻结 GPT-2 的 PHM 局部统计 prompt embedding 生成器（V3-A）。"""

from __future__ import annotations

import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer


class PHMPromptEmbedder(nn.Module):
    def __init__(self, model_name: str = "gpt2", device: torch.device | str = "cpu", sampling_rate: int = 12000):
        super().__init__()
        self.device = torch.device(device)
        self.sampling_rate = sampling_rate
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _statistics(signal, sampling_rate):
        x = signal.float()
        mean = x.mean()
        centered = x - mean
        std = centered.std(unbiased=False).clamp_min(1e-8)
        rms = torch.sqrt(torch.mean(x.square()))
        peak = x.abs().max()
        skew = torch.mean((centered / std).pow(3))
        kurt = torch.mean((centered / std).pow(4))
        crest = peak / rms.clamp_min(1e-8)
        spectrum = torch.fft.rfft(centered)
        idx = int(torch.argmax(spectrum.abs()[1:]).item()) + 1 if spectrum.numel() > 1 else 0
        return {
            "mean": mean.item(), "std": std.item(), "rms": rms.item(),
            "peak": peak.item(), "skewness": skew.item(),
            "kurtosis": kurt.item(), "crest": crest.item(),
            "dominant_frequency": idx * sampling_rate / x.numel(),
        }

    def _build_prompt(self, patch, load_hp, include_load_hp=True):
        """将一个同步 DE/FE 局部片段转为文本，不输入故障类别标签。

        ``include_load_hp`` 区分条件化跨工况实验与纯信号跨工况消融。该开关属于
        embedding 配方的一部分，调用方必须为两种配置使用不同的缓存目录。
        """
        if patch.ndim != 2 or patch.shape[1] != 2:
            raise ValueError(f"V4 prompt patch 必须为 [长度, 2] 的 DE/FE 数据，实际为 {tuple(patch.shape)}")
        de_stats = self._statistics(patch[:, 0], self.sampling_rate)
        fe_stats = self._statistics(patch[:, 1], self.sampling_rate)

        def describe(sensor, stats):
            return (
                f"{sensor}: mean {stats['mean']:.6g}, std {stats['std']:.6g}, "
                f"RMS {stats['rms']:.6g}, peak {stats['peak']:.6g}, "
                f"skewness {stats['skewness']:.6g}, kurtosis {stats['kurtosis']:.6g}, "
                f"crest factor {stats['crest']:.6g}, dominant frequency {stats['dominant_frequency']:.3f}. "
            )

        prefix = "A local synchronized bearing vibration patch. "
        if include_load_hp:
            prefix += f"Load {load_hp} horsepower. "
        return prefix + describe("Drive-end sensor", de_stats) + describe("Fan-end sensor", fe_stats)

    @torch.inference_mode()
    def patch_forward(self, signals, loads_hp, patch_len=256, patch_stride=128, include_load_hp=True):
        """为重叠信号 patch 生成 ``[B, E, P, 1]`` 的冻结文本 embedding。

        ``patch_stride`` 仅控制模型内部的局部 token，不改变 CWRU 数据集的 1024 点样本
        窗口及其文件级划分。对 V3-A 的 1024/256/128 配置，P 固定为 7。
        """
        b, length, _ = signals.shape
        if patch_len <= 0 or patch_stride <= 0 or length < patch_len:
            raise ValueError("invalid patch_len or patch_stride")
        if (length - patch_len) % patch_stride != 0:
            raise ValueError("the configured patch stride leaves an uncovered signal tail")
        if signals.ndim != 3 or signals.shape[2] != 2:
            raise ValueError(f"V4 signals 必须为 [B, 长度, 2] 的 DE/FE 数据，实际为 {tuple(signals.shape)}")
        # unfold 输出 [B, P, 2, patch_len]，换轴后按 DE、FE 计算每个局部片段的统计量。
        patches = signals.unfold(dimension=1, size=patch_len, step=patch_stride).permute(0, 1, 3, 2)
        prompts = []
        for i in range(b):
            for j in range(patches.shape[1]):
                prompts.append(self._build_prompt(patches[i, j], int(loads_hp[i]), include_load_hp))
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        hidden = self.model(**encoded).last_hidden_state
        pos = encoded["attention_mask"].sum(dim=1).sub(1)
        emb = hidden[torch.arange(hidden.size(0), device=self.device), pos]
        return emb.reshape(b, patches.shape[1], -1).permute(0, 2, 1).unsqueeze(-1)

    @torch.inference_mode()
    def forward(self, signals, loads_hp, include_load_hp=True):
        """Backward compatible global prompt embedding."""
        b, _, n = signals.shape
        if n != 2:
            raise ValueError(f"V4 signals 必须含 DE、FE 两个通道，实际通道数为 {n}")
        prompts = [self._build_prompt(signals[i], int(loads_hp[i]), include_load_hp) for i in range(b)]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        hidden = self.model(**encoded).last_hidden_state
        pos = encoded["attention_mask"].sum(dim=1).sub(1)
        emb = hidden[torch.arange(hidden.size(0), device=self.device), pos]
        return emb.unsqueeze(-1).unsqueeze(-1)
