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

    def _build_prompt(self, patch, load_hp):
        s = self._statistics(patch, self.sampling_rate)
        return (
            f"A local vibration patch contains statistical features. "
            f"Load {load_hp} horsepower. Mean {s['mean']:.6g}, std {s['std']:.6g}, "
            f"RMS {s['rms']:.6g}, peak {s['peak']:.6g}, skewness {s['skewness']:.6g}, "
            f"kurtosis {s['kurtosis']:.6g}, crest factor {s['crest']:.6g}. "
            f"Dominant frequency {s['dominant_frequency']:.3f}"
        )

    @torch.inference_mode()
    def patch_forward(self, signals, loads_hp, patch_len=256, patch_stride=128):
        """为重叠信号 patch 生成 ``[B, E, P, 1]`` 的冻结文本 embedding。

        ``patch_stride`` 仅控制模型内部的局部 token，不改变 CWRU 数据集的 1024 点样本
        窗口及其文件级划分。对 V3-A 的 1024/256/128 配置，P 固定为 7。
        """
        b, length, _ = signals.shape
        if patch_len <= 0 or patch_stride <= 0 or length < patch_len:
            raise ValueError("invalid patch_len or patch_stride")
        if (length - patch_len) % patch_stride != 0:
            raise ValueError("the configured patch stride leaves an uncovered signal tail")
        patches = signals.squeeze(-1).unfold(dimension=1, size=patch_len, step=patch_stride)
        prompts = []
        for i in range(b):
            for j in range(patches.shape[1]):
                prompts.append(self._build_prompt(patches[i, j], int(loads_hp[i])))
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        hidden = self.model(**encoded).last_hidden_state
        pos = encoded["attention_mask"].sum(dim=1).sub(1)
        emb = hidden[torch.arange(hidden.size(0), device=self.device), pos]
        return emb.reshape(b, patches.shape[1], -1).permute(0, 2, 1).unsqueeze(-1)

    @torch.inference_mode()
    def forward(self, signals, loads_hp):
        """Backward compatible global prompt embedding."""
        b, _, n = signals.shape
        prompts = [self._build_prompt(signals[i, :, 0], int(loads_hp[i])) for i in range(b)]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        hidden = self.model(**encoded).last_hidden_state
        pos = encoded["attention_mask"].sum(dim=1).sub(1)
        emb = hidden[torch.arange(hidden.size(0), device=self.device), pos]
        return emb.unsqueeze(-1).unsqueeze(-1)
