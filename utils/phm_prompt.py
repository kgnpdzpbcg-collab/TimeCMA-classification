"""为 PHM 振动窗口生成无标签泄漏的冻结 GPT-2 last-token embedding。"""

from __future__ import annotations

import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer


class PHMPromptEmbedder(nn.Module):
    """将窗口统计特征和已知工况转成 prompt，并返回每个通道的 GPT-2 最后 token 表示。

    不把故障类别、故障尺寸或人工诊断结果写进 prompt；这些信息在真实推理中不可得，
    写入会导致标签泄漏。完整 1024 点波形也不被序列化，以避免超过 GPT-2 上下文限制。
    """

    def __init__(self, model_name: str = "gpt2", device: torch.device | str = "cpu", sampling_rate: int = 12000) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.sampling_rate = sampling_rate
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)
        # 冻结且 eval，确保缓存可复现，并禁止 dropout 让同一窗口产生不同 embedding。
        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _statistics(signal: torch.Tensor, sampling_rate: int) -> dict[str, float]:
        """计算 prompt 所需的时域与主频统计量，输入为单个一维信号窗口。"""
        values = signal.float()
        mean = values.mean()
        centered = values - mean
        std = centered.std(unbiased=False).clamp_min(1e-8)
        rms = torch.sqrt(torch.mean(values.square()))
        peak = values.abs().max()
        skewness = torch.mean((centered / std).pow(3))
        kurtosis = torch.mean((centered / std).pow(4))
        crest = peak / rms.clamp_min(1e-8)
        spectrum = torch.fft.rfft(centered)
        # 忽略直流分量，避免均值残差被误判为故障频率。
        dominant_bin = int(torch.argmax(spectrum.abs()[1:]).item()) + 1 if spectrum.numel() > 1 else 0
        dominant_frequency = dominant_bin * sampling_rate / values.numel()
        return {
            "mean": mean.item(), "std": std.item(), "rms": rms.item(), "peak": peak.item(),
            "skewness": skewness.item(), "kurtosis": kurtosis.item(), "crest": crest.item(),
            "dominant_frequency": dominant_frequency,
        }

    def _build_prompt(self, signal: torch.Tensor, load_hp: int, sensor_name: str = "drive-end") -> str:
        stats = self._statistics(signal, self.sampling_rate)
        # 结尾保留数值，使最后 token 处于主要信号摘要之后，沿用 TimeCMA 的设计动机。
        return (
            f"A {signal.numel()} point vibration segment was collected from the {sensor_name} bearing sensor "
            f"at {self.sampling_rate} Hz under {load_hp} horsepower load. "
            f"Mean {stats['mean']:.6g}, standard deviation {stats['std']:.6g}, RMS {stats['rms']:.6g}, "
            f"absolute peak {stats['peak']:.6g}, skewness {stats['skewness']:.6g}, "
            f"kurtosis {stats['kurtosis']:.6g}, crest factor {stats['crest']:.6g}. "
            f"Dominant frequency {stats['dominant_frequency']:.3f}"
        )

    @torch.inference_mode()
    def forward(self, signals: torch.Tensor, loads_hp: torch.Tensor) -> torch.Tensor:
        """生成形状 [B,E,N,1] 的缓存 embedding，和原 CMA 的输入约定完全一致。"""
        if signals.ndim != 3:
            raise ValueError(f"signals 应为 [B,L,N]，实际为 {tuple(signals.shape)}")
        batch_size, _, num_nodes = signals.shape
        outputs: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            channel_embeddings: list[torch.Tensor] = []
            for channel_index in range(num_nodes):
                prompt = self._build_prompt(signals[batch_index, :, channel_index], int(loads_hp[batch_index]))
                encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.model.config.n_positions)
                encoded = {name: value.to(self.device) for name, value in encoded.items()}
                last_token = self.model(**encoded).last_hidden_state[:, -1, :].squeeze(0)
                channel_embeddings.append(last_token)
            outputs.append(torch.stack(channel_embeddings, dim=1))  # [E,N]
        return torch.stack(outputs, dim=0).unsqueeze(-1)  # [B,E,N,1]
