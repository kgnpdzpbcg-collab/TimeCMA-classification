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
        # GPT-2 原生没有 padding token；用 EOS 仅作为右侧填充，并由 attention_mask 屏蔽。
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
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
        prompts: list[tuple[int, int, str]] = []
        for batch_index in range(batch_size):
            for channel_index in range(num_nodes):
                prompt = self._build_prompt(signals[batch_index, :, channel_index], int(loads_hp[batch_index]))
                prompts.append((batch_index, channel_index, prompt))

        # 过去逐 prompt 调用 GPT-2 会让 CPU 预计算耗时极长；这里保持相同 prompt，
        # 仅将不同长度文本右侧补齐后合成一个 batch，并用 attention_mask 排除补齐 token。
        encoded = self.tokenizer(
            [prompt for _, _, prompt in prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.n_positions,
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        hidden_states = self.model(**encoded).last_hidden_state
        last_positions = encoded["attention_mask"].sum(dim=1).sub(1)
        last_tokens = hidden_states[torch.arange(hidden_states.size(0), device=self.device), last_positions]

        output = torch.empty(batch_size, hidden_states.size(-1), num_nodes, device=self.device)
        for prompt_index, (batch_index, channel_index, _prompt) in enumerate(prompts):
            output[batch_index, :, channel_index] = last_tokens[prompt_index]
        return output.unsqueeze(-1)  # [B,E,N,1]
