"""V6：冻结 GPT-2 的 PHM 证据型 Prompt embedding 生成器。

V6-A 只修改 Prompt 的信息组织，不改变 TimeCMA-FD 的模型结构、token 数、GPT-2
池化位置、Prompt Encoder、Cross-Attention 或 Signal branch。每个 patch Prompt
由三层信息组成：

1. Local evidence：当前 256 点 DE/FE patch 的局部统计与跨传感器关系；
2. Global evidence：同一 1024 点窗口共享的全局统计、频谱/包络证据与跨传感器关系；
3. Mechanism knowledge：所有样本完全相同的四类轴承状态机理知识。

固定机理知识同时包含 Normal / Ball / Inner / Outer，绝不根据真实标签选择文本，
因此不会把类别答案写入 Prompt。
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer


# 所有样本、所有 patch 共用同一份机理知识。这里故意不写任何样本级判断结果，
# 只提供“证据应如何与机械概念建立关系”的通用先验。
BEARING_MECHANISM_KNOWLEDGE = (
    "General bearing-fault mechanism knowledge shared by every sample: "
    "Healthy bearings usually do not exhibit persistent defect-related periodic impacts or stable "
    "fault-characteristic harmonic patterns. "
    "An outer-race defect is stationary relative to the housing; rolling elements repeatedly pass "
    "the defect and can generate periodic impacts associated with BPFO-related components and harmonics. "
    "An inner-race defect rotates with the shaft; repeated contacts can generate BPFI-related components "
    "and harmonics, and their amplitudes may be modulated by shaft rotation as the defect moves through "
    "the load zone. "
    "A rolling-element defect can generate BSF-related responses; because the damaged element contacts "
    "both races while rotating and orbiting, its impulsive and modulation patterns can be less stable and "
    "may contain cage- or shaft-related modulation. "
    "Fault characteristic frequencies depend on shaft speed and bearing geometry, so a single maximum "
    "peak in the raw FFT must not by itself be interpreted as BPFO, BPFI, or BSF evidence."
)


class PHMPromptEmbedder(nn.Module):
    def __init__(
        self,
        model_name: str = "gpt2",
        device: torch.device | str = "cpu",
        sampling_rate: int = 12000,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.sampling_rate = sampling_rate
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _statistics(signal: torch.Tensor, sampling_rate: int) -> dict[str, float]:
        """计算局部/全局都会使用的基础时域与原始频谱统计量。"""
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
        magnitude = spectrum.abs()
        if magnitude.numel() > 1:
            idx = int(torch.argmax(magnitude[1:]).item()) + 1
        else:
            idx = 0
        dominant_frequency = idx * sampling_rate / x.numel()

        frequencies = torch.fft.rfftfreq(x.numel(), d=1.0 / sampling_rate).to(x.device)
        spectral_mass = magnitude.sum().clamp_min(1e-8)
        spectral_centroid = torch.sum(frequencies * magnitude) / spectral_mass

        return {
            "mean": mean.item(),
            "std": std.item(),
            "rms": rms.item(),
            "peak": peak.item(),
            "skewness": skew.item(),
            "kurtosis": kurt.item(),
            "crest": crest.item(),
            "dominant_frequency": float(dominant_frequency),
            "spectral_centroid": spectral_centroid.item(),
        }

    @staticmethod
    def _envelope_dominant_frequency(signal: torch.Tensor, sampling_rate: int) -> float:
        """用 FFT 形式的 Hilbert 解析信号估计包络谱最大非直流峰。

        该量只在完整 1024 点窗口上使用；它仍只是观测证据，不被命名为 BPFO/BPFI/BSF。
        """
        x = signal.float()
        centered = x - x.mean()
        n = centered.numel()
        spectrum = torch.fft.fft(centered)
        hilbert_filter = torch.zeros(n, dtype=centered.dtype, device=centered.device)
        if n % 2 == 0:
            hilbert_filter[0] = 1.0
            hilbert_filter[n // 2] = 1.0
            hilbert_filter[1:n // 2] = 2.0
        else:
            hilbert_filter[0] = 1.0
            hilbert_filter[1:(n + 1) // 2] = 2.0
        analytic = torch.fft.ifft(spectrum * hilbert_filter)
        envelope = analytic.abs()
        envelope = envelope - envelope.mean()
        envelope_spectrum = torch.fft.rfft(envelope).abs()
        if envelope_spectrum.numel() <= 1:
            return 0.0
        idx = int(torch.argmax(envelope_spectrum[1:]).item()) + 1
        return float(idx * sampling_rate / n)

    @staticmethod
    def _cross_sensor_evidence(de_signal: torch.Tensor, fe_signal: torch.Tensor) -> dict[str, float]:
        """计算同一时间范围内 DE/FE 的相对响应，避免只罗列两个独立传感器。"""
        de = de_signal.float()
        fe = fe_signal.float()
        de_centered = de - de.mean()
        fe_centered = fe - fe.mean()
        de_std = de_centered.std(unbiased=False).clamp_min(1e-8)
        fe_std = fe_centered.std(unbiased=False).clamp_min(1e-8)
        de_rms = torch.sqrt(torch.mean(de.square())).clamp_min(1e-8)
        fe_rms = torch.sqrt(torch.mean(fe.square())).clamp_min(1e-8)
        de_peak = de.abs().max().clamp_min(1e-8)
        fe_peak = fe.abs().max().clamp_min(1e-8)
        correlation = torch.mean(de_centered * fe_centered) / (de_std * fe_std)

        return {
            "rms_ratio_de_to_fe": (de_rms / fe_rms).item(),
            "peak_ratio_de_to_fe": (de_peak / fe_peak).item(),
            "correlation": correlation.clamp(-1.0, 1.0).item(),
        }

    def _window_evidence(self, window: torch.Tensor) -> dict[str, object]:
        """提取同一 1024 点样本中由 7 个 patch 共享的全局证据。"""
        if window.ndim != 2 or window.shape[1] != 2:
            raise ValueError(f"V6 window 必须为 [长度, 2] 的 DE/FE 数据，实际为 {tuple(window.shape)}")
        de = window[:, 0]
        fe = window[:, 1]
        de_stats = self._statistics(de, self.sampling_rate)
        fe_stats = self._statistics(fe, self.sampling_rate)
        return {
            "de": de_stats,
            "fe": fe_stats,
            "de_envelope_frequency": self._envelope_dominant_frequency(de, self.sampling_rate),
            "fe_envelope_frequency": self._envelope_dominant_frequency(fe, self.sampling_rate),
            "cross_sensor": self._cross_sensor_evidence(de, fe),
        }

    @staticmethod
    def _describe_local_sensor(sensor: str, stats: dict[str, float]) -> str:
        """Local Evidence 保留 V5 的基础统计量，便于与历史 Prompt 对比。"""
        return (
            f"{sensor}: mean {stats['mean']:.6g}, std {stats['std']:.6g}, "
            f"RMS {stats['rms']:.6g}, peak {stats['peak']:.6g}, "
            f"skewness {stats['skewness']:.6g}, kurtosis {stats['kurtosis']:.6g}, "
            f"crest factor {stats['crest']:.6g}, raw-FFT dominant frequency "
            f"{stats['dominant_frequency']:.3f} Hz. "
        )

    def _build_prompt(
        self,
        patch: torch.Tensor,
        window_evidence: dict[str, object],
        load_hp: int,
        patch_index: int,
        num_patches: int,
        include_load_hp: bool = True,
    ) -> str:
        """构造 V6 三层 Prompt：局部证据 + 窗口证据 + 固定故障机理。

        当前版本没有从 MAT 读取可靠 RPM，也没有在代码中硬编码轴承几何参数，因此不会
        伪造 order、BPFO、BPFI 或 BSF 数值。后续若引入 RPM/几何信息，应在新的 Prompt
        template version 中显式加入，而不是把 raw-FFT 最大峰冒充为故障特征频率。
        """
        if patch.ndim != 2 or patch.shape[1] != 2:
            raise ValueError(f"V6 prompt patch 必须为 [长度, 2] 的 DE/FE 数据，实际为 {tuple(patch.shape)}")

        de_stats = self._statistics(patch[:, 0], self.sampling_rate)
        fe_stats = self._statistics(patch[:, 1], self.sampling_rate)
        local_cross = self._cross_sensor_evidence(patch[:, 0], patch[:, 1])

        global_de = window_evidence["de"]
        global_fe = window_evidence["fe"]
        global_cross = window_evidence["cross_sensor"]

        condition = (
            f"Operating condition: load {load_hp} horsepower. "
            if include_load_hp
            else "Operating condition: load value is intentionally withheld for cross-condition evaluation. "
        )

        # 末尾仍保留 ". "，刻意不在 V6-A 同时修改 GPT-2 last-token pooling，
        # 使本轮实验只比较 Prompt 信息内容的变化。
        return (
            "Bearing vibration diagnostic evidence. "
            f"{condition}"
            f"Local evidence for temporal patch {patch_index + 1} of {num_patches}: "
            + self._describe_local_sensor("Drive-end sensor", de_stats)
            + self._describe_local_sensor("Fan-end sensor", fe_stats)
            + (
                "Local cross-sensor relation: "
                f"DE/FE RMS ratio {local_cross['rms_ratio_de_to_fe']:.6g}, "
                f"DE/FE peak ratio {local_cross['peak_ratio_de_to_fe']:.6g}, "
                f"DE-FE waveform correlation {local_cross['correlation']:.6g}. "
            )
            + (
                "Global evidence shared by all patches in this 1024-sample observation: "
                f"DE RMS {global_de['rms']:.6g}, DE kurtosis {global_de['kurtosis']:.6g}, "
                f"DE crest factor {global_de['crest']:.6g}, DE spectral centroid "
                f"{global_de['spectral_centroid']:.3f} Hz, DE raw-FFT dominant frequency "
                f"{global_de['dominant_frequency']:.3f} Hz, DE envelope-spectrum dominant frequency "
                f"{window_evidence['de_envelope_frequency']:.3f} Hz; "
                f"FE RMS {global_fe['rms']:.6g}, FE kurtosis {global_fe['kurtosis']:.6g}, "
                f"FE crest factor {global_fe['crest']:.6g}, FE spectral centroid "
                f"{global_fe['spectral_centroid']:.3f} Hz, FE raw-FFT dominant frequency "
                f"{global_fe['dominant_frequency']:.3f} Hz, FE envelope-spectrum dominant frequency "
                f"{window_evidence['fe_envelope_frequency']:.3f} Hz. "
                "Global cross-sensor relation: "
                f"DE/FE RMS ratio {global_cross['rms_ratio_de_to_fe']:.6g}, "
                f"DE/FE peak ratio {global_cross['peak_ratio_de_to_fe']:.6g}, "
                f"DE-FE waveform correlation {global_cross['correlation']:.6g}. "
            )
            + BEARING_MECHANISM_KNOWLEDGE
            + " "
        )

    @torch.inference_mode()
    def patch_forward(
        self,
        signals: torch.Tensor,
        loads_hp: torch.Tensor,
        patch_len: int = 256,
        patch_stride: int = 128,
        include_load_hp: bool = True,
    ) -> torch.Tensor:
        """为重叠 signal patch 生成 [B, E, P, 1] 的 V6 Prompt embedding。

        仍然保持 7 个 Prompt token：每个 token 对应一个局部 patch；Global Evidence 与
        Mechanism Knowledge 被重复写入每个 Prompt，以保持 V5/V6 的模型结构完全一致。
        """
        if signals.ndim != 3 or signals.shape[2] != 2:
            raise ValueError(f"V6 signals 必须为 [B, 长度, 2] 的 DE/FE 数据，实际为 {tuple(signals.shape)}")
        b, length, _ = signals.shape
        if patch_len <= 0 or patch_stride <= 0 or length < patch_len:
            raise ValueError("invalid patch_len or patch_stride")
        if (length - patch_len) % patch_stride != 0:
            raise ValueError("the configured patch stride leaves an uncovered signal tail")

        patches = signals.unfold(dimension=1, size=patch_len, step=patch_stride).permute(0, 1, 3, 2)
        num_patches = patches.shape[1]
        prompts: list[str] = []
        for i in range(b):
            global_evidence = self._window_evidence(signals[i])
            for j in range(num_patches):
                prompts.append(
                    self._build_prompt(
                        patches[i, j],
                        global_evidence,
                        int(loads_hp[i]),
                        patch_index=j,
                        num_patches=num_patches,
                        include_load_hp=include_load_hp,
                    )
                )

        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        hidden = self.model(**encoded).last_hidden_state

        # V6-A 有意保持 V5 的 last-valid-token 规则不变；已知模板以 ". " 结束时，
        # 该位置可能是 GPT-2 空格 token。后续 pooling 修复应作为独立版本评估。
        pos = encoded["attention_mask"].sum(dim=1).sub(1)
        emb = hidden[torch.arange(hidden.size(0), device=self.device), pos]
        return emb.reshape(b, num_patches, -1).permute(0, 2, 1).unsqueeze(-1)

    @torch.inference_mode()
    def forward(
        self,
        signals: torch.Tensor,
        loads_hp: torch.Tensor,
        include_load_hp: bool = True,
    ) -> torch.Tensor:
        """Backward-compatible global Prompt embedding；主要供旧调用路径使用。"""
        if signals.ndim != 3 or signals.shape[2] != 2:
            raise ValueError(f"V6 signals 必须含 DE、FE 两个通道，实际形状为 {tuple(signals.shape)}")
        b = signals.shape[0]
        prompts = []
        for i in range(b):
            global_evidence = self._window_evidence(signals[i])
            prompts.append(
                self._build_prompt(
                    signals[i],
                    global_evidence,
                    int(loads_hp[i]),
                    patch_index=0,
                    num_patches=1,
                    include_load_hp=include_load_hp,
                )
            )
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        hidden = self.model(**encoded).last_hidden_state
        pos = encoded["attention_mask"].sum(dim=1).sub(1)
        emb = hidden[torch.arange(hidden.size(0), device=self.device), pos]
        return emb.unsqueeze(-1).unsqueeze(-1)
