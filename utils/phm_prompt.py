"""V7：冻结 GPT-2 的 PHM 证据型 Prompt embedding 生成器。

V7 只修改 Prompt 的信息顺序与池化方式，不改变 TimeCMA-FD 的模型结构、token 数、
Prompt Encoder、Cross-Attention 或 Signal branch。每个 patch Prompt 由三层信息组成：

1. Mechanism knowledge：所有样本完全相同的四类轴承状态机理知识；
2. Local evidence：当前 256 点 DE/FE patch 的局部统计与跨传感器关系；
3. Global evidence：同一 1024 点窗口共享的全局统计、频谱/包络证据与跨传感器关系。

与 V6 的两点差别：

- 机理知识前置到证据之前，使因果注意力下的证据 token 能读到它；
- 池化改为只在两段证据 token 上取平均，把逐样本相同的 header、工况与机理知识
  排除在池化范围之外。

V6 的 last-token 池化取在 170 token 的固定机理知识之后，所有样本的池化向量因此
被拉向同一方向（实测样本间余弦 1.0000），prompt_only 分支塌陷。V7 保留机理先验，
但不再让它进入池化。

另外，池化层从最后一层改为中间层：实测类间/类内结构在最后一层会塌陷
（V6 缓存 0.287、V7 同层 0.345），而 L0–L10 稳定在 0.60–0.70。

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


# GPT-2 small 有 12 层 Transformer，hidden_states 含 embedding 层共 13 个。
# 实测最后一层的类间/类内结构在 V6/V7 的 Prompt 上都会塌陷，故默认取中间层。
DEFAULT_POOLING_LAYER = 10
NUM_HIDDEN_STATES = 13


class PHMPromptEmbedder(nn.Module):
    def __init__(
        self,
        model_name: str = "gpt2",
        device: torch.device | str = "cpu",
        sampling_rate: int = 12000,
        pooling_layer: int = DEFAULT_POOLING_LAYER,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.sampling_rate = sampling_rate
        if not 0 <= pooling_layer < NUM_HIDDEN_STATES:
            raise ValueError(
                f"pooling_layer 必须在 [0, {NUM_HIDDEN_STATES - 1}] 内，实际为 {pooling_layer}"
            )
        self.pooling_layer = pooling_layer
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
    ) -> tuple[str, int]:
        """构造 V7 三层 Prompt：固定故障机理 + 局部证据 + 窗口证据。

        返回 ``(prompt, evidence_start_char)``。``evidence_start_char`` 是两段证据的
        第一个字符在 ``prompt`` 中的下标，供 :meth:`_masked_evidence_pool` 圈定池化
        范围；它之前的 header、工况与机理知识都不参与池化。

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

        # 机理知识前置到证据之前：因果注意力下，后面的证据 token 因此能读到它。
        # 分隔符用换行而不是空格——GPT-2 的 BPE 不跨换行合并，证据段因此有精确的
        # token 边界，_masked_evidence_pool 可以直接用字符位置圈定池化范围。
        prefix = (
            "Bearing vibration diagnostic evidence. "
            f"{condition}"
            + BEARING_MECHANISM_KNOWLEDGE
            + "\n"
        )
        evidence = (
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
        )
        return prefix + evidence, len(prefix)

    def _select_hidden(self, encoded) -> torch.Tensor:
        """取 ``pooling_layer`` 层的隐状态。

        实测本版本 transformers 的 ``hidden_states[-1]`` 已经是 ``ln_f`` 之后的输出
        （与 ``last_hidden_state`` 逐元素相等），所以这里直接取用、不再叠加 ``ln_f``：
        额外一层 ln_f 并不是恒等变换，会因 GPT-2 学习到的 gamma/beta 二次放大
        （实测范数 930 → 1919）。不叠加才使 ``pooling_layer = 12`` 精确复现换层之前
        的输入，层与层之间也只差 Transformer 深度。
        """
        out = self.model(**encoded, output_hidden_states=True)
        return out.hidden_states[self.pooling_layer]

    @staticmethod
    def _masked_evidence_pool(
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        offsets: torch.Tensor,
        evidence_starts: torch.Tensor,
    ) -> torch.Tensor:
        """只在证据段的 token 上做平均池化，返回 ``[B, 隐藏维]``。

        header、工况与机理知识都落在 ``evidence_start_char`` 之前，因此被排除在池化
        范围之外：机理先验仍被证据 token 通过因果注意力读到，但它逐样本相同的文本不再
        稀释样本间差异。

        判据用"字符区间与证据段相交"而不是"起点落在证据段内"——GPT-2 的 BPE 可能把
        分隔符与证据首词并成一个 token，用相交判据不会漏掉这个边界 token。
        """
        in_evidence = offsets[:, :, 1] > evidence_starts.unsqueeze(1)
        keep = attention_mask.to(torch.bool) & in_evidence
        counts = keep.sum(dim=1)
        if bool((counts == 0).any()):
            raise RuntimeError("存在没有任何证据 token 的 Prompt，无法计算掩码平均池化")
        weights = keep.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / counts.unsqueeze(-1).to(hidden.dtype)

    @torch.inference_mode()
    def patch_forward(
        self,
        signals: torch.Tensor,
        loads_hp: torch.Tensor,
        patch_len: int = 256,
        patch_stride: int = 128,
        include_load_hp: bool = True,
    ) -> torch.Tensor:
        """为重叠 signal patch 生成 [B, E, P, 1] 的 V7 Prompt embedding。

        仍然保持 7 个 Prompt token：每个 token 对应一个局部 patch；Global Evidence 与
        Mechanism Knowledge 被重复写入每个 Prompt，以保持模型结构完全一致。每个 token
        由该 Prompt 的**证据段**掩码平均池化得到，而非 last-token。
        """
        if signals.ndim != 3 or signals.shape[2] != 2:
            raise ValueError(f"DE/FE signals 必须为 [B, 长度, 2] 的数据，实际为 {tuple(signals.shape)}")
        b, length, _ = signals.shape
        if patch_len <= 0 or patch_stride <= 0 or length < patch_len:
            raise ValueError("invalid patch_len or patch_stride")
        if (length - patch_len) % patch_stride != 0:
            raise ValueError("the configured patch stride leaves an uncovered signal tail")

        patches = signals.unfold(dimension=1, size=patch_len, step=patch_stride).permute(0, 1, 3, 2)
        num_patches = patches.shape[1]
        prompts: list[str] = []
        evidence_starts: list[int] = []
        for i in range(b):
            global_evidence = self._window_evidence(signals[i])
            for j in range(num_patches):
                prompt, evidence_start = self._build_prompt(
                    patches[i, j],
                    global_evidence,
                    int(loads_hp[i]),
                    patch_index=j,
                    num_patches=num_patches,
                    include_load_hp=include_load_hp,
                )
                prompts.append(prompt)
                evidence_starts.append(evidence_start)

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        ).to(self.device)
        # offset_mapping 只用于圈定池化范围，不能转交给 GPT-2。
        offsets = encoded.pop("offset_mapping")
        hidden = self._select_hidden(encoded)
        emb = self._masked_evidence_pool(
            hidden,
            encoded["attention_mask"],
            offsets,
            torch.tensor(evidence_starts, dtype=torch.long, device=self.device),
        )
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
            raise ValueError(f"DE/FE signals 必须含两个通道，实际形状为 {tuple(signals.shape)}")
        b = signals.shape[0]
        prompts = []
        evidence_starts: list[int] = []
        for i in range(b):
            global_evidence = self._window_evidence(signals[i])
            prompt, evidence_start = self._build_prompt(
                signals[i],
                global_evidence,
                int(loads_hp[i]),
                patch_index=0,
                num_patches=1,
                include_load_hp=include_load_hp,
            )
            prompts.append(prompt)
            evidence_starts.append(evidence_start)
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        ).to(self.device)
        offsets = encoded.pop("offset_mapping")
        hidden = self._select_hidden(encoded)
        emb = self._masked_evidence_pool(
            hidden,
            encoded["attention_mask"],
            offsets,
            torch.tensor(evidence_starts, dtype=torch.long, device=self.device),
        )
        return emb.unsqueeze(-1).unsqueeze(-1)
