"""V8 的两类冻结 GPT-2 文本：类别故障原型和当前样本的诊断证据。"""

from __future__ import annotations

import hashlib
import json

import torch
from transformers import GPT2TokenizerFast

from utils.phm_prompt import PHMPromptEmbedder


V8_POOLING_LAYER = 10


# 顺序与 CWRUDataset.CLASS_TO_INDEX 完全一致。文本只描述概念与可能的机理；
# 不包含 CWRU、负载、传感器、故障尺寸或任何从当前样本计算的数值。
FAULT_PROTOTYPE_TEXTS = (
    "Fault concept: Healthy bearing. Mechanism: The rolling elements and raceways have no localized defect. "
    "Normal rotation does not create persistent defect-synchronous impacts or their repeated harmonics.",
    "Fault concept: Rolling-element defect. Mechanism: A damaged rolling element can contact both raceways "
    "while it spins and orbits. Repeated impacts can be related to ball spin frequency, with variable "
    "amplitude and cage-related modulation.",
    "Fault concept: Inner-race defect. Mechanism: A defect on the inner race rotates with the shaft. "
    "Rolling-element contacts can create periodic impacts related to inner-race pass frequency, with "
    "amplitude modulation as the defect travels through the load zone.",
    "Fault concept: Outer-race defect. Mechanism: A defect on the stationary outer race is crossed "
    "repeatedly by rolling elements. These contacts can create periodic impacts related to outer-race "
    "pass frequency and its harmonics.",
)


def prototype_text_sha256() -> str:
    """按类别顺序计算原型原文摘要，避免不改缓存名就换掉类别知识。"""
    canonical = json.dumps(FAULT_PROTOTYPE_TEXTS, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class V8PromptEmbedder(PHMPromptEmbedder):
    """复用 V7 的无标签物理统计提取，只替换 V8 的文本职责和池化入口。"""

    def __init__(
        self,
        model_name: str = "gpt2",
        device: torch.device | str = "cpu",
        sampling_rate: int = 12000,
        pooling_layer: int = V8_POOLING_LAYER,
    ):
        """加载 GPT-2 与支持字符 offset 的快速分词器。

        V8 需要从 ``Diagnostic evidence`` 起始字符定位到对应 token，慢速
        ``GPT2Tokenizer`` 不支持 ``return_offsets_mapping``。已核验 V8 原型和
        Evidence 模板在快、慢分词器下的 token id 完全一致，因此替换不会改变
        文本 token 序列，只是提供证据 span 所需的 offset。
        """
        super().__init__(
            model_name=model_name,
            device=device,
            sampling_rate=sampling_rate,
            pooling_layer=pooling_layer,
        )
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_evidence_prompt(
        self,
        patch: torch.Tensor,
        window_evidence: dict[str, object],
        patch_index: int,
        num_patches: int,
    ) -> tuple[str, int]:
        """返回样本证据全文及证据段起始字符，供 GPT-2 因果编码和掩码池化。"""
        de_stats = self._statistics(patch[:, 0], self.sampling_rate)
        fe_stats = self._statistics(patch[:, 1], self.sampling_rate)
        local_cross = self._cross_sensor_evidence(patch[:, 0], patch[:, 1])
        global_de = window_evidence["de"]
        global_fe = window_evidence["fe"]
        global_cross = window_evidence["cross_sensor"]

        # GPT-2 是因果模型：解释语境必须在证据之前，才能被证据 token 读到。
        # 不提供真值标签或负载，也不把四类原型文本复制到样本 Prompt 中。
        prefix = (
            "Task context: Bearing fault diagnosis.\n"
            "Dataset context: Vibration signals collected from bearings.\n"
            "Input: Synchronized drive-end (DE) and fan-end (FE) acceleration signals.\n"
            "Interpretation requirement: Relate observed evidence to possible fault mechanisms. "
            "Consider repeated impacts, modulation, amplitude distribution, and spectral structure; "
            "one raw-FFT peak alone does not identify a specific fault.\n"
            "Diagnostic evidence:\n"
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
                "Global evidence shared by this 1024-sample observation: "
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
                f"DE-FE waveform correlation {global_cross['correlation']:.6g}."
            )
        )
        return prefix + evidence, len(prefix)

    @torch.inference_mode()
    def encode_prototypes(self) -> torch.Tensor:
        """在冻结 GPT-2 第 pooling_layer 层平均有效 token，返回 `[4, 768]`。"""
        encoded = self.tokenizer(FAULT_PROTOTYPE_TEXTS, return_tensors="pt", padding=True).to(self.device)
        if encoded["input_ids"].shape[1] > self.model.config.n_positions:
            raise ValueError("故障原型文本超过 GPT-2 的最大上下文长度")
        hidden = self._select_hidden(encoded)
        keep = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * keep).sum(dim=1) / keep.sum(dim=1)

    @torch.inference_mode()
    def patch_forward(
        self, signals: torch.Tensor, patch_len: int = 256, patch_stride: int = 128,
    ) -> torch.Tensor:
        """生成 `[B, 768, 7, 1]`；V8 Prompt 仅依赖当前 DE/FE 窗口，不读取负载。"""
        if signals.ndim != 3 or signals.shape[2] != 2:
            raise ValueError(f"DE/FE 窗口应为 [B, 长度, 2]，实际为 {tuple(signals.shape)}")
        batch_size, length, _ = signals.shape
        if patch_len <= 0 or patch_stride <= 0 or length < patch_len:
            raise ValueError("patch_len、patch_stride 必须为正且不超过窗口长度")
        if (length - patch_len) % patch_stride:
            raise ValueError("patch_stride 会使窗口尾部不能被完整覆盖")

        patches = signals.unfold(1, patch_len, patch_stride).permute(0, 1, 3, 2)
        num_patches = patches.shape[1]
        prompts: list[str] = []
        evidence_starts: list[int] = []
        for i in range(batch_size):
            global_evidence = self._window_evidence(signals[i])
            for j in range(num_patches):
                prompt, start = self._build_evidence_prompt(patches[i, j], global_evidence, j, num_patches)
                prompts.append(prompt)
                evidence_starts.append(start)

        # 禁止静默截断；若固定前缀变长，应先改模板和缓存版本，再重新验证全部字段可见。
        encoded = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False, return_offsets_mapping=True,
        ).to(self.device)
        offsets = encoded.pop("offset_mapping")
        if encoded["input_ids"].shape[1] > self.model.config.n_positions:
            raise ValueError("V8 Evidence Prompt 超过 GPT-2 上下文长度，不能截断统计证据")
        hidden = self._select_hidden(encoded)
        embeddings = self._masked_evidence_pool(
            hidden, encoded["attention_mask"], offsets,
            torch.tensor(evidence_starts, dtype=torch.long, device=self.device),
        )
        return embeddings.reshape(batch_size, num_patches, -1).permute(0, 2, 1).unsqueeze(-1)

    def forward(
        self, signals: torch.Tensor, patch_len: int = 256, patch_stride: int = 128,
    ) -> torch.Tensor:
        """默认调用也进入 V8 的证据模板，避免继承 V7 的含负载全局 Prompt 路径。"""
        return self.patch_forward(signals, patch_len=patch_len, patch_stride=patch_stride)
