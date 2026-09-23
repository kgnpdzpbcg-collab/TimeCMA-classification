"""冻结文本 embedding 缓存的配置与完整性校验工具。

缓存目录只按“数据与 prompt 配方”组织，不按训练/验证/测试划分组织。这样同一个
窗口在随机划分、LOLO 或少样本协议中切换集合时，都能复用同一份冻结 GPT-2 输出。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CACHE_SPEC_FILENAME = "cache_spec.json"
# schema_version 1 与 2 共有的字段。
REQUIRED_CACHE_SPEC_FIELDS = {
    "schema_version", "dataset", "sensors", "window_size", "window_stride",
    "patch_len", "patch_stride", "num_patches", "embedding_dim", "gpt_model",
    "sampling_rate", "prompt_template_version", "include_load_hp",
}
# schema_version 2 新增：池化方式与池化层直接决定缓存里的数值，必须参与摘要，否则换
# 了池化仍会算出相同的 cache_spec_sha256，新旧缓存会被静默混用。
V2_CACHE_SPEC_FIELDS = {"pooling", "pooling_layer"}
# V8 同时保存固定类别原型，文本和原型池化配方必须进入摘要。
V3_CACHE_SPEC_FIELDS = {
    "prototype_template_version", "prototype_pooling", "prototype_transform",
    "prototype_text_sha256", "class_order",
}
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)


def cache_spec_digest(spec: dict[str, Any]) -> str:
    """返回规范化配置的 SHA-256，用于防止不同 prompt 配方混用同一目录。"""
    canonical = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_cache_spec(spec: dict[str, Any]) -> None:
    """验证训练、生成与迁移脚本共享的缓存描述字段。"""
    if "schema_version" not in spec:
        raise ValueError("cache_spec.json 缺少字段: ['schema_version']")
    if spec["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"不支持的缓存 schema_version: {spec['schema_version']}")
    required = set(REQUIRED_CACHE_SPEC_FIELDS)
    if spec["schema_version"] >= 2:
        required |= V2_CACHE_SPEC_FIELDS
    if spec["schema_version"] >= 3:
        required |= V3_CACHE_SPEC_FIELDS
    missing = required.difference(spec)
    if missing:
        raise ValueError(f"cache_spec.json 缺少字段: {sorted(missing)}")
    if spec["sensors"] != ["DE", "FE"]:
        raise ValueError(f"当前 CWRU 缓存仅支持同步 DE、FE，实际为: {spec['sensors']}")
    if spec["window_size"] <= 0 or spec["window_stride"] <= 0:
        raise ValueError("缓存窗口长度与步长必须为正数")
    if spec["patch_len"] <= 0 or spec["patch_stride"] <= 0 or spec["num_patches"] <= 0:
        raise ValueError("缓存 patch 配置必须为正数")
    if spec["embedding_dim"] <= 0:
        raise ValueError("缓存 embedding_dim 必须为正数")
    if spec["schema_version"] >= 3:
        if spec["class_order"] != ["normal", "ball", "inner", "outer"]:
            raise ValueError("V8 Prototype 的类别顺序必须为 normal/ball/inner/outer")
        if spec["include_load_hp"]:
            raise ValueError("V8 Evidence 不允许写入负载值")


def load_cache_spec(cache_root: str | Path) -> tuple[dict[str, Any], str]:
    """读取并验证缓存配方，返回配置及其内容摘要。"""
    root = Path(cache_root).expanduser().resolve()
    spec_path = root / CACHE_SPEC_FILENAME
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"缺少缓存配置文件: {spec_path}。旧 split 目录缓存需先执行迁移脚本，"
            "或重新生成 by_sample 缓存。"
        )
    with spec_path.open("r", encoding="utf-8") as file:
        spec = json.load(file)
    if not isinstance(spec, dict):
        raise ValueError(f"缓存配置必须是 JSON object: {spec_path}")
    validate_cache_spec(spec)
    return spec, cache_spec_digest(spec)


def write_cache_spec(cache_root: str | Path, spec: dict[str, Any]) -> str:
    """创建不可静默覆盖的缓存配置文件，返回其内容摘要。"""
    validate_cache_spec(spec)
    root = Path(cache_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec_path = root / CACHE_SPEC_FILENAME
    digest = cache_spec_digest(spec)
    if spec_path.exists():
        _existing, existing_digest = load_cache_spec(root)
        if existing_digest != digest:
            raise ValueError(
                f"缓存目录已有不同配方: {spec_path}。请为新 prompt 配方使用新的 embedding-root。"
            )
        return existing_digest
    with spec_path.open("w", encoding="utf-8") as file:
        json.dump(spec, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return digest


def build_v4_de_fe_cache_spec(
    *, model_name: str, window_size: int, window_stride: int, patch_len: int,
    patch_stride: int, sampling_rate: int, include_load_hp: bool, pooling: str,
    pooling_layer: int,
) -> dict[str, Any]:
    """构造 V7 DE+FE 证据型 Prompt 的标准缓存配方。

    ``pooling`` 与 ``pooling_layer`` 都没有默认值：换池化方式或池化层必须显式写出，
    否则摘要不变会导致新旧缓存混用。
    """
    if window_size < patch_len or (window_size - patch_len) % patch_stride != 0:
        raise ValueError("window_size、patch_len、patch_stride 不能完整生成 patch")
    if not pooling:
        raise ValueError("pooling 不能为空")
    if pooling_layer < 0:
        raise ValueError("pooling_layer 不能为负数")
    return {
        "schema_version": 2,
        "dataset": "CWRU",
        "sensors": ["DE", "FE"],
        "window_size": window_size,
        "window_stride": window_stride,
        "patch_len": patch_len,
        "patch_stride": patch_stride,
        "num_patches": 1 + (window_size - patch_len) // patch_stride,
        "embedding_dim": 768,
        "gpt_model": model_name,
        "sampling_rate": sampling_rate,
        "prompt_template_version": "v7_mechanism_first_evidence_v1",
        "include_load_hp": include_load_hp,
        "pooling": pooling,
        "pooling_layer": pooling_layer,
    }


def build_v8_cache_spec(
    *, model_name: str, window_size: int, window_stride: int, patch_len: int,
    patch_stride: int, sampling_rate: int, pooling_layer: int, prototype_text_sha256: str,
) -> dict[str, Any]:
    """构造 V8 证据与类别原型共用的冻结文本配方。

    训练仅凭这一份摘要就能确认 H5 与四个 Prototype 来自同一组模板、模型和池化层。
    文本原文保存在代码中；摘要来自四段原型文本的规范化 JSON。
    """
    if patch_len <= 0 or patch_stride <= 0 or window_stride <= 0 or sampling_rate <= 0:
        raise ValueError("窗口步长、patch 配置和采样率必须为正数")
    if window_size < patch_len or (window_size - patch_len) % patch_stride:
        raise ValueError("window_size、patch_len、patch_stride 不能完整生成 patch")
    if not 0 <= pooling_layer <= 12:
        raise ValueError("GPT-2 small 池化层必须在 0 到 12 之间")
    spec = {
        "schema_version": 3,
        "dataset": "CWRU",
        "sensors": ["DE", "FE"],
        "window_size": window_size,
        "window_stride": window_stride,
        "patch_len": patch_len,
        "patch_stride": patch_stride,
        "num_patches": 1 + (window_size - patch_len) // patch_stride,
        "embedding_dim": 768,
        "gpt_model": model_name,
        "sampling_rate": sampling_rate,
        "prompt_template_version": "v8_diagnostic_context_evidence_v1",
        "include_load_hp": False,
        "pooling": "evidence_span_mean",
        "pooling_layer": pooling_layer,
        "prototype_template_version": "v8_fault_concept_mechanism_v1",
        "prototype_pooling": "all_text_mean",
        "prototype_transform": "four_class_mean_center_l2",
        "prototype_text_sha256": prototype_text_sha256,
        "class_order": ["normal", "ball", "inner", "outer"],
    }
    validate_cache_spec(spec)
    return spec
