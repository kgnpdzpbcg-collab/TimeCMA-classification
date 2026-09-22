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
REQUIRED_CACHE_SPEC_FIELDS = {
    "schema_version", "dataset", "sensors", "window_size", "window_stride",
    "patch_len", "patch_stride", "num_patches", "embedding_dim", "gpt_model",
    "sampling_rate", "prompt_template_version", "include_load_hp",
}


def cache_spec_digest(spec: dict[str, Any]) -> str:
    """返回规范化配置的 SHA-256，用于防止不同 prompt 配方混用同一目录。"""
    canonical = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_cache_spec(spec: dict[str, Any]) -> None:
    """验证训练、生成与迁移脚本共享的缓存描述字段。"""
    missing = REQUIRED_CACHE_SPEC_FIELDS.difference(spec)
    if missing:
        raise ValueError(f"cache_spec.json 缺少字段: {sorted(missing)}")
    if spec["schema_version"] != 1:
        raise ValueError(f"不支持的缓存 schema_version: {spec['schema_version']}")
    if spec["sensors"] != ["DE", "FE"]:
        raise ValueError(f"当前 V4/V5/V6 仅支持同步 DE、FE 缓存，实际为: {spec['sensors']}")
    if spec["window_size"] <= 0 or spec["window_stride"] <= 0:
        raise ValueError("缓存窗口长度与步长必须为正数")
    if spec["patch_len"] <= 0 or spec["patch_stride"] <= 0 or spec["num_patches"] <= 0:
        raise ValueError("缓存 patch 配置必须为正数")
    if spec["embedding_dim"] <= 0:
        raise ValueError("缓存 embedding_dim 必须为正数")


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
    patch_stride: int, sampling_rate: int, include_load_hp: bool,
) -> dict[str, Any]:
    """构造 V6 DE+FE 证据型 Prompt 的标准缓存配方。"""
    if window_size < patch_len or (window_size - patch_len) % patch_stride != 0:
        raise ValueError("window_size、patch_len、patch_stride 不能完整生成 patch")
    return {
        "schema_version": 1,
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
        "prompt_template_version": "v6_local_global_mechanism_v1",
        "include_load_hp": include_load_hp,
    }
