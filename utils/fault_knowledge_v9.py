"""V9 的四类固定故障机理文本与可审计 Prototype 配方。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


# 类别顺序必须与 CWRUDataset.CLASS_TO_INDEX 一致。沿用 V8 的四段文本，
# 使 V8→V9 的主要变化集中在信号编码器和融合路径，而不是知识内容。
CLASS_ORDER = ("normal", "ball", "inner", "outer")
FAULT_TEXTS = (
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


def prototype_recipe() -> dict:
    """返回与样本、数据划分无关的文本编码配方，用于生成和校验缓存。"""
    return {
        "schema_version": 1,
        "class_order": list(CLASS_ORDER),
        "texts": list(FAULT_TEXTS),
        "model": "gpt2",
        "hidden_layer": 10,
        "pooling": "all_valid_tokens_mean",
        "transform": "four_class_mean_center_l2",
        "embedding_dim": 768,
    }


def recipe_sha256(recipe: dict) -> str:
    """对文本原文和全部编码参数计算稳定摘要，防止旧 Prototype 被误用。"""
    canonical = json.dumps(recipe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_fixed_prototypes(path: str | Path) -> torch.Tensor:
    """只读取四个冻结类别锚点，并检查文本、层数、变换及类别顺序。"""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    recipe = prototype_recipe()
    if saved.get("recipe") != recipe or saved.get("recipe_sha256") != recipe_sha256(recipe):
        raise ValueError(f"Prototype 与当前 V9 文本配方不一致: {path}")
    prototypes = saved.get("prototypes")
    if not isinstance(prototypes, torch.Tensor) or tuple(prototypes.shape) != (4, 768):
        raise ValueError(f"Prototype 必须为 [4, 768]: {path}")
    if not torch.isfinite(prototypes).all():
        raise ValueError(f"Prototype 含非有限数值: {path}")
    return prototypes.float()
