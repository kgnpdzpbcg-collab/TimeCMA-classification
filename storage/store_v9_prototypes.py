"""仅编码四段故障机理文本；V9 不读取 CWRU 样本，也不生成 Evidence H5。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from transformers import GPT2Model, GPT2TokenizerFast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.fault_knowledge_v9 import (
    FAULT_TEXTS, load_fixed_prototypes, prototype_recipe, recipe_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 V9 四类冻结 GPT-2 故障机理 Prototype")
    parser.add_argument("--output", type=Path, required=True, help="例如 Embeddings/CWRU_v9_prototypes/fault_prototypes.pt")
    parser.add_argument("--overwrite", action="store_true", help="显式重新生成已有文件")
    return parser.parse_args()


@torch.inference_mode()
def encode_prototypes(device: torch.device) -> torch.Tensor:
    """取冻结 GPT-2 第 10 层、对有效词元平均，再移除四段文本的公共方向。"""
    recipe = prototype_recipe()
    tokenizer = GPT2TokenizerFast.from_pretrained(recipe["model"])
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(recipe["model"]).to(device).eval()
    model.requires_grad_(False)

    encoded = tokenizer(FAULT_TEXTS, return_tensors="pt", padding=True).to(device)
    if encoded["input_ids"].shape[1] > model.config.n_positions:
        raise ValueError("故障机理文本超过 GPT-2 的最大上下文长度")
    # hidden_states[0] 是词嵌入；索引 10 已经过十个 GPT-2 Transformer Block。
    hidden = model(**encoded, output_hidden_states=True).hidden_states[recipe["hidden_layer"]]
    keep = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * keep).sum(dim=1) / keep.sum(dim=1)

    # 保持 V8 的无样本公共方向去除方式；整个变换只依赖四段固定文本。
    centered = pooled - pooled.mean(dim=0, keepdim=True)
    if bool((centered.norm(dim=-1) < 1e-6).any()):
        raise ValueError("四类机理文本去公共方向后重合，无法构造 Prototype")
    return F.normalize(centered, dim=-1).cpu()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        load_fixed_prototypes(args.output)
        print(f"已有 Prototype 配方一致，跳过: {args.output}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prototypes = encode_prototypes(device)
    recipe = prototype_recipe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"prototypes": prototypes, "recipe": recipe, "recipe_sha256": recipe_sha256(recipe)},
        args.output,
    )
    print(f"保存四类固定 Prototype: {args.output}")


if __name__ == "__main__":
    main()
