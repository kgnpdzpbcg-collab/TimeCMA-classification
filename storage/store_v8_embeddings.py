"""生成 V8 固定故障原型和 split 无关的样本证据 H5；不执行模型训练。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.cwru_dataset import CWRUDataset
from utils.embedding_cache import build_v8_cache_spec, write_cache_spec
from utils.phm_prompt_v8 import V8_POOLING_LAYER, V8PromptEmbedder, prototype_text_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 V8 类别原型与样本证据冻结 GPT-2 缓存")
    parser.add_argument("--data-root", type=Path, help="生成 evidence 时需要的 CWRU MAT 根目录")
    parser.add_argument("--text-root", type=Path, required=True, help="V8 专用 cache_spec、原型与 H5 的共同根目录")
    parser.add_argument("--artifact", choices=("all", "prototypes", "evidence"), default="all")
    parser.add_argument("--model-source", help="实际加载 GPT-2 的本地路径；缓存配方固定记为 gpt2")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--patch-len", type=int, default=256)
    parser.add_argument("--patch-stride", type=int, default=128)
    parser.add_argument("--sampling-rate", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def store_prototypes(
    embedder: V8PromptEmbedder, text_root: Path, spec_digest: str, overwrite: bool,
) -> None:
    """按固定类别顺序保存四个 `[768]` 文本锚点及其配方摘要。"""
    path = text_root / "fault_prototypes.pt"
    if path.exists() and not overwrite:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (
            saved["cache_spec_sha256"] != spec_digest
            or saved["prototype_text_sha256"] != prototype_text_sha256()
            or saved["class_order"] != ["normal", "ball", "inner", "outer"]
            or tuple(saved["prototypes"].shape) != (4, 768)
        ):
            raise ValueError(f"已有原型与当前 V8 文本配方不一致: {path}")
        print(f"原型已存在且配方一致，跳过: {path}")
        return
    raw_prototypes = embedder.encode_prototypes().cpu()
    if tuple(raw_prototypes.shape) != (4, 768) or not torch.isfinite(raw_prototypes).all():
        raise ValueError(f"故障原型形状或数值异常: {tuple(raw_prototypes.shape)}")
    # GPT-2 均值向量存在强公共方向。只用四段固定文本计算公共均值，去除后再做 L2；
    # 不接触任何 CWRU 样本、标签划分或验证/测试指标，且该变换写入 cache_spec 摘要。
    centered = raw_prototypes - raw_prototypes.mean(dim=0, keepdim=True)
    if bool((centered.norm(dim=-1) < 1e-6).any()):
        raise ValueError("四类原型去公共方向后重合，不能构造固定语义锚点")
    prototypes = F.normalize(centered, dim=-1)
    torch.save(
        {
            "prototypes": prototypes,
            "class_order": ["normal", "ball", "inner", "outer"],
            "prototype_text_sha256": prototype_text_sha256(),
            "cache_spec_sha256": spec_digest,
        },
        path,
    )
    print(f"保存原型: {path}")
    print("四类原始原型的余弦相似度矩阵：")
    raw_normalized = F.normalize(raw_prototypes, dim=-1)
    print((raw_normalized @ raw_normalized.T).numpy())
    print("去公共方向后的余弦相似度矩阵：")
    print((prototypes @ prototypes.T).numpy())


def store_evidence(
    args: argparse.Namespace, embedder: V8PromptEmbedder, spec_digest: str,
) -> None:
    """枚举全部 MAT 窗口，每个 sample_id 只存一份不依赖 split 的证据 embedding。"""
    if args.data_root is None:
        raise ValueError("生成 evidence 时必须指定 --data-root")
    output_dir = args.text_root / "by_sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CWRUDataset(args.data_root, "all", args.window_size, args.stride)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    created = skipped = 0
    for signals, _labels, _loads, sample_ids in tqdm(loader, desc="V8 evidence", unit="batch", dynamic_ncols=True):
        destinations = [output_dir / f"{sample_id}.h5" for sample_id in sample_ids]
        needed = [i for i, path in enumerate(destinations) if args.overwrite or not path.exists()]
        if not needed:
            skipped += len(sample_ids)
            continue
        selected = torch.tensor(needed)
        embeddings = embedder.patch_forward(
            signals.index_select(0, selected), patch_len=args.patch_len, patch_stride=args.patch_stride,
        ).cpu().numpy()
        for local_index, source_index in enumerate(needed):
            with h5py.File(destinations[source_index], "w") as handle:
                handle.create_dataset("embedding", data=embeddings[local_index], compression="gzip")
                handle.attrs["cache_spec_sha256"] = spec_digest
                handle.attrs["sample_id"] = sample_ids[source_index]
                handle.attrs["num_patches"] = embeddings.shape[2]
        created += len(needed)
        skipped += len(sample_ids) - len(needed)
    print(f"共 {len(dataset)} 个窗口；新生成 {created}，跳过已有 {skipped}。")


def main() -> None:
    args = parse_args()
    if args.artifact in {"all", "evidence"} and args.data_root is None:
        raise ValueError("生成 Evidence 时必须指定 --data-root")
    spec = build_v8_cache_spec(
        model_name="gpt2", window_size=args.window_size, window_stride=args.stride,
        patch_len=args.patch_len, patch_stride=args.patch_stride, sampling_rate=args.sampling_rate,
        pooling_layer=V8_POOLING_LAYER, prototype_text_sha256=prototype_text_sha256(),
    )
    spec_digest = write_cache_spec(args.text_root, spec)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = V8PromptEmbedder(
        args.model_source or "gpt2", device=device,
        sampling_rate=args.sampling_rate, pooling_layer=V8_POOLING_LAYER,
    )
    if args.artifact in {"all", "prototypes"}:
        store_prototypes(embedder, args.text_root, spec_digest, args.overwrite)
    if args.artifact in {"all", "evidence"}:
        store_evidence(args, embedder, spec_digest)


if __name__ == "__main__":
    main()
