"""按样本生成 split 无关的 V4 DE/FE 冻结 GPT-2 embedding。

本脚本总是枚举全部 CWRU 窗口；Train/Val/Test 的归属由训练阶段的 manifest 决定，
不会影响 embedding 的路径或内容。不同 prompt 配方必须使用不同 embedding-root。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.cwru_dataset import CWRUDataset
from utils.embedding_cache import build_v4_de_fe_cache_spec, write_cache_spec
from utils.phm_prompt import PHMPromptEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 split 无关的 CWRU DE/FE prompt embedding")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--patch-len", type=int, default=256)
    parser.add_argument("--patch-stride", type=int, default=128)
    parser.add_argument("--sampling-rate", type=int, default=12000)
    parser.add_argument("--include-load-hp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = build_v4_de_fe_cache_spec(
        model_name=args.model_name,
        window_size=args.window_size,
        window_stride=args.stride,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        sampling_rate=args.sampling_rate,
        include_load_hp=args.include_load_hp,
    )
    spec_digest = write_cache_spec(args.embedding_root, spec)
    output_dir = args.embedding_root / "by_sample"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = PHMPromptEmbedder(args.model_name, device=device, sampling_rate=args.sampling_rate)
    # split='all' 明确枚举 40 个 MAT 的所有窗口；不接收随机 seed 或 split manifest。
    dataset = CWRUDataset(args.data_root, "all", args.window_size, args.stride)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    created = 0
    skipped = 0
    progress = tqdm(loader, desc="Embedding all CWRU windows", unit="batch", dynamic_ncols=True)
    for signals, _labels, loads, sample_ids in progress:
        destinations = [output_dir / f"{sample_id}.h5" for sample_id in sample_ids]
        needed = [index for index, path in enumerate(destinations) if args.overwrite or not path.exists()]
        if not needed:
            skipped += len(sample_ids)
            progress.set_postfix(created=created, skipped=skipped)
            continue

        selected = torch.tensor(needed)
        embeddings = embedder.patch_forward(
            signals.index_select(0, selected),
            loads.index_select(0, selected),
            patch_len=args.patch_len,
            patch_stride=args.patch_stride,
            include_load_hp=args.include_load_hp,
        ).cpu().numpy()
        for local_index, source_index in enumerate(needed):
            with h5py.File(destinations[source_index], "w") as handle:
                handle.create_dataset("embedding", data=embeddings[local_index], compression="gzip")
                # 每个 H5 重复保存摘要，训练读取单文件时即可阻止错误混用缓存。
                handle.attrs["cache_spec_sha256"] = spec_digest
                handle.attrs["sample_id"] = sample_ids[source_index]
                handle.attrs["num_patches"] = embeddings.shape[2]
        created += len(needed)
        skipped += len(sample_ids) - len(needed)
        progress.set_postfix(created=created, skipped=skipped)

    print(f"全部 {len(dataset)} 个窗口：新生成 {created}，跳过已有 {skipped}。")


if __name__ == "__main__":
    main()
