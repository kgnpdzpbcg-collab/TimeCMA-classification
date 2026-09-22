"""离线生成 V4 DE/FE 双传感器重叠 patch 的 CWRU PHM GPT-2 embedding。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# 既支持 ``python -m storage.store_phm_embeddings``，也支持项目根目录中常用的
# ``python storage/store_phm_embeddings.py``。后者的默认 sys.path 是 storage/，
# 因此需显式补入项目根目录，避免 data_provider 导入失败。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.cwru_dataset import CWRUDataset
from utils.phm_prompt import PHMPromptEmbedder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, default=Path("Embeddings/CWRU_v4_de_fe_patch256_stride128"))
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    # 注意：patch stride 不等于数据集窗口 stride；前者只控制一个 1024 点样本内部的 token 重叠。
    parser.add_argument("--patch-len", type=int, default=256)
    parser.add_argument("--patch-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = PHMPromptEmbedder(args.model_name, device=device)

    for split in ("train", "val", "test"):
        dataset = CWRUDataset(args.data_root, split, args.window_size, args.stride, args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        output_dir = args.embedding_root / split
        output_dir.mkdir(parents=True, exist_ok=True)

        created = 0
        skipped = 0
        progress = tqdm(loader, desc=f"Embedding {split}", unit="batch", dynamic_ncols=True)
        for signals, _labels, loads, sample_ids in progress:
            destinations = [output_dir / f"{sid}.h5" for sid in sample_ids]
            needed = [i for i, p in enumerate(destinations) if args.overwrite or not p.exists()]
            if not needed:
                skipped += len(sample_ids)
                progress.set_postfix(created=created, skipped=skipped)
                continue
            idx = torch.tensor(needed)
            embeddings = embedder.patch_forward(
                signals.index_select(0, idx),
                loads.index_select(0, idx),
                patch_len=args.patch_len,
                patch_stride=args.patch_stride,
            ).cpu().numpy()

            for local_i, source_i in enumerate(needed):
                with h5py.File(destinations[source_i], "w") as f:
                    f.create_dataset("embedding", data=embeddings[local_i], compression="gzip")
                    # 缓存元数据让训练脚本可追溯其 token 几何，防止误用 V2 的 16-patch 文件。
                    f.attrs["version"] = "V4_DE_FE_overlap_patch_position_cls"
                    f.attrs["sensors"] = "DE,FE"
                    f.attrs["window_size"] = args.window_size
                    f.attrs["patch_len"] = args.patch_len
                    f.attrs["patch_stride"] = args.patch_stride
                    f.attrs["num_patches"] = embeddings.shape[2]
            created += len(needed)
            skipped += len(sample_ids) - len(needed)
            progress.set_postfix(created=created, skipped=skipped)

        print(f"{split}: 共 {len(dataset)} 个窗口，本次新生成 {created} 个，跳过已有缓存 {skipped} 个")


if __name__ == "__main__":
    main()
