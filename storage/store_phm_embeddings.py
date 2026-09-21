"""离线生成 CWRU PHM prompt embedding；训练阶段不会再次调用 GPT-2。"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader

from data_provider.cwru_dataset import CWRUDataset
from utils.phm_prompt import PHMPromptEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 CWRU 冻结 GPT-2 embedding")
    parser.add_argument("--data-root", type=Path, required=True, help="CWRU cwru_data 目录")
    parser.add_argument("--embedding-root", type=Path, default=Path("Embeddings/CWRU"))
    parser.add_argument("--model-name", default="gpt2", help="Hugging Face 模型名或本地模型目录")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0, help="Windows 推荐保持 0")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--overwrite", action="store_true", help="重新生成已存在的 embedding")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = PHMPromptEmbedder(args.model_name, device=device)
    for split in ("train", "val", "test"):
        dataset = CWRUDataset(args.data_root, split, args.window_size, args.stride, args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        output_dir = args.embedding_root / split
        output_dir.mkdir(parents=True, exist_ok=True)
        for signals, _labels, loads, sample_ids in loader:
            destinations = [output_dir / f"{sample_id}.h5" for sample_id in sample_ids]
            needed = [index for index, destination in enumerate(destinations) if args.overwrite or not destination.exists()]
            if not needed:
                continue
            indices = torch.tensor(needed, dtype=torch.long)
            embeddings = embedder(signals.index_select(0, indices), loads.index_select(0, indices)).cpu().numpy()
            for local_index, source_index in enumerate(needed):
                with h5py.File(destinations[source_index], "w") as handle:
                    handle.create_dataset("embedding", data=embeddings[local_index], compression="gzip")
                    handle.attrs["sample_id"] = sample_ids[source_index]
        print(f"{split}: 已缓存 {len(dataset)} 个 CWRU 窗口到 {output_dir}")


if __name__ == "__main__":
    main()
