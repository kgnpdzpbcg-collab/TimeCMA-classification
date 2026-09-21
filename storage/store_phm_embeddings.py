"""Offline generation of patch-level CWRU PHM GPT-2 embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader

from data_provider.cwru_dataset import CWRUDataset
from utils.phm_prompt import PHMPromptEmbedder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, default=Path("Embeddings/CWRU_patch_prompt"))
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
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

        for signals, _labels, loads, sample_ids in loader:
            destinations = [output_dir / f"{sid}.h5" for sid in sample_ids]
            needed = [i for i, p in enumerate(destinations) if args.overwrite or not p.exists()]
            if not needed:
                continue
            idx = torch.tensor(needed)
            embeddings = embedder.patch_forward(
                signals.index_select(0, idx),
                loads.index_select(0, idx),
            ).cpu().numpy()

            for local_i, source_i in enumerate(needed):
                with h5py.File(destinations[source_i], "w") as f:
                    f.create_dataset("embedding", data=embeddings[local_i], compression="gzip")
                    f.attrs["version"] = "V2_patch_prompt_alignment"

        print(f"{split}: cached {len(dataset)} samples")


if __name__ == "__main__":
    main()
