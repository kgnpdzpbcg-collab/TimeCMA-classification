"""生成可审计的 CWRU 留一负载（LOLO）文件级划分 manifest。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.cwru_dataset import CLASS_TO_INDEX, CWRUDataset, _parse_label, _parse_load, discover_cwru_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成严格的 CWRU LOLO 文件级 manifest")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-load", type=int, choices=range(4), required=True)
    parser.add_argument("--val-load", type=int, choices=range(4), required=True)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _class_counts(paths: list[Path]) -> dict[str, int]:
    """按四类标签统计 MAT 文件数，确保每个集合均具备完整任务标签。"""
    reverse = {value: key for key, value in CLASS_TO_INDEX.items()}
    counts = Counter(_parse_label(path) for path in paths)
    return {reverse[index]: counts[index] for index in range(len(reverse))}


def main() -> None:
    args = parse_args()
    if args.test_load == args.val_load:
        raise ValueError("LOLO 的 test-load 与 val-load 必须不同")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"manifest 已存在: {args.output}；确认后使用 --overwrite")

    root, all_files = discover_cwru_files(args.data_root)
    train_loads = [load for load in range(4) if load not in {args.test_load, args.val_load}]
    file_splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for path in all_files:
        load = _parse_load(path)
        split = "test" if load == args.test_load else "val" if load == args.val_load else "train"
        file_splits[split].append(path)

    for split, paths in file_splits.items():
        counts = _class_counts(paths)
        if any(count == 0 for count in counts.values()):
            raise ValueError(f"{split} 未覆盖全部四类标签: {counts}")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "protocol": "lolo_load_v1",
        "fold_name": f"test_{args.test_load}hp_val_{args.val_load}hp",
        "complete_partition": True,
        "require_all_classes": True,
        "dataset": "CWRU",
        "window_size": args.window_size,
        "window_stride": args.stride,
        "sensors": ["DE", "FE"],
        "train_loads": train_loads,
        "val_loads": [args.val_load],
        "test_loads": [args.test_load],
        "files": {
            split: [path.relative_to(root).as_posix() for path in sorted(paths)]
            for split, paths in file_splits.items()
        },
        "mat_file_counts": {split: _class_counts(paths) for split, paths in file_splits.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")

    # 使用与训练完全相同的 Dataset 建窗逻辑写入审计摘要，避免手算窗口数与实际不一致。
    window_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        dataset = CWRUDataset(root, split, args.window_size, args.stride, manifest_path=args.output)
        reverse = {value: key for key, value in CLASS_TO_INDEX.items()}
        counts = Counter(record.label for record in dataset.records)
        window_counts[split] = {"total": len(dataset), **{reverse[index]: counts[index] for index in range(4)}}
    manifest["window_counts"] = window_counts
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps({"manifest": str(args.output), "window_counts": window_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
