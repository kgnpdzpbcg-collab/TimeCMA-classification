"""将旧 ``train/val/test`` embedding 目录安全迁移为 split 无关 ``by_sample`` 缓存。

默认仅做只读预检。必须显式传入 ``--apply`` 才会复制或移动 H5 文件；默认复制，
以保留 V4 随机划分实验的原始缓存作为回退。迁移不会调用 GPT-2。
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys

import h5py

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.embedding_cache import build_v4_de_fe_cache_spec, write_cache_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="迁移旧 split 目录 CWRU embedding 到 by_sample")
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--include-load-hp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--patch-len", type=int, default=256)
    parser.add_argument("--patch-stride", type=int, default=128)
    parser.add_argument("--sampling-rate", type=int, default=12000)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--move", action="store_true", help="--apply 时移动而非复制；成功后旧缓存将不再保留")
    return parser.parse_args()


def collect_legacy_files(root: Path) -> list[Path]:
    """只读取旧缓存约定的三个 split 目录，并验证 sample_id 没有碰撞。"""
    files: list[Path] = []
    for split in ("train", "val", "test"):
        directory = root / split
        if not directory.is_dir():
            raise FileNotFoundError(f"旧缓存缺少 split 目录: {directory}")
        files.extend(sorted(directory.glob("*.h5")))
    duplicates = [name for name, count in Counter(path.name for path in files).items() if count > 1]
    if duplicates:
        raise ValueError(f"旧缓存存在跨 split 同名 sample_id，不能安全拍平: {duplicates[:5]}")
    if not files:
        raise RuntimeError("旧缓存中没有 H5 文件")
    return files


def validate_h5_files(files: list[Path], expected_shape: tuple[int, int, int]) -> None:
    """逐个检查 H5 数据集，避免将截断或错误版本的 embedding 迁入新缓存。"""
    for path in files:
        with h5py.File(path, "r") as handle:
            if "embedding" not in handle or tuple(handle["embedding"].shape) != expected_shape:
                actual = tuple(handle["embedding"].shape) if "embedding" in handle else None
                raise ValueError(f"缓存 shape 异常: {path}，期望 {expected_shape}，实际 {actual}")


def main() -> None:
    args = parse_args()
    legacy_root = args.legacy_root.expanduser().resolve()
    target_root = args.target_root.expanduser().resolve()
    files = collect_legacy_files(legacy_root)
    spec = build_v4_de_fe_cache_spec(
        model_name=args.model_name,
        window_size=args.window_size,
        window_stride=args.stride,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        sampling_rate=args.sampling_rate,
        include_load_hp=args.include_load_hp,
    )
    expected_shape = (spec["embedding_dim"], spec["num_patches"], 1)
    validate_h5_files(files, expected_shape)
    destination = target_root / "by_sample"
    existing = set(path.name for path in destination.glob("*.h5")) if destination.exists() else set()
    collisions = existing.intersection(path.name for path in files)
    if collisions:
        raise FileExistsError(f"目标 by_sample 已存在同名 H5，拒绝覆盖: {sorted(collisions)[:5]}")

    print(
        f"预检通过：{len(files)} 个 H5、{len(set(path.name for path in files))} 个唯一 sample_id、"
        f"shape={expected_shape}、mode={'move' if args.move else 'copy'}、apply={args.apply}"
    )
    if not args.apply:
        print("未传入 --apply；未创建目录、未复制、未移动任何文件。")
        return

    spec_digest = write_cache_spec(target_root, spec)
    destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        target = destination / source.name
        if args.move:
            shutil.move(str(source), str(target))
        else:
            shutil.copy2(source, target)
        # 旧 H5 没有配方摘要；迁移时只增加元数据，不改 embedding 数值。
        with h5py.File(target, "r+") as handle:
            handle.attrs["cache_spec_sha256"] = spec_digest
            handle.attrs["sample_id"] = target.stem
    print(f"迁移完成：{len(files)} 个 H5 写入 {destination}")


if __name__ == "__main__":
    main()
