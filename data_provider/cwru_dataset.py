"""CWRU 轴承故障诊断数据集。

本模块按原始 MAT 文件而非滑动窗口划分训练、验证和测试集，避免同一段振动记录
切窗后同时进入不同数据集所造成的数据泄漏。默认使用每个文件均具备的 DE 通道。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


# 标签名称固定为论文实验中常用的四类，数值标签由字典顺序稳定决定。
CLASS_TO_INDEX = {"normal": 0, "ball": 1, "inner": 2, "outer": 3}


@dataclass(frozen=True)
class WindowRecord:
    """一个由原始 MAT 文件及起始位置唯一确定的信号窗口。"""

    path: Path
    start: int
    label: int
    load_hp: int

    @property
    def sample_id(self) -> str:
        """用于 embedding 文件名的稳定 ID，不依赖 DataLoader 的遍历顺序。"""
        return f"{self.path.stem}__start_{self.start:07d}"


def _parse_label(path: Path) -> int:
    """根据 CWRU 官方文件名解析故障类别，不将故障尺寸作为标签输入模型。"""
    name = path.stem.lower()
    if name.startswith("normal_"):
        return CLASS_TO_INDEX["normal"]
    if "_b" in name:
        return CLASS_TO_INDEX["ball"]
    if "_ir" in name:
        return CLASS_TO_INDEX["inner"]
    if "_or" in name:
        return CLASS_TO_INDEX["outer"]
    raise ValueError(f"无法从文件名解析 CWRU 故障类别: {path.name}")


def _parse_load(path: Path) -> int:
    """从 0HP 至 3HP 目录名读取推理时可得的负载工况。"""
    match = re.fullmatch(r"([0-3])HP", path.parent.name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"无法从目录名解析负载工况: {path.parent}")
    return int(match.group(1))


def _split_files_by_label(paths: list[Path], seed: int, val_ratio: float, test_ratio: float) -> dict[str, set[Path]]:
    """按类别在 MAT 文件粒度进行确定性切分，保证每个集合都有每个类别。"""
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(paths):
        grouped[_parse_label(path)].append(path)

    splits: dict[str, set[Path]] = {"train": set(), "val": set(), "test": set()}
    rng = np.random.default_rng(seed)
    for label, class_paths in grouped.items():
        if len(class_paths) < 3:
            raise ValueError(f"类别 {label} 仅有 {len(class_paths)} 个 MAT 文件，无法进行 train/val/test 文件级划分")
        shuffled = list(class_paths)
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_ratio))
        n_val = max(1, round(len(shuffled) * val_ratio))
        if n_test + n_val >= len(shuffled):
            raise ValueError(f"类别 {label} 的划分比例使训练文件为空")
        splits["test"].update(shuffled[:n_test])
        splits["val"].update(shuffled[n_test:n_test + n_val])
        splits["train"].update(shuffled[n_test + n_val:])
    return splits


class CWRUDataset(Dataset):
    """读取 CWRU DE 振动窗口，并可选加载对应的冻结 GPT embedding。

    参数
    ----
    root_path:
        包含 0HP、1HP、2HP、3HP 的 ``cwru_data`` 目录。
    split:
        ``train``、``val`` 或 ``test``。切分始终在 MAT 文件粒度完成。
    embedding_root:
        非空时读取 ``embedding_root/split/<sample_id>.h5``；为空时用于生成 embedding。
    """

    def __init__(
        self,
        root_path: str | Path,
        split: str,
        window_size: int = 1024,
        stride: int = 1024,
        seed: int = 2024,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        embedding_root: str | Path | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split 必须为 train/val/test，实际为 {split}")
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size 和 stride 必须为正数")

        self.root_path = Path(root_path).expanduser().resolve()
        if not self.root_path.is_dir():
            raise FileNotFoundError(f"CWRU 数据目录不存在: {self.root_path}")
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.embedding_root = Path(embedding_root).expanduser().resolve() if embedding_root else None
        self.num_classes = len(CLASS_TO_INDEX)
        # 一个 MAT 文件通常对应上百个窗口；缓存避免每个 __getitem__ 重复解析同一文件。
        self._signal_cache: dict[Path, np.ndarray] = {}

        all_files = sorted(self.root_path.glob("[0-3]HP/*.mat"))
        if not all_files:
            raise FileNotFoundError(f"未在 {self.root_path} 找到 CWRU MAT 文件")
        selected_files = _split_files_by_label(all_files, seed, val_ratio, test_ratio)[split]
        self.records = self._build_records(sorted(selected_files))
        if not self.records:
            raise RuntimeError(f"{split} 集没有生成任何长度为 {window_size} 的信号窗口")

    @staticmethod
    def _read_de_signal(path: Path) -> np.ndarray:
        """从 MAT 中定位 Drive-End 字段，例如 ``X105_DE_time``。

        少数 normal 文件含两个相邻实验编号的 DE 字段，此时优先选择与文件名末尾编号一致的
        字段，例如 ``normal_2_99.mat`` 对应 ``X099_DE_time``。
        """
        contents = loadmat(path)
        candidates = [key for key in contents if key.lower().endswith("_de_time")]
        file_id = re.search(r"_(\d+)$", path.stem)
        expected_key = f"X{int(file_id.group(1)):03d}_DE_time" if file_id else None
        if expected_key in candidates:
            selected = expected_key
        elif len(candidates) == 1:
            selected = candidates[0]
        else:
            raise ValueError(f"{path.name} 中无法唯一匹配 DE 信号字段: {candidates}")
        signal = np.asarray(contents[selected], dtype=np.float32).reshape(-1)
        if signal.size == 0:
            raise ValueError(f"{path.name} 的 DE 信号为空")
        return signal

    def _load_signal(self, path: Path) -> np.ndarray:
        """懒加载并缓存原始 DE 信号，DataLoader 多进程时每个进程各自维护安全缓存。"""
        if path not in self._signal_cache:
            self._signal_cache[path] = self._read_de_signal(path)
        return self._signal_cache[path]

    def _build_records(self, paths: list[Path]) -> list[WindowRecord]:
        """预先建立窗口索引；仅保存索引，实际信号在访问时加载以控制内存占用。"""
        records: list[WindowRecord] = []
        for path in paths:
            signal_length = self._load_signal(path).size
            for start in range(0, signal_length - self.window_size + 1, self.stride):
                records.append(WindowRecord(path, start, _parse_label(path), _parse_load(path)))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        signal = self._load_signal(record.path)[record.start:record.start + self.window_size]
        # TimeCMA 的输入约定为 [时间长度, 传感器数]；第一版只使用 DE，故 N=1。
        signal_tensor = torch.from_numpy(signal[:, None].copy())
        label_tensor = torch.tensor(record.label, dtype=torch.long)

        if self.embedding_root is None:
            # 生成 embedding 时返回工况与稳定 ID；标签绝不传入 prompt 生成模块。
            return signal_tensor, label_tensor, torch.tensor(record.load_hp, dtype=torch.long), record.sample_id

        embedding_path = self.embedding_root / self.split / f"{record.sample_id}.h5"
        if not embedding_path.is_file():
            raise FileNotFoundError(f"缺少样本 embedding: {embedding_path}")
        with h5py.File(embedding_path, "r") as handle:
            embedding = torch.from_numpy(handle["embedding"][:].astype(np.float32, copy=False))
        # 存储格式固定为 [E, N, 1]，与原 TimeCMA 的 CMA 输入接口兼容。
        return signal_tensor, label_tensor, embedding
