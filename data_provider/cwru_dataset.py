"""CWRU DE/FE 双传感器窗口数据集与可审计数据划分读取器。

划分协议被保存为独立 manifest。数据集类不再根据随机种子自行决定 Train/Val/Test，
从而使随机文件级划分、留一负载（LOLO）和后续新协议共享同一套窗口与 embedding。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from utils.embedding_cache import load_cache_spec


CLASS_TO_INDEX = {"normal": 0, "ball": 1, "inner": 2, "outer": 3}
DATA_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class WindowRecord:
    """由 MAT 文件和时间起点唯一确定的一个信号窗口。"""

    path: Path
    start: int
    label: int
    load_hp: int

    @property
    def sample_id(self) -> str:
        """冻结 embedding 的全局稳定 ID，不包含任何 split 名称。"""
        return f"{self.path.stem}__start_{self.start:07d}"


def _parse_label(path: Path) -> int:
    """从 CWRU 文件名解析四类故障标签；故障尺寸不作为模型输入。"""
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
    """从 0HP 至 3HP 目录读取推理时可获得的负载工况元数据。"""
    match = re.fullmatch(r"([0-3])HP", path.parent.name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"无法从目录名解析负载工况: {path.parent}")
    return int(match.group(1))


def discover_cwru_files(root_path: str | Path) -> tuple[Path, list[Path]]:
    """发现并排序 CWRU MAT 文件，返回规范化根目录及文件列表。"""
    root = Path(root_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CWRU 数据目录不存在: {root}")
    files = sorted(root.glob("[0-3]HP/*.mat"))
    if not files:
        raise FileNotFoundError(f"未在 {root} 找到 CWRU MAT 文件")
    return root, files


def manifest_sha256(manifest_path: str | Path) -> str:
    """计算原始 manifest 摘要，写入 checkpoint 以追溯实验数据划分。"""
    path = Path(manifest_path).expanduser().resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split_manifest(root_path: str | Path, manifest_path: str | Path) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    """读取显式文件级划分，并阻止 MAT 或窗口跨集合泄漏。

    manifest 的 ``files`` 保存相对于 ``cwru_data`` 根目录的 POSIX 路径。若标记
    ``complete_partition=true``，则全部已发现 MAT 必须被 train/val/test 完整覆盖。
    """
    root, all_files = discover_cwru_files(root_path)
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"数据划分 manifest 不存在: {path}")
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError(f"manifest 必须含有 files object: {path}")

    listed_splits = manifest["files"]
    if set(listed_splits) != set(DATA_SPLITS):
        raise ValueError(f"manifest files 必须恰含 {DATA_SPLITS}，实际为 {sorted(listed_splits)}")

    all_by_relative = {item.relative_to(root).as_posix(): item for item in all_files}
    selected: dict[str, list[Path]] = {}
    seen: dict[Path, str] = {}
    for split in DATA_SPLITS:
        entries = listed_splits[split]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"manifest 的 {split} 必须是非空 MAT 相对路径列表")
        paths: list[Path] = []
        for relative in entries:
            if not isinstance(relative, str):
                raise ValueError(f"manifest {split} 含非字符串路径: {relative!r}")
            normalized = relative.replace("\\", "/")
            if normalized not in all_by_relative:
                raise ValueError(f"manifest 引用了不存在或不属于 CWRU 的文件: {relative}")
            item = all_by_relative[normalized]
            if item in seen:
                raise ValueError(f"MAT 文件跨集合泄漏: {normalized} 同时属于 {seen[item]} 和 {split}")
            seen[item] = split
            paths.append(item)
        if len(set(paths)) != len(paths):
            raise ValueError(f"manifest 的 {split} 存在重复 MAT 文件")
        selected[split] = sorted(paths)

        # LOLO manifest 声明了每个 split 应包含的工况；这里实际复算一次，防止有人
        # 手工编辑 files 后留下与协议名不一致的“伪跨工况”划分。
        declared_loads = manifest.get(f"{split}_loads")
        if declared_loads is not None:
            actual_loads = sorted({_parse_load(item) for item in paths})
            if actual_loads != sorted(declared_loads):
                raise ValueError(
                    f"manifest {split} 的文件工况 {actual_loads} 与声明 {sorted(declared_loads)} 不一致"
                )

    if manifest.get("complete_partition", False) and set(seen) != set(all_files):
        missing = sorted(item.relative_to(root).as_posix() for item in set(all_files).difference(seen))
        raise ValueError(f"complete_partition=true 但 manifest 未覆盖文件: {missing}")
    if manifest.get("require_all_classes", False):
        for split, paths in selected.items():
            labels = {_parse_label(item) for item in paths}
            if labels != set(CLASS_TO_INDEX.values()):
                raise ValueError(f"manifest {split} 未覆盖全部四类标签: {sorted(labels)}")
    return manifest, selected


class CWRUDataset(Dataset):
    """读取显式文件集合中的 CWRU DE/FE 窗口及 split 无关 embedding。

    ``split='all'`` 仅供 embedding 生成脚本枚举全部原始窗口；训练、验证和测试必须
    提供 ``manifest_path``，防止任何实验在运行时悄悄退回随机切分。
    """

    def __init__(
        self,
        root_path: str | Path,
        split: str,
        window_size: int = 1024,
        stride: int = 1024,
        manifest_path: str | Path | None = None,
        embedding_root: str | Path | None = None,
    ) -> None:
        if split not in {*DATA_SPLITS, "all"}:
            raise ValueError(f"split 必须为 train/val/test/all，实际为 {split}")
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size 和 stride 必须为正数")

        self.root_path, all_files = discover_cwru_files(root_path)
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.embedding_root = Path(embedding_root).expanduser().resolve() if embedding_root else None
        self.num_classes = len(CLASS_TO_INDEX)
        self._signal_cache: dict[Path, np.ndarray] = {}

        if split == "all":
            selected_files = all_files
            self.manifest: dict[str, Any] | None = None
        else:
            if manifest_path is None:
                raise ValueError("训练/验证/测试数据集必须显式提供 split manifest")
            self.manifest, file_splits = load_split_manifest(self.root_path, manifest_path)
            selected_files = file_splits[split]
        self.records = self._build_records(selected_files)
        if not self.records:
            raise RuntimeError(f"{split} 集没有生成任何长度为 {window_size} 的信号窗口")

        self.cache_spec: dict[str, Any] | None = None
        self.cache_spec_digest: str | None = None
        if self.embedding_root is not None:
            self.cache_spec, self.cache_spec_digest = load_cache_spec(self.embedding_root)
            self._validate_cache_geometry()

    def _validate_cache_geometry(self) -> None:
        """在第一个 H5 被读取前发现窗口、patch 或传感器配置不一致。"""
        assert self.cache_spec is not None
        spec = self.cache_spec
        expected_patches = 1 + (self.window_size - spec["patch_len"]) // spec["patch_stride"]
        if self.window_size != spec["window_size"] or self.stride != spec["window_stride"]:
            raise ValueError("训练窗口配置与 embedding cache_spec.json 不一致")
        if self.window_size < spec["patch_len"] or (self.window_size - spec["patch_len"]) % spec["patch_stride"] != 0:
            raise ValueError("训练窗口无法按缓存 patch 配置完整切分")
        if expected_patches != spec["num_patches"]:
            raise ValueError("训练窗口推导出的 patch 数与缓存不一致")

    @staticmethod
    def _read_sensor_signal(path: Path, sensor: str) -> np.ndarray:
        """读取指定 DE/FE 字段，并优先匹配文件名尾部的实验编号。"""
        sensor = sensor.upper()
        if sensor not in {"DE", "FE"}:
            raise ValueError(f"不支持的 CWRU 传感器: {sensor}")
        contents = loadmat(path)
        candidates = [key for key in contents if key.lower().endswith(f"_{sensor.lower()}_time")]
        file_id = re.search(r"_(\d+)$", path.stem)
        expected_key = f"X{int(file_id.group(1)):03d}_{sensor}_time" if file_id else None
        if expected_key in candidates:
            selected = expected_key
        elif len(candidates) == 1:
            selected = candidates[0]
        else:
            raise ValueError(f"{path.name} 中无法唯一匹配 {sensor} 信号字段: {candidates}")
        signal = np.asarray(contents[selected], dtype=np.float32).reshape(-1)
        if signal.size == 0:
            raise ValueError(f"{path.name} 的 {sensor} 信号为空")
        return signal

    def _load_signal(self, path: Path) -> np.ndarray:
        """懒加载同步 DE、FE 信号，返回 ``[时间长度, 2]``。"""
        if path not in self._signal_cache:
            de_signal = self._read_sensor_signal(path, "DE")
            fe_signal = self._read_sensor_signal(path, "FE")
            if de_signal.shape != fe_signal.shape:
                raise ValueError(f"{path.name} 的 DE/FE 长度不一致: {de_signal.size} vs {fe_signal.size}")
            self._signal_cache[path] = np.stack((de_signal, fe_signal), axis=-1)
        return self._signal_cache[path]

    def _build_records(self, paths: list[Path]) -> list[WindowRecord]:
        """先固定文件集合，再在每个文件内部建立不重叠窗口索引。"""
        records: list[WindowRecord] = []
        for path in paths:
            signal_length = self._load_signal(path).shape[0]
            for start in range(0, signal_length - self.window_size + 1, self.stride):
                records.append(WindowRecord(path, start, _parse_label(path), _parse_load(path)))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        signal = self._load_signal(record.path)[record.start:record.start + self.window_size]
        signal_tensor = torch.from_numpy(signal.copy())
        label_tensor = torch.tensor(record.label, dtype=torch.long)

        if self.embedding_root is None:
            return signal_tensor, label_tensor, torch.tensor(record.load_hp, dtype=torch.long), record.sample_id

        embedding_path = self.embedding_root / "by_sample" / f"{record.sample_id}.h5"
        if not embedding_path.is_file():
            raise FileNotFoundError(f"缺少样本 embedding: {embedding_path}")
        with h5py.File(embedding_path, "r") as handle:
            if handle.attrs.get("cache_spec_sha256") != self.cache_spec_digest:
                raise ValueError(f"embedding 缓存配方摘要不一致: {embedding_path}")
            embedding = torch.from_numpy(handle["embedding"][:].astype(np.float32, copy=False))
        assert self.cache_spec is not None
        expected_shape = (self.cache_spec["embedding_dim"], self.cache_spec["num_patches"], 1)
        if tuple(embedding.shape) != expected_shape:
            raise ValueError(f"embedding shape 不匹配: 期望 {expected_shape}，实际 {tuple(embedding.shape)}")
        return signal_tensor, label_tensor, embedding
