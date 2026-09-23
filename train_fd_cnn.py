"""用与 V8 相同的 CWRU 文件级 LOLO 协议训练纯信号 1D CNN 基线。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_provider.cwru_dataset import CWRUDataset, manifest_sha256
from models.CNN_FD import CWRUCNN
from utils.metrics import classification_metrics


def parse_args() -> argparse.Namespace:
    """沿用 V8 的数据窗口、优化器及选模默认值；每次运行指定一折 manifest。"""
    parser = argparse.ArgumentParser(description="CWRU DE/FE 纯信号 1D CNN 基线")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定模型初始化和训练集打乱；文件划分始终由 manifest 决定。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
    device: torch.device, num_classes: int, desc: str,
) -> dict:
    """在整个 split 上汇总预测后计算 Macro-F1 和混淆矩阵。"""
    model.eval()
    total_loss = 0.0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.inference_mode():
        for signals, labels, _loads, _sample_ids in tqdm(
            loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True,
        ):
            labels_device = labels.to(device)
            logits = model(signals.to(device))
            total_loss += criterion(logits, labels_device).item() * labels.shape[0]
            predictions.append(logits.argmax(dim=1).cpu())
            targets.append(labels)
    result = classification_metrics(torch.cat(predictions), torch.cat(targets), num_classes)
    result["loss"] = total_loss / len(loader.dataset)
    return result


def main() -> None:
    """仅按验证集 Macro-F1 保存最优模型；测试集只在训练结束后读取。"""
    args = parse_args()
    if args.window_size < 16 or args.stride <= 0 or args.batch_size <= 0:
        raise ValueError("窗口长度、步长或 batch size 不合法")
    if args.epochs <= 0 or args.patience <= 0 or args.num_workers < 0:
        raise ValueError("epochs、patience 或 num_workers 不合法")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("学习率或权重衰减不合法")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        split: CWRUDataset(
            args.data_root, split, args.window_size, args.stride,
            manifest_path=args.split_manifest, embedding_root=None,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=split == "train", num_workers=args.num_workers,
        )
        for split, dataset in datasets.items()
    }
    model = CWRUCNN(
        num_nodes=2, seq_len=args.window_size, num_classes=datasets["train"].num_classes,
        base_channels=args.base_channels, dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    best_f1, stale_epochs = -1.0, 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        progress = tqdm(
            loaders["train"], desc=f"CNN epoch {epoch}/{args.epochs}",
            unit="batch", dynamic_ncols=True,
        )
        for signals, labels, _loads, _sample_ids in progress:
            labels_device = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(signals.to(device))
            loss = criterion(logits, labels_device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item() * labels.shape[0]
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        val_metrics = evaluate(
            model, loaders["val"], criterion, device, datasets["val"].num_classes,
            f"CNN validation {epoch}",
        )
        train_loss = total_train_loss / len(datasets["train"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_metrics": val_metrics})
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1, stale_epochs = val_metrics["macro_f1"], 0
            torch.save(
                {
                    "model_state": model.state_dict(), "args": vars(args), "epoch": epoch,
                    "val_metrics": val_metrics,
                    "split_manifest_sha256": manifest_sha256(args.split_manifest),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"验证集 Macro-F1 连续 {args.patience} 个 epoch 未提升，提前停止。")
                break

    # 只载入验证集选出的 checkpoint；测试集不参与训练与模型选择。
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    split_metrics = {
        split: evaluate(model, loader, criterion, device, datasets[split].num_classes, f"Final {split}")
        for split, loader in loaders.items()
    }
    class_counts = {
        split: dict(sorted(Counter(record.label for record in dataset.records).items()))
        for split, dataset in datasets.items()
    }
    report = {
        "model": "cnn_1d_signal_only",
        "best_epoch": checkpoint["epoch"],
        "train_metrics": split_metrics["train"],
        "val_metrics": split_metrics["val"],
        "test_metrics": split_metrics["test"],
        "class_counts_by_index": class_counts,
        "history": history,
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": manifest_sha256(args.split_manifest),
        "seed": args.seed,
        "normalization": "per_window_per_sensor_revin_affine_false",
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({"best_epoch": checkpoint["epoch"], "test_metrics": split_metrics["test"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
