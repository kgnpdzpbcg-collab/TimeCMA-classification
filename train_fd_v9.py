"""V9 的 CWRU 四折训练入口：CNN 信号向量直接匹配固定 GPT-2 故障机理原型。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_provider.cwru_dataset import CWRUDataset, manifest_sha256
from models.PrototypeCNN_FD_V9 import PrototypeCNNFaultDiagnosisV9
from train_fd_cnn import evaluate, set_seed
from utils.fault_knowledge_v9 import load_fixed_prototypes, prototype_recipe, recipe_sha256


def parse_args() -> argparse.Namespace:
    """复用 CNN/V8 的数据、优化器和选模默认值，只增加固定 Prototype 路径。"""
    parser = argparse.ArgumentParser(description="训练 V9 CNN—故障机理 Prototype 双分支模型")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--prototype-path", type=Path, required=True)
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
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main() -> None:
    """逐折训练；只按验证集 Macro-F1 选 checkpoint，最后报告三个 split。"""
    args = parse_args()
    if args.window_size < 16 or args.stride <= 0 or args.batch_size <= 0:
        raise ValueError("窗口长度、步长或 batch size 不合法")
    if args.epochs <= 0 or args.patience <= 0 or args.num_workers < 0:
        raise ValueError("epochs、patience 或 num_workers 不合法")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("学习率或权重衰减不合法")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prototypes = load_fixed_prototypes(args.prototype_path)
    recipe_digest = recipe_sha256(prototype_recipe())
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
    model = PrototypeCNNFaultDiagnosisV9(
        prototypes=prototypes, num_nodes=2, seq_len=args.window_size,
        base_channels=args.base_channels, dropout=args.dropout, temperature=args.temperature,
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
            loaders["train"], desc=f"V9 epoch {epoch}/{args.epochs}",
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
            model, loaders["val"], criterion, device,
            datasets["val"].num_classes, f"V9 validation {epoch}",
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
                    "prototype_recipe_sha256": recipe_digest,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"验证集 Macro-F1 连续 {args.patience} 个 epoch 未提升，提前停止。")
                break

    # 测试集直到最佳 checkpoint 确定后才评估，避免目标负载影响模型选择。
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
        "model": "v9_cnn_fixed_fault_prototypes",
        "best_epoch": checkpoint["epoch"],
        "train_metrics": split_metrics["train"],
        "val_metrics": split_metrics["val"],
        "test_metrics": split_metrics["test"],
        "class_counts_by_index": class_counts,
        "history": history,
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": manifest_sha256(args.split_manifest),
        "prototype_path": str(args.prototype_path.resolve()),
        "prototype_recipe_sha256": recipe_digest,
        "seed": args.seed,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({"best_epoch": checkpoint["epoch"], "test_metrics": split_metrics["test"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
