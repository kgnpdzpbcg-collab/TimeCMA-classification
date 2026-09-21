"""CWRU TimeCMA 故障诊断训练入口。

与原 train.py 分离，确保 forecasting 基线保持不变；模型选择严格只使用验证集 Macro-F1。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_provider.cwru_dataset import CWRUDataset
from models.TimeCMA_FD import TimeCMAFaultDiagnosis
from utils.metrics import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 TimeCMA CMA 进行 CWRU 故障分类")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, default=Path("Embeddings/CWRU"))
    parser.add_argument("--output-dir", type=Path, default=Path("Results/CWRU_TimeCMA_FD"))
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0, help="Windows 推荐保持 0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--channel", type=int, default=64)
    parser.add_argument("--d-llm", type=int, default=768)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定常用随机源，确保文件切分、窗口顺序和模型初始化均可重现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int):
    """汇总整个数据集后再计算 Macro-F1，避免按 batch 平均造成统计偏差。"""
    model.eval()
    losses: list[float] = []
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.inference_mode():
        for signals, labels, embeddings in loader:
            logits = model(signals.to(device), embeddings.to(device))
            losses.append(criterion(logits, labels.to(device)).item())
            all_predictions.append(logits.argmax(dim=1).cpu())
            all_targets.append(labels.cpu())
    metrics = classification_metrics(torch.cat(all_predictions), torch.cat(all_targets), num_classes)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        split: CWRUDataset(args.data_root, split, args.window_size, args.stride, args.seed, embedding_root=args.embedding_root)
        for split in ("train", "val", "test")
    }
    loaders = {
        "train": DataLoader(datasets["train"], args.batch_size, shuffle=True, num_workers=args.num_workers),
        "val": DataLoader(datasets["val"], args.batch_size, shuffle=False, num_workers=args.num_workers),
        "test": DataLoader(datasets["test"], args.batch_size, shuffle=False, num_workers=args.num_workers),
    }
    model = TimeCMAFaultDiagnosis(
        num_nodes=1, seq_len=args.window_size, num_classes=datasets["train"].num_classes,
        channel=args.channel, d_llm=args.d_llm, e_layer=args.encoder_layers, head=args.heads, dropout=args.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1, stale_epochs = -1.0, 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for signals, labels, embeddings in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(signals.to(device), embeddings.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        val_metrics = evaluate(model, loaders["val"], criterion, device, datasets["val"].num_classes)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **val_metrics}
        history.append(row)
        print(f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} val_f1={row['macro_f1']:.4f} val_acc={row['accuracy']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1, stale_epochs = val_metrics["macro_f1"], 0
            torch.save({"model_state": model.state_dict(), "args": vars(args), "epoch": epoch, "val_metrics": val_metrics}, args.output_dir / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"验证集 Macro-F1 连续 {args.patience} 个 epoch 未提升，提前停止。")
                break

    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, loaders["test"], criterion, device, datasets["test"].num_classes)
    report = {"best_epoch": checkpoint["epoch"], "best_val_metrics": checkpoint["val_metrics"], "test_metrics": test_metrics, "history": history}
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({"best_epoch": checkpoint["epoch"], "test_metrics": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
