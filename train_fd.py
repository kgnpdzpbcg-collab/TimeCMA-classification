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
from tqdm.auto import tqdm

from data_provider.cwru_dataset import CWRUDataset, manifest_sha256
from models.TimeCMA_FD import TimeCMAFaultDiagnosis
from utils.embedding_cache import load_cache_spec
from utils.metrics import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 TimeCMA CMA 进行 CWRU 故障分类")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True, help="显式 MAT 文件级划分 manifest")
    parser.add_argument("--embedding-root", type=Path, required=True, help="含 cache_spec.json 与 by_sample/ 的缓存根目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="当前 fold 的独立结果目录")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--patch-len", type=int, default=256)
    parser.add_argument("--patch-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0, help="Windows 推荐保持 0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--channel", type=int, default=64)
    parser.add_argument("--d-llm", type=int, default=768)
    parser.add_argument("--align-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--ablation",
        choices=("dual", "signal_only", "prompt_only"),
        default="dual",
        help="模型输入消融模式；dual 保持 V4 原始双模态结构",
    )
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定模型初始化与 DataLoader 顺序；文件划分由 manifest 固定，不依赖随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unpack_batch(batch, ablation: str):
    """统一单模态与双模态 DataLoader 的返回结构。

    Signal-only 数据集不传入 embedding_root，因此返回信号、标签和仅用于追溯的
    负载/样本 ID；该模式不会打开任何逐样本 prompt H5。其余模式仍返回三元组。
    """
    if ablation == "signal_only":
        signals, labels, _loads, _sample_ids = batch
        return signals, labels, None
    signals, labels, embeddings = batch
    return signals, labels, embeddings


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    desc: str,
    ablation: str,
):
    """汇总整个数据集后再计算 Macro-F1，避免按 batch 平均造成统计偏差。"""
    model.eval()
    losses: list[float] = []
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.inference_mode():
        progress = tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True)
        for batch in progress:
            signals, labels, embeddings = unpack_batch(batch, ablation)
            prompt_inputs = embeddings.to(device) if embeddings is not None else None
            logits = model(signals.to(device), prompt_inputs)
            batch_loss = criterion(logits, labels.to(device)).item()
            losses.append(batch_loss)
            all_predictions.append(logits.argmax(dim=1).cpu())
            all_targets.append(labels.cpu())
            progress.set_postfix(loss=f"{batch_loss:.4f}")
    metrics = classification_metrics(torch.cat(all_predictions), torch.cat(all_targets), num_classes)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_spec, cache_spec_digest = load_cache_spec(args.embedding_root)
    expected = {
        "window_size": args.window_size,
        "window_stride": args.stride,
        "patch_len": args.patch_len,
        "patch_stride": args.patch_stride,
        "embedding_dim": args.d_llm,
    }
    for key, value in expected.items():
        if cache_spec[key] != value:
            raise ValueError(f"训练参数 {key}={value} 与缓存配置 {cache_spec[key]} 不一致")
    datasets = {
        split: CWRUDataset(
            args.data_root, split, args.window_size, args.stride,
            # Signal-only 仅使用波形，禁止为它加载逐样本 prompt；其余模式使用同一缓存。
            manifest_path=args.split_manifest,
            embedding_root=None if args.ablation == "signal_only" else args.embedding_root,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        "train": DataLoader(datasets["train"], args.batch_size, shuffle=True, num_workers=args.num_workers),
        "val": DataLoader(datasets["val"], args.batch_size, shuffle=False, num_workers=args.num_workers),
        "test": DataLoader(datasets["test"], args.batch_size, shuffle=False, num_workers=args.num_workers),
    }
    model = TimeCMAFaultDiagnosis(
        num_nodes=len(cache_spec["sensors"]), seq_len=args.window_size, num_classes=datasets["train"].num_classes,
        patch_len=args.patch_len, patch_stride=args.patch_stride,
        channel=args.channel, d_llm=args.d_llm, align_dim=args.align_dim,
        e_layer=args.encoder_layers, head=args.heads, dropout=args.dropout, ablation=args.ablation,
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
        progress = tqdm(loaders["train"], desc=f"Epoch {epoch}/{args.epochs}", unit="batch", dynamic_ncols=True)
        for batch in progress:
            signals, labels, embeddings = unpack_batch(batch, args.ablation)
            prompt_inputs = embeddings.to(device) if embeddings is not None else None
            optimizer.zero_grad(set_to_none=True)
            logits = model(signals.to(device), prompt_inputs)
            loss = criterion(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step()

        val_metrics = evaluate(
            model, loaders["val"], criterion, device, datasets["val"].num_classes,
            desc=f"Validation {epoch}/{args.epochs}", ablation=args.ablation,
        )
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **val_metrics}
        history.append(row)
        print(f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} val_f1={row['macro_f1']:.4f} val_acc={row['accuracy']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1, stale_epochs = val_metrics["macro_f1"], 0
            torch.save({
                "model_state": model.state_dict(), "args": vars(args), "epoch": epoch,
                "val_metrics": val_metrics, "split_manifest_sha256": manifest_sha256(args.split_manifest),
                "cache_spec_sha256": cache_spec_digest,
            }, args.output_dir / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"验证集 Macro-F1 连续 {args.patience} 个 epoch 未提升，提前停止。")
                break

    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(
        model, loaders["test"], criterion, device, datasets["test"].num_classes,
        desc="Final test", ablation=args.ablation,
    )
    report = {
        # 显式保存模式，防止后续汇总时把单模态结果误当成原始 V4 双模态结果。
        "ablation": args.ablation,
        "best_epoch": checkpoint["epoch"], "best_val_metrics": checkpoint["val_metrics"],
        "test_metrics": test_metrics, "history": history,
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": manifest_sha256(args.split_manifest),
        "cache_spec": cache_spec, "cache_spec_sha256": cache_spec_digest,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({"best_epoch": checkpoint["epoch"], "test_metrics": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
