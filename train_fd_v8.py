"""V8 三种语义对齐模式的 CWRU 训练入口；仅使用验证集 Macro-F1 选模。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_provider.cwru_dataset import CWRUDataset, manifest_sha256
from models.TimeCMA_FD_V8 import TimeCMAFaultDiagnosisV8, V8_MODES
from train_fd import set_seed
from utils.embedding_cache import load_cache_spec
from utils.metrics import classification_metrics
from utils.phm_prompt_v8 import V8_POOLING_LAYER, prototype_text_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 V8 故障知识原型 / 样本证据 / 两者组合")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--text-root", type=Path, required=True, help="含 cache_spec.json、原型和 by_sample/ 的 V8 根目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=V8_MODES, required=True)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--patch-len", type=int, default=256)
    parser.add_argument("--patch-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
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
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--prototype-weight", type=float, default=0.5, help="P+E 融合前 Signal→Prototype 辅助损失权重")
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def load_prototypes(text_root: Path, cache_digest: str, text_digest: str) -> torch.Tensor:
    """校验类别顺序、模板摘要与缓存配方后读取冻结原型。"""
    path = text_root / "fault_prototypes.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["cache_spec_sha256"] != cache_digest or saved["prototype_text_sha256"] != text_digest:
        raise ValueError(f"固定故障原型与当前 V8 缓存配方不一致: {path}")
    if saved["class_order"] != ["normal", "ball", "inner", "outer"]:
        raise ValueError("故障原型类别顺序与 CWRU 标签不一致")
    return saved["prototypes"]


def unpack_batch(batch, mode: str):
    """P 模式不读取任何 H5；E/P+E 从同一 H5 缓存读取证据。"""
    if mode == "prototype_only":
        signals, labels, _loads, _sample_ids = batch
        return signals, labels, None
    signals, labels, embeddings = batch
    return signals, labels, embeddings


def evaluate(
    model: TimeCMAFaultDiagnosisV8, loader: DataLoader, criterion: nn.Module,
    device: torch.device, num_classes: int, prototype_weight: float, desc: str,
) -> dict:
    """按整个 split 汇总指标；P+E 同时汇总融合前 Signal→Prototype 表现。"""
    model.eval()
    total_loss = 0.0
    final_predictions: list[torch.Tensor] = []
    signal_predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
            signals, labels, embeddings = unpack_batch(batch, model.mode)
            labels_device = labels.to(device)
            evidence_device = embeddings.to(device) if embeddings is not None else None
            logits, signal_logits = model(signals.to(device), evidence_device)
            loss = criterion(logits, labels_device)
            if signal_logits is not None:
                loss = loss + prototype_weight * criterion(signal_logits, labels_device)
                signal_predictions.append(signal_logits.argmax(dim=1).cpu())
            total_loss += loss.item() * labels.shape[0]
            final_predictions.append(logits.argmax(dim=1).cpu())
            targets.append(labels)
    truth = torch.cat(targets)
    result = classification_metrics(torch.cat(final_predictions), truth, num_classes)
    result["loss"] = total_loss / len(loader.dataset)
    if signal_predictions:
        result["signal_prototype_metrics"] = classification_metrics(
            torch.cat(signal_predictions), truth, num_classes,
        )
    return result


def main() -> None:
    args = parse_args()
    if args.mode == "prototype_evidence" and args.prototype_weight <= 0:
        raise ValueError("prototype-weight 必须大于 0，确保 P+E 训练融合前的信号对齐")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs 和 patience 必须大于 0")
    if args.patch_len <= 0 or args.patch_stride <= 0 or args.window_size < args.patch_len:
        raise ValueError("训练窗口与 patch 配置不合法")
    if (args.window_size - args.patch_len) % args.patch_stride:
        raise ValueError("patch_stride 不能完整覆盖训练窗口")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec, spec_digest = load_cache_spec(args.text_root)
    expected = {
        "schema_version": 3,
        "window_size": args.window_size,
        "window_stride": args.stride,
        "patch_len": args.patch_len,
        "patch_stride": args.patch_stride,
        "num_patches": 1 + (args.window_size - args.patch_len) // args.patch_stride,
        "embedding_dim": args.d_llm,
        "sensors": ["DE", "FE"],
        "sampling_rate": 12000,
        "include_load_hp": False,
        "gpt_model": "gpt2",
        "prompt_template_version": "v8_diagnostic_context_evidence_v1",
        "pooling": "evidence_span_mean",
        "pooling_layer": V8_POOLING_LAYER,
        "prototype_template_version": "v8_fault_concept_mechanism_v1",
        "prototype_pooling": "all_text_mean",
        "prototype_transform": "four_class_mean_center_l2",
        "prototype_text_sha256": prototype_text_sha256(),
    }
    for key, value in expected.items():
        if spec[key] != value:
            raise ValueError(f"V8 训练配置 {key}={value} 与文本缓存 {spec[key]} 不一致")
    uses_evidence = args.mode != "prototype_only"
    prototypes = None if args.mode == "evidence_only" else load_prototypes(
        args.text_root, spec_digest, spec["prototype_text_sha256"],
    )
    datasets = {
        split: CWRUDataset(
            args.data_root, split, args.window_size, args.stride,
            manifest_path=args.split_manifest,
            embedding_root=args.text_root if uses_evidence else None,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=split == "train", num_workers=args.num_workers,
        )
        for split, dataset in datasets.items()
    }
    model = TimeCMAFaultDiagnosisV8(
        mode=args.mode, prototypes=prototypes, num_nodes=len(spec["sensors"]),
        seq_len=args.window_size, num_classes=datasets["train"].num_classes,
        channel=args.channel, d_llm=args.d_llm, patch_len=args.patch_len,
        patch_stride=args.patch_stride, encoder_layers=args.encoder_layers,
        heads=args.heads, align_dim=args.align_dim, dropout=args.dropout,
        temperature=args.temperature,
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
        accumulated_loss = 0.0
        progress = tqdm(loaders["train"], desc=f"V8 {args.mode} epoch {epoch}/{args.epochs}", unit="batch", dynamic_ncols=True)
        for batch in progress:
            signals, labels, embeddings = unpack_batch(batch, args.mode)
            labels_device = labels.to(device)
            evidence_device = embeddings.to(device) if embeddings is not None else None
            optimizer.zero_grad(set_to_none=True)
            logits, signal_logits = model(signals.to(device), evidence_device)
            loss = criterion(logits, labels_device)
            if signal_logits is not None:
                loss = loss + args.prototype_weight * criterion(signal_logits, labels_device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            accumulated_loss += loss.item() * labels.shape[0]
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        val_metrics = evaluate(
            model, loaders["val"], criterion, device, datasets["val"].num_classes,
            args.prototype_weight, f"Validation {epoch}",
        )
        train_loss = accumulated_loss / len(datasets["train"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_metrics": val_metrics})
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1, stale_epochs = val_metrics["macro_f1"], 0
            torch.save(
                {
                    "model_state": model.state_dict(), "args": vars(args), "epoch": epoch,
                    "val_metrics": val_metrics,
                    "split_manifest_sha256": manifest_sha256(args.split_manifest),
                    "cache_spec_sha256": spec_digest,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"验证集 Macro-F1 连续 {args.patience} 个 epoch 未提升，提前停止。")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    split_metrics = {
        split: evaluate(model, loader, criterion, device, datasets[split].num_classes,
                        args.prototype_weight, f"Final {split}")
        for split, loader in loaders.items()
    }
    class_counts = {
        split: dict(sorted(Counter(record.label for record in dataset.records).items()))
        for split, dataset in datasets.items()
    }
    report = {
        "v8_mode": args.mode,
        "best_epoch": checkpoint["epoch"],
        "train_metrics": split_metrics["train"],
        "val_metrics": split_metrics["val"],
        "test_metrics": split_metrics["test"],
        "class_counts_by_index": class_counts,
        "history": history,
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": manifest_sha256(args.split_manifest),
        "cache_spec": spec,
        "cache_spec_sha256": spec_digest,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({"best_epoch": checkpoint["epoch"], "split_metrics": split_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
