"""汇总 dual、Signal-only、Prompt-only 的四折 LOLO × 多种子指标。

脚本只读取每个单元的 ``metrics.json``，不训练模型、不修改 checkpoint。三件事：

1. 校验同一 (fold, seed) 下三个模式使用完全相同的 manifest 与 embedding 缓存摘要；
2. 按“先种子内、后折间”的两层结构聚合，把初始化方差与划分方差分开；
3. 以 (fold, seed) 为单位和 dual 做配对差分，回答“prompt 分支是否带来增益”。

配对差分的离散度才是判断增益是否存在的依据：同一折两模式的共同难度会在相减时抵消，
直接比较两个均值会把这份共同难度误算成模式差异。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


FOLDS = (
    "test_0hp_val_1hp",
    "test_1hp_val_2hp",
    "test_2hp_val_3hp",
    "test_3hp_val_0hp",
)
MODES = ("dual", "signal_only", "prompt_only")
CLASS_NAMES = ("normal", "ball", "inner", "outer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 CWRU 四折 LOLO 模态消融结果")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("Results/CWRU_TimeCMA_FD/v7_prompt_lolo_load_v1"),
        help="含 <mode>/seed_<N>/<fold>/metrics.json 的结果根目录",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026], help="参与聚合的种子")
    parser.add_argument(
        "--modes", nargs="+", default=list(MODES), choices=list(MODES),
        help="参与聚合的模式；dual 始终需要，因为它是配对基准",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Results/CWRU_TimeCMA_FD/v7_prompt_lolo_load_v1/summary.md"),
        help="输出 Markdown 报告路径",
    )
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="允许某个 (mode, seed, fold) 缺失；默认缺失即报错，避免半成品被当成结果",
    )
    return parser.parse_args()


def _safe_std(values: list[float]) -> float:
    """样本标准差；只有 1 个观测时返回 0 而不是抛错。"""
    return stdev(values) if len(values) > 1 else 0.0


def _f1_by_class(confusion: list[list[int]]) -> dict[str, float]:
    """从“行是真实类别、列是预测类别”的混淆矩阵重算四类 F1。"""
    values: dict[str, float] = {}
    for index, name in enumerate(CLASS_NAMES):
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(len(CLASS_NAMES))) - true_positive
        false_negative = sum(confusion[index]) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        values[name] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return values


def cell_path(root: Path, mode: str, seed: int, fold: str) -> Path:
    """<root>/<mode>/seed_<N>/<fold>/metrics.json。"""
    return root / mode / f"seed_{seed}" / fold / "metrics.json"


def load_cell(path: Path, mode: str) -> dict[str, Any]:
    """读取一个单元并校验其记录的模式与调用方期望一致。"""
    if not path.is_file():
        raise FileNotFoundError(f"缺少结果文件: {path}")
    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    recorded = report.get("ablation", "dual")
    if recorded != mode:
        raise ValueError(f"{path} 记录的模式为 {recorded!r}，期望 {mode!r}")
    metrics = report["test_metrics"]
    return {
        "best_epoch": report["best_epoch"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "class_f1": _f1_by_class(metrics["confusion_matrix"]),
        "manifest_sha256": report["split_manifest_sha256"],
        "cache_spec_sha256": report["cache_spec_sha256"],
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    args = parse_args()
    modes = list(args.modes)
    if "dual" not in modes:
        raise ValueError("--modes 必须包含 dual：它是配对基准")

    # cells[mode][fold][seed] = 指标
    cells: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
        mode: {fold: {} for fold in FOLDS} for mode in modes
    }
    missing: list[str] = []
    for mode in modes:
        for fold in FOLDS:
            for seed in args.seeds:
                path = cell_path(args.result_root, mode, seed, fold)
                if not path.is_file():
                    missing.append(f"{mode}/seed_{seed}/{fold}")
                    continue
                cells[mode][fold][seed] = load_cell(path, mode)

    if missing:
        message = f"共 {len(missing)} 个单元缺失，例: " + ", ".join(missing[:6])
        if not args.allow_missing:
            raise FileNotFoundError(message + "（如需部分汇总请加 --allow-missing）")
        print(f"[warn] {message}")

    # 同一 (fold, seed) 下所有模式必须来自同一份划分与同一份 prompt 缓存。
    for fold in FOLDS:
        for seed in args.seeds:
            reference = cells["dual"][fold].get(seed)
            if reference is None:
                continue
            for mode in modes:
                current = cells[mode][fold].get(seed)
                if current is None:
                    continue
                if current["manifest_sha256"] != reference["manifest_sha256"]:
                    raise ValueError(f"{mode}/seed_{seed}/{fold} 的 manifest 与 dual 不一致")
                if current["cache_spec_sha256"] != reference["cache_spec_sha256"]:
                    raise ValueError(f"{mode}/seed_{seed}/{fold} 的 embedding cache 与 dual 不一致")

    lines: list[str] = [
        "# CWRU 四折 LOLO 模态消融汇总",
        "",
        f"- 种子: {', '.join(str(s) for s in args.seeds)}",
        "- 每格 = 该 (mode, fold) 在可用种子上的 mean ± sample std（**初始化方差**）。",
        "- 四折聚合 = 4 个 fold-mean 的 mean ± sample std（**划分方差**）。",
        "- Δ = 该模式减同一 (fold, seed) 的 dual，先配对再聚合；配对消去了每折的共同难度。",
        "- 校验：同一 (fold, seed) 下各模式的 manifest 与 embedding cache 摘要必须完全一致。",
        "",
    ]

    # 表 1：每 (mode, fold) 跨种子
    lines += [
        "## 表 1 每 (模式, 折) 跨种子",
        "",
        "| Mode | Fold | n_seed | Accuracy | Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for mode in modes:
        for fold in FOLDS:
            seed_cells = cells[mode][fold]
            if not seed_cells:
                lines.append(f"| {mode} | {fold} | 0 | — | — |")
                continue
            acc = [c["accuracy"] for c in seed_cells.values()]
            f1 = [c["macro_f1"] for c in seed_cells.values()]
            lines.append(
                f"| {mode} | {fold} | {len(seed_cells)} | "
                f"{pct(mean(acc))} ± {_safe_std(acc) * 100:.2f} | "
                f"{pct(mean(f1))} ± {_safe_std(f1) * 100:.2f} |"
            )

    # 表 2：四折聚合 + 类别 F1
    lines += [
        "",
        "## 表 2 四折聚合与类别 F1",
        "",
        "| Mode | Accuracy | Macro-F1 | Normal F1 | Ball F1 | Inner F1 | Outer F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in modes:
        fold_acc: list[float] = []
        fold_f1: list[float] = []
        class_fold: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}
        for fold in FOLDS:
            seed_cells = cells[mode][fold]
            if not seed_cells:
                continue
            fold_acc.append(mean(c["accuracy"] for c in seed_cells.values()))
            fold_f1.append(mean(c["macro_f1"] for c in seed_cells.values()))
            for name in CLASS_NAMES:
                class_fold[name].append(mean(c["class_f1"][name] for c in seed_cells.values()))
        if not fold_f1:
            lines.append(f"| {mode} | — | — | — | — | — | — |")
            continue
        class_cells = [
            f"{pct(mean(class_fold[name]))} ± {_safe_std(class_fold[name]) * 100:.2f}" if class_fold[name] else "—"
            for name in CLASS_NAMES
        ]
        lines.append(
            f"| {mode} | {pct(mean(fold_acc))} ± {_safe_std(fold_acc) * 100:.2f} | "
            f"{pct(mean(fold_f1))} ± {_safe_std(fold_f1) * 100:.2f} | "
            + " | ".join(class_cells)
            + " |"
        )

    # 表 3：配对 Δ（相对 dual）
    lines += [
        "",
        "## 表 3 与 dual 的配对差 Δ（每折先算 seed 内均值，再跨折聚合）",
        "",
        "| Mode | " + " | ".join(FOLDS) + " | 折间 mean ± std | 同向折数 |",
        "| --- | " + " | ".join("---:" for _ in FOLDS) + " | ---: | ---: |",
    ]
    for mode in modes:
        if mode == "dual":
            continue
        per_fold_deltas: list[float] = []
        row_cells: list[str] = []
        for fold in FOLDS:
            paired = [
                cells[mode][fold][seed]["macro_f1"] - cells["dual"][fold][seed]["macro_f1"]
                for seed in args.seeds
                if seed in cells[mode][fold] and seed in cells["dual"][fold]
            ]
            if not paired:
                row_cells.append("—")
                continue
            fold_delta = mean(paired)
            per_fold_deltas.append(fold_delta)
            row_cells.append(f"{fold_delta * 100:+.2f} ± {_safe_std(paired) * 100:.2f}")
        if not per_fold_deltas:
            lines.append(f"| {mode} | " + " | ".join(row_cells) + " | — | — |")
            continue
        positive = sum(1 for d in per_fold_deltas if d > 0)
        negative = sum(1 for d in per_fold_deltas if d < 0)
        lines.append(
            f"| {mode} | " + " | ".join(row_cells)
            + f" | {mean(per_fold_deltas) * 100:+.2f} ± {_safe_std(per_fold_deltas) * 100:.2f} pp"
            + f" | {positive}正/{negative}负 |"
        )

    lines += [
        "",
        "> 判读方式：若 Δ 的折间 mean 落在 ±1 个折间 std 内、且同向折数接近 2:2，则四个折",
        "> 无法把这组差异与初始化和划分噪声区分开；此时应结合 `docs/V4_PROMPT_BRANCH_AUDIT.md`",
        "> 的反事实测试结果一起判断，而不是把该 Δ 当作增益证据。",
        "",
        "## 表 4 每 (模式, 折, 种子) 明细",
        "",
        "| Mode | Fold | Seed | Best epoch | Accuracy | Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in modes:
        for fold in FOLDS:
            for seed in sorted(cells[mode][fold]):
                c = cells[mode][fold][seed]
                lines.append(
                    f"| {mode} | {fold} | {seed} | {c['best_epoch']} | {pct(c['accuracy'])} | {pct(c['macro_f1'])} |"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入汇总报告: {args.output}")


if __name__ == "__main__":
    main()
