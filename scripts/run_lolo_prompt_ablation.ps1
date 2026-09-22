<#!
运行 TimeCMA-FD 四折 LOLO 的三模态 × 多种子对照实验。

本脚本只调用已经存在的 train_fd.py，不修改任何数据、embedding 或历史结果。
三个模式共用同一批 manifest、窗口参数、优化器与训练预算，唯一变化的是 --ablation，
因此 dual 与 signal_only 的配对差值可以直接回答“prompt 分支是否带来增益”。

结果布局把 seed 放进路径，避免多种子互相覆盖：

    Results\CWRU_TimeCMA_FD\v6_prompt_lolo_load_v1\<mode>\seed_<N>\<fold>\

示例（从项目根目录执行）：
    .\scripts\run_lolo_prompt_ablation.ps1 -Mode dual -Seeds 2025,2026
    .\scripts\run_lolo_prompt_ablation.ps1 -Mode signal_only
    .\scripts\run_lolo_prompt_ablation.ps1 -Mode all -SkipExisting
#>

[CmdletBinding()]
param(
    # all 会依次执行 dual、Signal-only 与 Prompt-only；单模式用于断点续跑。
    [ValidateSet('dual', 'signal_only', 'prompt_only', 'all')]
    [string]$Mode = 'all',

    # 三个模式必须使用同一组种子，否则模式间的 Δ 会混入初始化方差。
    [int[]]$Seeds = @(2024, 2025, 2026),

    # 固定 LOLO 已使用的数据源和缓存，避免消融额外改变输入分布。
    [string]$DataRoot = 'D:\project\公开数据集\a8c15-main\CWRU轴承数据\cwru_data',
    [string]$EmbeddingRoot = 'Embeddings\CWRU_v6_prompt\evidence-mechanism-with-load',
    [string]$ResultRoot = 'Results\CWRU_TimeCMA_FD\lolo_load_v1',

    # 已存在 metrics.json 时跳过，用于中断后续跑而不重算已完成的单元。
    [switch]$SkipExisting
)

$ErrorActionPreference = 'Stop'

# 四个测试负载与已完成的 dual 基线一一对应。manifest 已保证每折 MAT 文件不重叠。
$folds = @(
    'test_0hp_val_1hp',
    'test_1hp_val_2hp',
    'test_2hp_val_3hp',
    'test_3hp_val_0hp'
)

$modes = if ($Mode -eq 'all') { @('dual', 'signal_only', 'prompt_only') } else { @($Mode) }

foreach ($ablation in $modes) {
    foreach ($seed in $Seeds) {
        foreach ($fold in $folds) {
            $manifest = "splits\cwru\lolo_load_v1\$fold.json"
            $outputDir = Join-Path $ResultRoot "$ablation\seed_$seed\$fold"

            if ($SkipExisting -and (Test-Path (Join-Path $outputDir 'metrics.json'))) {
                Write-Host "[SKIP] mode=$ablation seed=$seed fold=$fold"
                continue
            }

            Write-Host "[START] mode=$ablation seed=$seed fold=$fold"
            uv run python train_fd.py `
                --data-root $DataRoot `
                --split-manifest $manifest `
                --embedding-root $EmbeddingRoot `
                --output-dir $outputDir `
                --window-size 1024 `
                --stride 1024 `
                --patch-len 256 `
                --patch-stride 128 `
                --align-dim 128 `
                --batch-size 64 `
                --epochs 50 `
                --patience 10 `
                --seed $seed `
                --ablation $ablation

            # Python 训练失败时立即停止，避免后续单元产生不完整且难比较的结果。
            if ($LASTEXITCODE -ne 0) {
                throw "训练失败：mode=$ablation seed=$seed fold=$fold，exit code=$LASTEXITCODE"
            }
        }
    }
}
