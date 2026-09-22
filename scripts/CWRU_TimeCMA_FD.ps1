# 在项目根目录执行。本模板只给出一次明确 manifest 的实验，避免隐式随机切分。
# 先生成或迁移 split 无关缓存；随后仅替换 $manifest 与 $outputDir 即可切换 LOLO fold。
$dataRoot = 'D:\project\公开数据集\a8c15-main\CWRU轴承数据\cwru_data'
$manifest = 'splits\cwru\lolo_load_v1\test_3hp_val_0hp.json'
$embeddingRoot = 'Embeddings\CWRU_v4_de_fe\prompt-v1-with-load'
$outputDir = 'Results\CWRU_TimeCMA_FD\lolo_load_v1\test_3hp_val_0hp'

# 仅在目标缓存不存在时运行；该命令生成全部 5927 个窗口，路径不含 train/val/test。
uv run python -m storage.store_phm_embeddings `
    --data-root $dataRoot `
    --embedding-root $embeddingRoot `
    --window-size 1024 `
    --stride 1024 `
    --patch-len 256 `
    --patch-stride 128 `
    --include-load-hp

# 训练时必须给出 manifest；模型选择严格只使用 val，test 只做最终一次评估。
uv run python train_fd.py `
    --data-root $dataRoot `
    --split-manifest $manifest `
    --embedding-root $embeddingRoot `
    --output-dir $outputDir `
    --window-size 1024 `
    --stride 1024 `
    --patch-len 256 `
    --patch-stride 128 `
    --align-dim 128 `
    --batch-size 64 `
    --epochs 50
