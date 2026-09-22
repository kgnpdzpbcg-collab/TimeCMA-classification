# 在项目根目录执行：powershell -ExecutionPolicy Bypass -File scripts\CWRU_TimeCMA_FD.ps1
# 第一阶段只生成冻结 GPT-2 embedding；首次执行会下载模型权重并在本地缓存。
$dataRoot = 'D:\project\公开数据集\a8c15-main\CWRU轴承数据\cwru_data'
$embeddingRoot = 'Embeddings\CWRU_v3a_patch256_stride128'

uv run python -m storage.store_phm_embeddings `
    --data-root $dataRoot `
    --embedding-root $embeddingRoot `
    --window-size 1024 `
    --stride 1024 `
    --patch-len 256 `
    --patch-stride 128 `
    --batch-size 8

# 第二阶段训练分类模型；分类 checkpoint 只由验证集 Macro-F1 决定。
uv run python train_fd.py `
    --data-root $dataRoot `
    --embedding-root $embeddingRoot `
    --window-size 1024 `
    --stride 1024 `
    --patch-len 256 `
    --patch-stride 128 `
    --align-dim 128 `
    --batch-size 64 `
    --epochs 50
