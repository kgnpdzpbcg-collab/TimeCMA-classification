# V9：CNN 信号分支 + 固定故障机理 Prototype

V9 保留两个信息来源：当前窗口的 DE/FE 波形，以及四段类别级故障机理文本。样本级 Evidence Prompt、Evidence Transformer、CLS/位置编码和 CMA 均不进入 V9。V8 和已有 CNN 基线代码保留，便于追溯。

```text
DE/FE [B,1024,2] → 与 CNN 基线相同的逐窗口标准化及 1D CNN → [B,512]
                                                        ↓ LayerNorm + Linear → L2
四段固定机理文本 → 冻结 GPT-2 第 10 层 → 均值池化 → 去公共方向 → L2
                                                        ↓
                                               四类余弦得分 / 温度
```

四段机理文本及其顺序与 V8 相同：Normal、Ball、Inner、Outer。GPT-2 只在生成四个固定 Prototype 时运行一次；训练和推理期间不读取样本级 H5，也不更新 GPT-2 或 Prototype。V9 不设普通线性分类头，所有最终类别得分都必须通过与四个固定文本向量比较得到。信号编码器复用 CNN 基线的卷积及池化结构，但去掉其分类头。

## 数据与评估

沿用 `splits/cwru/lolo_load_v1/` 的四个显式文件级划分。每个样本是同步 DE/FE 的 1024 点窗口，窗口步长 1024；默认 seed 2024、batch 64、最多 50 epoch、patience 10、AdamW 学习率 1e-4、权重衰减 1e-3。每轮仅在验证集按 Macro-F1 选最优 checkpoint，最后报告 Train/Val/Test 的准确率、Macro-F1 与混淆矩阵。每个 fold 使用独立输出目录。

## 运行入口

先在项目独立 uv 环境中生成四类固定向量（不需要 CWRU 数据路径）：

```powershell
uv run python storage/store_v9_prototypes.py `
  --output Embeddings/CWRU_v9_prototypes/fault_prototypes.pt
```

然后逐折训练。以下仅示范留出 0HP；其余三折替换 manifest 文件名和输出目录即可：

```powershell
uv run python train_fd_v9.py `
  --data-root "D:\project\公开数据集\a8c15-main\CWRU轴承数据\cwru_data" `
  --split-manifest splits/cwru/lolo_load_v1/test_0hp_val_1hp.json `
  --prototype-path Embeddings/CWRU_v9_prototypes/fault_prototypes.pt `
  --output-dir Results/CWRU_TimeCMA_FD/v9_cnn_prototype_lolo_load_v1/seed_2024/test_0hp_val_1hp
```

Prototype 文件记录完整文本配方及摘要；训练入口会校验类别顺序、GPT-2 层数、池化和变换是否匹配。生成的向量文件、checkpoint 和指标都属于本地实验产物，不纳入 Git。

当前 CWRU 四分类跨负载协议中，纯 CNN 已接近满分。因此 V9 在该协议上的准确率不能单独证明 GPT-2 提供了有效的故障机理知识；这需要后续更有区分力的评估来验证。
