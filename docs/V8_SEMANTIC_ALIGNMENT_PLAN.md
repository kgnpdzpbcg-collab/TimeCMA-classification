# V8：故障机理原型与样本证据分离

## 研究问题

V7 把四类通用机理和当前样本统计量写在同一段 Prompt 中，再与振动信号做跨注意力。V8 分别编码固定的类别知识和当前样本证据，检验固定知识原型是否能约束信号表示，以及证据跨注意力是否提供额外收益。

V8 是四类、有监督的 CWRU 跨负载诊断实验。四个固定文本向量是类别原型；`Signal → Prototype` 使用另外三类作为负类。这个目标不等同于 CLIP 式大规模成对预训练，也不单独证明 GPT-2 已具备可靠的轴承物理推理能力。

## 固定的数据和输入

- 沿用现有四个 `splits/cwru/lolo_load_v1/*.json`：按 MAT 文件先划分 train/val/test，再在文件内按 1024 点、步长 1024 建窗。
- 输入为同步 DE/FE 两通道，形状 `[B, 1024, 2]`；信号 patch 长 256、步长 128，得到 7 个局部 token。
- Evidence 对每个 patch 生成一个冻结 GPT-2 embedding，形状 `[B, 768, 7, 1]`。四折和三个 V8 模式使用相同的 Evidence 配方及缓存目录。
- Prototype 仅包含类别概念与机理；四类的固定文本、类别顺序、GPT-2 模型及池化层在全部 fold 中一致，不从任何 MAT 文件或测试标签生成。
- Evidence 文本只使用当前窗口的统计量以及通用诊断语境；不写 CWRU 名称、负载值、真实类别或四类具体机理段落。GPT-2 为冻结编码器，不生成诊断结论。

## 文本配方

Prototype 格式为 `Fault concept: ... Mechanism: ...`。Normal、Ball、Inner、Outer 四段英文原文固定在 `utils/phm_prompt_v8.py`；它们只陈述可能的物理机制，不包含数据集、负载、传感器和样本特征。取 GPT-2 第 10 层的有效文本 token 均值，得到 `[4, 768]`。预检发现原始四向量两两余弦为 0.9855–0.9948，因此仅用四段固定文本的均值去公共方向，再逐行 L2 归一化，保存于 `fault_prototypes.pt`。此变换不使用 CWRU 样本或划分信息，并写入缓存配方；训练中原型不可更新。去公共方向后各类余弦约为 -0.662 到 -0.004，其几何形状仍需通过后续知识对照实验验证语义合理性。

Evidence 按以下顺序构造，确保因果 GPT-2 的证据 token 能读到前面的诊断语境：

```text
Task context: Bearing fault diagnosis.
Dataset context: Vibration signals collected from bearings.
Input: Synchronized drive-end (DE) and fan-end (FE) acceleration signals.
Interpretation requirement: Relate observed evidence to possible fault mechanisms. Consider repeated impacts, modulation, amplitude distribution and spectral structure; one raw-FFT peak alone does not identify a specific fault.
Diagnostic evidence:
  Local evidence for patch p of 7: ...
  Global evidence shared by the 1024-sample window: ...
```

局部和全局数值字段沿用 V7 的提取方式，以集中检验框架变化；原始 FFT 峰和包络谱最大峰仍只称为观测值，不称为 BPFO/BPFI/BSF。Evidence 只对 `Diagnostic evidence` 标题之后的内容 token 取第 10 层均值；固定前缀本身不进入平均。这里的 `Interpretation requirement` 是冻结 GPT-2 的文本上下文，不应未经验证便声称模型执行了显式推理。任何截断到证据末尾的 Prompt 都报错，不静默丢字段。

V8 的 H5 和原型文件位于新目录 `Embeddings/CWRU_v8_semantic/evidence-context-v1/`。`cache_spec.json` 记录全部文本与池化配方，并由 H5、原型文件和训练 checkpoint 保存摘要；V7 缓存不能混用。

## 三种模型

三种模式使用同一个 Signal Transformer 与输入几何。令 `h_s` 为融合前信号 CLS；固定归一化原型为 `K_c`。Signal 和融合后的表示经同一个可训练投影 `P` 与归一化后，用余弦相似度及固定温度 `τ` 得到四类得分。

| 模式 | 分类路径 | 损失 |
| --- | --- | --- |
| `prototype_only`（P） | `Signal → P(h_s) → sim(K_c)` | `CE(signal_logits, y)` |
| `evidence_only`（E） | `Signal + Evidence → Cross Attention → P(h_f) → trainable class vectors` | `CE(fused_logits, y)` |
| `prototype_evidence`（P+E） | `Signal + Evidence → Cross Attention → P(h_f) → sim(K_c)` | `CE(fused_logits, y) + λ CE(signal_logits, y)` |

E 模式的四个可训练分类向量与固定 Prototype 使用同样的余弦打分形式，但不输入类别机理文本。P+E 的第二项始终在融合前的 `h_s` 上计算，阻止模型仅依赖 Evidence 完成对齐。Cross Attention 以信号 token 为 Query、Evidence token 为 Key/Value，并对信号表示做残差更新；P 模式不建立 Prompt Encoder 或 Cross Attention。

这三组是框架探索，不应把 P 与 E 的直接差值解释为单个模块的纯因果效应。现有 V7 Signal-only 可作历史参照，但其分类头不同，不能充当严格的第四个析因实验。

## 训练、评估和判定边界

- 模式间保持相同 manifest、窗口、patch、随机种子、batch、训练预算与验证集选模规则；只按验证集 Macro-F1 选 checkpoint，测试集不参与模型、模板或池化层选择。
- 最佳 checkpoint 加载后分别报告 Train/Val/Test accuracy、Macro-F1 和混淆矩阵；P+E 另外报告融合前 Signal→Prototype 的分类指标。
- 首先做形状、梯度、文本无截断、缓存摘要和单 fold 的运行检查；在代码完成之前不启动四折实验。本轮代码提交不包含原始 MAT、H5 或训练结果。
- Prompt 的数值字段冗余、RevIN 对原始振幅的影响和固定 GPT-2 原型是否真正承载故障语义仍是后续问题。框架稳定后需用无机理文本或随机固定向量对照验证“知识”的贡献。

## 运行入口（PowerShell）

先在项目独立 uv 环境中生成 V8 文本产物。下列命令会生成全量 Evidence H5，耗时取决于本地 CPU/GPU；只验证原型时可加 `--artifact prototypes`，不读取 MAT 或生成 H5。V8 目录与 V7 完全分开。

```powershell
uv run python storage/store_v8_embeddings.py `
  --data-root "D:\project\公开数据集\a8c15-main\CWRU轴承数据\cwru_data" `
  --text-root "Embeddings/CWRU_v8_semantic/evidence-context-v1"
```

以下示例只运行一个 fold、一种模式。另两个模式依次把 `--mode` 改为 `evidence_only`、`prototype_evidence`，并使用各自独立的 `--output-dir`。四折 manifest 位于 `splits/cwru/lolo_load_v1/`；训练命令不会自行生成或改写划分。

```powershell
uv run python train_fd_v8.py `
  --data-root "D:\project\公开数据集\a8c15-main\CWRU轴承数据\cwru_data" `
  --split-manifest "splits/cwru/lolo_load_v1/test_2hp_val_3hp.json" `
  --text-root "Embeddings/CWRU_v8_semantic/evidence-context-v1" `
  --output-dir "Results/CWRU_TimeCMA_FD/v8_semantic_lolo_v1/prototype_only/seed_2024/test_2hp_val_3hp" `
  --mode prototype_only --seed 2024
```
