<div align="center">
  <h2><b> (AAAI'25) TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment </b></h2>
</div>

This repository contains the code for our AAAI 2025 [paper](https://arxiv.org/abs/2406.01638), where we propose an intuitive yet effective framework for MTSF via cross-modality alignment.

> If you find our work useful in your research. Please consider giving a star ⭐ and citation 📚:

```bibtex
@inproceedings{liu2025timecma,
  title={{TimeCMA}: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment},
  author={Liu, Chenxi and Xu, Qianxiong and Miao, Hao and Yang, Sun and Zhang, Lingzheng and Long, Cheng and Li, Ziyue and Zhao, Rui},
  booktitle={AAAI},
  year={2025}
}
```

## Abstract
Multivariate time series forecasting (MTSF) aims to learn temporal dynamics among variables to forecast future time series. Existing statistical and deep learning-based methods suffer from limited learnable parameters and small-scale training data. Recently, large language models (LLMs) combining time series with textual prompts have achieved promising performance in MTSF. However, we discovered that current LLM-based solutions fall short in learning *disentangled* embeddings. We introduce TimeCMA, an intuitive yet effective framework for MTSF via cross-modality alignment. Specifically, we present a dual-modality encoding with two branches: the time series encoding branch extracts *disentangled yet weak* time series embeddings, and the LLM-empowered encoding branch wraps the same time series with text as prompts to obtain *entangled yet robust* prompt embeddings. As a result, such a cross-modality alignment retrieves *both disentangled and robust* time series embeddings, ``the best of two worlds'', from the prompt embeddings based on time series and prompt modality similarities. As another key design, to reduce the computational costs from time series with their length textual prompts, we design an effective prompt to encourage the most essential temporal information to be encapsulated in the last token: only the last token is passed to downstream prediction. We further store the last token embeddings to accelerate inference speed. Extensive experiments on eight real datasets demonstrate that TimeCMA outperforms state-of-the-arts.

<p align="center">
  <img width="900" alt="image" src="https://github.com/user-attachments/assets/f7359297-5781-4f09-b7b6-aa82f0df817d" />
</p>

## Dependencies

* Python 3.11
* PyTorch 2.1.2
* CUDA 12.1
* torchvision 0.8.0

```bash
> conda env create -f env_{ubuntu,windows}.yaml
```

## Datasets
Datasets can be obtained from [TimesNet](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) and [TFB](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view).

## Usages
* ### Last token embedding storage

```bash
bash Store_{data_name}.sh
```

* ### Train and inference
   
```bash
bash {data_name}.sh
```

## CWRU Fault Diagnosis

该分支保留 TimeCMA 的 Time-Series Encoder、冻结 GPT-2 prompt branch 与 Cross-Modality
Alignment（CMA），并将 forecasting decoder 替换为四类故障分类头：normal、ball、inner、outer。
原 forecasting 代码和脚本不受影响。

### 1. 创建独立 uv 环境

```powershell
uv sync
```

本机没有 CUDA 时会使用 CPU PyTorch；这只影响运行速度，不改变模型逻辑。

### 2. 缓存冻结 GPT-2 embedding

```powershell
uv run python -m storage.store_phm_embeddings `
  --data-root 'D:\project\公开数据集\a8c15-main\CWRU轴承数据\cwru_data' `
  --embedding-root Embeddings\CWRU `
  --window-size 1024 --stride 1024
```

Prompt 仅由信号窗口统计量、采样率和已知负载工况构成，不包含故障标签或故障尺寸；完整
波形不会直接写入文本，以避免 GPT-2 上下文截断。GPT-2 只在本步骤运行一次，训练阶段读取
缓存的 last-token embedding。

### 3. 训练和测试

```powershell
uv run python train_fd.py `
  --data-root 'D:\project\公开数据集\a8c15-main\CWRU轴承数据\cwru_data' `
  --embedding-root Embeddings\CWRU
```

数据在原始 MAT 文件粒度切分，防止同一记录的不同窗口跨训练、验证、测试集合泄漏。模型仅以
验证集 Macro-F1 保存最佳 checkpoint，测试集只在训练结束后评估一次。
