import numpy as np
import torch

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))

def MSE(pred, true):
    return torch.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mse = MSE(pred, true).item()
    mae = MAE(pred, true).item()
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    
    # return mae,mse,rmse,mape,mspe
    return mse,mae


def classification_metrics(predictions, targets, num_classes):
    """计算故障诊断指标，输入为一维预测标签和真实标签。

    实现只依赖 PyTorch，避免为了评估额外引入 sklearn。Macro-F1 对 CWRU 类别样本数
    不平衡更稳健，因此被训练脚本用作验证集选模指标。
    """
    predictions = torch.as_tensor(predictions, dtype=torch.long).reshape(-1)
    targets = torch.as_tensor(targets, dtype=torch.long).reshape(-1)
    if predictions.numel() != targets.numel():
        raise ValueError("predictions 和 targets 的样本数必须一致")
    if predictions.numel() == 0:
        raise ValueError("不能对空预测计算分类指标")

    encoded = targets * num_classes + predictions
    confusion = torch.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    true_positive = confusion.diag().float()
    false_positive = confusion.sum(dim=0).float() - true_positive
    false_negative = confusion.sum(dim=1).float() - true_positive
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "accuracy": (predictions == targets).float().mean().item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "confusion_matrix": confusion.cpu().tolist(),
    }
