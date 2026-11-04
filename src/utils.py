# -*- coding: utf-8 -*-
"""通用工具：日志、随机种子、评估指标、绘图。"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import config  # noqa: F401 - 先导入以设置缓存/临时目录

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def setup_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


class Logger(object):
    """同时输出到控制台和日志文件。"""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def __call__(self, message=""):
        text = str(message)
        print(text, flush=True)
        self.file.write(text + "\n")
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(log=None):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if log:
        if device.type == "cuda":
            log("使用 GPU：%s（显存 %.1f GB）" % (
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))
        else:
            log("未检测到 CUDA，使用 CPU 训练（速度会较慢）")
    return device


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_seconds(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "%dh %02dm %02ds" % (h, m, s)
    if m:
        return "%dm %02ds" % (m, s)
    return "%ds" % s


def compute_metrics(y_true, y_pred, y_prob=None):
    """二分类评估：Accuracy / Precision / Recall / F1 / AUC。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=config.LABEL_NAMES,
            output_dict=True, zero_division=0),
    }
    if y_prob is not None:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            result["roc_auc"] = None
    result["n_samples"] = int(len(y_true))
    return result


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def plot_history(history, path, title):
    """训练曲线：loss 与 accuracy。"""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epochs, history["train_loss"], "o-", label="Train loss")
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], "o-", label="Train acc")
    axes[1].plot(epochs, history["val_acc"], "s-", label="Val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, path, title):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=config.LABEL_NAMES, yticklabels=config.LABEL_NAMES,
           xlabel="Predicted label", ylabel="True label", title=title)
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), "d"), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(str(path), dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(rows, path):
    """rows: [{name, accuracy, precision, recall, f1}] -> grouped bar chart."""
    names = [r.get("name", r.get("model", "")) for r in rows]
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = [r.get(metric, 0) for r in rows]
        ax.bar(x + (i - 1.5) * width, values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Test performance comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160, bbox_inches="tight")
    plt.close(fig)
