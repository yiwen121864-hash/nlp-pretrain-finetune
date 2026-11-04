# -*- coding: utf-8 -*-
"""汇总三个模型的测试指标，生成对比表与柱状图。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import utils

import pandas as pd

DISPLAY = {
    "scratch": "Scratch Transformer",
    "frozen": "DistilBERT (frozen)",
    "finetune": "DistilBERT (fine-tuned)",
}


def main():
    utils.setup_console()
    rows = []
    for mode, name in DISPLAY.items():
        metrics = utils.load_json(config.PRED_DIR / ("%s_test_metrics.json" % mode))
        history = utils.load_json(config.LOG_DIR / ("%s_history.json" % mode))
        if not metrics:
            print("[跳过] %s 还没有评估结果" % mode)
            continue
        rows.append({
            "mode": mode,
            "model": name,
            "trainable_params": history.get("trainable_params") if history else None,
            "total_params": history.get("total_params") if history else None,
            "train_seconds": round(history.get("train_seconds", 0), 1) if history else None,
            "best_epoch": history.get("best_epoch") if history else None,
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "roc_auc": round(metrics["roc_auc"], 4) if metrics.get("roc_auc") else None,
        })

    if not rows:
        raise SystemExit("没有可汇总的结果，请先训练并评估。")

    df = pd.DataFrame(rows)
    csv_path = config.OUTPUT_DIR / "model_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    lines = ["# 三个模型测试集对比", "",
             "| 模型 | 可训练参数 | 训练时间(s) | Accuracy | Precision | Recall | F1 | ROC-AUC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        lines.append("| %s | %s | %s | %.4f | %.4f | %.4f | %.4f | %s |" % (
            r["model"],
            format(int(r["trainable_params"]), ",") if pd.notna(r["trainable_params"]) else "-",
            r["train_seconds"],
            r["accuracy"], r["precision"], r["recall"], r["f1"],
            ("%.4f" % r["roc_auc"]) if pd.notna(r["roc_auc"]) else "-"))
    md = "\n".join(lines) + "\n"
    (config.OUTPUT_DIR / "model_comparison.md").write_text(md, encoding="utf-8")
    print(md)

    utils.plot_model_comparison(rows, config.FIG_DIR / "model_comparison.png")
    print("对比表：%s" % csv_path)
    print("对比图：%s" % (config.FIG_DIR / "model_comparison.png"))


if __name__ == "__main__":
    main()
