# -*- coding: utf-8 -*-
"""在测试集上评估模型：准确率/P/R/F1/AUC、混淆矩阵、错误样例。

用法：
    python src/evaluate.py --all
    python src/evaluate.py --mode finetune
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import utils
import data
import model as model_lib

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scratch", "frozen", "finetune"])
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def load_model_and_testset(mode):
    test_df = pd.read_csv(config.TEST_CSV)
    if mode == "scratch":
        vocab = data.ScratchVocab.load(config.CKPT_DIR / "scratch_vocab.json")
        test_ds = data.ScratchDataset(test_df["text"], test_df["label"],
                                      vocab, config.MAX_WORDS)
        model = model_lib.ScratchTransformerClassifier(vocab_size=len(vocab))
    else:
        tokenizer_path = config.MODEL_DIR / "tokenizer"
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            tokenizer = data.get_bert_tokenizer()
        test_ds = data._load_or_encode(test_df["text"], test_df["label"], tokenizer,
                                       config.BERT_MAX_LEN, "test", True)
        model = model_lib.BertClassifier()
    return model, test_ds, test_df


def evaluate(mode, device=None):
    utils.setup_console()
    device = device or utils.get_device()
    log = utils.Logger(config.LOG_DIR / ("%s_evaluate.log" % mode))
    log("=" * 70)
    log("测试集评估：%s" % mode)

    ckpt_path = config.CKPT_DIR / ("%s_best.pt" % mode)
    if not ckpt_path.exists():
        log("[跳过] 找不到模型：%s，请先训练" % ckpt_path)
        log.close()
        return None

    model, test_ds, test_df = load_model_and_testset(mode)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    log("加载模型：%s（epoch=%s, val F1=%.4f）"
        % (ckpt_path, ckpt.get("epoch"), ckpt.get("val_f1", -1)))

    loader = data.make_loader(test_ds, 64, shuffle=False)
    probs, preds, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask)
            prob = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            probs.extend(prob.tolist())
            preds.extend((prob >= 0.5).astype(int).tolist())
            labels.extend(batch["label"].numpy().tolist())

    metrics = utils.compute_metrics(labels, preds, probs)
    metrics.update({
        "mode": mode,
        "checkpoint": str(ckpt_path),
        "test_size": int(len(labels)),
    })
    utils.save_json(metrics, config.PRED_DIR / ("%s_test_metrics.json" % mode))
    utils.plot_confusion(metrics["confusion_matrix"],
                         config.FIG_DIR / ("%s_confusion.png" % mode),
                         "%s test confusion matrix" % mode)

    # 预测明细（用于报告中的错误分析）
    pred_df = pd.DataFrame({
        "text": test_df["text"].astype(str),
        "label": labels,
        "pred": preds,
        "prob_positive": probs,
    })
    pred_df["correct"] = (pred_df["label"] == pred_df["pred"])
    pred_df.to_csv(config.PRED_DIR / ("%s_test_predictions.csv" % mode),
                   index=False, encoding="utf-8-sig")

    log("\n测试结果（%d 条）：" % len(labels))
    log("  Accuracy  : %.4f" % metrics["accuracy"])
    log("  Precision : %.4f" % metrics["precision"])
    log("  Recall    : %.4f" % metrics["recall"])
    log("  F1        : %.4f" % metrics["f1"])
    if metrics.get("roc_auc") is not None:
        log("  ROC-AUC   : %.4f" % metrics["roc_auc"])
    log("  Confusion : %s" % metrics["confusion_matrix"])

    report_txt = config.PRED_DIR / ("%s_classification_report.txt" % mode)
    report = metrics["classification_report"]
    lines = ["%s test classification report" % mode, ""]
    lines.append("%-12s %9s %9s %9s %9s" % ("", "precision", "recall", "f1", "support"))
    for label in config.LABEL_NAMES + ["macro avg", "weighted avg"]:
        if label in report:
            r = report[label]
            lines.append("%-12s %9.4f %9.4f %9.4f %9d"
                         % (label, r["precision"], r["recall"], r["f1-score"], r["support"]))
    report_txt.write_text("\n".join(lines), encoding="utf-8")

    # 展示 5 个高置信度错误
    wrong = pred_df[~pred_df["correct"]].copy()
    wrong["confidence"] = np.where(wrong["pred"] == 1,
                                   wrong["prob_positive"], 1 - wrong["prob_positive"])
    wrong = wrong.sort_values("confidence", ascending=False).head(5)
    log("\n典型错误样例（按置信度排序）：")
    for _, row in wrong.iterrows():
        log("  true=%s pred=%s prob=%.4f | %s"
            % (config.LABEL_NAMES[int(row["label"])],
               config.LABEL_NAMES[int(row["pred"])],
               row["prob_positive"], row["text"][:180].replace("\n", " ")))
    log("\n结果文件：%s" % (config.PRED_DIR / ("%s_test_metrics.json" % mode)))
    log.close()
    return metrics


if __name__ == "__main__":
    args = parse_args()
    modes = ["scratch", "frozen", "finetune"] if args.all else [args.mode]
    if not modes or modes == [None]:
        raise SystemExit("请指定 --mode 或 --all")
    device = utils.get_device()
    for m in modes:
        evaluate(m, device)
    print("\n评估完成。")
