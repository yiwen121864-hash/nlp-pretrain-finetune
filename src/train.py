# -*- coding: utf-8 -*-
"""训练脚本：从零 Transformer / DistilBERT 冻结 / DistilBERT 全量微调。

用法：
    python src/train.py --mode scratch
    python src/train.py --mode frozen
    python src/train.py --mode finetune
    python src/train.py --mode finetune --epochs 1 --max-train-batches 50   # 冒烟测试
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import utils
import data
import model as model_lib

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def parse_args():
    parser = argparse.ArgumentParser(description="IMDB 情感分类训练")
    parser.add_argument("--mode", required=True,
                        choices=["scratch", "frozen", "finetune"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="只用前 N 条训练样本（调试用）")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="每个 epoch 只跑 N 个 batch（调试用）")
    return parser.parse_args()


def defaults_for(mode):
    if mode == "scratch":
        return dict(epochs=config.SCRATCH_EPOCHS,
                    batch_size=config.SCRATCH_BATCH_SIZE,
                    lr=config.SCRATCH_LR)
    if mode == "frozen":
        return dict(epochs=config.FROZEN_EPOCHS,
                    batch_size=config.BERT_BATCH_SIZE,
                    lr=config.FROZEN_LR)
    return dict(epochs=config.FINETUNE_EPOCHS,
                batch_size=config.BERT_BATCH_SIZE,
                lr=config.FINETUNE_LR)


def build_optimizer(model, lr, mode):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": config.WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    grouped = [g for g in grouped if g["params"]]
    return AdamW(grouped, lr=lr)


def batch_to_device(batch, device):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return input_ids, attention_mask, labels


def run_epoch(model, loader, optimizer, scheduler, scaler, criterion,
              device, training, mode, log=None, epoch=0, max_batches=None):
    model.train(training)
    if training and mode == "frozen":
        model.encoder.eval()  # 冻结时编码器保持 eval，避免 dropout 干扰

    total_loss, correct, total = 0.0, 0, 0
    all_probs = []
    use_amp = device.type == "cuda"

    for step, batch in enumerate(loader, 1):
        if max_batches and step > max_batches:
            break
        input_ids, attention_mask, labels = batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

        probs = torch.softmax(logits.float(), dim=-1)[:, 1]
        preds = (probs >= 0.5).long()
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_probs.extend(probs.detach().cpu().numpy().tolist())

        if training and log and step % 100 == 0:
            log("    epoch %d step %d/%d | loss %.4f | acc %.4f"
                % (epoch, step, len(loader), total_loss / max(total, 1),
                   correct / max(total, 1)))

    return total_loss / max(total, 1), correct / max(total, 1), all_probs


def main():
    args = parse_args()
    utils.setup_console()
    utils.set_seed(config.SEED)

    d = defaults_for(args.mode)
    epochs = args.epochs or d["epochs"]
    batch_size = args.batch_size or d["batch_size"]
    lr = args.lr or d["lr"]
    tag = args.mode

    log = utils.Logger(config.LOG_DIR / ("%s_train.log" % tag))
    log("=" * 70)
    log("训练模式：%s" % tag)
    log("epochs=%d, batch_size=%d, lr=%g, max_len=%d, seed=%d"
        % (epochs, batch_size, lr, config.MAX_WORDS, config.SEED))
    device = utils.get_device(log)
    log("PyTorch %s | CUDA available=%s" % (torch.__version__, torch.cuda.is_available()))

    # ---------------- 数据 ----------------
    t0 = time.time()
    if args.mode == "scratch":
        vocab, train_ds, val_ds, _ = data.build_scratch_loaders(args.max_train_samples)
        train_loader = data.make_loader(train_ds, batch_size, shuffle=True)
        val_loader = data.make_loader(val_ds, batch_size, shuffle=False)
        vocab.save(config.CKPT_DIR / "scratch_vocab.json")
        model = model_lib.ScratchTransformerClassifier(vocab_size=len(vocab))
        log("词表大小：%d" % len(vocab))
    else:
        tokenizer, train_ds, val_ds, _ = data.build_bert_datasets(args.max_train_samples)
        train_loader = data.make_loader(train_ds, batch_size, shuffle=True)
        val_loader = data.make_loader(val_ds, batch_size, shuffle=False)
        model = model_lib.BertClassifier()
        if args.mode == "frozen":
            model.freeze_encoder()
        log("预训练模型：%s" % config.BERT_MODEL_NAME)

    model = model.to(device)
    log("数据准备完成：train=%d, val=%d，用时 %s"
        % (len(train_ds), len(val_ds), utils.format_seconds(time.time() - t0)))
    log("可训练参数：%s" % format(utils.count_parameters(model), ","))

    # ---------------- 优化器 ----------------
    optimizer = build_optimizer(model, lr, args.mode)
    steps_per_epoch = len(train_loader)
    if args.max_train_batches:
        steps_per_epoch = min(steps_per_epoch, args.max_train_batches)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup = int(total_steps * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    criterion = nn.CrossEntropyLoss()

    # ---------------- 训练 ----------------
    history = {"train_loss": [], "train_acc": [], "val_loss": [],
               "val_acc": [], "val_f1": [], "epoch_time": []}
    best_f1 = -1.0
    best_epoch = -1
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        log("\nEpoch %d/%d" % (epoch, epochs))
        train_loss, train_acc, _ = run_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion,
            device, True, args.mode, log, epoch, args.max_train_batches)
        val_loss, val_acc, val_probs = run_epoch(
            model, val_loader, None, None, scaler, criterion,
            device, False, args.mode, None, epoch, None)

        y_true = val_ds.labels if hasattr(val_ds, "labels") else \
            [int(val_ds[i]["label"]) for i in range(len(val_ds))]
        y_pred = [1 if p >= 0.5 else 0 for p in val_probs]
        val_f1 = utils.compute_metrics(y_true, y_pred, val_probs)["f1"]
        elapsed = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["epoch_time"].append(elapsed)
        log("  train loss %.4f | train acc %.4f | val loss %.4f | val acc %.4f | val F1 %.4f | %s"
            % (train_loss, train_acc, val_loss, val_acc, val_f1,
               utils.format_seconds(elapsed)))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "mode": args.mode,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "model_name": config.BERT_MODEL_NAME if args.mode != "scratch" else None,
            }, str(config.CKPT_DIR / ("%s_best.pt" % tag)))
            log("  [保存] 目前最佳模型：%s（val F1=%.4f）"
                % (config.CKPT_DIR / ("%s_best.pt" % tag), val_f1))

    total_time = time.time() - train_start
    log("\n训练完成：共 %s，最佳 epoch=%d，val F1=%.4f"
        % (utils.format_seconds(total_time), best_epoch, best_f1))

    result = {
        "mode": tag,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_len": config.BERT_MAX_LEN if tag != "scratch" else config.MAX_WORDS,
        "trainable_params": utils.count_parameters(model),
        "total_params": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "train_seconds": total_time,
        "history": history,
        "device": str(device),
        "model_name": config.BERT_MODEL_NAME if tag != "scratch" else "scratch-transformer",
    }
    utils.save_json(result, config.LOG_DIR / ("%s_history.json" % tag))
    utils.plot_history(history, config.FIG_DIR / ("%s_curves.png" % tag),
                       "%s training curves" % tag)
    log("历史记录：%s" % (config.LOG_DIR / ("%s_history.json" % tag)))
    log("训练曲线：%s" % (config.FIG_DIR / ("%s_curves.png" % tag)))
    log.close()


if __name__ == "__main__":
    main()
