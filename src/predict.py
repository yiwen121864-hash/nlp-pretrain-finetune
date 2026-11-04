# -*- coding: utf-8 -*-
"""加载训练好的模型，对任意英文影评进行情感预测。

用法：
    python src/predict.py --text "This movie is absolutely wonderful."
    python src/predict.py          # 交互式输入
    python src/predict.py --mode frozen
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import utils

import torch
from transformers import AutoTokenizer


def load_predictor(mode):
    model_path = config.CKPT_DIR / ("%s_best.pt" % mode)
    if not model_path.exists():
        raise FileNotFoundError("找不到模型 %s，请先运行训练脚本" % model_path)

    if mode == "scratch":
        import data
        import model as model_lib
        vocab = data.ScratchVocab.load(config.CKPT_DIR / "scratch_vocab.json")
        model = model_lib.ScratchTransformerClassifier(vocab_size=len(vocab))
        ckpt = torch.load(str(model_path), map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        def predict(text):
            ids = torch.tensor([vocab.encode(text, config.MAX_WORDS)], dtype=torch.long)
            with torch.no_grad():
                logits = model(ids)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            return prob
        return predict, model_path

    import model as model_lib
    tokenizer_path = config.MODEL_DIR / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    model = model_lib.BertClassifier()
    ckpt = torch.load(str(model_path), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def predict(text):
        encoded = tokenizer(text, truncation=True, padding="max_length",
                            max_length=config.BERT_MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            logits = model(encoded["input_ids"], encoded["attention_mask"])
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        return prob
    return predict, model_path


def main():
    utils.setup_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="finetune",
                        choices=["scratch", "frozen", "finetune"])
    parser.add_argument("--text", default=None)
    args = parser.parse_args()

    predict, model_path = load_predictor(args.mode)
    print("加载模型：%s" % model_path)

    if args.text:
        texts = [args.text]
    else:
        texts = []
        print("输入英文影评（空行结束）：")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            texts.append(line)

    for text in texts:
        prob = predict(text)
        label = config.LABEL_NAMES[1 if prob >= 0.5 else 0]
        print("\n文本：%s" % text[:200])
        print("预测：%s（positive 概率 %.4f）" % (label, prob))


if __name__ == "__main__":
    main()
