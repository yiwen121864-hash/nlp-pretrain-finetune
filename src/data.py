# -*- coding: utf-8 -*-
"""数据读取、从零模型的词表/数据集、BERT 数据集与 DataLoader。"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config
import utils

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")


# ---------------------------------------------------------------------------
# 读取 CSV
# ---------------------------------------------------------------------------
def load_splits():
    frames = {}
    for name, path in (("train", config.TRAIN_CSV),
                       ("val", config.VAL_CSV),
                       ("test", config.TEST_CSV)):
        if not path.exists():
            raise FileNotFoundError("找不到 %s，请先运行 src/download_data.py" % path)
        frames[name] = pd.read_csv(path)
    return frames["train"], frames["val"], frames["test"]


# ---------------------------------------------------------------------------
# 从零 Transformer 使用的简单分词与词表
# ---------------------------------------------------------------------------
def simple_tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


class ScratchVocab(object):
    PAD, UNK = 0, 1

    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}

    @classmethod
    def build(cls, texts, max_size=20000, min_freq=2):
        counter = Counter()
        for text in texts:
            counter.update(simple_tokenize(text))
        words = [w for w, c in counter.most_common() if c >= min_freq][:max_size - 2]
        word2idx = {"<pad>": cls.PAD, "<unk>": cls.UNK}
        for w in words:
            word2idx[w] = len(word2idx)
        return cls(word2idx)

    def encode(self, text, max_len):
        ids = [self.word2idx.get(w, self.UNK) for w in simple_tokenize(text)]
        ids = ids[:max_len]
        ids = ids + [self.PAD] * (max_len - len(ids))
        return ids

    def save(self, path):
        Path(path).write_text(json.dumps(self.word2idx, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __len__(self):
        return len(self.word2idx)


class ScratchDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len
        self.encoded = [vocab.encode(t, max_len) for t in self.texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.encoded[index], dtype=torch.long),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# BERT 分词与数据集
# ---------------------------------------------------------------------------
_BERT_TOKENIZER = None


def get_bert_tokenizer():
    global _BERT_TOKENIZER
    if _BERT_TOKENIZER is None:
        from transformers import AutoTokenizer
        cache_dir = config.MODEL_DIR / "pretrained"
        print("加载 tokenizer：%s（目录 %s）" % (config.BERT_MODEL_NAME, cache_dir))
        _BERT_TOKENIZER = AutoTokenizer.from_pretrained(
            config.BERT_MODEL_NAME, cache_dir=str(cache_dir))
        _BERT_TOKENIZER.save_pretrained(str(config.MODEL_DIR / "tokenizer"))
    return _BERT_TOKENIZER


class BertDataset(Dataset):
    """预先把文本编码成 input_ids / attention_mask，避免每个 batch 重复分词。"""

    def __init__(self, texts, labels, tokenizer, max_len=256, batch_size=1024):
        self.input_ids = []
        self.attention_mask = []
        texts = list(texts)
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            encoded = tokenizer(chunk, truncation=True, padding="max_length",
                                max_length=max_len, return_tensors="pt")
            self.input_ids.append(encoded["input_ids"])
            self.attention_mask.append(encoded["attention_mask"])
        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_mask = torch.cat(self.attention_mask, dim=0)
        self.labels = torch.tensor([int(y) for y in labels], dtype=torch.long)

    @classmethod
    def from_tensors(cls, input_ids, attention_mask, labels):
        obj = cls.__new__(cls)
        obj.input_ids = input_ids
        obj.attention_mask = attention_mask
        obj.labels = labels
        return obj

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "label": self.labels[index],
        }


def _load_or_encode(texts, labels, tokenizer, max_len, name, use_cache):
    """完整数据集编码结果缓存到 data/processed，避免每次训练重复分词。"""
    cache_path = config.PROCESSED_DIR / ("bert_%s_%d.pt" % (name, max_len))
    if use_cache and cache_path.exists():
        print("  [缓存] 读取已分词数据：%s" % cache_path)
        obj = torch.load(str(cache_path))
        return BertDataset.from_tensors(obj["input_ids"], obj["attention_mask"],
                                        obj["labels"])
    print("  分词 %s（%d 条，max_len=%d）..." % (name, len(texts), max_len))
    dataset = BertDataset(texts, labels, tokenizer, max_len)
    if use_cache:
        torch.save({
            "input_ids": dataset.input_ids,
            "attention_mask": dataset.attention_mask,
            "labels": dataset.labels,
        }, str(cache_path))
        print("  [缓存] 已保存：%s" % cache_path)
    return dataset


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
def make_loader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,          # Windows 下避免多进程 DataLoader 的问题
        pin_memory=torch.cuda.is_available(),
    )


def build_scratch_loaders(max_train=None, include_test=False):
    train_df, val_df, test_df = load_splits()
    if max_train:
        train_df = train_df.iloc[:max_train]
    vocab = ScratchVocab.build(train_df["text"], config.SCRATCH_VOCAB_SIZE,
                               config.SCRATCH_MIN_FREQ)
    train_ds = ScratchDataset(train_df["text"], train_df["label"], vocab, config.MAX_WORDS)
    val_ds = ScratchDataset(val_df["text"], val_df["label"], vocab, config.MAX_WORDS)
    test_ds = None
    if include_test:
        test_ds = ScratchDataset(test_df["text"], test_df["label"], vocab, config.MAX_WORDS)
    return vocab, train_ds, val_ds, test_ds


def build_bert_datasets(max_train=None, include_test=False):
    train_df, val_df, test_df = load_splits()
    if max_train:
        train_df = train_df.iloc[:max_train]
    tokenizer = get_bert_tokenizer()
    use_cache = max_train is None
    train_ds = _load_or_encode(train_df["text"], train_df["label"], tokenizer,
                               config.BERT_MAX_LEN, "train", use_cache)
    val_ds = _load_or_encode(val_df["text"], val_df["label"], tokenizer,
                             config.BERT_MAX_LEN, "val", use_cache)
    test_ds = None
    if include_test:
        test_ds = _load_or_encode(test_df["text"], test_df["label"], tokenizer,
                                  config.BERT_MAX_LEN, "test", use_cache)
    return tokenizer, train_ds, val_ds, test_ds
