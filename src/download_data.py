# -*- coding: utf-8 -*-
"""下载 IMDB 数据集并划分 train / val / test。

数据来源：Hugging Face `stanfordnlp/imdb`（原始数据来自 Stanford IMDB Large Movie Review Dataset）。
下载通过 hf-mirror.com，全部保存在项目 data/ 目录。
"""

import argparse
import html
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import utils


HTML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
SPACE_RE = re.compile(r"\s+")


def clean_text(text):
    """去掉 IMDB 评论中的 HTML 标签（如 <br />），反转义实体，压缩空白。"""
    text = str(text)
    text = text.replace("<br />", " ").replace("<br/>", " ")
    text = HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def download_file(url, dest):
    import requests

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        print("  [缓存] %s 已存在（%.1f MB）" % (dest.name, dest.stat().st_size / 1048576.0))
        return dest

    print("  [下载] %s" % url)
    session = requests.Session()
    session.trust_env = False  # 绕过失效的系统代理
    with session.get(url, stream=True, timeout=60, headers={
        "User-Agent": "Mozilla/5.0 (NLP-course-design)"}) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write("\r    %.1f%%  %.1f/%.1f MB" % (
                        done * 100.0 / total, done / 1048576.0, total / 1048576.0))
                    sys.stdout.flush()
        sys.stdout.write("\n")
        tmp.replace(dest)
    print("  [完成] %s（%.1f MB）" % (dest.name, dest.stat().st_size / 1048576.0))
    return dest


def build_splits(force=False):
    if all(p.exists() for p in [config.TRAIN_CSV, config.VAL_CSV, config.TEST_CSV]) and not force:
        print("处理后的数据已存在，跳过（加 --force 重新生成）")
    else:
        train_raw = download_file(config.IMDB_TRAIN_URL, config.RAW_DIR / "imdb_train.parquet")
        test_raw = download_file(config.IMDB_TEST_URL, config.RAW_DIR / "imdb_test.parquet")

        print("读取 parquet ...")
        train_df = pd.read_parquet(train_raw)
        test_df = pd.read_parquet(test_raw)
        print("原始训练集 %d 条，测试集 %d 条；列：%s"
              % (len(train_df), len(test_df), list(train_df.columns)))

        for df in (train_df, test_df):
            df["text"] = df["text"].map(clean_text)
            df["label"] = df["label"].astype(int)
            df.dropna(subset=["text"], inplace=True)
            df.drop_duplicates(subset=["text"], inplace=True)

        # 训练集 20000 / 验证集 5000（分层抽样，保证正负样本比例一致）
        train_df, val_df = train_test_split(
            train_df, train_size=config.TRAIN_SIZE,
            stratify=train_df["label"], random_state=config.SEED)
        # 测试集抽 5000 条
        test_df, _ = train_test_split(
            test_df, train_size=config.TEST_SIZE,
            stratify=test_df["label"], random_state=config.SEED)

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df.to_csv(config.TRAIN_CSV, index=False, encoding="utf-8")
        val_df.to_csv(config.VAL_CSV, index=False, encoding="utf-8")
        test_df.to_csv(config.TEST_CSV, index=False, encoding="utf-8")
        print("已保存：%s / %s / %s" % (config.TRAIN_CSV, config.VAL_CSV, config.TEST_CSV))

    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    def describe(df):
        lengths = df["text"].str.split().map(len)
        return {
            "n_samples": int(len(df)),
            "label_distribution": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
            "avg_words": float(lengths.mean()),
            "median_words": float(lengths.median()),
            "max_words": int(lengths.max()),
            "min_words": int(lengths.min()),
        }

    stats = {
        "source": "Hugging Face stanfordnlp/imdb (Stanford IMDB Large Movie Review Dataset)",
        "seed": config.SEED,
        "max_seq_len": config.MAX_WORDS,
        "train": describe(train_df),
        "val": describe(val_df),
        "test": describe(test_df),
        "samples": [
            {"split": "train", "label": int(train_df.iloc[i]["label"]),
             "text": train_df.iloc[i]["text"][:300]} for i in range(3)
        ],
    }
    utils.save_json(stats, config.DATA_STATS)

    print("\n数据统计：")
    for split in ("train", "val", "test"):
        s = stats[split]
        print("  %-5s: %5d 条 | 正:%d 负:%d | 平均 %.1f 词 | 最长 %d 词"
              % (split, s["n_samples"], s["label_distribution"].get("1", 0),
                 s["label_distribution"].get("0", 0),
                 s["avg_words"], s["max_words"]))
    print("统计已保存：%s" % config.DATA_STATS)


if __name__ == "__main__":
    utils.setup_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="重新生成处理后的数据")
    args = parser.parse_args()
    build_splits(force=args.force)
