# -*- coding: utf-8 -*-
"""实验一：基于预训练模型的 IMDB 情感分类 —— 全局配置。

所有缓存、临时文件、模型、数据都指向项目目录（E 盘），不写 C 盘。
本文件必须在 import torch / transformers / matplotlib 之前导入。
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "_cache"
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
FIG_DIR = OUTPUT_DIR / "figures"
PRED_DIR = OUTPUT_DIR / "predictions"
REPORT_DIR = BASE_DIR / "report"

# ---------------------------------------------------------------------------
# 必须在导入第三方库之前设置的环境变量
# ---------------------------------------------------------------------------
os.environ["HF_HOME"] = str(CACHE_DIR / "hf")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "matplotlib")
os.environ["TMP"] = str(CACHE_DIR / "tmp")
os.environ["TEMP"] = str(CACHE_DIR / "tmp")
os.environ["TMPDIR"] = str(CACHE_DIR / "tmp")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUTF8", "1")
# 绕过失效的系统代理（Windows 注册表中的代理关闭后仍会被 requests 读取）
for _key in ("no_proxy", "NO_PROXY"):
    os.environ.setdefault(_key, "*")

for _p in (CACHE_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR,
           OUTPUT_DIR, LOG_DIR, CKPT_DIR, FIG_DIR, PRED_DIR, REPORT_DIR,
           REPORT_DIR / "figures", CACHE_DIR / "tmp", CACHE_DIR / "matplotlib"):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 数据与实验参数
# ---------------------------------------------------------------------------
SEED = 42

IMDB_TRAIN_URL = (
    "https://hf-mirror.com/datasets/stanfordnlp/imdb/resolve/main/"
    "plain_text/train-00000-of-00001.parquet"
)
IMDB_TEST_URL = (
    "https://hf-mirror.com/datasets/stanfordnlp/imdb/resolve/main/"
    "plain_text/test-00000-of-00001.parquet"
)

TRAIN_SIZE = 20000      # 从原始 25000 条训练集抽取
VAL_SIZE = 5000         # 从原始训练集抽取（与训练集不重叠）
TEST_SIZE = 5000        # 从原始 25000 条测试集抽取
MAX_WORDS = 256         # 两条路线统一最大长度

# 从零 Transformer
SCRATCH_VOCAB_SIZE = 20000
SCRATCH_MIN_FREQ = 2
SCRATCH_D_MODEL = 128
SCRATCH_NHEAD = 4
SCRATCH_LAYERS = 2
SCRATCH_FF = 512
SCRATCH_DROPOUT = 0.1
SCRATCH_BATCH_SIZE = 64
SCRATCH_EPOCHS = 3
SCRATCH_LR = 1e-3

# 预训练 DistilBERT
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_MAX_LEN = 256
BERT_BATCH_SIZE = 32
FINETUNE_EPOCHS = 2
FINETUNE_LR = 2e-5
FROZEN_EPOCHS = 3
FROZEN_LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LABEL_NAMES = ["negative", "positive"]

# 文件路径
TRAIN_CSV = PROCESSED_DIR / "imdb_train.csv"
VAL_CSV = PROCESSED_DIR / "imdb_val.csv"
TEST_CSV = PROCESSED_DIR / "imdb_test.csv"
DATA_STATS = PROCESSED_DIR / "data_stats.json"
