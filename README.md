# 课程设计 实验一：基于预训练模型的 IMDB 情感分类

本项目完成《自然语言处理课程设计》实验一（预训练模型及微调方法应用）。

- 任务：IMDB 电影评论 **二分类情感分析**（negative / positive）
- 模型对比：
  1. **从零 Transformer**：Embedding + 位置编码 + 2 层 Transformer Encoder + 分类头
  2. **DistilBERT 冻结编码器**：只训练分类头
  3. **DistilBERT 全量微调**：预训练参数 + 分类头一起训练
- 评价指标：Accuracy、Precision、Recall、F1、ROC-AUC、混淆矩阵、训练曲线

> 所有数据、模型、缓存、临时文件均位于本目录（E 盘），不会写入 C 盘。

---

## 1. 目录结构

```
课程设计_实验一_预训练微调/
├─ .venv/                     # Python 3.8 虚拟环境
├─ _cache/                    # pip / HF / torch / matplotlib 缓存（E 盘）
├─ config.py                  # 路径、超参数、环境变量
├─ src/
│  ├─ download_data.py        # 下载 IMDB，清洗，划分 train/val/test
│  ├─ data.py                 # 词表、Dataset、DataLoader
│  ├─ model.py                # 从零 Transformer + DistilBERT 分类器
│  ├─ train.py                # 三种模式的训练脚本
│  ├─ evaluate.py             # 测试集评估、混淆矩阵、错误样例
│  ├─ summarize.py            # 汇总对比表与柱状图
│  └─ predict.py              # 单条影评预测 / 交互式演示
├─ data/
│  ├─ raw/                    # IMDB parquet
│  └─ processed/              # imdb_train/val/test.csv + data_stats.json
├─ models/
│  ├─ pretrained/             # HF 下载缓存
│  └─ tokenizer/              # 本地 tokenizer
├─ outputs/
│  ├─ logs/                   # 训练/评估日志、history.json
│  ├─ checkpoints/            # *_best.pt、scratch_vocab.json
│  ├─ figures/                # 训练曲线、混淆矩阵、模型对比图
│  ├─ predictions/            # 测试指标、分类报告、预测明细
│  └─ model_comparison.{csv,md}
├─ report/                    # 课程设计报告 docx + 答辩 PPT
```

## 2. 环境安装（全部写入 E 盘）

```bash
cd /d E:\自然语言处理\课程设计_实验一_预训练微调

set PIP_CACHE_DIR=E:\自然语言处理\课程设计_实验一_预训练微调\_cache\pip
set TMP=E:\自然语言处理\课程设计_实验一_预训练微调\_cache\tmp
set TEMP=E:\自然语言处理\课程设计_实验一_预训练微调\_cache\tmp

.venv\Scripts\python.exe -m pip install torch==2.4.1+cu121 ^
  --index-url https://download.pytorch.org/whl/cu121 ^
  --extra-index-url https://mirrors.aliyun.com/pypi/simple/

.venv\Scripts\python.exe -m pip install -r requirements.txt ^
  -i https://mirrors.aliyun.com/pypi/simple/
```

`config.py` 会在导入 torch/transformers 之前设置 `HF_HOME`、`HF_ENDPOINT`、
`TORCH_HOME`、`TMP`、`MPLCONFIGDIR` 等，保证运行时也不写 C 盘。

## 3. 运行

```bash
cd /d E:\自然语言处理\课程设计_实验一_预训练微调

# 1) 数据：下载 + 清洗 + 划分（已生成可跳过）
.venv\Scripts\python.exe src\download_data.py

# 2) 训练三个模型
.venv\Scripts\python.exe src\train.py --mode scratch
.venv\Scripts\python.exe src\train.py --mode frozen
.venv\Scripts\python.exe src\train.py --mode finetune

# 3) 测试集评估
.venv\Scripts\python.exe src\evaluate.py --all

# 4) 汇总对比表 / 图
.venv\Scripts\python.exe src\summarize.py

# 5) 交互式预测
.venv\Scripts\python.exe src\predict.py --text "This movie is wonderful and moving."
```

调试时可加 `--max-train-samples 2000 --epochs 1` 快速验证。

## 4. 实验设置

| 项目 | 设置 |
|---|---|
| 数据 | IMDB（Hugging Face `stanfordnlp/imdb`），清洗 HTML 标签 |
| 训练 / 验证 / 测试 | 20,000 / 4,902 / 5,000（分层抽样，seed=42） |
| 最大长度 | 256 token |
| 从零 Transformer | d_model=128, heads=4, layers=2, FFN=512, batch=64, lr=1e-3, epochs=3 |
| DistilBERT 冻结 | 只训练分类头, batch=32, lr=1e-3, epochs=3 |
| DistilBERT 微调 | 全部参数, batch=32, lr=2e-5, epochs=2, warmup=10% |
| 硬件 | RTX 4060 Laptop 8GB / 32GB RAM（CPU 也可运行） |
| 随机种子 | 42 |

## 5. 结果

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 从零 Transformer | 0.8404 | 0.8467 | 0.8325 | 0.8396 | 0.9227 |
| DistilBERT（冻结） | 0.8358 | 0.8561 | 0.8086 | 0.8317 | 0.9190 |
| **DistilBERT（全量微调）** | **0.9140** | **0.9176** | **0.9103** | **0.9139** | **0.9729** |

运行完成后查看：

- `outputs/model_comparison.md`：三个模型的指标对比表
- `outputs/figures/model_comparison.png`：指标柱状图
- `outputs/figures/*_curves.png`：训练/验证曲线
- `outputs/figures/*_confusion.png`：混淆矩阵
- `outputs/predictions/*_test_metrics.json`：完整指标
- `outputs/predictions/*_test_predictions.csv`：每条测试样本的预测

具体数值见运行日志与报告。

## 6. 报告与答辩

- `report/自然语言处理课程设计报告.docx`：13 页，按学院模板章节撰写
- `report/课程设计答辩_v2.pptx`：**推荐使用**，16 页，按 pptx skill 的设计规范重做
- `report/课程设计答辩.pptx`：v1 版本，保留用于对比

v2 的视觉要点：深色注意力热力图封面、章节标签胶囊、数据卡片、
去掉了模板化的页眉色条；每页至少一个图表/结构图/数据卡。
