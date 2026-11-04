# -*- coding: utf-8 -*-
"""模型定义：从零 Transformer 分类器 + DistilBERT 分类器。"""

import torch
import torch.nn as nn

import config


# ===========================================================================
# 一、从零构建 Transformer 文本分类器（对应课程 PPT 的 block + 整体模型）
# ===========================================================================
class TransformerBlock(nn.Module):
    """一个标准 Transformer Encoder block。

    结构：多头自注意力 -> 残差 + LayerNorm -> 前馈网络 -> 残差 + LayerNorm。
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.self_attn(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class ScratchTransformerClassifier(nn.Module):
    """Embedding + 位置编码 + 多个 TransformerBlock + 分类头。"""

    def __init__(self, vocab_size, num_classes=2, max_len=None,
                 d_model=None, nhead=None, num_layers=None,
                 dim_feedforward=None, dropout=None):
        super(ScratchTransformerClassifier, self).__init__()
        max_len = max_len or config.MAX_WORDS
        d_model = d_model or config.SCRATCH_D_MODEL
        nhead = nhead or config.SCRATCH_NHEAD
        num_layers = num_layers or config.SCRATCH_LAYERS
        dim_feedforward = dim_feedforward or config.SCRATCH_FF
        dropout = config.SCRATCH_DROPOUT if dropout is None else dropout

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch, seq_len)；padding 位置不参与注意力
        # [CLS] 前要补一个 False，使 mask 长度与加入 CLS 后的序列一致
        pad_mask = input_ids.eq(0)
        cls_mask = torch.zeros(input_ids.size(0), 1, dtype=torch.bool,
                               device=input_ids.device)
        key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1)
        x = self.embedding(input_ids)
        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)[:, :self.position.size(1), :]
        x = self.dropout(x + self.position[:, :x.size(1), :])

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        pooled = x[:, 0]              # 取 [CLS] 位置的表示
        return self.classifier(pooled)


# ===========================================================================
# 二、预训练 DistilBERT + 分类头（微调 / 冻结编码器两种用法）
# ===========================================================================
class BertClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2):
        super(BertClassifier, self).__init__()
        from transformers import AutoModel

        cache_dir = config.MODEL_DIR / "pretrained"
        self.encoder = AutoModel.from_pretrained(
            config.BERT_MODEL_NAME, cache_dir=str(cache_dir))
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]   # [CLS] 向量
        return self.classifier(self.dropout(pooled))
