"""
语言模型训练脚本

模型: Decoder-only Transformer（GPT风格）
训练: 纯文本，学习预测下一个字

这就是GPT的核心原理：
    给文本 → 每个位置预测下一个字 → 学会语言规律
"""
import sys
import os

# 添加decoder_only目录到路径，以便import模型
sys.path.append(os.path.join(os.path.dirname(__file__), "../decoder_only"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SortDecoderTransformer
from vocab import BertVocab
from dataset import TextDataset


# ============ 超参数配置 ============
DATA_PATH = "data.json"     # 文本数据文件
MAX_LEN = 64                # 最大序列长度

# 模型参数（调小以适合CPU）
D_MODEL = 64                # 隐藏维度
N_LAYERS = 2                # 层数
N_HEADS = 2                 # 注意力头数

# 训练参数
BATCH_SIZE = 8              # 批大小
EPOCHS = 100                # 训练轮数
LR = 0.001                  # 学习率

# ============ 加载词表 ============
print("\n===== 加载词表 =====")
vocab = BertVocab()
VOCAB_SIZE = len(vocab)
print(f"vocab_size = {VOCAB_SIZE}")

# ============ 加载数据集 ============
print("\n===== 加载数据集 =====")
dataset = TextDataset(DATA_PATH, vocab, max_len=MAX_LEN)

# DataLoader需要自定义collate_fn处理不同长度的序列
def collate_fn(batch):
    """
    处理不同长度的序列

    将batch中的序列padding到相同长度
    """
    xs, ys = [], []
    for x, y in batch:
        xs.append(x)
        ys.append(y)

    # 找到最大长度
    max_len = max(len(x) for x in xs)

    # Padding
    padded_xs = []
    padded_ys = []
    for x, y in zip(xs, ys):
        pad_len = max_len - len(x)
        padded_x = torch.cat([x, torch.full((pad_len,), vocab.PAD_IDX)])
        padded_y = torch.cat([y, torch.full((pad_len,), vocab.PAD_IDX)])
        padded_xs.append(padded_x)
        padded_ys.append(padded_y)

    return torch.stack(padded_xs), torch.stack(padded_ys)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 测试数据格式
print("\n===== 数据格式测试 =====")
for x, y in loader:
    print(f"x形状: {x.shape}")
    print(f"y形状: {y.shape}")
    print(f"x样例: {vocab.decode(x[0].tolist())}")
    print(f"y样例: {vocab.decode(y[0].tolist())}")
    break

# ============ 创建模型 ============
print("\n===== 创建模型 =====")
model = SortDecoderTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    max_len=MAX_LEN
)

loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"模型参数量: {count_parameters(model):,}")


# ============ 训练函数 ============
def train():
    print("\n===== 开始训练 =====")
    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = len(loader)

        for x, y in loader:
            # 模型预测
            logits = model(x)  # [batch, seq_len, vocab_size]

            # 计算loss
            logits_flat = logits.view(-1, VOCAB_SIZE)
            y_flat = y.view(-1)

            loss_value = loss_fn(logits_flat, y_flat)

            # 反向传播
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item()

        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    train()

    # 保存模型
    torch.save(model.state_dict(), "mini_chat_model.pth")
    print("\n模型已保存到 mini_chat_model.pth")