"""
语言模型训练脚本

模型: Decoder-only Transformer（GPT风格）
训练: 纯文本，学习预测下一个字

这就是GPT的核心原理：
    给文本 → 每个位置预测下一个字 → 学会语言规律
"""
import sys
import os
import time

# 添加decoder_only目录到路径，以便import模型
sys.path.append(os.path.join(os.path.dirname(__file__), "../decoder_only"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SortDecoderTransformer
from vocab import BertVocab
from dataset import TextDataset


BASE_DIR = os.path.dirname(__file__)
TRAIN_DATA_PATH = os.getenv(
    "TRAIN_DATA_PATH",
    os.path.abspath(os.path.join(BASE_DIR, "../dataset/lccc_base_train.jsonl.gz")),
)
VALID_DATA_PATH = os.getenv(
    "VALID_DATA_PATH",
    os.path.abspath(os.path.join(BASE_DIR, "../dataset/lccc_base_valid.jsonl.gz")),
)
SAVE_PATH = os.getenv("SAVE_PATH", os.path.join(BASE_DIR, "mini_chat_model.pth"))

MAX_LEN = int(os.getenv("MAX_LEN", "256"))
D_MODEL = int(os.getenv("D_MODEL", "64"))
N_LAYERS = int(os.getenv("N_LAYERS", "2"))
N_HEADS = int(os.getenv("N_HEADS", "2"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
EPOCHS = int(os.getenv("EPOCHS", "100"))
LR = float(os.getenv("LR", "1e-3"))
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "200000"))
VAL_MAX_SAMPLES = int(os.getenv("VAL_MAX_SAMPLES", "20000"))
PATIENCE = int(os.getenv("PATIENCE", "3"))
MIN_DELTA = float(os.getenv("MIN_DELTA", "1e-4"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# ============ 加载词表 ============
print("\n===== 加载词表 =====")
vocab = BertVocab()
VOCAB_SIZE = len(vocab)
print(f"vocab_size = {VOCAB_SIZE}")

# ============ 加载数据集 ============
print("\n===== 加载数据集 =====")
print(f"数据路径: {TRAIN_DATA_PATH}")
dataset = TextDataset(TRAIN_DATA_PATH, vocab, max_len=MAX_LEN, max_samples=MAX_SAMPLES)
print(f"验证路径: {VALID_DATA_PATH}")
valid_dataset = TextDataset(
    VALID_DATA_PATH, vocab, max_len=MAX_LEN, max_samples=VAL_MAX_SAMPLES
)
print(f"设备: {DEVICE}")

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
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

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
    max_len=MAX_LEN,
)
model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"模型参数量: {count_parameters(model):,}")


def evaluate():
    """在验证集上计算平均 loss（不更新参数）。"""
    model.eval()
    total_loss = 0.0
    num_batches = len(valid_loader)
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            logits_flat = logits.view(-1, VOCAB_SIZE)
            y_flat = y.view(-1)
            loss_value = loss_fn(logits_flat, y_flat)
            total_loss += loss_value.item()
    return total_loss / max(1, num_batches)


# ============ 训练函数 ============
def train():
    print("\n===== 开始训练 =====")
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    epoch_times = []

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        num_batches = len(loader)

        for batch_idx, (x, y) in enumerate(loader, start=1):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
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
            if batch_idx % 100 == 0 or batch_idx == num_batches:
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | "
                    f"Batch {batch_idx}/{num_batches} | "
                    f"loss={loss_value.item():.4f}"
                )

        avg_loss = total_loss / num_batches
        val_loss = evaluate()
        elapsed = time.time() - epoch_start
        epoch_times.append(elapsed)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remain_epochs = EPOCHS - (epoch + 1)
        eta_seconds = avg_epoch_time * remain_epochs

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | "
            f"time={elapsed:.1f}s | eta={eta_seconds/60:.1f}min"
        )

        # val_loss 下降超过 min_delta 才算有效改进
        if val_loss < (best_val_loss - MIN_DELTA):
            best_val_loss = val_loss
            best_epoch = epoch + 1
            no_improve_epochs = 0
            save_dir = os.path.dirname(SAVE_PATH)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> 保存最佳模型: epoch={best_epoch}, val_loss={best_val_loss:.4f}")
        else:
            no_improve_epochs += 1
            print(f"  -> 验证集未提升: {no_improve_epochs}/{PATIENCE}")

        if no_improve_epochs >= PATIENCE:
            print(
                f"\nEarly stopping 触发：连续 {PATIENCE} 轮验证集无明显提升。"
                f"\n最佳模型在 epoch {best_epoch}, val_loss={best_val_loss:.4f}"
            )
            break


if __name__ == "__main__":
    train()
    print(f"\n训练结束，最佳模型已保存到 {SAVE_PATH}")