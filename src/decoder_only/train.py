import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SortDecoderTransformer
from dataset import SortDecoderDatasetV2


# ============ 超参数配置 ============
NUM_SAMPLES = 2000      # 样本数量
SEQ_LEN = 20            # 输入序列长度（输出长度相同）
MAX_VALUE = 9           # 数字最大值，范围 0-9
VOCAB_SIZE = MAX_VALUE + 1  # 词表大小 = 10
D_MODEL = 32            # 隐藏维度，所有向量都用这个维度
N_LAYERS = 3            # Transformer 层数，堆叠多层让模型更强大
N_HEADS = 2             # 注意力头数，多头可以关注不同模式
MAX_LEN = SEQ_LEN * 2   # 序列最大长度 = 输入(20) + 输出(20) = 40

BATCH_SIZE = 32         # 每批次样本数
EPOCHS = 200            # 训练轮数
LR = 0.001              # 学习率

# ============ 创建数据集 ============
# 数据格式: [输入序列(20个)] + [输出序列(20个)] = 40 个 token
# 训练目标: 每个位置预测下一个 token
# Loss Mask: 只计算输出部分（从位置19开始），让模型学习"输入→输出"过渡
train_dataset = SortDecoderDatasetV2(
    num_samples=NUM_SAMPLES,
    seq_len=SEQ_LEN,
    max_value=MAX_VALUE
)

# DataLoader: 批量加载数据，shuffle=True 表示每个 epoch 随机打乱
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 测试数据格式
print("\n===== 数据格式测试 =====")
for x, y, mask in train_loader:
    print(f"x 形状: {x.shape}")        # [32, 40] 完整序列
    print(f"y 形状: {y.shape}")        # [32, 40] 预测目标（每个位置的下一个token）
    print(f"mask 形状: {mask.shape}")  # [32, 40] 只计算输出部分的 loss
    print(f"x 样例: {x[0].tolist()}")  # 输入+输出 拼接在一起
    print(f"y 样例: {y[0].tolist()}")  # 每个位置往后移一位，最后填 -1
    print(f"mask 样例: {mask[0].tolist()}")  # 前面输入部分为0，后面输出部分为1
    break

# ============ 创建模型 ============
# Decoder-only Transformer，类似 GPT
# 输出 logits: [batch, seq_len, vocab_size]，每个位置的 10 维向量对应数字 0-9
model = SortDecoderTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    max_len=MAX_LEN
)

# CrossEntropyLoss: 交叉熵损失，用于分类任务
# ignore_index=-1: y=-1 的位置不计算 loss（序列最后一位）
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

# Adam 优化器：自适应学习率，训练更稳定
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def count_parameters(model):
    """计算模型参数总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    """
    训练函数

    流程:
        1. 模型预测 logits
        2. 用 Loss Mask 选择哪些位置计算 loss
        3. 计算损失，反向传播，更新参数

    Loss Mask 的作用:
        mask=0 的位置，把 y 设为 -1，CrossEntropyLoss 自动忽略
        mask=1 的位置，正常计算 loss

        位置19（输入最后一位）mask=1，学习"输入→输出"过渡
    """
    print("\n===== 开始训练 =====")
    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = len(train_loader)

        for x, y, mask in train_loader:
            # 模型预测
            # logits 形状: [batch, seq_len*2, vocab_size] = [32, 40, 10]
            logits = model(x)

            # 计算 loss：只计算输出部分（mask 为 1 的位置）
            # flatten 成 [batch*seq_len, vocab_size] 和 [batch*seq_len]
            logits_flat = logits.view(-1, VOCAB_SIZE)  # [1280, 10]
            y_flat = y.view(-1)  # [1280]

            # Loss Mask 处理：
            # mask 为 0 的位置（输入部分），把 y 设为 -1
            # CrossEntropyLoss(ignore_index=-1) 会自动忽略这些位置
            y_masked = y_flat.clone()
            mask_flat = mask.view(-1)
            y_masked[mask_flat == 0] = -1

            # 计算损失
            loss_value = loss_fn(logits_flat, y_masked)

            # 反向传播
            optimizer.zero_grad()  # 清空梯度
            loss_value.backward()  # 计算梯度
            optimizer.step()       # 更新参数

            total_loss += loss_value.item()

        # 打印平均损失
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    print(f"模型参数总数: {count_parameters(model):,}")

    train()

    # 保存模型权重
    torch.save(model.state_dict(), "sort_decoder_model.pth")
    print("\n模型已保存到 sort_decoder_model.pth")