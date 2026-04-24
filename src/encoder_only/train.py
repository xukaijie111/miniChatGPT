
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SortTransformer
from dataset import SortDataset


NUM_ARRAYS = 2000
DIM_SIZE = 20
MAX_VALUE = 9
VOCAB_SIZE = MAX_VALUE + 1
D_MODEL = 32

BATCH_SIZE = 32
EPOCHS = 200


train_dataset = SortDataset(num_arrays=NUM_ARRAYS,dim_size=DIM_SIZE,max_value=MAX_VALUE)
train_loader = DataLoader(train_dataset,batch_size = BATCH_SIZE,shuffle=True)

# 测试一下
for x, y in train_loader:
    print("输入形状:", x.shape)   # (32, 10)
    print("标签形状:", y.shape)   # (32, 10)
    print("输入样例:", x[0])
    print("标签样例:", y[0])
    break

model = SortTransformer(vocab_size=VOCAB_SIZE,d_model = D_MODEL,max_len=DIM_SIZE)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)



def count_parameters(model):
    """计算模型参数总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = len(train_loader)

        for x, y in train_loader:
            logits = model(x)
            loss_value = loss(logits.view(-1, VOCAB_SIZE), y.view(-1))

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item()

        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")


def test():
    model.eval()
    with torch.no_grad():
        # 测试训练数据
        print("\n===== 训练数据测试 (前5个样本) =====")
        correct_count = 0
        for i in range(min(5, len(train_dataset))):
            train_x, train_y = train_dataset[i]
            train_x = train_x.unsqueeze(0)
            train_y = train_y.unsqueeze(0)

            logits = model(train_x)
            pred = torch.argmax(logits, dim=-1)
            is_correct = torch.equal(pred, train_y)
            correct_count += int(is_correct)

            print(f"样本{i+1} | 输入: {train_x[0].tolist()}")
            print(f"       | 预测: {pred[0].tolist()}")
            print(f"       | 正确: {train_y[0].tolist()}")
            print(f"       | 结果: {'✓' if is_correct else '✗'}\n")

        print(f"训练数据正确率: {correct_count}/5")

        # 测试新数据
        print("\n===== 新数据测试 (10个样本) =====")

        # 收集训练数据集的所有样本，用于去重
        train_data_set = set()
        for idx in range(len(train_dataset)):
            sample = tuple(train_dataset[idx][0].tolist())
            train_data_set.add(sample)

        correct_count = 0
        for i in range(10):
            # 生成测试数据，确保不在训练集中
            while True:
                test_x = torch.randint(0, VOCAB_SIZE, (1, DIM_SIZE))
                if tuple(test_x[0].tolist()) not in train_data_set:
                    break

            test_y = torch.sort(test_x, dim=-1).values

            logits = model(test_x)
            pred = torch.argmax(logits, dim=-1)
            is_correct = torch.equal(pred, test_y)
            correct_count += int(is_correct)

            print(f"样本{i+1} | 输入: {test_x[0].tolist()}")
            print(f"       | 预测: {pred[0].tolist()}")
            print(f"       | 正确: {test_y[0].tolist()}")
            print(f"       | 结果: {'✓' if is_correct else '✗'}\n")

        print(f"新数据正确率: {correct_count}/10")


if __name__ == "__main__":
    train()
    test()
    print(f"\n模型参数总数: {count_parameters(model):,}")

    # 保存模型
    torch.save(model.state_dict(), "sort_model.pth")
    print("模型已保存到 sort_model.pth")
