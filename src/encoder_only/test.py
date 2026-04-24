import torch
from model import SortTransformer

# 参数配置（需要和训练时一致）
VOCAB_SIZE = 10
D_MODEL = 32
DIM_SIZE = 20

# 加载模型
model = SortTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=DIM_SIZE)
model.load_state_dict(torch.load("sort_model.pth"))
model.eval()
print("模型加载成功！\n")


def predict(arr):
    """
    输入一个数组，返回排序结果

    参数:
        arr: 列表或tensor，包含0-9的数字

    返回:
        排序后的列表
    """
    if isinstance(arr, list):
        x = torch.tensor([arr])
    else:
        x = arr.unsqueeze(0) if arr.dim() == 1 else arr

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=-1)

    return pred[0].tolist()


def interactive_test():
    """交互式测试"""
    print("===== 交互式测试 =====")
    print("输入10个数字（0-9），用空格分隔，输入 q 退出")

    while True:
        user_input = input("\n请输入数字: ").strip()

        if user_input.lower() == 'q':
            print("退出测试")
            break

        try:
            arr = list(map(int, user_input.split()))
            if len(arr) != DIM_SIZE:
                print(f"请输入恰好 {DIM_SIZE} 个数字")
                continue

            result = predict(arr)
            correct = sorted(arr)
            is_correct = result == correct

            print(f"输入:   {arr}")
            print(f"预测:   {result}")
            print(f"正确:   {correct}")
            print(f"结果:   {'✓ 正确' if is_correct else '✗ 错误'}")

        except ValueError:
            print("请输入有效的数字")


def batch_test(num_samples=10):
    """批量随机测试"""
    print(f"\n===== 批量测试 ({num_samples}个样本) =====")

    correct_count = 0
    for i in range(num_samples):
        x = torch.randint(0, VOCAB_SIZE, (1, DIM_SIZE))
        y = torch.sort(x, dim=-1).values

        pred = predict(x[0].tolist())
        correct = y[0].tolist()
        is_correct = pred == correct
        correct_count += int(is_correct)

        print(f"样本{i+1} | 输入: {x[0].tolist()}")
        print(f"       | 预测: {pred}")
        print(f"       | 正确: {correct}")
        print(f"       | 结果: {'✓' if is_correct else '✗'}\n")

    print(f"正确率: {correct_count}/{num_samples}")


if __name__ == "__main__":
    # 批量测试
    batch_test(10)

    # 交互式测试
    # interactive_test()
