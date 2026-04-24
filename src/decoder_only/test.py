"""
Decoder-only Transformer 测试脚本

测试流程:
    1. 加载训练好的模型
    2. 输入待排序的数字序列（20个）
    3. 自回归生成排序结果（逐步预测下一个 token）
    4. 比较预测结果和正确排序结果

自回归生成:
    输入 [3, 1, 4, 1, 5]  →  预测下一个 1  →  拼接 [3, 1, 4, 1, 5, 1]
    输入 [3, 1, 4, 1, 5, 1]  →  预测下一个 1  →  拼接 [3, 1, 4, 1, 5, 1, 1]
    ...重复20次...
    最终得到 [输入(20个), 排序结果(20个)]
"""
import torch
from model import SortDecoderTransformer


# ============ 参数配置（需要和训练时一致）============
VOCAB_SIZE = 10    # 词表大小，数字 0-9
D_MODEL = 32       # 隐藏维度
SEQ_LEN = 20       # 输入序列长度
MAX_LEN = SEQ_LEN * 2  # 序列最大长度 = 输入 + 输出
N_LAYERS = 3       # Transformer 层数
N_HEADS = 2        # 注意力头数

# ============ 加载模型 ============
model = SortDecoderTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    max_len=MAX_LEN
)

# 加载训练好的权重
model.load_state_dict(torch.load("sort_decoder_model.pth"))

# 设置为评估模式（关闭 dropout 等）
model.eval()
print("模型加载成功！\n")


def generate(model, input_seq, max_new_tokens):
    """
    自回归生成排序结果

    自回归的含义:
        当前输出 → 拼接 → 作为下一次输入 → 产生新输出 → 再拼接...
        序列自己"喂养"自己，一步一步生成

    参数:
        input_seq: 输入序列 [seq_len]，例如 [3, 1, 4, 1, 5, ...]（20个）
        max_new_tokens: 要生成的 token 数量，等于 seq_len = 20

    返回:
        生成的完整序列 [seq_len + max_new_tokens]
        例如 [3, 1, 4, 1, 5, ..., 1, 1, 3, 4, 5, ...]（40个）

    流程:
        Step 1: 输入 [输入(20)] → logits[19] 预测下一个 → 拼接
        Step 2: 输入 [输入(20), 预测1] → logits[20] 预测下一个 → 拼接
        ...重复...
        最终: [输入(20), 排序结果(20)]
    """
    generated = input_seq.clone()

    with torch.no_grad():  # 不计算梯度，节省内存
        for step in range(max_new_tokens):
            # 模型预测整个序列的 logits
            # logits 形状: [1, current_len, vocab_size]
            logits = model(generated.unsqueeze(0))

            # 取最后一个位置的 logits，预测下一个 token
            # logits[0, -1, :] 形状: [vocab_size] = [10]
            # 每个维度对应数字 0-9 的预测得分
            next_token_logits = logits[0, -1, :]

            # argmax: 取得分最高的数字作为预测结果
            next_token = torch.argmax(next_token_logits)

            # 拼接到序列末尾，序列长度 +1
            generated = torch.cat([generated, next_token.unsqueeze(0)])

    return generated


def predict(arr):
    """
    输入一个数组，返回排序结果

    参数:
        arr: 列表，包含 0-9 的数字，长度等于 SEQ_LEN
        例如 [3, 1, 4, 1, 5, 0, 9, 2, 6, 7, ...]

    返回:
        排序后的列表，长度等于 SEQ_LEN
        例如 [0, 1, 1, 2, 3, 4, 5, 6, 7, 9, ...]
    """
    x = torch.tensor(arr)

    # 自回归生成：输入20个，生成20个
    generated = generate(model, x, SEQ_LEN)

    # 取输出部分（后半部分，位置 20-39）
    output = generated[SEQ_LEN:].tolist()

    return output


def interactive_test():
    """
    交互式测试

    用户手动输入数字序列，模型输出排序结果
    方便测试特定输入
    """
    print("===== 交互式测试 =====")
    print(f"输入 {SEQ_LEN} 个数字（0-9），用空格分隔，输入 q 退出")

    while True:
        user_input = input("\n请输入数字: ").strip()

        if user_input.lower() == 'q':
            print("退出测试")
            break

        try:
            arr = list(map(int, user_input.split()))
            if len(arr) != SEQ_LEN:
                print(f"请输入恰好 {SEQ_LEN} 个数字")
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
    """
    批量随机测试

    生成多个随机输入，测试模型正确率
    不展示生成过程，只显示最终结果

    参数:
        num_samples: 测试样本数量，默认 10
    """
    print(f"\n===== 批量测试 ({num_samples}个样本) =====")

    correct_count = 0
    for i in range(num_samples):
        # 生成随机输入: [SEQ_LEN] 个数字，范围 0-9
        x = torch.randint(0, VOCAB_SIZE, (SEQ_LEN,))

        # 正确排序结果（用 torch.sort 计算）
        y_sorted = torch.sort(x).values

        # 自回归生成预测
        generated = generate(model, x, SEQ_LEN)

        # 取输出部分（后 SEQ_LEN 个）
        pred_output = generated[SEQ_LEN:]

        # 比较预测和正确结果
        is_correct = torch.equal(pred_output, y_sorted)
        correct_count += int(is_correct)

        print(f"样本{i+1} | 输入: {x.tolist()}")
        print(f"       | 预测: {pred_output.tolist()}")
        print(f"       | 正确: {y_sorted.tolist()}")
        print(f"       | 结果: {'✓' if is_correct else '✗'}\n")

    print(f"正确率: {correct_count}/{num_samples}")


def step_by_step_test():
    """
    逐步生成测试（展示自回归过程）

    测试 1 个样本，详细展示每一步的预测过程
    方便理解自回归生成原理

    流程:
        初始: [输入(20)]
        Step 1: logits[19] → 预测数字 X → 拼接
        Step 2: logits[20] → 预测数字 Y → 拼接
        ...每一步打印预测的数字和当前序列...
        最终: [输入(20), 排序结果(20)]
    """
    print("\n===== 逐步生成测试 =====")

    # 生成随机输入
    x = torch.randint(0, VOCAB_SIZE, (SEQ_LEN,))
    y_sorted = torch.sort(x).values

    print(f"输入序列: {x.tolist()}")
    print(f"正确排序: {y_sorted.tolist()}")
    print("\n生成过程:")

    generated = x.clone()
    print(f"初始: {generated.tolist()}")

    with torch.no_grad():
        for step in range(SEQ_LEN):
            # 模型预测
            logits = model(generated.unsqueeze(0))

            # 取最后一个位置的 logits
            next_token_logits = logits[0, -1, :]

            # 预测下一个数字
            next_token = torch.argmax(next_token_logits)

            # 拼接
            generated = torch.cat([generated, next_token.unsqueeze(0)])

            # 打印每一步
            print(f"步骤 {step+1}: 预测数字 {next_token.item()}, 当前序列长度: {len(generated)}")

    # 最终结果
    pred_output = generated[SEQ_LEN:].tolist()
    correct = y_sorted.tolist()
    is_correct = pred_output == correct

    print(f"\n最终预测: {pred_output}")
    print(f"正确结果: {correct}")
    print(f"结果: {'✓ 正确' if is_correct else '✗ 错误'}")


if __name__ == "__main__":
    # 批量测试（默认）
    batch_test(10)

    # 逐步生成测试（可选，展示生成过程）
    # step_by_step_test()

    # 交互式测试（可选）
    # interactive_test()