"""
语言模型测试脚本

交互式文本续写:
    用户输入文本开头 → 模型自回归生成后续

这就是GPT的使用方式：
    给开头 → 模型续写后面的内容
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../decoder_only"))

import torch
from model import SortDecoderTransformer
from vocab import BertVocab


# ============ 参数配置（和训练时一致）============
VOCAB_SIZE = 21128
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 2
MAX_LEN = 64
MAX_GEN_LEN = 32  # 最大续写长度

# ============ 加载模型 ============
print("加载词表...")
vocab = BertVocab()

print("加载模型...")
model = SortDecoderTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    max_len=MAX_LEN
)

model.load_state_dict(torch.load("mini_chat_model.pth"))
model.eval()
print("模型加载成功！\n")


def generate(model, text, max_new_tokens=MAX_GEN_LEN):
    """
    自回归续写文本

    流程:
        1. 编码输入文本
        2. 输入模型，取最后位置logits
        3. argmax得到下一个字
        4. 拼接，继续预测
        5. 遇到EOS或达到最大长度停止

    参数:
        text: 用户输入的文本开头
        max_new_tokens: 最多续写的字数

    返回:
        续写后的完整文本
    """
    # 编码输入
    input_ids = vocab.encode(text)
    generated = torch.tensor(input_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 输入模型
            logits = model(generated.unsqueeze(0))

            # 取最后位置的logits，预测下一个字
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits)

            # 遇到EOS就停止
            if next_token == vocab.EOS_IDX:
                break

            # 拼接到序列末尾
            generated = torch.cat([generated, next_token.unsqueeze(0)])

    # 解码完整文本
    result = vocab.decode(generated.tolist())
    return result


def chat():
    """
    交互式文本续写

    用户输入开头 → 模型续写
    """
    print("===== 语言模型续写 =====")
    print("输入文本开头，模型会续写后面的内容")
    print("输入 'q' 退出")
    print("=" * 30)

    while True:
        user_input = input("\n输入: ").strip()

        if user_input.lower() == "q":
            print("再见！")
            break

        if not user_input:
            continue

        # 续写文本
        result = generate(model, user_input)
        print(f"续写: {result}")


if __name__ == "__main__":
    chat()