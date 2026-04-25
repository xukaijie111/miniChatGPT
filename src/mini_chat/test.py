"""
语言模型测试脚本

交互式文本续写:
    用户输入文本开头 → 模型自回归生成后续

这就是GPT的使用方式：
    给开头 → 模型续写后面的内容
"""
import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "../decoder_only"))

import torch
from model import SortDecoderTransformer
from vocab import BertVocab


BASE_DIR = os.path.dirname(__file__)
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "mini_chat_model.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="mini_chat 文本续写测试")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="模型权重路径")
    parser.add_argument("--max-len", type=int, default=256, help="训练时使用的最大长度")
    parser.add_argument("--d-model", type=int, default=64, help="隐藏维度")
    parser.add_argument("--n-layers", type=int, default=2, help="Transformer 层数")
    parser.add_argument("--n-heads", type=int, default=2, help="注意力头数")
    parser.add_argument("--max-gen-len", type=int, default=32, help="最大续写长度")
    return parser.parse_args()

# ============ 加载模型 ============
print("加载词表...")
vocab = BertVocab()
VOCAB_SIZE = len(vocab)
args = parse_args()

print("加载模型...")
model = SortDecoderTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=args.d_model,
    n_layers=args.n_layers,
    n_heads=args.n_heads,
    max_len=args.max_len,
)

model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
model.eval()
print("模型加载成功！\n")


def generate(model, text, max_new_tokens):
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
            # 只保留最后 max_len 个 token，避免超过位置编码长度
            context = generated[-args.max_len :]
            logits = model(context.unsqueeze(0))

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
        result = generate(model, user_input, max_new_tokens=args.max_gen_len)
        print(f"续写: {result}")


if __name__ == "__main__":
    chat()