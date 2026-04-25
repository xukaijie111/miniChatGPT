"""
开源词表 - 使用 BERT 中文 Tokenizer

使用 transformers 库的 BertTokenizer，无需自己构建词表。

词表大小: ~21128（包含常用汉字和符号）

优点:
    - 开源稳定，无需自己维护
    - 包含所有常用汉字，不用担心缺失
    - 社区广泛使用

需要安装:
    pip install transformers
"""
from transformers import BertTokenizer


class BertVocab:
    """
    BERT 中文词表封装

    特殊token:
        [PAD] = 0   填充
        [UNK] = 100 未知
        [CLS] = 101 句子开始
        [SEP] = 102 句子分隔/结束

    使用 bert-base-chinese 预训练词表
    """

    def __init__(self):
        # 加载BERT中文tokenizer
        # 会自动下载词表文件到本地缓存
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        # 特殊token索引
        self.PAD_IDX = self.tokenizer.pad_token_id  # 0
        self.UNK_IDX = self.tokenizer.unk_token_id  # 100
        self.EOS_IDX = self.tokenizer.sep_token_id  # 102 (SEP作为结束符)
        self.SEP_IDX = self.tokenizer.sep_token_id  # 102

        print(f"词表大小: {len(self.tokenizer)}")
        print(f"PAD索引: {self.PAD_IDX}")
        print(f"UNK索引: {self.UNK_IDX}")
        print(f"EOS索引: {self.EOS_IDX}")

    def encode(self, text, max_len=None):
        """
        编码：文本 → 索引列表

        例如:
            "你好" → [872, 1962]

        add_special_tokens=False: 不自动添加[CLS]和[SEP]
        """
        if max_len is None:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )

    def decode(self, indices):
        """
        解码：索引列表 → 文本

        例如:
            [872, 1962] → "你好"

        遇到 PAD 和 EOS 停止
        """
        # 过滤掉特殊token
        filtered = []
        for idx in indices:
            if idx == self.PAD_IDX:
                continue
            if idx == self.EOS_IDX:
                break
            filtered.append(idx)

        return self.tokenizer.decode(filtered)

    def __len__(self):
        return len(self.tokenizer)


def test_vocab():
    """测试词表"""
    vocab = BertVocab()

    # 测试编码解码
    text = "你好呀，我是一个聊天机器人"
    encoded = vocab.encode(text)
    decoded = vocab.decode(encoded)

    print(f"\n测试编码解码:")
    print(f"原文: {text}")
    print(f"编码: {encoded}")
    print(f"编码长度: {len(encoded)}")
    print(f"解码: {decoded}")


if __name__ == "__main__":
    test_vocab()