"""
语言模型数据集

数据格式: 纯文本序列
训练目标: 每个位置预测下一个字符

这就是语言模型的核心：
    输入: 今天天气真好
    目标: 天天气真好啊
    每个位置学习预测下一个字
"""
import json
import torch
from vocab import BertVocab


class TextDataset:
    """
    纯文本数据集

    数据处理流程:
        1. 加载JSON文本数据
        2. 用词表编码成数字序列
        3. x = 文本序列
        4. y = x的下一个字符（左移一位）
    """

    def __init__(self, data_path, vocab, max_len=64):
        """
        参数:
            data_path: JSON数据文件路径
            vocab: BertVocab词表对象
            max_len: 最大序列长度（超出截断）
        """
        self.vocab = vocab
        self.max_len = max_len

        # 加载数据
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # 预处理：编码所有文本
        self.samples = []
        for item in self.data:
            text = item["text"]

            # 编码: 文本 → 索引列表，末尾加EOS
            encoded = vocab.encode(text)
            full_seq = encoded + [vocab.EOS_IDX]

            # 截断（如果太长）
            if len(full_seq) > max_len:
                full_seq = full_seq[:max_len]

            # 目标y: 每个位置预测下一个字符
            # y[i] = x[i+1]，最后一位填PAD
            y = full_seq[1:] + [vocab.PAD_IDX]

            # 转成tensor
            self.samples.append({
                "x": torch.tensor(full_seq),
                "y": torch.tensor(y),
            })

        print(f"加载了 {len(self.samples)} 条文本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回训练数据

        x: [seq_len] 输入序列
        y: [seq_len] 目标序列（预测下一个字符）
        """
        sample = self.samples[idx]
        return sample["x"], sample["y"]


def test_dataset():
    """测试数据集"""
    vocab = BertVocab()

    dataset = TextDataset("data.json", vocab, max_len=64)

    print(f"\n数据集大小: {len(dataset)}")

    # 看第一条数据
    x, y = dataset[0]
    print(f"\n第一条数据:")
    print(f"x: {vocab.decode(x.tolist())}")
    print(f"y: {vocab.decode(y.tolist())}")


if __name__ == "__main__":
    test_dataset()