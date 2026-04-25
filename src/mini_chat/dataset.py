"""
语言模型数据集（支持 LCCC jsonl.gz）

数据格式:
    - json: [{"text": "..."}]
    - jsonl / jsonl.gz: 每行一个对话列表，如 ["你好", "你好呀", ...]

训练目标:
    每个位置预测下一个 token（Causal LM）。
"""
import gzip
import json
import os
import torch


class TextDataset:
    """
    纯文本数据集

    数据处理流程:
        1. 加载JSON文本数据
        2. 用词表编码成数字序列
        3. x = 文本序列
        4. y = x的下一个字符（左移一位）
    """

    def __init__(self, data_path, vocab, max_len=64, max_samples=None):
        """
        参数:
            data_path: JSON数据文件路径
            vocab: BertVocab词表对象
            max_len: 最大序列长度（超出截断）
            max_samples: 最多加载样本数，None 表示不限制
        """
        self.vocab = vocab
        self.max_len = max_len

        # 预处理：编码所有文本
        self.samples = []
        for text in self._iter_texts(data_path):
            # 编码阶段就截断，避免 transformers 的 512 长度告警
            encoded = vocab.encode(text, max_len=max_len - 1)
            if not encoded:
                continue

            full_seq = encoded + [vocab.EOS_IDX]

            # y[i] = x[i+1]，最后一位填 PAD（loss 会 ignore PAD）
            y = full_seq[1:] + [vocab.PAD_IDX]

            self.samples.append(
                {
                    "x": torch.tensor(full_seq, dtype=torch.long),
                    "y": torch.tensor(y, dtype=torch.long),
                }
            )
            if max_samples is not None and len(self.samples) >= max_samples:
                break

        print(f"加载了 {len(self.samples)} 条文本")

    def _iter_texts(self, data_path):
        """根据文件类型迭代文本样本。"""
        lower = data_path.lower()
        if lower.endswith(".json"):
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                if isinstance(item, dict) and "text" in item:
                    text = str(item["text"]).strip()
                    if text:
                        yield text
            return

        if lower.endswith(".jsonl") or lower.endswith(".jsonl.gz"):
            opener = gzip.open if lower.endswith(".gz") else open
            with opener(data_path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # LCCC: 每行通常是多轮对话 list[str]
                    if isinstance(obj, list):
                        turns = []
                        for turn in obj:
                            turn = str(turn).strip()
                            if not turn:
                                continue
                            # LCCC 常见空格分词，改成自然句子更适合字符级建模
                            turns.append(turn.replace(" ", ""))
                        if len(turns) >= 2:
                            # 把整段对话拼成一个训练样本，SEP 作为轮次分隔
                            text = "[SEP]".join(turns)
                            if text:
                                yield text
                    elif isinstance(obj, dict) and "text" in obj:
                        text = str(obj["text"]).strip()
                        if text:
                            yield text
            return

        raise ValueError(
            f"不支持的数据格式: {os.path.basename(data_path)}。仅支持 .json/.jsonl/.jsonl.gz"
        )

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
    from vocab import BertVocab
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