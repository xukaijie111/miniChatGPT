import torch


class SortDecoderDataset:
    """
    Decoder-only 排序数据集

    数据格式: [输入序列] + [输出序列]
    例如: [3, 1, 4, 1, 5, 1, 1, 3, 4, 5]

    总长度 = seq_len * 2

    训练目标: 每个位置预测下一个 token
    - x = 完整序列 [输入, 输出]
    - y = x 的下一个 token（x 向后移一位）
    """

    def __init__(self, num_samples=100, seq_len=20, max_value=9):
        """
        参数:
            num_samples: 样本数量
            seq_len:     输入序列长度（输出长度相同）
            max_value:   数字最大值 (0 ~ max_value)
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.max_value = max_value

        # 生成数据
        self._generate_data()

    def _generate_data(self):
        """生成随机数据并拼接"""
        # 生成随机输入: [num_samples, seq_len]
        self.inputs = torch.randint(0, self.max_value + 1, (self.num_samples, self.seq_len))

        # 排序得到输出: [num_samples, seq_len]
        self.targets, _ = torch.sort(self.inputs, dim=1)

        # 拼接成完整序列: [num_samples, seq_len * 2]
        self.data = torch.cat([self.inputs, self.targets], dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        返回训练数据

        x: [seq_len * 2] 完整序列，作为模型输入
        y: [seq_len * 2] 目标序列，每个位置是下一个 token

        例如 seq_len=5:
        data[idx] = [3, 1, 4, 1, 5, 1, 1, 3, 4, 5]
        x = [3, 1, 4, 1, 5, 1, 1, 3, 4, 5]   ← 模型输入
        y = [1, 4, 1, 5, 1, 1, 3, 4, 5, -1]  ← 每个位置预测下一个，最后一位填 -1

        训练时: 位置 i 的 logits 预测 y[i]
        """
        x = self.data[idx]
        # y 是 x 的下一个 token，最后一个位置用 -1 填充（训练时忽略）
        y = torch.cat([x[1:], torch.tensor([-1])])

        return x, y


class SortDecoderDatasetV2:
    """
    Decoder-only 排序数据集（版本2：只计算输出部分的 loss）

    数据格式: [输入序列] + [输出序列]

    只在输出部分计算 loss，输入部分不参与 loss 计算
    这样模型只需要学会：在 seq_len 位置开始输出排序结果
    """

    def __init__(self, num_samples=100, seq_len=20, max_value=9):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.max_value = max_value
        self._generate_data()

    def _generate_data(self):
        self.inputs = torch.randint(0, self.max_value + 1, (self.num_samples, self.seq_len))
        self.targets, _ = torch.sort(self.inputs, dim=1)
        self.data = torch.cat([self.inputs, self.targets], dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        返回训练数据

        x: [seq_len * 2] 完整序列
        y: [seq_len * 2] 目标序列
        mask: [seq_len * 2] 从 seq_len-1 开始计算 loss，包含过渡位置

        例如 seq_len=5:
        x     = [3, 1, 4, 1, 5, 1, 1, 3, 4, 5]
        y     = [1, 4, 1, 5, 1, 1, 3, 4, 5, -1]
        mask  = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  ← 从位置4(输入最后)开始计算

        关键：位置 seq_len-1 的 loss 也要计算，学习"输入→输出"的过渡
        """
        x = self.data[idx]
        y = torch.cat([x[1:], torch.tensor([-1])])

        # 创建 mask：从输入最后一个位置开始计算 loss
        # 这样模型能学习"输入结束 → 输出开始"的过渡
        mask = torch.zeros(self.seq_len * 2)
        mask[self.seq_len - 1:] = 1  # 从位置 seq_len-1 开始设为 1

        return x, y, mask