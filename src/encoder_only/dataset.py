import torch


class SortDataset:
    """排序用数据集类"""

    def __init__(self, num_arrays=100, dim_size=10, max_value=10):
        """
        初始化并自动生成数据集

        参数:
            num_arrays: 数组数量
            dim_size: 每份数组的维度大小
            max_value: 每个维度的最大数字
        """
        self.num_arrays = num_arrays
        self.dim_size = dim_size
        self.max_value = max_value
        self.data = None
        self.targets = None
        self._generate_data()

    def _generate_data(self):
        """自动生成随机数据并排序存入 targets"""
        # 生成随机数据张量 (num_arrays, dim_size)
        self.data = torch.randint(0, self.max_value + 1, (self.num_arrays, self.dim_size))

        # 对每个样本排序，存入 targets
        self.targets, _ = torch.sort(self.data, dim=1)

    def __getitem__(self, idx):
        """通过索引获取数据和目标"""
        return self.data[idx], self.targets[idx]

    def __len__(self):
        """获取数据集大小"""
        return self.num_arrays