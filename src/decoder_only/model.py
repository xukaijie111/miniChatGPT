import torch
import torch.nn as nn


class SortDecoderTransformer(nn.Module):
    """
    Decoder-only Transformer 用于排序任务（类似 GPT）

    输入格式: [输入序列] + [SEP] + [输出序列]
    例如: [3, 1, 4, 1, 5, SEP, 1, 1, 3, 4, 5]

    特点:
    - Causal Mask: 每个位置只能看到当前位置之前的信息
    - 自回归生成: 推理时逐个预测下一个 token
    """

    def __init__(self, vocab_size=11, d_model=32, n_layers=3, n_heads=2, max_len=42):
        """
        参数说明:
        vocab_size: 词表大小 = 数字范围(0-9) + 分隔符(SEP=10) = 11
        d_model:    隐藏维度
        n_layers:   Transformer 层数
        n_heads:    多头注意力的头数（这次真的用多头）
        max_len:    序列最大长度 = 输入长度 + SEP + 输出长度 + 1
                    例如: 20 + 1 + 20 + 1 = 42
        """
        super().__init__()

        # 保存超参数
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len

        # TODO: 下面我们一步一步添加组件

        assert d_model % n_heads == 0 
        self.head_dim = d_model / n_heads

        self.token_embedding = nn.Embedding(vocab_size,d_model)

        

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入序列 [batch_size, seq_len]

        返回:
            logits: [batch_size, seq_len, vocab_size]
        """
        # TODO: 一步一步实现
        pass