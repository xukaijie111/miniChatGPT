import torch
import torch.nn as nn


class SortTransformer(nn.Module):
    """
    Encoder-only Transformer 用于排序任务

    输入: 随机数字序列 [3, 1, 4, 1, 5]
    输出: 排序后的序列 [1, 1, 3, 4, 5]

    特点: 双向注意力，每个位置可以看到所有位置的信息
    """

    def __init__(self, vocab_size=6, d_model=10, n_layers=3, n_heads=1, max_len=10):
        """
        参数说明:
        vocab_size: 词表大小，数字范围 0 ~ vocab_size-1
                    例如 vocab_size=10 表示数字 0-9
        d_model:    模型的隐藏维度，所有向量都用这个维度
                    例如 d_model=32，则 embedding、Q/K/V 都是 32 维
        n_layers:   Transformer 层数，堆叠多层让模型更强大
                    通常 3-6 层对小任务足够
        n_heads:    多头注意力的头数（当前代码未实现多头，只是预留参数）
        max_len:    序列最大长度，position embedding 的范围
                    序列长度不能超过这个值
        """
        super().__init__()

        # ============ 保存超参数 ============
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len

        # ============ Embedding 层 ============
        # Token Embedding: 把每个数字映射成一个向量
        # 输入: 数字索引 (如 3) → 输出: d_model 维向量 (如 [32维])
        # 相当于一个查找表: vocab_size 行，每行是一个 d_model 维向量
        self.vocab_embedding = nn.Embedding(vocab_size, d_model)

        # Position Embedding: 给每个位置一个独特的向量
        # 位置 0 的向量 + 位置 1 的向量 + ... 让模型知道"顺序"
        # 为什么需要: Attention 本身不知道位置，打乱顺序结果一样
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # ============ Attention 投影层 (每层各有一套) ============
        # Q (Query): "我想找什么信息"
        # K (Key):   "我有什么信息可以匹配"
        # V (Value): "我的实际内容"
        # 每个都是 Linear(d_model → d_model)，把向量投影到另一个空间
        self.q_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.k_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.v_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])

        # Attention 输出投影: 把注意力结果再投影一次
        # 相当于让模型有更多表达能力，融合信息
        self.attn_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])

        # ============ LayerNorm 层 (每层各有两个) ============
        # norm1: Attention 后的归一化
        # norm2: FFN 后的归一化
        # 作用: 让数值稳定在均值附近，防止梯度爆炸/消失
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # ============ FFN (Feed Forward Network) 层 ============
        # 结构: Linear(d → 4d) → GELU → Linear(4d → d)
        # 作用: 给模型非线性表达能力，"思考"提取的信息
        # 为什么扩展 4 倍: 更大的"工作空间"处理复杂信息
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),  # 扩展维度: 32 → 128
                nn.GELU(),                        # 激活函数: 比 ReLU 更平滑
                nn.Linear(d_model * 4, d_model),  # 回归维度: 128 → 32
            )
            for _ in range(n_layers)
        ])

        # ============ 输出层 ============
        # 把 d_model 维向量投影回 vocab_size 维
        # 输出 logits: 每个位置对每个数字的"得分"
        # 后接 softmax 得到概率，argmax 得到预测数字
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        前向传播: 输入序列 → 输出 logits

        参数:
            x: 输入序列 [batch_size, seq_len]，如 [32, 20]
               每个元素是数字索引 (0 ~ vocab_size-1)

        返回:
            logits: [batch_size, seq_len, vocab_size]
                   每个位置对每个数字的预测得分
        """
        # ============ Step 1: Embedding ============
        batch_size, seq_len = x.shape

        # 生成位置索引: [0, 1, 2, ..., seq_len-1]
        # device=x.device 确保在同一个设备 (CPU/GPU)
        positions = torch.arange(seq_len, device=x.device)

        # Token Embedding + Position Embedding
        # vocab_embedding(x): 把每个数字转成向量 [batch, seq_len, d_model]
        # pos_embedding(positions): 每个位置的向量 [seq_len, d_model]
        # 相加后得到每个位置的完整表示
        h = self.vocab_embedding(x) + self.pos_embedding(positions)
        # h 形状: [batch_size, seq_len, d_model]

        # ============ Step 2: 多层 Transformer 处理 ============
        for i in range(self.n_layers):
            # ----- 2.1 计算 Q, K, V -----
            # 每个都是 [batch, seq_len, d_model]
            Q = self.q_proj[i](h)  # Query: 查询向量
            K = self.k_proj[i](h)  # Key:   键向量
            V = self.v_proj[i](h)  # Value: 值向量

            # ----- 2.2 计算注意力分数 -----
            # Q × K^T: 矩阵乘法，计算每个位置对其他位置的"相关性"
            # K.transpose(-2, -1): 交换最后两维，[batch, d_model, seq_len]
            # 结果: [batch, seq_len, seq_len]，每行是某位置对所有位置的分数
            scores = torch.matmul(Q, K.transpose(-2, -1))

            # 缩放: 除以 sqrt(d_model)
            # 为什么: 当 d_model 大时，Q×K 数值大，softmax 会梯度消失
            # 缩放后数值稳定，softmax 分布更合理
            scores = scores / (self.d_model ** 0.5)

            # ----- 2.3 Softmax 得到注意力权重 -----
            # dim=-1: 对最后一维 (每个位置的分数) 做 softmax
            # 结果: 每行变成概率分布，和为 1
            attn_weights = torch.softmax(scores, dim=-1)
            # attn_weights 形状: [batch, seq_len, seq_len]

            # ----- 2.4 加权求和 V -----
            # 每个位置 = 所有位置 V 的加权平均，权重是 attn_weights
            # attn_weights × V: 矩阵乘法
            attn_output = torch.matmul(attn_weights, V)
            # attn_output 形状: [batch, seq_len, d_model]

            # ----- 2.5 Attention 输出投影 -----
            # 再投影一次，让模型有更多表达能力
            attn_output = self.attn_proj[i](attn_output)

            # ----- 2.6 残差连接 + LayerNorm -----
            # Post-Norm 风格: 先相加，再归一化
            # h + attn_output: 残差连接，保留原始信息，梯度更容易传播
            # norm1: LayerNorm，稳定数值
            h = self.norm1[i](h + attn_output)

            # ----- 2.7 FFN + 残差连接 + LayerNorm -----
            # FFN: 非线性变换，让模型"思考"提取的信息
            ffn_output = self.ffn[i](h)

            # 残差 + LayerNorm
            h = self.norm2[i](h + ffn_output)

        # ============ Step 3: 输出投影 ============
        # 把 d_model 维向量投影到 vocab_size 维
        # logits[i, j, k] = 第 i 个样本，第 j 个位置，数字 k 的得分
        logits = self.out_proj(h)
        # logits 形状: [batch_size, seq_len, vocab_size]

        return logits