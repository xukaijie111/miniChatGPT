"""
Decoder-only Transformer 架构图 (GPT 风格)

┌─────────────────────────────────────────────────────────────────────┐
│                        整体架构                                       │
└─────────────────────────────────────────────────────────────────────┘

数据格式: [输入序列] + [输出序列]
例如: [3, 1, 4, 1, 5, 1, 1, 3, 4, 5]
      └──────────────┘  └──────────────┘
         输入(20)          输出(20)

vocab_size = 10 (数字 0-9，无分隔符)
max_len = 40 (输入长度 20 + 输出长度 20)

输入序列 [batch, 40]
         │
         ▼
┌─────────────────┐
│ Token Embedding │  [batch, 40] → [batch, 40, d_model]
└─────────────────┘
         │
         + ←─────────────────────┐
         │                       │
         ▼                       │
┌─────────────────┐              │
│ Pos Embedding   │  [40] → [40, d_model]
└─────────────────┘              │
         │                       │
         + ←─────────────────────┘ 相加
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Transformer Layer × n_layers                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │    h ─────────────────────────────────────────┐              │  │
│  │    │                                          │              │  │
│  │    ▼                                          │              │  │
│  │  ┌─────────────┐                              │              │  │
│  │  │ Q_proj      │                              │              │  │
│  │  │ K_proj      │                              │              │  │
│  │  │ V_proj      │                              │              │  │
│  │  └─────────────┘                              │              │  │
│  │    │                                          │              │  │
│  │    │         ┌─────────────────────┐          │              │  │
│  │    │         │ Causal Mask         │          │              │  │
│  │    │         │ (上三角 = -∞)        │          │              │  │
│  │    │         └─────────────────────┘          │              │  │
│  │    │                 │                        │              │  │
│  │    ▼                 ▼                        │              │  │
│  │  ┌─────────────────────────────────────┐     │              │  │
│  │  │ Masked Self-Attention (单向)         │     │              │  │
│  │  │                                     │     │              │  │
│  │  │   Q × K^T → scores                  │     │              │  │
│  │  │   scores + causal_mask ← 加上掩码   │     │              │  │
│  │  │   scores / sqrt(d)                  │     │              │  │
│  │  │   softmax(scores) → attn_weights    │     │              │  │
│  │  │   attn_weights × V → attn_output    │     │              │  │
│  │  └─────────────────────────────────────┘     │              │  │
│  │    │                                          │              │  │
│  │    ▼                                          │              │  │
│  │  ┌─────────────┐                              │              │  │
│  │  │ attn_proj   │                              │              │  │
│  │  └─────────────┘                              │              │  │
│  │    │                                          │              │  │
│  │    + ←────────────────────────────────────────┘ 残差连接     │  │
│  │    │                                                        │  │
│  │    ▼                                                        │  │
│  │  ┌─────────────┐                                            │  │
│  │  │ LayerNorm 1 │                                            │  │
│  │  └─────────────┘                                            │  │
│  │    │                                                        │  │
│  │    ├────────────────────────────────────────┐               │  │
│  │    │                                        │               │  │
│  │    ▼                                        │               │  │
│  │  ┌─────────────────────┐                    │               │  │
│  │  │ FFN                  │                    │               │  │
│  │  │ Linear(d → 4d)       │                    │               │  │
│  │  │ GELU                 │                    │               │  │
│  │  │ Linear(4d → d)       │                    │               │  │
│  │  └─────────────────────┘                    │               │  │
│  │    │                                        │               │  │
│  │    + ←──────────────────────────────────────┘ 残差连接      │  │
│  │    │                                                        │  │
│  │    ▼                                                        │  │
│  │  ┌─────────────┐                                            │  │
│  │  │ LayerNorm 2 │                                            │  │
│  │  └─────────────┘                                            │  │
│  │    │                                                        │  │
│  │    ▼ → 下一层或输出                                          │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ final_norm      │  ← GPT 风格：所有层结束后再 norm 一次
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ out_proj        │  [batch, 40, d_model] → [batch, 40, vocab_size]
└─────────────────┘
         │
         ▼
输出 logits [batch, 40, 10]
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    logits 含义                                       │
└─────────────────────────────────────────────────────────────────────┘

每个位置输出 10 个值，对应数字 0-9 的概率得分：

logits[位置i] = [值0, 值1, 值2, 值3, 值4, 值5, 值6, 值7, 值8, 值9]

softmax → 概率分布
argmax → 预测数字

例如：logits[位置5] = [0.1, 0.8, 0.2, ...]
      softmax → 数字1的概率最大
      argmax → 预测数字 1


┌─────────────────────────────────────────────────────────────────────┐
│                    Causal Mask 示意图                                 │
└─────────────────────────────────────────────────────────────────────┘

单向注意力（Decoder）- Causal Mask:

        位置0  位置1  位置2  位置3  位置4
        ┌───────────────────────────────┐
位置0   │  ✓     ✗     ✗     ✗     ✗   │  ← 位置0 只能看到位置0
位置1   │  ✓     ✓     ✗     ✗     ✗   │  ← 位置1 只能看到位置0,1
位置2   │  ✓     ✓     ✓     ✗     ✗   │  ← 位置2 只能看到位置0,1,2
位置3   │  ✓     ✓     ✓     ✓     ✗   │  ← 位置3 只能看到位置0~3
位置4   │  ✓     ✓     ✓     ✓     ✓   │  ← 位置4 可以看到全部(过去)
        └───────────────────────────────┘

✓ = 可以看到    ✗ = 不能看到（被 mask 遮住）

Mask矩阵数值：
        位置0  位置1  位置2  位置3  位置4
        ┌───────────────────────────────┐
位置0   │  0    -∞    -∞    -∞    -∞   │
位置1   │  0     0    -∞    -∞    -∞   │
位置2   │  0     0     0    -∞    -∞   │
位置3   │  0     0     0     0    -∞   │
位置4   │  0     0     0     0     0   │
        └───────────────────────────────┘

-∞ 在 softmax 后变成 0，相当于"遮住"了未来信息


┌─────────────────────────────────────────────────────────────────────┐
│                    自回归生成过程                                     │
└─────────────────────────────────────────────────────────────────────┘

推理时（生成排序结果）：

输入: [3, 1, 4, 1, 5]  (长度 20，示例简化为 5)

Step 1: 输入 [3, 1, 4, 1, 5]
        模型预测下一个 token → 1
        拼接后: [3, 1, 4, 1, 5, 1]

Step 2: 输入 [3, 1, 4, 1, 5, 1]
        模型预测下一个 token → 1
        拼接后: [3, 1, 4, 1, 5, 1, 1]

Step 3: 输入 [3, 1, 4, 1, 5, 1, 1]
        模型预测下一个 token → 3
        拼接后: [3, 1, 4, 1, 5, 1, 1, 3]

... 重复直到生成 20 个 token

最终输出: [3, 1, 4, 1, 5, 1, 1, 3, 4, 5, ...]
          └──────────────┘  └──────────────┘
             输入(20)          生成的排序结果(20)


┌─────────────────────────────────────────────────────────────────────┐
│                    训练 Loss 计算                                    │
└─────────────────────────────────────────────────────────────────────┘

只计算输出部分（位置 20~39）的 Loss，输入部分不参与：

x = [输入(0~19), 输出(20~39)]
y = [下一个token，输出部分的下一个token，-1]
mask = [0,0,...,0, 1,1,...,1]  ← 只有后 20 个位置为 1

loss = CrossEntropyLoss(logits, y, ignore_index=-1)
       只计算 mask=1 的位置
"""
import torch
import torch.nn as nn


class SortDecoderTransformer(nn.Module):
    """
    Decoder-only Transformer 用于排序任务（类似 GPT）

    输入格式: [输入序列] + [输出序列]
    例如: [3, 1, 4, 1, 5, 1, 1, 3, 4, 5]（输入5 + 输出5，实际是20+20=40）

    特点:
    - Causal Mask: 每个位置只能看到当前位置之前的信息（上三角遮住）
    - 自回归生成: 推理时逐个预测下一个 token，拼接后继续预测

    输出 logits:
    - 形状: [batch, seq_len, vocab_size] = [batch, 40, 10]
    - 每个位置的 logits 是 10 维向量，对应数字 0-9 的预测得分

    训练目标:
    - 每个位置预测下一个 token
    - 位置19（输入最后）预测位置20（输出第一），这是关键过渡位置
    """

    def __init__(self, vocab_size=10, d_model=32, n_layers=3, n_heads=2, max_len=40):
        """
        参数说明:
        vocab_size: 词表大小 = 数字范围(0-9) = 10
        d_model:    隐藏维度，所有向量都用这个维度
        n_layers:   Transformer 层数，堆叠多层让模型更强大
        n_heads:    多头注意力的头数（多头可以关注不同模式）
        max_len:    序列最大长度 = 输入长度 + 输出长度 = 20 + 20 = 40
        """
        super().__init__()

        # 保存超参数
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len

        # 检查：d_model 必须能被 n_heads 整除，否则多头无法均匀分配
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads  # 每个头的维度，如 32 // 2 = 16

        # ============ Embedding 层 ============
        # Token Embedding: 把每个数字映射成向量
        # 输入数字索引(如 3) → 输出 d_model 维向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Position Embedding: 给每个位置一个独特的向量
        # 让模型知道"顺序"，因为 attention 本身不知道位置
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # ============ Attention 投影层（每层各有一套）============
        # Q (Query): "我想找什么信息"
        # K (Key):   "我有什么信息可以匹配"
        # V (Value): "我的实际内容"
        self.q_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.k_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.v_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])

        # Attention 输出投影: 把注意力结果再投影一次，增加表达能力
        self.attn_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])

        # LayerNorm（每层两个：Attention 后一个，FFN 后一个）
        # 作用: 稳定数值，防止梯度爆炸/消失
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # ============ FFN（Feed Forward Network）============
        # 结构: Linear(d → 4d) → GELU → Linear(4d → d)
        # 作用: 给模型非线性表达能力，"思考"提取的信息
        # 扩展 4 倍是为了有更大的"工作空间"处理复杂信息
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),  # 扩展维度: 32 → 128
                nn.GELU(),                        # 激活函数: 比 ReLU 更平滑
                nn.Linear(d_model * 4, d_model),  # 回归维度: 128 → 32
            )
            for _ in range(n_layers)
        ])

        # ============ 输出层 ============
        # Final LayerNorm: GPT 风格，所有层结束后再 norm 一次
        self.final_norm = nn.LayerNorm(d_model)

        # 输出投影: d_model → vocab_size
        # logits 的每个维度对应一个数字(0-9)的预测得分
        self.out_proj = nn.Linear(d_model, vocab_size)

    def _create_causal_mask(self, seq_len, device):
        """
        创建因果掩码（Causal Mask）：上三角矩阵

        作用：让每个位置只能看到"当前位置之前"的信息
              防止 decoder "偷看"未来，实现自回归

        例如 seq_len=4:
            位置0  位置1  位置2  位置3
            [ 0    -∞    -∞    -∞  ]  ← 位置0只能看到自己
            [ 0     0    -∞    -∞  ]  ← 位置1能看到位置0,1
            [ 0     0     0    -∞  ]  ← 位置2能看到位置0,1,2
            [ 0     0     0     0  ]  ← 位置3能看到全部

        -∞ 在 softmax 后变成 0，相当于"遮住"未来
        """
        # torch.triu: 取上三角，diagonal=1 表示不包含对角线
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        # 把 1 变成 -∞，softmax 后就是 0
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x):
        """
        前向传播：输入序列 → 输出 logits

        参数:
            x: 输入序列 [batch_size, seq_len]
               例如 [batch, 40]，包含输入(20)和输出(20)

        返回:
            logits: [batch_size, seq_len, vocab_size]
                   每个位置的 logits 是 10 维向量，对应数字 0-9 的预测得分
                   例如 [batch, 40, 10]

        流程:
            1. Embedding (token + position)
            2. 创建 Causal Mask
            3. 多层 Transformer (attention + FFN)
            4. Final LayerNorm
            5. 输出投影 → logits
        """
        batch_size, seq_len = x.shape

        # ============ Step 1: Embedding ============
        # 生成位置索引: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device)

        # Token Embedding + Position Embedding
        # token_embedding(x): 把数字索引转成向量 [batch, seq_len, d_model]
        # pos_embedding(positions): 每个位置的向量 [seq_len, d_model]
        # 相加后得到每个位置的完整表示
        h = self.token_embedding(x) + self.pos_embedding(positions)
        # h 形状: [batch_size, seq_len, d_model] = [batch, 40, 32]

        # ============ Step 2: 创建 Causal Mask ============
        # 上三角为 -∞ 的矩阵，遮住未来信息
        causal_mask = self._create_causal_mask(seq_len, x.device)
        # mask 形状: [seq_len, seq_len] = [40, 40]

        # ============ Step 3: 多层 Transformer 处理 ============
        for i in range(self.n_layers):
            # ----- 3.1 计算 Q, K, V -----
            # 每个都是 [batch, seq_len, d_model]
            Q = self.q_proj[i](h)  # Query: 查询向量
            K = self.k_proj[i](h)  # Key:   键向量
            V = self.v_proj[i](h)  # Value: 值向量

            # ----- 3.2 计算注意力分数 -----
            # Q × K^T: 矩阵乘法，计算每个位置对其他位置的"相关性"
            # 结果: [batch, seq_len, seq_len] = [batch, 40, 40]
            scores = torch.matmul(Q, K.transpose(-2, -1))

            # 缩放: 除以 sqrt(d_model)，稳定数值
            scores = scores / (self.d_model ** 0.5)

            # ----- 3.3 加上 Causal Mask -----
            # scores + mask：上三角变成 -∞
            scores = scores + causal_mask

            # ----- 3.4 Softmax 得到注意力权重 -----
            # -∞ 变成 0，相当于"遮住"未来
            # 每行变成概率分布，和为 1
            attn_weights = torch.softmax(scores, dim=-1)
            # attn_weights 形状: [batch, seq_len, seq_len]

            # ----- 3.5 加权求和 V -----
            # 每个位置 = 所有位置 V 的加权平均（未来位置权重为 0）
            attn_output = torch.matmul(attn_weights, V)
            # attn_output 形状: [batch, seq_len, d_model]

            # ----- 3.6 Attention 输出投影 -----
            # 再投影一次，增加表达能力
            attn_output = self.attn_proj[i](attn_output)

            # ----- 3.7 残差连接 + LayerNorm -----
            # h + attn_output: 残差连接，保留原始信息，梯度更容易传播
            # norm1: LayerNorm，稳定数值
            h = self.norm1[i](h + attn_output)

            # ----- 3.8 FFN + 残差连接 + LayerNorm -----
            # FFN: 非线性变换，让模型"思考"提取的信息
            ffn_output = self.ffn[i](h)
            # 残差 + LayerNorm
            h = self.norm2[i](h + ffn_output)

        # ============ Step 4: Final LayerNorm ============
        # GPT 风格：所有层结束后再 norm 一次
        h = self.final_norm(h)

        # ============ Step 5: 输出投影 ============
        # 把 d_model 维向量投影到 vocab_size 维
        # logits 的每个维度对应一个数字(0-9)的预测得分
        logits = self.out_proj(h)
        # logits 形状: [batch_size, seq_len, vocab_size] = [batch, 40, 10]

        return logits