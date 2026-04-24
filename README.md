# miniTransformer - Transformer 教学项目

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个用于学习 Transformer 原理的教学项目，用排序任务演示 Encoder-only (BERT) 和 Decoder-only (GPT) 两种架构。

**关键词**: Transformer, GPT, BERT, Self-Attention, Causal Mask, 自回归生成, PyTorch 教学项目, 深度学习入门

## 项目概述

本项目用 Transformer 模型学习"排序"任务：输入一个随机数字序列，输出排序后的序列。

```
输入: [3, 1, 4, 1, 5, 0, 9, 2, 6, 7, ...]  (20个随机数字)
输出: [0, 1, 1, 2, 3, 4, 5, 6, 7, 9, ...]  (排序后的结果)
```

通过这个简单的任务，帮助理解 Transformer 的核心组件：
- Embedding（Token + Position）
- Self-Attention（注意力机制）
- Causal Mask（因果掩码，Decoder 专用）
- FFN（前馈网络）
- 残差连接 + LayerNorm

## 文件结构

```
src/
├── encoder_only/          # Encoder-only Transformer (BERT 风格)
│   ├── model.py           # 模型定义（含详细注释和架构图）
│   ├── dataset.py         # 数据集
│   ├── train.py           # 训练脚本
│   └── test.py            # 测试脚本
│   └── sort_model.pth     # 训练好的模型权重
│
├── decoder_only/          # Decoder-only Transformer (GPT 风格)
│   ├── model.py           # 模型定义（含详细注释和架构图）
│   ├── dataset.py         # 数据集（含 Loss Mask）
│   ├── train.py           # 训练脚本
│   ├── test.py            # 测试脚本（含自回归生成）
│   └── sort_decoder_model.pth  # 训练好的模型权重
│
└── __init__.py
```

## 两种架构对比

### Encoder-only (BERT 风格)

```
输入: [3, 1, 4, 1, 5]  (20个)
      ↓
模型处理（双向注意力，每个位置能看到所有位置）
      ↓
输出: [1, 1, 3, 4, 5]  (一次性全部输出)

特点:
- 双向注意力（无 Mask）
- 一次输入 → 一次输出
- 适合理解任务（分类、标注）
```

### Decoder-only (GPT 风格)

```
训练时:
输入: [3, 1, 4, 1, 5, 1, 1, 3, 4, 5]  (输入20 + 输出20 = 40个)
      ↓
模型处理（单向注意力，只能看过去）
      ↓
每个位置预测下一个 token

推理时（自回归）:
输入: [3, 1, 4, 1, 5]  (只有输入20个)
      ↓
预测下一个 → 1，拼接
      ↓
输入: [3, 1, 4, 1, 5, 1]
      ↓
预测下一个 → 1，拼接
      ↓
...重复20次...
      ↓
最终: [3, 1, 4, 1, 5, 1, 1, 3, 4, 5, ...]
输出部分 = [1, 1, 3, 4, 5]  (排序结果)

特点:
- 单向注意力 + Causal Mask（上三角遮住）
- 自回归生成（一步一步预测）
- 适合生成任务
```

### 核心区别

| 特性 | Encoder-only | Decoder-only |
|------|--------------|--------------|
| 注意力方向 | 双向（能看所有位置） | 单向（只能看过去） |
| Causal Mask | 无 | 有（上三角 = -∞） |
| 输入长度 | seq_len = 20 | seq_len * 2 = 40 |
| 输出方式 | 一次全部输出 | 自回归逐个生成 |
| logits 形状 | [batch, 20, 10] | [batch, 40, 10] |
| 每个位置预测 | 该位置的排序结果 | 下一个 token |

## 运行方法

### Encoder-only

```bash
cd src/encoder_only

# 训练
python train.py

# 测试
python test.py
```

### Decoder-only

```bash
cd src/decoder_only

# 训练
python train.py

# 测试（默认批量测试10个样本）
python test.py
```

### 测试结果

两种架构都能达到 **100% 正确率**：
```
===== 批量测试 (10个样本) =====
样本1 | 输入: [3, 7, 0, 7, ...]
       | 预测: [0, 0, 1, 1, ...]
       | 正确: [0, 0, 1, 1, ...]
       | 结果: ✓
...
正确率: 10/10
```

## 关键概念详解

### logits 含义

每个位置的 logits 是一个 **10 维向量**，对应数字 0-9 的预测得分：

```
logits[位置i] = [值0, 值1, 值2, 值3, 值4, 值5, 值6, 值7, 值8, 值9]

softmax → 概率分布
argmax → 预测数字

例如: logits[19] = [0.1, 0.8, 0.2, ...]
     softmax → 数字1概率最大
     argmax → 预测数字 1
```

### Causal Mask (因果掩码)

Decoder-only 的核心，防止"偷看"未来：

```
位置:    0     1     2     3     4
       [ 0    -∞    -∞    -∞    -∞ ]  ← 位置0只能看自己
       [ 0     0    -∞    -∞    -∞ ]  ← 位置1能看位置0,1
       [ 0     0     0    -∞    -∞ ]  ← 位置2能看位置0,1,2
       [ 0     0     0     0    -∞ ]  ← 位置3能看位置0~3
       [ 0     0     0     0     0 ]  ← 位置4能看全部

-∞ 在 softmax 后变成 0，相当于"遮住"未来
```

### Loss Mask (损失掩码)

Decoder-only 训练时使用，只计算输出部分的 loss：

```
位置:   0   1  ...  18  19  20  21  ...  39
      [输入序列]     [输出序列]

mask: [0,  0, ...,  0,  1,  1,  1, ...,  1]
                        ↑
                   位置19 mask=1

位置19预测什么？
  y[19] = x[20] = 输出第一个数字

所以 mask[19]=1 必须计算，学习"输入→输出"过渡
```

### 自回归生成

Decoder 推理时的生成过程：

```
Step 1: 输入 [输入(20)]
        logits[19] 预测下一个 → 拼接
        序列变成 [输入(20), 预测1]

Step 2: 输入 [输入(20), 预测1]
        logits[20] 预测下一个 → 拼接
        序列变成 [输入(20), 预测1, 预测2]

...重复20次...

最终: [输入(20), 排序结果(20)]
```

## 模型参数

```python
vocab_size = 10     # 词表大小（数字0-9）
d_model = 32        # 隐藏维度
n_layers = 3        # Transformer层数
n_heads = 2         # 注意力头数
seq_len = 20        # 输入序列长度
max_len = 40        # Decoder最大长度（输入+输出）

# Encoder 参数量: ~40,000
# Decoder 参数量: ~40,000
```

## 依赖

```bash
pip install torch
```

## 学习建议

1. **先看 Encoder-only**：理解 Transformer 基础组件
2. **再看 Decoder-only**：理解 Causal Mask 和自回归
3. **对比两种架构**：理解为什么 Decoder 用单向注意力
4. **运行代码**：修改参数观察效果
5. **阅读注释**：每个文件都有详细解释

## 参考资料

- [minGPT](https://github.com/karpathy/minGPT) - Karpathy 的极简 GPT 实现，本项目 Decoder 部分参考了该项目的结构
- [Transformer 原论文](https://arxiv.org/abs/1706.03762) - Attention Is All You Need
- [GPT 原论文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Language Models are Unsupervised Multitask Learners
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Transformer 可视化讲解

## License

MIT

---

## 🔍 关于本项目

本项目适合：
- 深度学习初学者理解 Transformer 原理
- 想要了解 Encoder-only 和 Decoder-only 区别的人
- PyTorch 实践者
- NLP 入门学习

**为什么用排序任务？**
排序任务足够简单，能快速验证模型是否正确学习；同时又足够复杂，需要模型理解序列的全局信息。是学习 Transformer 的理想入门任务。

**项目特点**：
- 📚 详细代码注释，每行都有解释
- 📊 架构图可视化，直观理解模型结构
- 🎯 100% 测试正确率，验证模型有效性
- 🔬 对比两种架构，深入理解设计差异