---
layout: post
title: "SLP3 阅读笔记（三）：从逻辑回归到 Embedding——学习出来的表示是什么"
description: "沿 logits、softmax、交叉熵、向量语义和 MLP 的计算链，解释分类器如何演化成神经语言模型的基础部件。"
date: 2026-08-27 20:03:00 +0800
categories: [nlp-language-models]
category_name: "NLP 与语言模型"
tags: [Logistic Regression, Embedding, Neural Network, Cross Entropy]
series: "SLP3 2026 阅读笔记"
series_part: 3
reading_time: "25 分钟"
---

> 对应 SLP3 第 4–6 章。重点不是复现一个情感分类器，而是看清现代语言模型仍反复使用的三个构件：线性打分、归一化概率、可学习表示。

## 1. 逻辑回归建立了整条训练链

对输入特征向量 `x`，二分类器先计算：

```text
z = w·x + b
p(y=1|x) = sigmoid(z)
```

`z` 是 logit：它是未归一化分数，不是概率。sigmoid 把任意实数映射到 `(0,1)`。多分类时，为每个类别计算一个 logit，再用 softmax：

```text
p_k = exp(z_k) / Σ_j exp(z_j)
```

训练通常最小化交叉熵。若真实类别是 `y`，单样本损失就是 `-log p_y`。它对“自信但错误”的预测惩罚很大，并且梯度有简洁形式：softmax 输出减 one-hot 标签。

这条链在大模型中几乎原样保留：最后一个 hidden state 经过词表投影得到数万 logits，softmax 得到下一个 token 的分类分布。LLM 不是脱离分类器的神秘系统，而是在每个位置反复解决一个超大规模多分类问题。

## 2. 特征工程与表示学习的分界

传统文本分类把输入变成设计好的特征，例如词频、二元词组、否定标记。模型只学习如何加权这些特征：

```text
human chooses φ(text)
model learns w
```

神经模型进一步学习特征本身：

```text
model learns h = f(text; θ)
model also learns classifier on h
```

这就是 representation learning。它的强大之处是任务相关表示可以由数据优化；风险也随之而来：数据偏差、捷径特征和领域相关性都可能被压进 `h`，但人很难直接读出。

## 3. Embedding 是查表，也是坐标系

设词表大小为 `|V|`，embedding 维度为 `d`，矩阵：

```text
E ∈ R^(|V| × d)
```

token ID `i` 的 embedding 就是 `E[i]`。计算上是查一行；学习上，每一行都是通过反向传播更新的参数。相似 token 在训练目标中承担相似作用时，其向量往往靠近。

“一个向量代表意义”需要谨慎理解。分布假说认为，出现在相似上下文中的词倾向于有相似含义。由此学习到的是**使用模式的几何结构**，不是字典定义本身。静态 embedding 还把一个词的所有语境压成一个向量，多义词会被混合；上下文化 embedding 才让同一 token 在不同句子中得到不同表示。

余弦相似度：

```text
cos(u,v) = (u·v) / (||u|| ||v||)
```

消除了长度影响，常用于比较方向。但“余弦高”只说明在当前表示空间接近，不自动等于同义、可替换或事实相关。空间的各向异性、词频和训练目标都会扭曲它。

## 4. MLP 为什么需要非线性

一个前馈层可以写成：

```text
h = g(Wx + b)
```

若 `g` 仍是线性函数，多层线性变换可以合并成一层，深度没有增加表达能力。ReLU、GELU 等非线性让网络能把输入空间折叠成复杂决策区域。

两层 MLP 常见形式：

```text
h = g(W₁x + b₁)
z = W₂h + b₂
```

第一层扩展并重组特征，第二层投影到输出空间。Transformer block 中的 FFN/MLP 也是逐 token 应用这种变换；attention 负责 token 间信息混合，MLP 负责对每个位置的表示做非线性重写。二者分工不同，缺一不可。

## 5. 反向传播到底在传什么

前向计算产生 loss；链式法则把 `∂loss/∂parameter` 从输出层逐层传回。优化器据此更新参数：

```text
θ ← θ - η ∇θ L
```

需要区分三件事：

- **梯度**是当前 batch 对参数的局部敏感度；
- **优化器状态**（如 Adam 的矩估计）不是模型知识本身；
- **学习率**决定沿估计方向走多远，过大可能发散，过小则收敛缓慢。

训练 loss 下降，只说明模型越来越能拟合训练目标。泛化要靠独立验证/测试集判断；若不断根据测试集改模型，测试集也会被间接训练。

## 6. 一个统一视角：输入矩阵与输出矩阵

语言模型开头用 embedding 矩阵把离散 ID 变成连续向量，结尾用 unembedding 把 hidden state 变回词表 logits：

```text
token id → E[id] → neural layers → h_t → h_t U → logits over V
```

一些模型会将输入 embedding 与输出投影权重绑定。这减少参数，也表达一种对偶关系：同一个词表坐标系既用于“读入 token”，也用于“判断下一个 token”。但中间层的表示并不因此天然可读；它们只需要为最终目标提供有用计算。

## 7. 给资深工程师的诊断框架

当分类或生成模型异常时，可以沿四层检查：

1. **样本层**：标签是否一致，类别先验是否漂移；
2. **表示层**：tokenization、截断和 embedding 是否丢失关键区别；
3. **目标层**：loss 是否与线上代价一致，class weighting 是否合理；
4. **决策层**：概率是否校准，阈值是否根据业务成本选择。

Accuracy 可能掩盖类别不平衡；概率排序正确也可能严重不校准；embedding 检索召回高也不等于生成器会采用证据。每一层必须有自己的观测指标。

下一篇把这些部件装进 decoder-only Transformer，并从一次前向传播推导训练、prefill、decode 与 KV cache。

## 资料入口

- [SLP3 第 4 章：Logistic Regression and Text Classification](https://web.stanford.edu/~jurafsky/slp3/4.pdf)
- [SLP3 第 5 章：Embeddings](https://web.stanford.edu/~jurafsky/slp3/5.pdf)
- [SLP3 第 6 章：Neural Networks](https://web.stanford.edu/~jurafsky/slp3/6.pdf)

