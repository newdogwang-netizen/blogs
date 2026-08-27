---
layout: post
title: "SLP3 阅读笔记（二）：Token 与 N-gram——模型先决定看见什么"
description: "从 Unicode、形态素和 BPE 到 N-gram、平滑与困惑度，解释 tokenizer 为什么同时影响模型能力、上下文容量和算力成本。"
date: 2026-08-27 20:02:00 +0800
categories: [nlp-language-models]
category_name: "NLP 与语言模型"
tags: [Tokenization, BPE, N-gram, Perplexity]
series: "SLP3 2026 阅读笔记"
series_part: 2
reading_time: "24 分钟"
---

> 对应 SLP3 第 2–3 章。本文把“切分”和“概率估计”放在一起，因为语言模型的样本空间由 tokenizer 先定义。

语言模型不直接读取“词义”。它读取整数 ID 序列。原始字符串到 ID 的管线通常是：

```text
bytes → Unicode code points → normalization → pre-tokenization
      → subword segmentation → vocabulary IDs
```

任何一步不同，后面的概率事件就不同。讨论模型容量之前，先要明确模型究竟在预测什么。

## 1. “一个词”不是跨语言稳定的计算单位

英语用空格切词只是局部便利。中文没有天然空格，德语可长距离组合，土耳其语等黏着语会把大量语法信息压进一个表面词，语音系统面对的甚至不是字符而是连续波形。

因此需要区分：

- **word type**：词表中的不同形式；
- **token**：某个形式在语料中的一次出现；
- **morpheme**：携带意义或语法功能的较小单位；
- **subword token**：为统计覆盖率与计算效率构造的模型单位。

子词不必等于形态素。它由语料频率和训练算法决定，可能碰巧对应词根，也可能只是常见字符片段。把 token 解释成“模型理解的词”会产生过度语义化。

## 2. BPE 优化的是压缩式折中

BPE 可以从字符或字节级符号开始，反复合并训练语料中高频相邻对。直觉上，它在两个极端之间找平衡：

| 粒度 | 优点 | 代价 |
|---|---|---|
| 字符/字节 | 几乎没有 OOV，跨语言覆盖好 | 序列长，长期依赖更难，推理步数多 |
| 整词 | 序列短，常见词语义完整 | 词表巨大，稀有词与形态变化严重 |
| 子词 | 开放词表与序列长度的折中 | 切分依赖训练语料，边界不等于语言结构 |

合并规则学到以后，推理时必须按同一规则执行；token ID 还必须与 embedding 矩阵的行严格对应。换 tokenizer 不是简单预处理替换，它等于改变输入词表、输出类别集合和已有参数的语义。

### 工程账：上下文按 token 收费

如果一段文本在 tokenizer A 下是 1,000 tokens，在 B 下是 1,600 tokens，那么差异不只在 API 计费：

- 自回归 decode 多出约 600 个预测步；
- KV cache 按 token 数增长；
- 全注意力 prefill 的交互项随序列长度近似二次增长；
- 固定上下文窗口能容纳的原始信息减少。

所以 tokenizer 对某种语言切得更碎，实质上给该语言施加了额外的时延、显存和上下文税。

## 3. N-gram：把不可估计的问题压缩到局部

链式法则把序列概率拆成：

```text
P(x₁…xₙ) = ∏ P(xᵢ | x₁…xᵢ₋₁)
```

直接统计任意长历史几乎不可能。N-gram 做 Markov 近似，只保留最近的 `N-1` 个 token：

```text
P(xᵢ | x₁…xᵢ₋₁) ≈ P(xᵢ | xᵢ₋ₙ₊₁…xᵢ₋₁)
```

最大似然估计就是“某上下文后出现目标 token 的次数 / 该上下文总次数”。这很透明，也暴露了核心矛盾：上下文越长，表达力越强，但组合越稀疏。

N-gram 的历史意义不只是旧模型。它把所有语言模型都必须面对的问题显式化了：**如何把概率质量分给训练集中没见过、但测试时合理的序列？**

## 4. 平滑不是给零概率加一个小数

简单加一平滑会从高频事件拿走过多质量。更成熟的方法把证据拆开：

- 当前高阶上下文有可靠计数时，信任高阶估计；
- 计数不足时，把一部分质量回退到短上下文；
- 插值则始终混合不同阶数；
- Kneser–Ney 还区分“一个词很常见”和“它能出现在很多不同上下文之后”。

最后一点很关键。某个词总频率高，不代表它在新上下文中也应有高概率。continuation count 衡量的是上下文多样性，而不是裸频次。现代神经语言模型用参数共享和向量相似性泛化，解决手段不同，但目标相同：让相似上下文共享统计强度。

## 5. Perplexity 到底衡量什么

给定测试序列，平均负对数似然为：

```text
NLL = -(1/T) Σ log P(x_t | x_<t)
```

困惑度是其指数形式：

```text
PP = exp(NLL)
```

它可直观理解为模型在每一步面对的“有效分支数”。概率越集中在真实 token 上，PP 越低。

但比较困惑度必须满足至少三个条件：相同测试集、相同 tokenization、相同概率口径。不同 tokenizer 改变了 `T` 和事件空间，直接比较每 token PP 往往没有意义。跨模型时可考虑转换到每 byte/character 的交叉熵，仍要说明规范化方式。

更重要的是，PP 衡量预测分布，不直接衡量事实性、帮助性、安全性或长任务成功率。它是训练目标附近的内在指标，不是产品质量总分。

## 6. 从 N-gram 到 Transformer，什么变了

| 问题 | N-gram | Transformer LM |
|---|---|---|
| 上下文 | 固定窗口、离散匹配 | 有限上下文窗口、连续表示 |
| 泛化 | 回退/插值/平滑 | 参数共享与表示相似性 |
| 可解释性 | 计数可查 | 内部表示分布式、难追因 |
| 计算 | 查表为主 | 大规模矩阵乘与内存访问 |
| 根本目标 | 下一个 token 概率 | 下一个 token 概率 |

最后一行没有变化。Transformer 极大增强了条件分布估计器，但它仍从语料共现中学习。这个事实解释了为什么“生成得像”与“对世界为真”之间始终存在缺口。

## 7. 实战检查表

遇到多语言质量或成本问题，先记录：Unicode 规范化形式、tokenizer 版本、词表大小、每种语言的 chars/token 或 bytes/token、截断率、特殊 token 规则。遇到困惑度报告，则补问测试集是否泄漏、是否包含 BOS/EOS、loss 是否忽略 padding、平均是在 token 还是 sequence 上做。

下一篇将从逻辑回归开始，推导 logits、softmax、交叉熵、embedding 和 MLP 为什么可以自然拼成神经语言模型。

## 资料入口

- [SLP3 第 2 章：Words and Tokens](https://web.stanford.edu/~jurafsky/slp3/2.pdf)
- [SLP3 第 3 章：N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
- [SLP3 附录 C：Kneser–Ney Smoothing](https://web.stanford.edu/~jurafsky/slp3/C.pdf)

