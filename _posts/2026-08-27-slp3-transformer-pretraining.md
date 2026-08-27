---
layout: post
title: "SLP3 阅读笔记（四）：Transformer 与预训练——一次前向传播里发生了什么"
description: "从 Q/K/V、因果掩码、残差流、MLP 到词表投影，推导 decoder-only Transformer 的训练、prefill、decode 与 KV cache。"
date: 2026-08-27 20:04:00 +0800
categories: [nlp-language-models]
category_name: "NLP 与语言模型"
tags: [Transformer, Attention, Pretraining, KV Cache]
series: "SLP3 2026 阅读笔记"
series_part: 4
reading_time: "30 分钟"
---

> 对应 SLP3 第 7 章。本文只讨论 decoder-only 自回归主线；硬件容量与集群并行见本站「AI 与 GPU」分类，概念上关联，分类上保持独立。

<figure class="architecture-figure">
  <a href="{{ '/assets/images/transformer-forward-pass.svg' | relative_url }}"><img src="{{ '/assets/images/transformer-forward-pass.svg' | relative_url }}" alt="自回归 Transformer 单步前向过程"></a>
  <figcaption>同一网络，训练时并行预测所有位置；生成时一次追加一个 token。</figcaption>
</figure>

## 1. Attention 解决的是“从哪些位置取信息”

输入表示矩阵 `X ∈ R^(T×d)` 经过三个线性投影：

```text
Q = XW_Q,  K = XW_K,  V = XW_V
```

对每个 query，与所有 key 做相似度，再缩放、掩码和 softmax：

```text
A = softmax((QKᵀ / √d_k) + M)
H = AV
```

这里容易混淆两种角色：`QKᵀ` 决定**从哪里读、读多少**，`V` 承载**实际被搬运的内容**。attention 权重不是完整解释；某位置权重大，只表示 value 路径影响可能较强，还要考虑 value 内容、后续投影、残差与非线性。

因果掩码 `M` 把未来位置的分数置为负无穷，使位置 `t` 只能读取 `≤t` 的 token。没有它，训练时模型会偷看标签。

## 2. Multi-head 不是简单复制多次

每个 head 有自己的投影子空间，可以学习不同的匹配与搬运模式。各 head 输出拼接后再投影回模型维度：

```text
MHA(X) = Concat(head₁…head_h) W_O
```

“某个 head 专门负责语法”通常是事后观察，不是架构保证。功能可能分布在多个 head 与 MLP 中，也可能存在冗余。做机制分析时必须区分相关性、可解码性与因果必要性。

## 3. 残差流是更好的整体视角

Transformer block 不是把表示彻底替换，而是把子层输出加回主干：

```text
x ← x + Attention(Norm(x))
x ← x + MLP(Norm(x))
```

可以把残差流理解为共享工作区：attention 从其他位置读信息并写回，MLP 在当前位置变换信息，许多 block 逐步累积更新。LayerNorm/RMSNorm 控制尺度，让深层优化更稳定。

这也说明为什么只看单个 attention map 不够：最终 logit 是许多层、许多路径在残差流中叠加的结果。

## 4. 训练为何并行，生成为何串行

训练语料给出了完整序列。借助 causal mask，模型可以一次算出每个位置对下一个 token 的预测，loss 在多个位置求和或平均。位置之间在图上有因果限制，但矩阵计算仍可并行。

生成时，`x_(t+1)` 还不存在，必须先从 `P(x_(t+1)|x_≤t)` 选出它，再作为输入计算下一步。这形成严格的 token 级串行依赖。批处理能并行多个请求，却不能消除单条序列内部的时间链。

## 5. Prefill 与 Decode 是两个不同负载

**Prefill**处理已有 prompt，可用大矩阵并行计算，通常更偏算力吞吐。**Decode**每步只新增一个 query，却要读取所有历史 K/V，矩阵形状小、重复读缓存，常更偏 HBM 带宽和调度开销。

KV cache 保存每层历史 token 的 key 与 value。粗略容量与以下量成正比：

```text
batch × sequence_length × layers × kv_heads × head_dim × 2(K,V) × bytes
```

它不缓存新 token 的最终答案，也不免除新 query 与历史 key 的注意力计算。MQA/GQA 通过让多个 query heads 共享较少的 K/V heads，直接缩小缓存与读带宽。

## 6. 位置为何必须显式进入模型

纯 attention 对输入排列本身没有顺序感。模型必须加入绝对位置 embedding、旋转位置编码（RoPE）或其他相对位置机制。位置方案会影响长度外推，但“支持更长位置”不代表模型能稳定使用所有远距离信息；训练长度分布、注意力稀释与检索行为同样重要。

## 7. 预训练学到什么，没承诺什么

自回归预训练最小化语料上的 next-token cross-entropy。它促使模型压缩语言规律、世界共现与任务模式，因为这些信息有助于预测后续 token。但目标没有直接要求：

- 回答必须引用可追溯事实；
- 不知道时必须拒答；
- 遵循某个用户意图而不是续写网页风格；
- 多步工具调用最终必须成功。

这些能力可能涌现为预测副产品，却不是 loss 的硬约束。后训练和系统设计之所以必要，原因就在目标错位，而不是预训练“还不够大”。

## 8. Decoding 会改变可见行为

greedy 每次取最大概率，稳定但可能陷入重复；temperature 调整 logits 的尖锐程度；top-k/top-p 截断候选集合后采样。它们不会给模型增加知识，只改变已有分布被读取的方式。

因此评测生成模型时必须固定 decoding 参数和随机种子范围。把一次随机输出差异归因于模型版本，是常见的实验错误。

## 9. 从模型公式到系统指标

建议至少分开观察：

- prefill latency / TTFT（首 token 时间）；
- decode inter-token latency 与 tokens/s；
- prompt、output token 数分布；
- KV cache 占用与命中/换页；
- batch 调度、尾延迟和取消请求浪费；
- task quality，而不是仅报告吞吐。

下一篇讨论预训练之后的行为塑形：SFT、参数高效微调、偏好优化、可验证奖励，以及我们到底能从内部表示推断多少。

## 资料入口

- [SLP3 第 7 章：Large Language Models with Transformers](https://web.stanford.edu/~jurafsky/slp3/7.pdf)
- [本站 GPU 笔记：显存、精度与 KV Cache]({{ '/ai-gpu/memory-precision-kv-cache/' | relative_url }})
