---
layout: post
title: "SLP3 阅读笔记（六）：从 BM25 到 RAG——外部证据如何进入生成链"
description: "拆开索引、召回、重排、上下文组装和生成五个阶段，解释稀疏与稠密检索的互补、RAG 的可观测性和端到端评测。"
date: 2026-08-27 20:06:00 +0800
categories: [nlp-language-models]
category_name: "NLP 与语言模型"
tags: [Information Retrieval, BM25, Dense Retrieval, RAG]
series: "SLP3 2026 阅读笔记"
series_part: 6
reading_time: "28 分钟"
---

> 对应 SLP3 第 11 章。RAG 不是“向量库 + prompt”的产品名，而是一条可分解、可独立评测的信息检索与生成管线。

参数内知识有三个结构性限制：可能记错，无法覆盖私有数据，会随时间陈旧。RAG 的基本动作是先从受控语料找证据，再让模型在证据条件下生成。它改善可更新性和可追溯性，却不会自动消除幻觉。

## 1. 五阶段管线

```text
documents → parse/chunk/index
query → retrieve candidates → rerank → assemble context → generate/cite
```

每一箭头都是独立故障点：解析丢表格，chunk 切断定义，召回漏掉目标，重排被伪相关文本干扰，上下文超预算，生成器忽略证据。只看最终回答无法定位根因。

## 2. 稀疏检索：词面匹配仍是强基线

TF‑IDF 用 term frequency 表示词在文档内的重要性，用 inverse document frequency 降低全局常见词权重。BM25 进一步让词频收益饱和，并校正文档长度。

它的优点是索引成熟、速度快、分数可解释，对产品编号、错误码、人名和罕见术语尤其强。缺点是同义改写没有词面重合时召回弱。

不要因为系统包含 LLM 就跳过 BM25。很多企业查询包含精确实体，稀疏检索常是最难替代的召回通道。

## 3. 稠密检索：把查询和文档映射到同一空间

双编码器分别产生查询向量 `e_q` 和文档向量 `e_d`，用 dot product 或 cosine 打分：

```text
score(q,d) = e_q · e_d
```

文档向量可离线计算，并用近似最近邻索引搜索。优势是能召回语义相近但用词不同的文本；代价是 embedding 模型有领域分布，ANN 近似会损失召回，向量更新与权限过滤也增加系统复杂度。

稀疏与稠密不是二选一。常见可靠方案是 hybrid recall：两路各取候选，再做分数融合或 rank fusion，最后交给更昂贵的 cross-encoder/LLM reranker。

## 4. Chunking 是隐含的数据建模

chunk 太小，答案需要的前提与结论被拆开；太大，向量表示主题混杂、上下文浪费。固定字符数只是起点，技术文档更适合尊重标题层级、代码块、表格和段落边界，并保留父标题与来源 metadata。

应根据问题需要的证据跨度调参，而不是只比较“每块 512 还是 1,024 tokens”。可测指标包括 answer-bearing chunk recall、相邻块合并率和每个回答实际使用的证据比例。

## 5. 召回和重排回答不同问题

召回强调高 recall：目标证据尽量不要漏；重排强调在有限上下文里提高 precision。用同一个 top-k 最终答案指标同时调两者，会失去诊断信息。

建议分层评测：

| 层 | 关键问题 | 示例指标 |
|---|---|---|
| 语料/解析 | 正确信息是否进入索引 | ingestion coverage、解析错误率 |
| 召回 | gold evidence 是否进入候选 | Recall@k、MRR |
| 重排 | gold 是否被推到前面 | nDCG@k、MRR |
| 生成 | 回答是否被证据支持 | correctness、faithfulness、citation precision |
| 端到端 | 用户任务是否完成 | task success、latency、cost |

## 6. “有证据”不等于“忠于证据”

生成器可能混入参数记忆、误读否定、把多个来源拼成不存在的结论，或引用一个相关但不支撑陈述的段落。citation presence 只表示有链接，citation correctness 才检查该来源是否真正支持对应 claim。

对高风险场景，应要求：答案中的原子事实能映射到证据；冲突来源被显式呈现；证据不足时允许 abstain；检索语料、时间戳和访问权限进入审计记录。

## 7. Query 侧也需要建模

用户问题可能省略上下文、包含代词或把多个信息需求混在一起。query rewriting、decomposition 和 multi-hop retrieval 可以改善召回，但也可能改错意图。保留原始 query，并让改写结果可观察；对每个子查询记录命中文档，才能复盘。

多轮对话尤其不能把整段历史直接当查询。应先解析当前信息需求和必要实体，再生成检索查询；否则闲聊 token 会稀释检索信号。

## 8. Production RAG 的最小可观测集合

每次请求至少记录：索引/embedding/reranker 版本、query 与 rewrite、候选 IDs 和分数、最终上下文、token 预算、模型版本、生成参数、引用映射、耗时与用户反馈。涉及隐私时做脱敏和最小保留。

没有这些中间量，线上错误只能被粗暴归因于“模型幻觉”。有了分层 trace，才能判断应该补文档、改 chunk、训练 retriever、调 reranker，还是约束生成器。

## 9. 一个关键结论

RAG 把“模型知道什么”变成“系统此刻能检索并正确使用什么”。知识新鲜度因此从训练周期问题转成索引周期问题；但正确性也从单模型问题扩展成数据、检索、排序、上下文与生成的联合可靠性问题。

最后一篇回到经常被 LLM 叙事遮住的两条主线：语音链路，以及词法—句法—语义—篇章结构。它们为生成质量提供比“读起来不错”更细的检测坐标。

## 资料入口

- [SLP3 第 11 章：Information Retrieval and Retrieval-Augmented Generation](https://web.stanford.edu/~jurafsky/slp3/11.pdf)

