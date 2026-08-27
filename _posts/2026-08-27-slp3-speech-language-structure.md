---
layout: post
title: "SLP3 阅读笔记（七）：语音与语言结构——LLM 之外仍需保留的坐标系"
description: "把翻译、ASR/TTS、序列标注、句法、语义角色、指代与篇章放回一张任务地图，说明结构化评测为何仍不可替代。"
date: 2026-08-27 20:07:00 +0800
categories: [nlp-language-models]
category_name: "NLP 与语言模型"
tags: [Speech, ASR, Syntax, Semantics, Discourse]
series: "SLP3 2026 阅读笔记"
series_part: 7
reading_time: "26 分钟"
---

> 对应 SLP3 第 13–26 章。本文是后半卷的结构化导航，不用一篇文章假装替代十四章细节；后续专题将从真实系统问题继续展开。

统一的生成接口很方便，却容易把不同任务的错误压成一个“答案不好”。SLP3 后半卷提供了一组更细的观测层：声音是否识别对，实体边界是否正确，谁对谁做了什么，代词指向谁，句子之间是否连贯。

## 1. 语音不是“文本模型前后各加一个盒子”

语音输入是随时间变化的声学信号。ASR 要从波形中提取表示，处理说话人、口音、噪声、语速与发音变化，再产生词序列；TTS 则从文本、音素和韵律条件生成声学表示与波形。

一条实际语音助手链路可能是：

```text
audio → VAD/feature encoder → ASR → language model/tool
      → response text → pronunciation/prosody → vocoder → audio
```

端到端模型可以联合其中多步，但评测仍要拆开。ASR 的 word error rate 可分 substitution、deletion、insertion；TTS 还需可懂度、自然度、说话人相似度和韵律评价。最终任务失败可能源于首个实体就被 ASR 听错，而不是 LLM 推理差。

## 2. 翻译是有源条件的生成

机器翻译建模 `P(target|source)`。与自由生成相比，目标不仅要流畅，还要忠实覆盖源信息。漏译、增译、术语漂移和指代错位都可能在“很自然”的译文里隐藏。

自动指标适合规模化回归，但不能把一个 corpus-level 分数当成全部质量。应按领域、语言方向、句长、实体、否定和术语分桶，并对关键场景做人工错误分析。

## 3. 序列标注给文本加局部结构

POS tagging 和 NER 为每个 token 预测标签。BIO 等标记方案还编码实体边界。现代 LLM 可以用生成形式输出实体，但结构约束并没有消失：边界是否合法、标签是否来自 schema、字符 offset 是否能回到原文，都需要确定性验证。

在生产信息抽取中，结构化 decoder 或后处理常比“让模型输出一段看似 JSON 的文本”更可靠。评测也应区分 exact span、partial overlap、entity type 与 normalization。

## 4. 两种句法回答两类问题

**Constituency** 把句子组织成嵌套短语，适合描述 NP、VP 等成分边界；**dependency** 用有向关系连接中心词与依存词，更直接表示修饰和论元关系。

句法不是为了画树。它能暴露 attachment ambiguity：一个介词短语到底修饰动词还是名词？当生成模型在长句中把限定范围挂错，表面语法仍可能完全通顺。

## 5. 从“有哪些实体”到“谁对谁做了什么”

信息抽取逐层增加结构：

```text
entities → relations → events → temporal relations
```

Semantic Role Labeling 则围绕谓词寻找论元，如施事、受事、工具、地点。它提供一个特别实用的事实一致性检查：生成摘要是否保留了正确的参与者与角色，还是把动作发起者和承受者颠倒？

这类错误无法靠流畅度检测。即使最终系统采用 LLM，也可用结构化抽取器或规则作为独立审计通道。

## 6. 指代与实体链接处理“它是谁”

coreference 把同一篇文本中指向同一实体的 mention 聚成簇；entity linking 进一步把 mention 对齐到知识库中的唯一实体。两者常被混淆：前者主要是文内一致性，后者是文外消歧。

长文摘要、会议纪要和多轮对话中，代词、简称和角色变化会积累错误。只做逐句事实检查可能全部通过，但跨句合并后却把两个同名人物混为一人。

## 7. 篇章与对话高于单句

coherence 研究句子如何组成可理解的整体，包括实体延续、主题推进和修辞关系。conversation structure 还要处理 turn-taking、speech acts、grounding 和对话状态。

聊天模型输出单轮漂亮，不代表多轮任务成功。应记录用户目标、已确认事实、待补槽位、工具状态与承诺事项；对话评测要检查跨轮一致性、纠错能力和任务完成，而不只是最后一句偏好分。

## 8. 结构任务在 LLM 时代的三个新角色

1. **诊断标签**：把模糊的生成失败分解为实体、关系、论元、指代或篇章错误；
2. **约束接口**：通过 schema、grammar、validator 或工具参数限制输出空间；
3. **独立验证器**：用不同模型/规则检测生成结果，降低同源错误。

传统任务未必继续以独立产品形态存在，但作为观测、约束和评测层，它们反而更重要。

## 9. 一张选择表

| 线上症状 | 首先检查的结构层 |
|---|---|
| 人名、型号漏掉或边界错误 | tokenization / NER |
| 主客体颠倒 | dependency / SRL |
| 事件先后错乱 | event & temporal extraction |
| “他/它”指错对象 | coreference / entity linking |
| 每句正确但全文矛盾 | discourse coherence |
| 多轮反复询问已给信息 | dialogue state / grounding |
| 语音命令实体听错 | ASR 分桶 WER + entity WER |

## 10. 这套笔记如何继续

前七篇建立了从输入表示到系统评测的主干。后续不机械地“一章一篇”，而会围绕可落地问题深化，例如：多语言 tokenizer 公平性、RAG 评测集构造、ASR 错误如何传播到 Agent、结构化信息抽取如何做生成验证。

SLP3 最值得保留的不是某个 2026 年模型名称，而是分层建模的习惯：明确随机变量、表示、假设、搜索空间和评价指标。模型会换，这套分析骨架不会很快过时。

## 资料入口

- [SLP3 官方目录：Volume II 与 Volume III](https://web.stanford.edu/~jurafsky/slp3/)
- [第 13 章：Machine Translation](https://web.stanford.edu/~jurafsky/slp3/13.pdf)
- [第 15 章：Phonetics and Speech Feature Extraction](https://web.stanford.edu/~jurafsky/slp3/15.pdf)
- [第 16 章：Automatic Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/16.pdf)
- [第 18 章：Sequence Labeling](https://web.stanford.edu/~jurafsky/slp3/18.pdf)
- [第 26 章：Conversation Structure](https://web.stanford.edu/~jurafsky/slp3/26.pdf)

