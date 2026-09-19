---
layout: post
title: "Agent 的 KV Cache：什么值得保留，什么时候应该重新计算？"
description: "从 UCM 的块匹配、加载与保存路径，理解跨请求缓存的兼容契约，以及命中率之外的延迟和容量成本。"
date: 2026-09-19 09:09:00 +0800
categories: [cloud-native-ai-infra]
category_name: "云原生与 AI 基础设施"
tags: [UCM, KV Cache, Agent, 推理存储]
series: "2026 上海大会技术研究笔记"
series_part: 9
track: extension
reading_time: "11–14 分钟"
technical_series: true
toc: true
topic_url: /cloud-native-ai-infra/
---

一个代码 Agent 连续处理同一仓库的多个问题。每次请求都包含相似的系统指令、文件内容和历史消息。如果这些前缀反复 prefill，GPU 就在重复计算已经做过的工作。把 KV 留在进程内能够复用一部分，但进程重启、请求被路由到别处或显存压力上升，都可能让复用消失。

大会 [UCM 分层 KV Cache 分享](https://www.lfopensource.cn/kubecon-cloudnativecon-openinfra-summit-pytorch-conference-china/program/schedule/?id=1223545)讨论了进程之外的缓存复用。以下沿 UCM 的公开架构看一次查询、加载和保存，再计算读取 KV 与重算 prefill 的成本。文中算例使用假设值，没有运行推理 benchmark。

## 一、相同文本，不一定对应相同 KV

KV 是模型执行产生的状态，不是原始文本的压缩文件。同一段文字放在不同前缀后面，因果 attention 已经看到了不同历史，通常不能直接复用相同 KV。即使 token 序列一致，权重、adapter、位置处理、dtype、并行切分和缓存布局变化，也可能使缓存不兼容。

UCM 在常规 vLLM 路径中使用具有前缀关联的块标识，并把模型与执行配置纳入标识和布局处理。文档特别提醒：相同模型目录名并不能代替权重和 tokenizer 的版本治理；不同布局也不能统一套用连续前缀规则。[How Caching Works](https://ucm.readthedocs.io/en/latest/developer-guide/capability-principles/)

从工程上，可以把“允许共享”看成四项同时成立：语义相同、执行布局兼容、数据完整、授权允许。最后一项容易被性能讨论遗漏：不同租户恰好拥有相同 token，并不自动意味着平台允许它们共享可观察的缓存状态或时序信息。

因此，模型版本升级时不应只清理进程内缓存，还要明确外部缓存命名空间如何隔离。一个内容指纹可避免误读旧数据，却不能替代访问控制。

## 二、查到缓存之后，worker 还要等什么

UCM 的角色分工值得单独学习：引擎拥有请求、批次与设备内 KV；Connector 理解引擎钩子、块身份和布局；Store 接收块与传输描述，负责相应存储操作。Store 不应该靠猜测模型结构决定某块内存该怎么解释。[UCM 架构](https://ucm.readthedocs.io/en/latest/developer-guide/architecture/)

{% include infra-figure.html file="kv-lookup-load-save" alt="调度侧查找可复用块，worker 加载到引擎分配的缓冲并等待完成，attention 才能消费；保存是另一条异步状态路径" caption="图 1 · 常规外部 KV 复用路径重绘。lookup 命中、load 完成和 save 完成是三个事件。" %}

调度侧查到命中后，可以减少计划重算的 token；worker 仍要将数据装入引擎提供的目标缓冲，并在 attention 使用前确认加载完成。查找之后缓存可能被淘汰，传输也可能失败，所以命中不能当作对未来读取的无条件保证。

一个具体接口细节体现了这个边界：常规路径的 `lookup_on_prefix()` 返回最后连续命中的块索引，首块未命中时是 -1，而不是直接返回块数。若上层把索引当数量，会产生偏移错误。混合布局还可能使用不同的状态查找方式，不能把这个规则推广到所有模型。[块匹配语义](https://ucm.readthedocs.io/en/latest/developer-guide/capability-principles/)

## 三、分层存储要看数据实际经过哪里

GPU、主机内存、本地文件与远端存储具有不同容量和访问代价。把 KV 放到更便宜的一层，只说明保存成本可能降低，不说明再读回来一定划算。

例如 UCM 的 `Cache|Posix` 路径由 Cache 管理主机缓冲及设备/主机传输，由 Posix 管理文件 I/O。不同引擎的集成也不完全相同：SGLang 的适配路径会与其 HiCache 分工，不能照搬 vLLM Connector 的所有配置与钩子。[组件职责与引擎差异](https://ucm.readthedocs.io/en/latest/developer-guide/architecture/)

这里应特别关注 staging 内存。外部缓存减少 HBM 中长期保留的状态，却可能增加主机内存、锁页缓冲、磁盘与网络压力。如果多个 worker 同时恢复大块 KV，系统瓶颈可能从 prefill 计算转移到共享读取路径。

## 四、什么时候读取缓存比重算更便宜

设可跳过的 prefill 关键路径耗时为 `Tcompute`，查找、排队、读取、传输、布局处理与同步的新增关键路径为 `Tcache`。一次命中真正有益，至少需要 `Tcache < Tcompute`。可重叠的部分应按实际时间线扣除，不能把各阶段峰值带宽简单相加。

{% include infra-figure.html file="kv-reuse-decision" alt="缓存先经过兼容与授权检查，再比较读取关键路径和重算关键路径；命中但不划算时仍可选择重算" caption="图 2 · 作者的复用决策模型。存在缓存不等于必须使用；最后一项还取决于引擎是否支持相应选择和回退。" %}

一个仅用于推导的例子：若可复用 KV 为 8 GiB，有效瓶颈吞吐为 20 GiB/s，则光传输就有约 0.4 秒的量级，还未计入查找、排队和同步。如果可跳过的 prefill 只需 0.15 秒，这次远端命中可能更慢；如果需要 1.5 秒，才有进一步优化的空间。这里的数字不是硬件或 UCM 实测。

因此，淘汰策略不宜只考虑“最近是否用过”。同样一 GiB 缓存，长前缀、重计算昂贵且近期可能复用的部分，价值可能高于短而便宜的部分。可以用“预计复用概率 × 可节省的关键路径成本 / 占用容量”作为讨论起点，但真实策略还要考虑前缀依赖、存储层级与公平性。

## 五、命中率为什么不是最终答案

请求命中率、token 命中率和字节命中率会给出不同印象。大量短请求命中，可能让请求命中率很高，却没有节省多少 GPU 工作；少量长前缀命中，可能大幅减少 prefill，但一次大读取又推高其他请求的尾延迟。

更完整的观察链路是：命中的 token → 实际加载成功的 token → 真正跳过的计算 → 端到端 TTFT 与吞吐变化。UCM 文档也区分查询、传输和实际执行阶段，并提供[支持矩阵](https://ucm.readthedocs.io/en/latest/user-guide/support-matrix/)；某种引擎有接入路径，不代表所有模型、并行配置和存储后端组合都已验证。

验收应在同样输入和并发下比较进程内缓存、外部冷缓存、外部热缓存，再加入多租户和存储拥塞。不要把首轮冷运行与第二轮热运行的差异全部归给一个新缓存组件。

## 六、把缓存失败降级成什么

理想情况下，非必需的复用失败可以退回重算。但是否能在调度已预算命中、目标缓冲已分配之后安全重算，取决于引擎集成；不能仅凭缓存“本来可丢弃”就假设任何时刻都能透明回退。

值得测试的断点包括：lookup 后删除缓存、读取中断、部分块损坏、模型升级后读到旧命名空间、同一前缀并发保存，以及存储不可达时大量请求同时重算。最后一种会形成计算侧的突发洪峰，需要限流或退避，不能无限放行。

是否启用某一层外部缓存，最后要回到同一组请求上比较：省了多少 prefill，增加了多少读取等待，P99 是否还能接受。对于读取比重算更慢的前缀，即使命中率很高，也没有必要为了保住这项指标继续保留。
