---
layout: post
title: "异步后训练：rollout 越快，为什么训练也未必越有效？"
description: "沿 veRL 的异步样本队列与参数同步，解释长尾、背压和策略滞后；区分生成吞吐与达到目标质量的时间。"
date: 2026-09-19 09:11:00 +0800
categories: [cloud-native-ai-infra]
category_name: "云原生与 AI 基础设施"
tags: [veRL, 强化学习, 异步训练, Rollout]
series: "2026 上海大会技术研究笔记"
series_part: 11
track: extension
reading_time: "12–15 分钟"
technical_series: true
toc: true
topic_url: /cloud-native-ai-infra/
---

后训练系统同时运行两种节奏完全不同的工作：rollout 像在线推理，样本长度和工具调用时间高度不均；learner 像训练，希望拿到足够的数据后稳定执行批量计算。把它们接在一起，不只是部署一个训练任务加一个推理服务。

大会 [veRL 优化分享](https://www.lfopensource.cn/kubecon-cloudnativecon-openinfra-summit-pytorch-conference-china/program/schedule/?id=1222557)以及 [SGLang/Miles 分享](https://www.lfopensource.cn/kubecon-cloudnativecon-openinfra-summit-pytorch-conference-china/program/schedule/?id=1325851)都涉及后训练系统。本文沿 veRL `3efe38c` 的 fully async recipe 研究一条具体路径；Miles 作为另一条大会线索，不把两个框架的功能拼成同一套实现。没有运行 RL 训练实验。

## 一、同步流水线为什么会等待最慢样本

假设一批 rollout 中大多数回答很快结束，少数样本需要长推理或多轮工具调用。如果 learner 等整批完成才能更新，那么已经完成的 GPU 工作无法让训练提前开始，长尾样本决定了这轮屏障的结束时间。

简单增加 rollout GPU 可以提高并行生成能力，却不一定缩短单个最长样本。把这类等待视为普通容量不足，容易付出更多资源却保留同样的同步瓶颈。

异步化的直接思路是让完成的样本先进入队列，learner 累积到所需批次就训练，rollouter 同时继续生成。veRL 的公开 recipe 用 Rollouter、MessageQueue、Trainer 和 ParameterSynchronizer 分工实现这一方向。[固定提交的设计说明](https://github.com/verl-project/verl/blob/3efe38c759c14622fd1b2c9e3679f2d02f86bdac/docs/advance/fully_async.md)

{% include infra-figure.html file="rl-data-weight-loops" alt="rollouter 生成样本经奖励和队列交给 learner；learner 通过参数同步器把新策略送回 rollouter，形成方向相反的两条路径" caption="图 1 · 数据流与权重流。奖励阶段是后训练逻辑的一部分，具体执行位置随算法/框架配置变化。" %}

## 二、样本入队后，策略还在更新

同步训练较容易维持一批样本与某个策略版本的对应关系。异步情况下，rollouter 可能还在使用版本 v 生成，而 learner 已经更新到 v+2。样本来自旧策略，不再只是队列里的无状态任务。

定义 `Δv = 消费时学习策略版本 − 样本生成策略版本`，可以帮助观察策略滞后。但它不是唯一指标：两次更新可能很小，也可能改变很大；同样的版本差，在不同学习率和训练阶段不等价。

此外，一个多轮轨迹可能跨越参数同步。若框架支持保存部分 rollout 并继续生成，还要明确是否可能在一条轨迹内使用多个策略版本，以及相应 log probability 和 mask 怎样记录。保留已生成文本，不等于自动保留正确的训练语义。

## 三、队列越长，不一定越安全

设样本平均产生速率为 λ、消费速率为 μ。若 λ 长期大于 μ，队列就会增长；增加队列容量只延后溢出，并不能消除不平衡。更重要的是，旧样本等待越久，可能越偏离当前策略。

{% include infra-figure.html file="rl-staleness-backpressure" alt="旧版本样本在队列中等待时 learner 继续更新；背压限制继续生成，过期样本需要按算法规则处理" caption="图 2 · 策略滞后示意，无实测分布。队列深度是资源指标，样本年龄与版本差是训练语义指标。" %}

用 Little 定律作一个平均值算例：队列里平均有 600 条等待样本，稳态消费速率为每秒 100 条，则平均等待约 6 秒。如果 learner 每秒更新一次，样本仅在队列里就可能落后约六次更新；还没算生成期间使用的旧版本。这不代表六步一定不可接受，只说明“缓存了更多样本”同时改变了训练条件。

合理的控制需要背压：限制新的生成、调整资源配比、加快消费，或按算法规则丢弃/处理过期样本。它不是单纯的系统限流，因为改变样本保留策略也可能改变训练数据分布。

## 四、不要把 staleness 参数猜成版本差

veRL 这个 recipe 的 `staleness_threshold` 与允许额外生成的旧样本预算相关，文档给出了它和 `trigger_parameter_sync_step`、`require_batches`、mini-batch 大小之间的关系。**它不是一个简单的“最多落后 N 个版本”的整数字段。** `partial_rollout` 的生效条件也与异步配置有关。[参数语义](https://github.com/verl-project/verl/blob/3efe38c759c14622fd1b2c9e3679f2d02f86bdac/docs/advance/fully_async.md)

不同框架的 async 参数不宜直接照搬：限制轮次、限制样本预算和限制队列长度，控制的是不同的量。参数名相似，也需要沿队列和同步代码确认其生效位置。

另一个需要固定的条件是后端支持。本文引用的 recipe 对引擎组合和 server/AgentLoop 模式有具体说明；仓库中其他实验路径或新版本支持更多组合，不等于当前 recipe 的每项能力都自动覆盖它们。

## 五、训练和推理的概率计算必须能对上

rollout 与 learner 常常使用不同执行栈。即使权重版本相同，dtype、量化、采样参数、tokenizer、attention 实现和数值误差也可能使概率计算不完全一致。

许多策略优化方法会使用行为策略与当前策略的概率关系。如果保存的 log probability 不是实际产生样本时的那一份，或者 token 对齐与 mask 出错，系统看上去仍在高速训练，却可能优化了错误的目标。

因此，性能验收前应有一组小规模语义对照：固定输入与采样条件，检查 token 序列、有效 token 范围、行为 log probability、奖励与截断标记。重要性权重或其他校正可以处理特定偏差，但不是把任意陈旧或错误样本变成有效数据的通用修复器。

## 六、资源比例要围绕质量目标调整

rollout 与 learner 分开使用 GPU 后，两阶段单独耗时可能都比独占整池时更长，但由于重叠，总时间仍可能减少。反过来，如果拆得过细，权重同步、排队和资源碎片也可能抵消重叠收益。

评估至少同时记录有效样本/秒、队列年龄分布、策略滞后、丢弃比例、参数同步时间、奖励评估成本和验证集质量。最重要的比较是：在同样总 GPU 预算下，达到同一质量阈值需要多长时间，而不是某个阶段输出了多少 token。

实验可以从同步基线开始，逐步增加异步预算，保持算法与样本条件可比。如果 token 吞吐上升了，达到目标质量却更慢，应先查样本版本、概率记录和数据分布。此时继续增加 rollout GPU，可能只是让队列里积压更多旧样本。
