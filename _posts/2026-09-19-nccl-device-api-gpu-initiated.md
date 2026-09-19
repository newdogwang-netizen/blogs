---
layout: post
title: "NCCL Beyond Collectives：通信进入 GPU kernel 后，谁保证顺序？"
description: "沿大会 LSA、GIN 与 MegaMoE 教程，解释对称窗口、远端访问和可见性；也说明为什么通信计算融合不一定更快。"
date: 2026-09-19 09:06:00 +0800
categories: [cloud-native-ai-infra]
category_name: "云原生与 AI 基础设施"
tags: [NCCL, Device API, GIN, MoE]
series: "2026 上海大会技术研究笔记"
series_part: 6
track: main
reading_time: "14–18 分钟"
technical_series: true
toc: true
topic_url: /cloud-native-ai-infra/
---

过去学习 NCCL，通常从 AllReduce、AllGather 和 ReduceScatter 开始：应用在主机侧提交一个 collective，再通过 CUDA stream 组织计算与通信。大会的高级教程把观察点向下一层移动——如果 GPU kernel 可以直接访问 peer 内存，甚至发起网络操作，通信还能怎样组织？

本文固定讲师仓库提交 [`53e0951`](https://github.com/jeffhammond/pytorch-china-2026-nccl-tutorial/tree/53e095107fc1bac4c9912213f48e4c9d54043264)，从 `lsa-simple` 读起，再连接 `barrier` 与 `megamoe`。官方 API 说明参考当前页面标识的 NCCL 2.31.2；教程中 LSA 示例有 **NCCL ≥ 2.29** 的具体检查，不能用“Device API 从 2.28 开始提供”替代每个示例的版本要求。本文只做代码与机制分析，没有运行 GPU 带宽测试。

## 一、为什么需要比 collective 更细的组织方式

collective 给应用一个很强的抽象：参与者、输入输出和操作类型确定之后，库负责选择通信算法。这对大量规则通信非常合适，主机发起也并不意味着每个数据包都经过 CPU。

新问题出现在数据依赖越来越细时。例如 MoE 中某些 token 的路由与打包已经完成，另一些还在计算。如果必须等待完整阶段结束，再启动统一通信，就可能错过细粒度流水的机会。把数据产生、发送和消费放得更近，有机会减少阶段间缓冲和等待。

但这只是机会。新 kernel 也会消耗寄存器、共享内存与 SM 执行资源；融合后占用率下降，可能把省下的启动成本重新花在计算上。应该比较的是整个算子的完成时间，而不是“少启动了几个 kernel”。

## 二、对称窗口不是自动一致的大显存

对称窗口为参与 rank 建立有共同组织方式的内存区域，使程序能够通过窗口、偏移与 peer 身份定位数据。它解决地址与注册契约，不会自动提供任意访问模式下的一致性，更不会把所有显存变成统一低延迟内存。[NCCL Device API 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)

从取得地址到安全读取，中间还有几步：路径必须允许访问，写入必须发生，数据必须对接收端可见，参与者还要确认本轮数据已经完整。拿到了 peer 指针，后面这些条件仍需要逐项满足。

最小 LSA 程序可以按这个顺序读。每个 rank 分配并注册同样大小的窗口，随后取得 peer 指针表。rank s 往每个目的 rank 的第 s 个槽位写入由源和目的身份共同决定的签名。接收端检查全部槽位，就能发现缺写与映射错误；仅看 kernel 是否报错，检查不到这些问题。[LSA 示例源码](https://github.com/jeffhammond/pytorch-china-2026-nccl-tutorial/blob/53e095107fc1bac4c9912213f48e4c9d54043264/lsa-simple/lsa_tutorial.cu)

## 三、沿最小例子看“完成”如何被证明

{% include infra-figure.html file="nccl-lsa-ordering" alt="LSA 示例先注册窗口，GPU 执行远端写与系统范围 fence，主机同步本地设备，再经 MPI barrier 后读取验证" caption="图 1 · 按教程重绘的验证顺序。MPI 用于进程协调，不承载这个示例的 GPU payload；fence 本身不是所有 rank 的集合屏障。" %}

该示例在 kernel 中使用系统范围 fence，并在主机侧等待本地 kernel 完成，然后让各 rank 经过 MPI barrier，再读取本地窗口验证结果。要点是这一整条 happens-before 关系，而不是把某一个调用当作万能的“刷新远端内存”。

fence 约束内存操作的顺序与作用范围；barrier 协调参与者到达阶段。两者不是同义词。一个参与者还没有执行写入，其他参与者做再强的本地 fence，也不会替它把数据写出来。

教程选择主机同步，是为了让最小程序的正确性更容易检查。走向持久 kernel 或设备侧流水后，必须重新设计通知、等待、复用缓冲的规则，不能直接删去主机 barrier 并假定语义不变。

## 四、从 LSA 到 GIN：发起位置与传输后端分别看

LSA（Load/Store Accessible）让设备程序在支持的 peer 路径上执行 load/store。官方描述不把它严格限定为一种互联，具体可达性由平台决定；本教程最小例子的目标路径是 NVLink。

跨节点网络则引入 GIN（GPU-Initiated Networking）。设备代码可以发起相应通信操作，但“GPU 发起”不等于“整条路径完全没有 CPU”。官方文档区分包括 CPU proxy 与 GDAKI 在内的后端，平台、驱动、网络和版本支持需要分别确认。[GIN 与平台条件](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)

{% include infra-figure.html file="nccl-access-paths" alt="域内 LSA 走可访问 peer 内存路径；跨节点 GIN 经具体后端和 NIC，可能使用 CPU proxy 或硬件直接发起能力" caption="图 2 · 两类数据路径。host 初始化与资源管理仍然存在，绿色 payload 路径不代表整个系统没有 CPU 参与。" %}

讲师的 `barrier` 教程分别组织 full 与 rail 布局，说明参与者集合和网络组织会改变同步实现。阅读时要追问：谁向谁发送到达信号，谁等待哪些参与者，一轮完成后怎样避免把上轮的信号当作新轮信号。[GIN barrier 教程](https://github.com/jeffhammond/pytorch-china-2026-nccl-tutorial/blob/53e095107fc1bac4c9912213f48e4c9d54043264/barrier/README.md)

这也解释为什么设备侧通信更需要关注版本。NCCL 文档对 GIN 已编译 kernel 的升级兼容有单独提示：升级库时需要按要求重新编译，不能把普通 host API 的兼容经验直接推广到设备代码。

## 五、通知先到、数据没准备好，会发生什么

设生产者写一块 payload，再更新一个 ready 标记，消费者等待标记后读取。如果实现只保证“程序里先写数据再写标记”，却没有保证适用内存模型下的发布与获取顺序，消费者就可能看到标记已经变化，却仍读到旧数据。

下面只是协议意图，不是可直接编译的 NCCL 调用序列：

```text
生产者：获得可写缓冲 → 写 payload → 满足发布顺序 → 通知本轮完成
消费者：等待本轮通知 → 满足获取顺序 → 读 payload → 确认可以复用
```

真实代码还要明确通知属于哪一轮、缓冲由谁拥有、等待是否有超时，以及某个 rank 故障时谁终止其他参与者。双缓冲能减少覆盖风险，但只有在所有权交接正确时才成立；多一个缓冲区不会自动修复协议。

还有一个容易忽略的 GPU 侧问题：等待者可能占据执行资源，而让它继续前进的生产者尚未得到调度。设备侧循环不是免费等待。设计持久 kernel、CTA 协作与跨 GPU barrier 时，需要证明参与者具备实际前进条件，而非仅证明逻辑图中存在一条路径。

## 六、MoE 为什么是合适的研究对象

MoE 的 token 路由产生大小不均的消息，专家计算又依赖特定输入片段。把打包、交换、计算和结果合并组织成更细的流水，有机会减少中间阶段等待。教程提供了 SM100 NVFP4 MegaMoE 的 LSA 集成例子，并另有 SM90 路径；它们的硬件、数值格式和实现条件不能混成一个通用配置。[MegaMoE 示例说明](https://github.com/jeffhammond/pytorch-china-2026-nccl-tutorial/blob/53e095107fc1bac4c9912213f48e4c9d54043264/megamoe/README.md)

收益依赖消息大小分布、专家负载平衡和执行资源竞争。消息很小，协议固定开销可能占主导；专家严重偏斜，少数 GPU 仍会决定尾部完成时间；融合占用过多寄存器，又可能损害本来很快的矩阵计算。

因此，优化目标应是代表性 token 路由下的端到端算子吞吐与尾延迟，不能用某次连续大块传输的带宽结果代替。

## 七、正确性通过之后再谈性能

实验可以从两张支持目标访问路径的 GPU 开始，先跑签名验证，再增加 rank、消息尺寸和重复轮数。必须覆盖空消息、非整齐边界、不同到达顺序和缓冲轮转。测试框架还应设置超时，把“无限等待”记录为失败，而不是没有结果的慢测试。

性能对照至少包括成熟 collective/现有算子、未融合的设备通信实现，以及融合实现。分别记录初始化、通信、计算和整个算子的时间，保留不同消息分布的结果。没有硬件条件时可以继续源码研究，但不能据此宣布某种路径快多少。

如果现有 collective 已经能充分重叠通信与计算，未必需要改写。只有 profile 显示阶段屏障、中间缓冲或启动开销值得进一步消除时，才值得尝试设备侧组织；实现时要把通知轮次、缓冲复用和等待者的前进条件一起检查。
