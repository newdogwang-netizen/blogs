---
layout: post
title: "GPU 进阶笔记（七）：Scale-up 与 Scale-out——从链路语义到 Collective 成本"
description: "把 PCIe、NVLink、UALink、InfiniBand/RoCE 放回真实数据路径，并推导 Ring、Tree、All-to-All 与 rail topology 的性能边界。"
date: 2026-08-27 11:00:00 +0800
categories: [ai-gpu]
category_name: "AI 与 GPU"
tags: [NVLink, UALink, PCIe 6.0, NCCL, AllReduce, RoCE]
series: "2026 GPU 进阶笔记"
series_part: 3
reading_time: "31 分钟"
---

> 资料状态：2026-08-27。文中 `GB/s` 为 byte/s，`Gb/s` 为 bit/s；“双向聚合”绝不等同于单向吞吐。

把 GPU 互联排成一张“带宽排行榜”几乎必然误导。PCIe、NVLink、UALink 和 RDMA 网络服务于不同距离、语义和故障域。真正的问题不是哪条链路数字大，而是一次张量传输经过哪些桥、占用哪些共享端口，以及同步算法在关键路径上搬了多少字节。

## 1. 先画数据路径，再谈带宽

跨机架 GPU-to-GPU 传输的典型路径是：

```text
source HBM
  → source GPU memory subsystem
  → PCIe / C2C / on-package I/O
  → source HCA DMA engine
  → leaf → spine → leaf
  → destination HCA
  → PCIe / C2C
  → destination HBM
```

机架内 NVLink/UALink 路径则更短：

```text
source HBM → GPU → scale-up link → switch → link → GPU → target HBM
```

端到端时间由最慢环节和固定开销共同决定：

```text
T ≈ T_setup + T_queue + bytes / min(B_source, B_link, B_switch, B_sink)
```

若 collective 有多步，还要乘 step 数，并考虑多条链路能否真正并行。端口线速只是这个式子中的一个上限。

## 2. 四类互联的语义不同

| 层级 | 代表技术 | 常见语义 | 距离/规模 | 主要用途 |
|---|---|---|---|---|
| HBM fabric | GPU memory controller | cache line / burst | 封装内 | 本地权重、KV、激活 |
| Scale-up | NVLink、UALink、Infinity Fabric | GPU load/store/atomic、P2P | 板内到数个机架 | TP/EP/CP、细粒度共享 |
| Host I/O | PCIe、CXL、NVLink-C2C | device I/O、coherent memory（视协议） | 主机/机箱 | CPU、NIC、SSD、扩展内存 |
| Scale-out | InfiniBand、RoCE/UET | message/RDMA/collective | Pod/数据中心 | 跨节点、跨机架训练 |

“支持 RDMA”表示 NIC 能直接访问注册内存，减少 CPU 拷贝和软件路径；它不自动提供 GPU cache coherence。类似地，load/store 语义让 GPU 可以寻址远端内存，也不意味着远端访问等价于本地 HBM。

## 3. 单位、方向和编码：规格表的三重陷阱

忽略编码和协议开销：

```text
400 Gb/s ÷ 8 = 50 GB/s 单向
800 Gb/s ÷ 8 = 100 GB/s 单向
1.6 Tb/s ÷ 8 = 200 GB/s 单向
```

而“1.8 TB/s NVLink per GPU”通常是多个端口的**双向聚合**。若链路对称，单向端口总和约为一半，但应用能否同时吃满所有端口还取决于 traffic pattern。

需要至少标明四项：

```text
line rate / payload rate
one-way / bidirectional aggregate
per port / per device aggregate
peak / sustainable measured
```

PCIe 6.0 就是典型例子：64GT/s、x16 的 256GB/s 是双向合计，单向原始字节带宽约 128GB/s；PAM4、FLIT、FEC/CRC 与实现效率使实际 DMA 小于该值。来源：[PCI-SIG PCIe 6.0](https://pcisig.com/pci-express-6.0-specification)。

## 4. PCIe/NUMA：GPU 到 NIC 的隐藏瓶颈

GPUDirect RDMA 把路径从“GPU→host memory→NIC”缩短为 NIC DMA 直接读写 GPU memory，但数据仍要经过 GPU I/O、PCIe switch/root complex 和 NIC。

一个节点有 8 GPU 和 8×400G NIC 时，不能只看数量匹配，还要核对：

- 每张 GPU 最近的 HCA 是哪一张；
- GPU 与 HCA 是否共享 PCIe switch 上行；
- 是否跨 CPU socket/NUMA；
- x16 是否实际训练成 x16 与目标代际；
- IOMMU/ACS、BAR 和 peer-memory 配置是否允许最短 P2P；
- 一个 rail 故障后流量会怎样重映射。

可将路径代价写成矩阵 `C[gpu][nic]`，元素包含带宽和延迟。rank mapper 应优先选择代价最小的 GPU-HCA 对，而不是简单用设备编号相同。编号是软件枚举，不是物理拓扑。

## 5. NVLink/NVSwitch：高端口带宽之外的关键是交换语义

NVLink 4/5/6 每 GPU 双向聚合公开值分别为 0.9、1.8、3.6TB/s。真正让系统从 8 GPU 扩到 72 GPU 的是 NVSwitch fabric、地址映射与软件控制面，而不是单根 lane 加速。

在 GB300 NVL72 中，每 GPU 的 18 条 NVLink 5 分别连接 18 个 NVSwitch ASIC。任意 GPU 对之间通过一个交换 ASIC 到达，形成单跳、非过订阅的 scale-up 域。

不过，下列三种数字仍不能互换：

- **NVLink line rate**：物理端口收发上限；
- **P2P bandwidth**：特定源/目标、读写方向与 copy engine 下的有效值；
- **collective algorithm bandwidth**：`message_size / elapsed_time`。

硬件 multicast/reduction 还会改变“每字节经过多少端口”的假设。使用 NVLink SHARP/NVLS 等硬件加速时，传统 ring 校正出来的 `busbw` 不再代表真实线上字节率，应该优先报告 operation time 和 `algbw`，并注明算法。

NVLink 代际数据：[NVIDIA NVLink](https://www.nvidia.com/en-us/data-center/nvlink/)。

## 6. UALink：为什么“基于 Ethernet PHY”不等于 Ethernet transport

UALink 1.0 使用 Ethernet PHY 生态，却定义了自己的 scale-up protocol：

- 200GT/s/lane，x1/x2/x4 station；x4 每方向 800Gb/s；
- 64B 与 640B 固定 FLIT；
- credit-based flow control 与 link-level retry；
- load/store/atomic memory semantics；
- 软件维护一致性，不宣称 CPU 式硬件 cache coherence；
- request 有顺序规则，但 completion 不保证天然同序；
- 57-bit vendor physical address、source/destination routing；
- 目标 round-trip latency 小于 1µs、约 4m 内、1–4 racks、最多 1K endpoints；
- 支持端到端认证和加密。

这套设计在“短距、低延迟、无丢包、内存语义”上与数据中心 Ethernet/RDMA 的目标不同。UALoE 是把 UALink 能力映射到相应 Ethernet 物理/交换实现的路线，也不能直接等同 RoCEv2。

标准文本：[UALink 1.0 White Paper](https://ualinkconsortium.org/wp-content/uploads/2025/04/UALink-1.0-White_Paper_v3.pdf)与 [UALink Specification Overview](https://ualinkconsortium.org/blog/ualink-200g-1-0-specification-overview-802/)。

## 7. Scale-out：无损不是目标，低尾延迟才是目标

InfiniBand 与 RoCE 都能提供 RDMA。差异更多落在端到端生态、路由、拥塞控制、遥测和运维，而不是“一个能 RDMA、另一个不能”。

RoCE 网络常见的风险链条是：

```text
incast / hash collision
  → queue growth
  → ECN mark or PFC pause
  → congestion reaction
  → pause propagation / head-of-line blocking
  → collective tail latency
```

PFC 只避免特定 priority 丢包，不解决拥塞本身；buffer 越大也不一定越好，因为排队延迟会增加。ECN/DCQCN 或更新的拥塞控制要足够早地反馈，同时避免对短暂 burst 过度降速。

AI 网络设计更关心：

- 多个同步 collective 同时启动时的 incast；
- elephant flow 与短控制消息是否互相阻塞；
- adaptive routing 是否会造成乱序及重传；
- 单个慢端口如何放大全组 barrier 时间；
- 背景存储流量是否与训练共享 priority/queue。

UEC/UET 试图在 Ethernet 上面向 AI/HPC 重新设计 transport、拥塞控制和安全；规范存在不等于具体 NIC/switch 已达到同等成熟度，仍需按版本和实现验收。[UEC 1.0](https://ultraethernet.org/uec-1-0-spec)。

## 8. Collective 的字节账：先于 benchmark 的理论检查

设每 rank 输入张量为 `S` byte，共 `p` ranks。

### 8.1 Ring AllReduce

Ring 由 ReduceScatter 与 AllGather 两阶段组成，各 `p-1` 步，每步传 `S/p`：

```text
bytes_sent_per_rank = 2 × (p-1)/p × S
steps               = 2 × (p-1)
```

大消息下它接近 bandwidth-optimal，但 `p` 增大时 step 数线性增长。近似时间：

```text
T_ring ≈ 2(p-1)α + 2(p-1)/p × S/B
```

### 8.2 Tree AllReduce

平衡树的 step 数约 `2 log₂p`，小消息延迟更好：

```text
T_tree ≈ 2log₂(p)α + C_tree × S/B
```

其中 `C_tree` 取决于分块、双树和端口并行度。Tree 不是无条件优于 ring：大消息时若不能充分使用多条链路，带宽可能更低。

### 8.3 ReduceScatter 与 AllGather

每 rank 的理想数据量分别是：

```text
(p-1)/p × S
```

这也是 FSDP/ZeRO 常把 AllReduce 拆为 ReduceScatter + 参数 AllGather 的原因：它让状态分片和计算重叠成为可能，而不是减少两阶段的总理论字节。

### 8.4 All-to-All

若每 rank 总发送 `S`，自己保留 `S/p`：

```text
bytes_sent_per_rank = (p-1)/p × S
```

总字节不比 AllReduce 大，但它同时产生 `p(p-1)` 个逻辑 src-dst 流，容易形成 incast、交换热点和小消息低效率。MoE 的 token 路由又会打破均匀假设，所以 All-to-All 常比字节公式显示得更难。

## 9. 怎样正确读 NCCL/RCCL tests

NCCL tests 报告：

```text
algbw = S / time
```

经典 ring AllReduce 的校正值：

```text
busbw = algbw × 2(p-1)/p
```

AllGather/ReduceScatter/All-to-All 的校正因子是 `(p-1)/p`。这个 `busbw` 是为了让传统点到点算法更容易与硬件瓶颈比较，并非网卡计数器读数。

当 NCCL 自动选择 hierarchical、NVLS 或 in-network reduction 时，固定校正因子不再等于真实线上流量。因此性能报告应至少包含：

- message size、rank 数、节点/机架数；
- operation time、`algbw`、`busbw`；
- NCCL/RCCL 版本与实际选中的算法/protocol；
- GPU-NIC mapping、rail、channel 数；
- 平均值以及 P95/P99/max rank completion time。

定义来源：[NCCL Tests Performance](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)与 [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。

## 10. 一个 AllReduce 下界算例

假设 8 ranks 对 `S=140GB` 的 BF16 梯度做 ring AllReduce：

```text
bytes/rank = 2 × 7/8 × 140GB = 245GB
```

若有效单向带宽分别为 50GB/s、100GB/s、1.8TB/s，忽略延迟和一切争用：

| 假设有效带宽 | 纯传输下界 |
|---:|---:|
| 50GB/s | 4.90s |
| 100GB/s | 2.45s |
| 1.8TB/s | 0.136s |

最后一行只是数学上限，不能把 NVLink 双向端口聚合直接当 ring 单向有效带宽。算例真正告诉我们的是：若完整梯度频繁落到 400G/800G scale-out，通信很难被几百毫秒的 step 隐藏。工程上必须使用状态分片、多维并行、分桶和 overlap。

## 11. Rail topology：把算法坐标映射到物理坐标

一个 8-GPU 节点有 8 个 HCA 时，可把相同 local rank 的 HCA 连接到同一 rail：

```text
node0 GPU0—HCA0 ─ rail0 ─ HCA0—GPU0 node1
node0 GPU1—HCA1 ─ rail1 ─ HCA1—GPU1 node1
...
```

分层 AllReduce 先在节点/机架内 ReduceScatter，再由各 rail 并行交换对应 shard，最后本地 AllGather。理想情况下每条 rail 只承载自己的分片，避免任意 GPU 流量在 leaf/spine 上横向混洗。

rail 优化失效的常见原因：

- scheduler 改变了 GPU device order；
- 容器里看到的 NIC 名称与宿主拓扑不一致；
- 某个 rail 降级后 hash 没有均匀重分布；
- rank mapper 不知道实际 GPU-NIC 代价；
- 多租户 job 同时占用相同 rail。

因此拓扑、调度和 collective runtime 必须共享同一份物理事实。

## 12. 网络验收应回答哪些因果问题

不要只运行一个 1GB AllReduce。至少做以下扫描：

1. **消息尺寸**：8B 到数 GiB，找到 latency/bandwidth 转折点；
2. **规模**：单 GPU pair、单节点、单机架、跨 2/4/8 机架；
3. **操作**：P2P、AllReduce、AllGather、ReduceScatter、All-to-All；
4. **方向**：单向、双向、读、写；
5. **负载**：空载、背景存储流量、多 job 并发；
6. **故障**：端口 flap、rail 降级、ECN/PFC 异常、交换机维护；
7. **遥测关联**：把 collective 尾延迟与 queue、ECN、pause、replay、symbol error 对齐到同一时间线。

最终要能回答：“这次 step 变慢，是 GPU kernel、HBM、scale-up、PCIe、HCA、leaf queue 还是某个慢 rank？”如果遥测不能完成这条因果链，再高的线速也难以形成可运营的集群。

## 参考资料

- [PCI Express 6.0](https://pcisig.com/pci-express-6.0-specification)
- [NVIDIA NVLink](https://www.nvidia.com/en-us/data-center/nvlink/)
- [UALink 1.0 White Paper](https://ualinkconsortium.org/wp-content/uploads/2025/04/UALink-1.0-White_Paper_v3.pdf)
- [UEC Specification 1.0](https://ultraethernet.org/uec-1-0-spec)
- [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [NCCL Tests Performance](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- [NVL72 Network Logical Architecture](https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/network-logical-architecture.html)
