---
layout: post
title: "GPU 进阶笔记（五·前篇）：HGX B300——8-GPU 节点为什么仍是主流边界"
description: "拆解 HGX B300 baseboard、双 NVSwitch、1:1 ConnectX-8、双平面网络与 DGX B300 整机，解释它相对 NVL72 更现实的部署边界。"
date: 2026-08-27 13:00:00 +0800
categories: [ai-gpu]
category_name: "AI 与 GPU"
tags: [HGX B300, Blackwell Ultra, NVSwitch, ConnectX-8, DGX B300]
series: "2026 GPU 进阶笔记"
series_part: 0
reading_time: "32 分钟"
---

> 资料状态：2026-08-27。本文以 NVIDIA 2026-05-18 更新的 HGX AI Factory 参考架构和 DGX B300 运维文档为主。容量使用十进制 GB/TB；网络带宽明确区分 bit/s、byte/s 和双向聚合。

72-GPU NVL72 展示了机架级 scale-up 的上限，但很多数据中心更现实的采购单位仍是 **8-GPU HGX B300 服务器**：它保留传统双路 x86、单 OS 节点和可独立更换服务器的运维边界，同时把 Blackwell Ultra、NVLink 5、NVSwitch 与每 GPU 800Gb/s 网络放进一个节点。

这不是“缩水版 NVL72”。两者的 CPU、内存语义、故障域、网络比例和适用并行策略都不同。理解 HGX B300，要先把三个经常混用的名称拆开。

## 1. HGX B300、NVIDIA-Certified System、DGX B300 不是同一个产品

| 名称 | 实际含义 | 哪些规格固定 |
|---|---|---|
| HGX B300 baseboard | 8 GPU、NVSwitch、板载互联与 I/O 的平台基板 | GPU、HBM、NVLink/NVSwitch、基础 PCIe/NIC 能力 |
| NVIDIA-Certified HGX B300 system | OEM 基于 HGX baseboard 设计并认证的完整服务器 | 满足参考架构下限；CPU、机箱、电源、冷却可因 OEM 而异 |
| NVIDIA DGX B300 | NVIDIA 自己交付的一种具体 HGX B300 整机 | 10RU、双 Intel CPU、14.5kW、具体存储/网卡/软件栈 |

因此，下面两句话只有第一句可以泛化：

- “HGX B300 有 8 张 B300、2.304TB HBM、节点内 NVSwitch”——是平台事实；
- “HGX B300 是 10RU、14.5kW”——不严谨，这是 **DGX B300 整机**的公开规格，不代表所有 OEM HGX 系统。

采购时应同时写清 baseboard generation、OEM system SKU、CPU、冷却、PSU、NIC/DPU、固件矩阵与认证状态。

## 2. Baseboard 的物理结构：8 GPU + 2 NVSwitch

HGX B300 节点内的核心结构可以抽象为：

```text
GPU 0 ─┬─ 9 × NVLink ─ NVSwitch 0
       └─ 9 × NVLink ─ NVSwitch 1

GPU 1 ─┬─ 9 × NVLink ─ NVSwitch 0
       └─ 9 × NVLink ─ NVSwitch 1
       ...
GPU 7 ─┬─ 9 × NVLink ─ NVSwitch 0
       └─ 9 × NVLink ─ NVSwitch 1
```

每张 B300 有 18 条 NVLink，平均分到两个 NVSwitch ASIC。任意 GPU 间通信通过交换 fabric 完成，而不是组成一条必须逐卡转发的 ring。

每 GPU NVLink 双向聚合带宽为 1.8TB/s，因此节点口径是：

```text
8 GPU × 1.8 TB/s/GPU = 14.4 TB/s
```

这正是参考架构写出的 total aggregate bandwidth。它是所有 GPU 端口的双向聚合，不表示某一 GPU pair 独占 14.4TB/s，也不等于 NCCL AllReduce 的算法带宽。

关于代际名称，NVIDIA 不同文档的措辞并不完全一致：HGX AI Factory 组件页称“第五代 NVSwitch 和第五代 NVLink”，Fabric Manager 文档则把 B200/B300 的两颗 NVSwitch ASIC 归入第四代 NVSwitch。本文不根据这个标签推导能力，而以更可验证的事实为准：**两颗 NVSwitch、每 GPU 各 9 条链路、总计 18 条、1.8TB/s/GPU。**

拓扑依据：[NVIDIA Fabric Manager：HGX B200/B300 GPU Baseboard](https://docs.nvidia.com/hgx-platforms/fabric-manager-user-guide/index.html)与 [HGX B300 Components](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html#nvidia-hgx-b300-baseboard)。

## 3. HBM 账：2.304TB 容量与 64TB/s 不是一块共享显存

每张 B300 配置 288GB HBM3e、最高 8TB/s：

```text
node HBM capacity  = 8 × 288 GB  = 2,304 GB = 2.304 TB
node HBM bandwidth = 8 × 8 TB/s  = 64 TB/s aggregate peak
```

8 张 GPU 仍各自拥有 288GB 物理 HBM。CUDA 进程可以通过 NVLink P2P 和 collective 操作跨 GPU 搬运数据，但不会自动得到一块硬件一致的 2.304TB 单体内存。

这直接影响模型放置：

- 单 GPU 权重和 KV 必须落在 288GB 边界内；
- tensor parallel 把权重分到 2/4/8 GPU，同时引入每层 collective；
- pipeline parallel 让各 GPU 持有不同层，引入 stage 间 activation；
- 数据并行复制整份模型，吞吐高但容量不相加。

以 300B 参数、NVFP4 约 4.5 effective bit/weight 粗算：

```text
300e9 × 4.5 / 8 = 168.75 GB
```

裸权重可落在一张 288GB B300，还剩约 119GB 给 scale 对齐、运行时、workspace 和 KV；具体能否满足长上下文并发必须继续做 KV 容量实测。若用 BF16，同一模型仅裸权重就约 600GB，至少需要多 GPU 分片。

所以“大 HBM”带来的第一收益，常常不是单次 GEMM 更快，而是减少 TP degree 或让更多 KV 留在本地。

## 4. B300 GPU：不要把 144 PFLOPS 当作无条件算力

Blackwell Ultra 的芯片要点包括：

- 双 reticle 设计，通过约 10TB/s NV-HBI 作为一个 CUDA device 工作；
- 最多 160 SM、640 个第五代 Tensor Core；
- 288GB HBM3e、最高 8TB/s；
- NVLink 5 为 1.8TB/s/GPU；
- 支持 NVFP4 的 16-value micro-block 与两级 scale；
- 单 GPU TGP 最高 1,400W。

DGX B300 规格表给出系统 72 PFLOPS FP8 training、144 PFLOPS FP4 inference；HGX 参考架构也写“up to 144 petaflops”。但 NVIDIA 的 Blackwell Ultra 芯片说明还给出 full implementation 最高 15 PFLOPS dense NVFP4/GPU。

这些数字不能直接混算，因为 SKU、dense/sparse、数据格式、频率和测量定义可能不同。资深用户应要求供应商明确：

```text
precision + dense/sparse + SKU/SM count + clock/power cap
+ input shape + accumulator + measured/peak
```

文章后续的容量与通信推导不依赖“144”这一容易混淆的峰值，而使用 HBM、NVLink 和网络这些定义更明确的数字。

架构来源：[Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)。

## 5. Host 边界：HGX 是 x86 节点，不是 Grace Superchip

HGX B300 reference system 采用双 CPU socket，支持 Intel 或 AMD OEM 平台；参考要求包括：

- 每 socket 至少 48 个物理 core，建议 56 core；
- 两 socket 合计至少 2TB system memory；
- system memory bandwidth 至少 500GB/s；
- 内存应对称填满 CPU memory controller channels；
- baseboard 提供 8 条 PCIe Gen5 x16 与一条 Gen4 x2；
- DPU/SuperNIC/adapter 使用 Gen5 x16；
- PCIe root port 应在双 socket 之间均衡分布。

DGX B300 是一个具体实现：双 Intel Xeon Platinum 6776P、2TB 默认且最高 4TB host memory。OEM HGX 可能选择 AMD EPYC 或其他认证 CPU，因此应用不应假设所有 HGX B300 的 NUMA、core 数和 host bandwidth 相同。

### 5.1 为什么 2TB host memory 仍然重要

GPU HBM 很大，但 host memory 仍承载：

- 数据预处理、tokenization、page-locked staging buffer；
- checkpoint 组装与异步写入；
- CPU optimizer/offload（若使用）；
- filesystem cache 与远端存储 client；
- 容器、通信 runtime 与 control plane。

如果 DIMM 没有对称填充，或数据线程、GPU 和 NIC 跨 NUMA，瓶颈会出现在 host path，而不是 B300。`GPU-Util` 看似很高也不能排除间歇性供数断流。

## 6. 1:1 GPU-to-NIC：板载 800G 并不等于端到端 800G

HGX B300 baseboard 集成 8 个 ConnectX-8 SuperNIC，目标是每 GPU 对应一个 NIC：

```text
8 GPU : 8 ConnectX-8 = 1 : 1
each ConnectX-8 = up to 800 Gb/s = 100 GB/s line rate
```

reference architecture 给出两种部署：

| 方案 | 每 GPU 外部端口 | 每节点端口总和 | 特点 |
|---|---:|---:|---|
| Single plane | 1×400Gb/s | 8×400G = 3.2Tb/s = 400GB/s | 成本较低，带宽减半，单 fabric |
| Dual plane | 2×400Gb/s breakout | 16×400G = 6.4Tb/s = 800GB/s | 两个独立 fabric，带宽与故障路径更好 |

这里的 800GB/s 是 8 张 NIC 的节点单向端口线速合计，不是 800Gb/s；原始文档正好同时使用两种单位，阅读时很容易差 8 倍。

Dual-plane 中，每个 400G port 连接不同 leaf/fabric。健康状态下两个 plane 各承载约 50%；一个 plane 故障后流量转到另一个 plane，可用带宽会近似线性下降。NCCL 与 ConnectX runtime 基于全局通信信息做分流，比静态 LAG hash 更适合低 entropy 的 collective flow。

但端到端有效带宽仍受：GPU I/O、PCIe path、NIC protocol、leaf/spine、对端和 collective algorithm 的共同限制。`800G NIC` 不是 `100GB/s NCCL algbw` 的承诺。

## 7. 三张网络为什么仍要分开

HGX B300 集群与 NVL72 一样，参考架构区分三类 physical fabric：

| Fabric | 设备 | 流量 | 设计目标 |
|---|---|---|---|
| GPU Compute E/W | 8×ConnectX-8/node | 跨节点 RDMA、collective、训练 | rail-optimized、non-blocking fat tree |
| CPU Converged N/S | BlueField-3 B3240 | 存储、客户入口、in-band 管理 | 与训练 fabric 隔离，双端口冗余 |
| OOB | BMC、DPU/NIC 管理口 | Redfish、固件、PXE、救援 | 数据面故障时仍可管理 |

参考系统使用一张 BlueField-3 B3240，提供两个 400GbE port；DGX B300 的具体 feature summary 则列出两张 dual-port BlueField-3、总计两个 400Gb/s 对外连接。再次说明：**平台最低要求与具体 DGX 料单不能混为一张表。**

N/S 网络文档给出每 node 最高约 40GB/s 的存储吞吐目标。两个 400G port 的物理线速合计是 100GB/s，40GB/s 是更接近系统数据路径的设计目标，两者差异体现了协议、路径与用途，而非文档矛盾。

## 8. 4-node Scalable Unit：集群为什么从 4 台开始复制

HGX B300 Enterprise RA 把 4 个 server node 定义为一个 scalable unit（SU）：

```text
4 nodes × 8 GPU             = 32 GPU
4 nodes × 8 ConnectX-8      = 32 SuperNIC
32 NIC × 2 × 400Gb/s        = 64 × 400G endpoints
64 × 400Gb/s                = 25.6Tb/s compute port aggregate

4 nodes × 1 BlueField-3
× 2 × 400Gb/s               = 8 × 400G = 3.2Tb/s converged aggregate
```

SU 不是新的共享内存域，而是网络端口、线缆和交换机 radix 便于重复设计的容量块。参考架构再以 8/16/32 个 SU 扩到：

| Nodes | GPUs | SUs | Dual-plane compute leaf | Compute spine |
|---:|---:|---:|---:|---:|
| 32 | 256 | 8 | 8 | 4 |
| 64 | 512 | 16 | 16 | 8 |
| 128 | 1024 | 32 | 32 | 16 |

表中 leaf/spine 数是参考拓扑，而不是任何集群的唯一答案。它的重要信息是：endpoint 每加倍，光模块、线缆、交换端口、故障域和运维对象也近似成倍增长。GPU 预算若不包含 fabric BOM，就不是完整集群预算。

## 9. Rail-optimized 怎样对应 8-GPU 节点

理想映射让每个 local GPU index 使用对应 rail：

```text
node0 GPU0—NIC0 ─ rail0 ─ NIC0—GPU0 node1
node0 GPU1—NIC1 ─ rail1 ─ NIC1—GPU1 node1
...
node0 GPU7—NIC7 ─ rail7 ─ NIC7—GPU7 node1
```

dual-plane 再把每个 rail 分成 A/B 两个独立 fabric。分层 AllReduce 可以先在节点内用 NVSwitch ReduceScatter，再让 8 个 NIC/rail 各自交换一个 shard，最后回到节点内 AllGather。

这解释了为什么 HGX B300 同时需要 14.4TB/s node-scale fabric 和每 GPU 800G scale-out：

- NVSwitch 负责高频、低延迟的 TP/节点内归约；
- 8 个 NIC 并行搬运已经切好的跨节点 shard；
- rail topology 避免任意 GPU 流量在交换网络横向洗牌；
- rank mapping 错误会让通信先绕路，再进入正确 NIC。

因此验收必须保存 `nvidia-smi topo -m`、NIC/PCI BDF、NUMA 和交换端口的完整映射，而不是只看设备数量。

## 10. HGX B300 与 GB300 NVL72：边界差异比 GPU 数更重要

| 维度 | HGX B300 system | GB300 NVL72 |
|---|---:|---:|
| Scale-up GPUs | 8 | 72 |
| CPU | 双路 x86，OEM 可选 | 36 Grace CPU |
| OS/故障边界 | 单 server node | 多个 compute node/tray |
| GPU HBM | 2.304TB aggregate | 20.736TB aggregate |
| NVSwitch ASIC | 2 | 18 |
| NVLink port aggregate | 14.4TB/s 双向 | 129.6TB/s 双向 |
| Scale-up 终点 | 单服务器 | 整机架 |
| 典型外部网络 | 每 GPU 1×或2×400G | 每 GPU 对应 800G 级 HCA 配置 |
| 设施示例 | DGX B300：10RU、14.5kW | 参考整架：最高约142kW |

HGX 的优点：

- 单节点软件与故障边界清晰；
- 可沿用 x86、普通 scheduler 和成熟的 8-GPU 运维模型；
- 可以按 1/2/4/8 GPU 分配，减少整架调度碎片；
- 服务器、交换网络和机房可分阶段建设；
- 单次 tray/server 故障只影响 8 GPU，而非 72-GPU fabric。

NVL72 的优点：

- 把 TP/EP/CP 的高频通信保留在 72-GPU NVLink domain；
- 大模型和 MoE 不必在 8 GPU 后立刻进入 HCA；
- 单跳 scale-up 更适合细粒度跨节点 GPU memory access。

二者不存在抽象意义上的“谁更先进”。如果模型能在 8 GPU 内高效切分，HGX 往往是更易部署的边界；只有当跨 8-GPU 的 exposed communication 成为主要瓶颈，72-GPU scale-up 的成本才可能兑现为业务价值。

## 11. DGX B300 给出的设施现实

DGX B300 作为一种具体整机，公开规格为：

- 10RU，深度约 904mm；
- PSU 版本约 168kg，busbar 版本约 123kg；
- 最大功耗 14.5kW，热输出约 49,476 BTU/h；
- 12 个 3.2kW PSU（AC 版本），或 54V DC busbar；
- 约 1,500 CFM airflow（PSU 版本，特定风扇工况）；
- 8×3.84TB E1.S cache NVMe 与 2×1.92TB M.2 boot drive。

用最简单的上限估算，4 台占 40RU、最大 IT power 达 58kW，还没有计算交换机、管理设备和冗余余量。这仍远低于 NVL72 的 142kW 整架，但已经超过很多传统风冷机架的供电密度。

OEM HGX B300 可能采用不同 U 数、液冷或混合冷却，因此设施验收必须以所购整机 data sheet 为准，不能拿 DGX 数字替代。

来源：[DGX B300 Physical and Operational Specifications](https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html)。

## 12. 哪些工作负载更适合 HGX B300

### 12.1 单节点大模型推理

当模型能在 1–8 GPU 内容纳，TP traffic 全部走 NVSwitch。多节点可以部署独立 replica，通过 service layer 做数据并行扩容，不必为每个 token 引入 scale-out collective。

NVIDIA 参考架构因此允许纯推理部署省略 compute fabric，但这个判断有严格前提：请求之间独立、模型与 KV 不跨 node、没有训练/微调和分布式 prefill/decode。未来若要增加这些 workload，再补建 fabric 会有显著停机和布线成本。

### 12.2 训练与微调

TP≤8 留在节点内，DP 跨节点走 8 rails，是成熟的 2D/3D parallelism 映射。梯度大消息更容易利用 400G/800G bandwidth；高频 TP collective 不越过 Ethernet。

### 12.3 不适合的情况

- 每层需要 TP>8 且 collective 无法隐藏；
- MoE expert group 远大于 8，All-to-All 成为 step 主导；
- 单个模型/上下文要求跨多个 node 频繁共享 KV；
- 业务只能接受类似 scale-up 的远端 load/store latency。

这些才是考虑 NVL72 的技术理由，而不是“72 比 8 大”。

## 13. 到货验收：先证明 node，再证明 cluster

### 13.1 节点内

1. 枚举 8 GPU、2 NVSwitch、8 ConnectX-8、DPU、NVMe 与固件版本；
2. 检查每 GPU 的 18 条 NVLink、replay/error counter；
3. 跑完整 8×8 P2P bandwidth/latency matrix；
4. 扫描 AllReduce/AllGather/ReduceScatter/All-to-All 消息尺寸；
5. 验证 HBM、业务 GEMM/attention shape 与持续功率/温度；
6. 验证 Fabric Manager 初始化和 node reboot 后恢复。

### 13.2 节点到网络

1. 对齐 GPU↔NIC↔NUMA↔rail，不依赖 device index 猜测；
2. 分别测试 plane A、plane B 与双 plane；
3. 单 rail 到 8 rails 扫描，检查线性扩展；
4. 注入一个 400G port/plane 故障，验证带宽下降与作业行为；
5. 同时施加 N/S storage traffic，确认不会污染 E/W compute fabric；
6. 将 collective P99 与 ECN、queue、FEC、replay 和最慢 rank 对齐。

### 13.3 业务层

最终必须用生产模型报告 TTFT、TPOT、吞吐、质量、功耗和 HBM/KV 使用；训练则报告 time-to-quality、step breakdown、checkpoint 和故障恢复。只有证明“8-GPU 节点内快、跨节点按设计扩展、故障可隔离”，HGX B300 才从一张 baseboard 变成可运营的系统。

## 结语：8-GPU 边界没有过时，它只是更清楚地暴露了网络成本

HGX B300 延续了一个经过多年验证的系统分层：8 GPU 在节点内用 NVSwitch 形成高带宽域，节点外用 1:1 ConnectX-8 和 rail-optimized fabric 扩展。Blackwell Ultra 把单节点容量推到 2.304TB、HBM 聚合带宽推到 64TB/s，使许多以前必须跨节点的模型重新落回一个 OS 边界。

NVL72 的方向是扩大 scale-up domain；HGX B300 的方向则是让传统 node-scale 架构保持足够强。对于能够把高频通信限制在 8 GPU 内的 workload，后者往往更容易采购、部署、切分和维修——这正是它在 2026 年仍值得单独研究的原因。

## 参考资料

- [NVIDIA HGX B300 Components](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html#nvidia-hgx-b300-baseboard)
- [HGX B300 Networking Physical Topologies](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/networking-physical-topologies.html)
- [HGX B300 Networking Logical Architecture](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/network-logical-architecture.html)
- [HGX B300 Node Configurations](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/appendix-node-configurations.html)
- [NVIDIA Fabric Manager: HGX B200/B300 Baseboard](https://docs.nvidia.com/hgx-platforms/fabric-manager-user-guide/index.html)
- [NVIDIA DGX B300 User Guide](https://docs.nvidia.com/dgx/dgxb300-user-guide/)
- [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
