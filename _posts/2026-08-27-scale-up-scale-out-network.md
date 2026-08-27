---
layout: post
title: "GPU 进阶笔记（七）：Scale-up 与 Scale-out——NVLink、UALink、PCIe 和 800G 网络"
description: "拆解 GPU 集群的四层数据路径，厘清 NVLink、UALink、PCIe、InfiniBand、RoCE 和 UEC 的边界。"
date: 2026-08-27 11:00:00 +0800
categories: [ai-gpu]
category_name: "AI 与 GPU"
tags: [NVLink 6, UALink, PCIe 6.0, InfiniBand, RoCE, UEC]
series: "2026 GPU 进阶笔记"
series_part: 3
reading_time: "17 分钟"
---

> 发布建议日期：2026-08-27  
> 关键词：NVLink 6、UALink、Infinity Fabric、PCIe 6.0、InfiniBand、RoCE、UEC、AllReduce

GPU 集群里最常见的误解之一，是把所有高速链路都放进一张“带宽排行榜”。PCIe、NVLink、CXL、InfiniBand 和 Ethernet 的用途、语义、距离与拓扑都不同，不能只按 GB/s 排序。

## 1. 先看一条数据可能经过的路径

```text
GPU HBM
  │ 22 TB/s 级（Rubin 示例）
GPU
  ├─ Scale-up：NVLink / Infinity Fabric / UALink / ICI
  │     └─ 同服务器或同机架内的 GPU
  │
  ├─ Host I/O：PCIe / CXL / NVLink-C2C
  │     └─ CPU、系统内存、NIC、SSD
  │
  └─ Scale-out：InfiniBand / RoCE / UEC Ethernet
        └─ 跨机架、跨 Pod
```

越靠近计算核心，目标通常是更低延迟和更细粒度访问；距离越远，系统更依赖分组网络、拥塞控制和路由。

## 2. 单位先算对：Gb/s 与 GB/s

网络常用 bit/s，GPU 互联和内存常用 Byte/s。忽略协议开销时：

| 标称速率 | 理论单向字节带宽 |
|---:|---:|
| 400Gb/s | 50GB/s |
| 800Gb/s | 100GB/s |
| 1.6Tb/s | 200GB/s |
| 1.8TB/s | 14.4Tb/s |
| 3.6TB/s | 28.8Tb/s |

实际有效带宽还会扣除编码、FEC、协议头、流控、拥塞和软件开销。

## 3. PCIe：通用 I/O，不是 NVLink 的同义词

PCIe 连接 CPU、GPU、NIC 和 SSD。PCIe 6.0 将原始速率提高到 64GT/s，x16 双向聚合带宽最高 256GB/s，并引入 PAM4、轻量 FEC 和 CRC。这里的 256GB/s 是双向合计；单方向约 128GB/s。参见 [PCI-SIG PCIe 6.0](https://pcisig.com/pci-express-6.0-specification)。

典型误区：

- `PCIe 6.0 x16 = 256GB/s` 不表示 GPU 到 NIC 单向可传 256GB/s；
- 多个设备挂在同一个 PCIe switch/CPU root complex 下时可能共享上行；
- 跨 NUMA 会增加延迟并消耗 CPU 间链路；
- GPUDirect RDMA 可以绕过 CPU 数据拷贝，但不会消除 PCIe 物理带宽上限。

CXL 建立在 PCIe 物理层上，重点是缓存一致性、内存扩展与池化。CXL 3.1 扩展了 fabric 管理和跨机架内存能力，但 CXL 内存不能自动替代 GPU 本地 HBM：延迟和带宽等级仍不同。参见 [CXL 3.1 发布说明](https://computeexpresslink.org/wp-content/uploads/2024/01/CXL_3.1-Specification-Release_FINAL.pdf)。

## 4. NVLink：NVIDIA 的 scale-up 路线

| NVLink 代际 | 代表平台 | 每 GPU 双向聚合带宽 | 典型 domain |
|---|---|---:|---:|
| 第 4 代 | Hopper | 900GB/s | 8 GPU |
| 第 5 代 | Blackwell / Blackwell Ultra | 1.8TB/s | 8 或 72 GPU |
| 第 6 代 | Rubin | 3.6TB/s | 8 或 72 GPU |

来源：[NVIDIA NVLink 规格](https://www.nvidia.com/en-us/data-center/nvlink/)。第 6 代及 Rubin 数字仍以官方初步规格为准。

NVLink Switch 的价值不仅是端口速度，而是让多个 GPU 形成非阻塞、全互联的 scale-up domain。Vera Rubin NVL72 的 72 GPU 总聚合 NVLink 带宽为 260TB/s。

注意：`3.6TB/s per GPU` 是 GPU 连接整个 fabric 的聚合双向带宽，不是任意两张 GPU 之间各自独占 3.6TB/s。

## 5. Infinity Fabric、UALink 与 ICI

### 5.1 AMD Infinity Fabric

MI355X 的 8 OAM 平台使用 Infinity Fabric 组成全互联。MI455X 则公布了 3.6TB/s 每 GPU 的 scale-up 带宽，并在 Helios 参考设计中引入 UALink/UALoE 路线。不要把 MI355X 的 8 卡拓扑直接外推为 MI455X 72 卡拓扑；两代产品的系统设计不同。

### 5.2 UALink

UALink 是开放的加速器 scale-up 规范。1.0 版支持最高 200G/lane，目标是在一个 AI pod 内连接最多 1,024 个加速器；x4 station 可提供每方向 800Gb/s 的原始带宽。它使用内存语义，支持 load/store/atomic，并以低延迟短距互联为目标。参见 [UALink 1.0 规范页](https://ualinkconsortium.org/specification/)和 [白皮书](https://ualinkconsortium.org/wp-content/uploads/2025/04/UALink-1.0-White_Paper_v3.pdf)。

规范发布不等于生态立即成熟。交换芯片、NIC/接口 IP、线缆、固件、collective library 和整机互操作都需要时间。UALink 联盟给出的行业常见窗口是规范发布后 1–2 年出现产品。

### 5.3 Google ICI

TPU7x 每芯片 ICI 双向带宽为 1.2TB/s，使用 3D torus 拓扑，最大 Pod 9,216 芯片。Torus 与全互联的编程和性能特征不同：通信映射必须考虑三维拓扑、邻接关系和 hop 数。参见 [TPU7x 系统架构](https://docs.cloud.google.com/tpu/docs/tpu7x)。

## 6. 400G/800G：Scale-out 的主战场

跨机架通常使用 InfiniBand 或 Ethernet/RoCE。2026 年高端平台进入 800Gb/s 时代；NVIDIA Quantum-X800 提供 144 个 800Gb/s 端口，ConnectX 系列最高可向每 GPU 提供 1.6Tb/s 连接。参见 [Quantum-X800](https://www.nvidia.com/en-us/networking/products/infiniband/quantum-x800/)。

但“网卡 800G”不等于端到端 800G：

```text
有效吞吐 = min(
  GPU 到 NIC 的 PCIe/C2C 带宽,
  NIC 线速,
  leaf 上联容量,
  spine 容量,
  对端路径,
  collective 和拥塞控制效率
)
```

还要检查每 GPU 是独占一张 NIC、共享 NIC，还是双端口拆分。

## 7. InfiniBand、RoCE 与 UEC：不要再背固定百分比

“InfiniBand 一定比 RoCE 快 20%”不是可迁移的规律。结果取决于网络代际、拓扑、交换机 buffer、ECN/PFC/DCQCN、adaptive routing、NIC offload、消息大小和集群负载。

- **InfiniBand**：端到端协议和运维体系一致，集合通信与 in-network computing 成熟。
- **RoCEv2**：利用 Ethernet 生态，成本与供应选择更广，但需要严谨的无损/低损设计和拥塞控制。
- **UEC / UET**：在 Ethernet 上为 AI/HPC 重新设计 transport、拥塞控制、安全和 in-network collectives。UEC 1.0 已于 2025 年发布，但部署成熟度仍要按具体产品验证。参见 [UEC 1.0 公告](https://ultraethernet.org/ultra-ethernet-consortium-uec-launches-specification-1-0-transforming-ethernet-for-ai-and-hpc-at-scale/)。

选型应做相同拓扑、相同 collective、相同规模的实测，而不是套用固定比例。

## 8. 用 AllReduce 算一个通信下界

Ring AllReduce 对每个 rank 的理论通信量约为：

```text
bytes_per_rank ≈ 2 × (N - 1) / N × message_size
```

假设 8 卡对 70B 模型的 BF16 梯度做一次 AllReduce，消息大小粗略取 140GB：

```text
2 × 7 / 8 × 140GB = 245GB/rank
```

只用线速除，不计任何协议和同步开销：

| 链路 | 理论传输时间下界 |
|---|---:|
| 400Gb/s（50GB/s） | 4.9s |
| 800Gb/s（100GB/s） | 2.45s |
| 1.8TB/s NVLink 5 每 GPU 聚合（仅作理想上限） | 约 0.136s |

这不是实际 benchmark；NVLink 的聚合链路带宽也不能直接当作 ring 的可持续 bus bandwidth。这个计算只用于说明：训练中的梯度同步若落到 scale-out 网络，链路层级可能造成数量级差异。实际框架会采用分片、reduce-scatter、all-gather、通信计算重叠和多维并行来避免一次搬完整梯度。

## 9. 2026 年组网的三个实用原则

1. **把通信最重的并行维度放进最快的 domain。** Tensor parallel、expert parallel 通常比 data parallel 更怕延迟和带宽不足。
2. **按 rail 对齐 GPU 与 NIC。** 让每个 GPU 或 GPU 组稳定使用最近的 NIC，避免跨 NUMA 和热点。
3. **把平均带宽和尾延迟分开测。** 一条慢链路或一次拥塞就可能让同步 collective 的所有 rank 等待。

GPU 集群不是“计算节点 + 一张大网”。它是 HBM、scale-up、host I/O、scale-out 四层带宽共同构成的流水线，任何一层失衡都会让昂贵的计算单元空转。

## 参考资料

- [PCIe 6.0 Specification](https://pcisig.com/pci-express-6.0-specification)
- [NVIDIA NVLink](https://www.nvidia.com/en-us/data-center/nvlink/)
- [UALink Specifications](https://ualinkconsortium.org/specification/)
- [UEC Specification 1.0](https://ultraethernet.org/uec-1-0-spec)
- [NVIDIA GB300 NVL72 双平面网络参考架构](https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/network-logical-architecture.html)
