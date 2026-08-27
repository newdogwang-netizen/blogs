# GPU 进阶笔记（五）：2026 数据中心 GPU，为什么从 8 卡服务器进入 72 卡机架

> 发布建议日期：2026-08-27  
> 关键词：Blackwell Ultra、Rubin、MI355X、MI455X、Ironwood、Trainium3、NVL72、Helios

2023 年讨论 GPU 服务器，最自然的单位是“一台 8 卡机”。到 2026 年，只看单卡或单机已经很容易得出错误结论：高端 AI 系统的设计单位正在变成整机架，甚至多个机架组成的 Pod。

变化的本质不是“把更多 GPU 塞在一起”，而是同时重做五件事：计算芯片、HBM、卡间互联、CPU-GPU 互联，以及机架的供电和液冷。

## 1. 先统一几个层级

```text
GPU / Accelerator
  └─ Superchip 或 OAM 模块
       └─ Compute Tray（计算托盘）
            └─ Rack（机架，常见 72 GPU 级）
                 └─ Pod / SuperPOD（多机架集群）
```

- **GPU / accelerator**：真正执行矩阵、向量和标量计算的芯片或模块。
- **Superchip**：CPU 与一片或多片 GPU 通过高带宽一致性链路组合成的模块。
- **Tray**：可插拔的计算单元，通常包含 CPU、GPU、网卡和本地存储。
- **Scale-up domain**：用低延迟、高带宽互联构成的“像一台大机器”的范围。
- **Scale-out domain**：通过数据中心网络把多个 scale-up 单元组成更大的集群。

这里最重要的区分是：**72 卡在一个机架里，不代表 72 卡共享一块物理显存；它代表 72 个独立地址空间之间有足够快的互联，软件能高效做张量并行、专家并行和集合通信。**

## 2. 8 卡机仍然存在，但不再是唯一标准答案

H100/H200 时代的典型思路是：8 张 SXM GPU 通过 NVSwitch 组成单机全互联，再给每张 GPU 配一张 400Gb/s 左右的计算网卡。其优点是边界清晰、故障域较小、部署成熟。

2026 年仍有类似产品。NVIDIA HGX B300 是 8 卡基板；AMD MI355X 平台也是 8 个 OAM 模块。它们适合：

- 企业私有部署；
- 中等规模微调和推理；
- 不希望一次引入百千瓦级机架的机房；
- 需要沿用传统服务器运维模型的团队。

两者的公开硬件要点如下。峰值算力口径不同，故意不放在同一列比较。

| 2025–2026 量产平台 | 单 GPU 显存 | 单 GPU HBM 带宽 | 单机 GPU 数 | 单机总显存 |
|---|---:|---:|---:|---:|
| NVIDIA B300 SXM / HGX B300 | 288GB HBM3e | 最高 8TB/s | 8 | 2.30TB |
| AMD MI355X / UBB 2.0 | 288GB HBM3E | 8TB/s | 8 | 2.30TB |

数据来自 [NVIDIA HGX 参考架构](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)和 [AMD MI355X 产品页](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)。

相似的容量不表示性能相同。还要继续比较：实际支持的数据类型、集合通信效率、内核成熟度、网络、功耗、可获得性和价格。

## 3. Blackwell Ultra：从 HGX B300 到 GB300 NVL72

### 3.1 两种系统边界

同一代 GPU 可以组成两类系统：

```text
HGX B300（传统服务器边界）
  8 GPU ── NVSwitch ── 8 GPU
      │ 每 GPU 800Gb/s 级外部网络
      └────────────── 多机扩展

GB300 NVL72（机架边界）
  18 个 compute tray × 4 GPU = 72 GPU
       │
  9 个 NVLink switch tray
       │
  72 GPU 构成一个非阻塞 NVLink domain
```

GB300 NVL72 把 72 张 Blackwell Ultra GPU 和 36 颗 Grace CPU 放进一个液冷机架。每个计算托盘包含 4 GPU、2 CPU 和 4 个 ConnectX-8 适配器；9 个 NVLink switch tray 负责 72 GPU 的全互联。官方参考架构给出的整架 NVLink 聚合带宽为 130TB/s，整架最大功率约 142kW。参见 [GB300 NVL72 系统组件说明](https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/components.html)。

### 3.2 为什么要把 NVLink 拉到机架范围

MoE 模型的 token 会被路由到不同专家，专家并行产生大量 all-to-all 通信；长上下文推理又要求更大的 KV Cache 和更频繁的数据搬运。若 8 卡之后立刻进入以太网/InfiniBand，通信边界来得太早。

NVL72 的工程选择是：先用高成本、低延迟的 scale-up fabric 覆盖 72 GPU，再用 800G 级网络连接多个机架。这样不是消灭 scale-out，而是把 scale-out 边界向外推。

### 3.3 B300 和 GB300 不是两个 GPU 架构

- B300 通常指 Blackwell Ultra GPU 或 8 卡 HGX 形态。
- GB300 强调 Grace CPU + Blackwell Ultra GPU 的组合。
- GB300 NVL72 是包含 CPU、GPU、NVLink switches、网卡、供电和液冷的整架系统。

采购时若只写“要 B300”，信息远远不够。至少要写清：SXM/HGX 还是 NVL72、每 GPU HBM、NIC 数量和速率、冷却方式、机架功率，以及软件版本。

## 4. Rubin：2026 的下一代机架级平台

截至 2026-08-27，NVIDIA 已宣布 Vera Rubin 进入量产爬坡，但官方规格页仍将详细数值标为 **preliminary**，因此这些数字可用于理解架构，不能直接当作最终验收值。

| 指标 | 单 Rubin GPU | Vera Rubin NVL72 |
|---|---:|---:|
| GPU 数 | 1 | 72 |
| HBM | 288GB HBM4 | 20.7TB HBM4 |
| HBM 带宽 | 最高 22TB/s | 最高 1,580TB/s |
| NVLink | 第 6 代，3.6TB/s/GPU | 260TB/s 聚合 |
| Scale-out 网络 | 官方表列 0.4TB/s/GPU（双向聚合口径） | 28.8TB/s 聚合 |

来源：[Vera Rubin NVL72 规格页](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)与 [NVLink 代际规格](https://www.nvidia.com/en-us/data-center/nvlink/)。

相对 Blackwell Ultra，Rubin 值得关注的不是单一 FLOPS 倍数，而是三条带宽一起增长：

1. HBM3e → HBM4，单 GPU 带宽从约 8TB/s 提到最高 22TB/s；
2. NVLink 5 → NVLink 6，从 1.8TB/s/GPU 提到 3.6TB/s/GPU；
3. 通过 ConnectX-9 把单 GPU scale-out 连接推到 1.6Tb/s 级；若按对称收发链路折算，官方表中的双向聚合写法是 0.4TB/s。

这体现了 2026 年架构设计的关键词：**平衡**。计算单元继续变快，但内存和互联若不同比例增长，低精度 Tensor Core 只会更频繁地等数据。

## 5. AMD：8 卡 MI355X 与 72 卡 Helios

### 5.1 MI355X：传统 8 卡形态中的高容量方案

MI355X 基于 CDNA 4，单卡 288GB HBM3E、8TB/s 带宽、最高 1400W。8 卡 UBB 2.0 平台总计 2.3TB HBM；AMD 的验收文档建议配置 8 张 400G 后端网卡。参见 [MI355X 规格](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)和 [平台验收指南](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi355x.html)。

### 5.2 MI455X + Helios：开放标准路线的机架级设计

MI455X 基于 CDNA 5，官方列出的单卡规格包括 432GB HBM4、23.3TB/s 内存带宽和 3.6TB/s scale-up 带宽。72 卡 Helios 参考设计合计 31TB HBM4、约 1.67PB/s HBM 带宽。

Helios 和 NVL72 的共同点是 72 GPU、液冷、机架级 scale-up；差异在于 Helios 强调 OCP Open Rack Wide、UALink/UALoE 和 UEC 等开放标准。AMD 明确说明 Helios 是参考设计而非直接销售的整机产品，并预计合作伙伴系统在 2026 年下半年规模部署。参见 [AMD MI400/Helios 产品页](https://www.amd.com/en/products/accelerators/instinct/mi400.html)。

因此，2026 年 8 月更严谨的状态描述是：

- MI355X：已发布并进入实际平台部署；
- MI455X：已正式列出产品规格；
- Helios：参考设计已公布，合作伙伴量产部署处于 2026 下半年计划窗口。

不能把“参考设计公布”写成“所有 OEM 整机已经普遍现货”。

## 6. 不要把所有 AI 加速器都叫 GPU

### 6.1 Google TPU7x（Ironwood）

Ironwood 是第七代 TPU，不是 GPU。Google Cloud 文档列出的单芯片规格为 192GiB HBM、7.38TB/s HBM 带宽、4.614PFLOPS FP8、1.2TB/s 双向 ICI；最大 Pod 为 9,216 芯片，采用 3D torus 拓扑。它支持 JAX 和 PyTorch，但官方页面明确写明 TPU7x 不支持 TensorFlow。参见 [TPU7x 文档](https://docs.cloud.google.com/tpu/docs/tpu7x)。

TPU 的优势来自硬件与 XLA 编译器、JAX/PyTorch-XLA、Pallas 内核的共同设计。迁移成本也主要发生在这个软件边界，而不是“有没有类似 CUDA core”。

### 6.2 AWS Trainium3

Trainium3 同样不是 GPU。AWS 公布的单芯片规格是 144GB HBM3e、4.9TB/s、2.52PFLOPS FP8；Trn3 UltraServer 可扩到 144 颗芯片，并通过 Neuron SDK 使用。参见 [EC2 Trn3 UltraServers GA 公告](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/)。

选择这类云专用加速器时，关键问题不是硬件是否“更快”，而是：模型算子覆盖率、编译时间、调试工具、分布式运行时、云锁定程度和实际价格。

## 7. 2026 年选型时先问什么

不要先问“哪张卡 FLOPS 最高”，先问下面七个问题：

1. 模型权重、KV Cache、激活和优化器状态合计需要多少 HBM？
2. 工作负载是 prefill 重、decode 重、MoE 通信重，还是 FP64 重？
3. 需要的 scale-up domain 是 8、72，还是更大？
4. 超出 scale-up domain 后，每 GPU 有多少真实可用网络带宽？
5. 软件栈是否已有稳定内核和已验证模型？
6. 机房能否提供 100kW 以上机架供电和液冷？
7. 故障时能否降级运行，备件和现场服务是否跟得上？

结论很朴素：2026 年的“GPU 产品”已经是芯片、内存、互联、网络、冷却和软件共同组成的系统。单卡参数只回答了其中很小一部分。

## 参考资料

- [原始文章：GPU 进阶笔记（一）](https://arthurchiao.art/blog/gpu-advanced-notes-1-zh/)
- [NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)
- [NVIDIA Vera Rubin 平台](https://www.nvidia.com/en-us/data-center/technologies/rubin/)
- [AMD MI350 Series](https://www.amd.com/en/products/accelerators/instinct/mi350.html)
- [AMD MI400 Series / Helios](https://www.amd.com/en/products/accelerators/instinct/mi400.html)
- [Google Cloud TPU7x](https://docs.cloud.google.com/tpu/docs/tpu7x)
- [AWS Trainium3](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/)
