---
layout: post
title: "GPU 进阶笔记（六）：从 Roofline 推导 HBM、FP4 与 KV Cache 的真实上限"
description: "不再停留于‘显存更大、位宽更低’：逐项推导 prefill/decode 的算术强度、有效位宽、KV 容量和训练状态。"
date: 2026-08-27 10:00:00 +0800
categories: [ai-gpu]
category_name: "AI 与 GPU"
tags: [Roofline, HBM4, NVFP4, MXFP4, KV Cache, FlashAttention]
series: "2026 GPU 进阶笔记"
series_part: 2
reading_time: "29 分钟"
---

> 资料状态：2026-08-27。文中的上限都是用于定位瓶颈的模型，不是产品性能承诺。

低精度宣传常把三件不同的事压成一个数字：Tensor Core 每秒能做多少乘加、HBM 每秒能搬多少字节、模型在质量约束下实际能使用什么格式。三者没有同时对齐时，“FP4 是 BF16 的四倍”没有端到端含义。

本文从最小性能模型开始，把权重、KV Cache、激活和通信一层层加回来。目标是看到一个 GPU 规格后，可以自行推断它更可能改善 prefill、decode、长上下文并发，还是训练容量。

## 1. Roofline 不是一张图，而是一个判定式

定义算术强度 `I`：

```text
I = 实际执行的 FLOP / 从目标内存层实际搬运的 byte
```

若 GPU 的计算峰值是 `P_peak`，HBM 持续带宽是 `BW_HBM`，则：

```text
P_attainable ≤ min(P_peak, BW_HBM × I)
```

两条屋顶相交处称为 machine balance：

```text
I_ridge = P_peak / BW_HBM   [FLOP/byte]
```

- `I < I_ridge`：即使计算单元无限快，也受 HBM 限制；
- `I > I_ridge`：数据复用足够高，才可能进入计算受限区；
- 正好越过交点也不等于到达峰值，还受 shape、occupancy、调度和非 Tensor Core 算子影响。

这里的 byte 必须针对正确内存层。数据若在 L2 命中，就不应重复计入 HBM；若跨 GPU 读取，又要增加 fabric roof。现代 GPU 实际上是多层 Roofline：

```text
compute roof
  min with register/shared-memory roof
  min with L2 roof
  min with HBM roof
  min with scale-up / scale-out roof
```

## 2. 为什么 prefill 与 decode 像两种不同的工作负载

对含 `P` 个 dense 参数的 Transformer，忽略 embedding 等细节，一 token 前向大约执行 `2P` FLOP。差别在于同一批权重能服务多少 token。

假设每个 decode step 同时处理 `B` 个序列，每参数有效存储 `b` byte，且权重理想地只从 HBM 读取一次：

```text
FLOP per step   ≈ 2P × B
weight bytes    ≈ P × b
I_weight        ≈ 2B / b  FLOP/byte
```

于是：

| 权重格式 | `b` 的理想值 | `I_weight` |
|---|---:|---:|
| BF16 | 2 B | `B` FLOP/B |
| FP8 | 1 B | `2B` FLOP/B |
| 4 bit（暂不计 scale） | 0.5 B | `4B` FLOP/B |

单请求 decode 的 `B=1` 时，哪怕 4 bit 也只有约 4 FLOP/B，通常远低于现代 Tensor Core/HBM 的 machine balance。它天然偏带宽受限。

prefill 则一次把长 prompt 的很多 token 组成大矩阵，权重被大量 token 复用，`B` 可理解为远大的 token batch，算术强度迅速提高。因此同一张卡可能出现：

- prefill 接近计算受限，低精度 Tensor FLOPS 很重要；
- decode 仍受 HBM/collective 限制，更高 FLOPS 贡献有限；
- 增大 continuous batch 能提高 decode 吞吐，但排队和 token 间延迟会恶化。

这就是吞吐与延迟的第一组根本冲突。

## 3. HBM 容量和带宽解决的是两个正交问题

| 代表 GPU | HBM 容量/GPU | 峰值带宽/GPU | 主要意义 |
|---|---:|---:|---|
| H200 | 141GB HBM3e | 4.8TB/s | 更大的 Hopper 容量边界 |
| B300 / MI355X | 288GB HBM3e | 最高 8TB/s | 少切分权重或容纳更多 KV |
| Rubin | 288GB HBM4 | 初步规格最高 22TB/s | 容量不增、带宽大增 |
| MI455X | 432GB HBM4 | 23.3TB/s | 更大单卡容量与 HBM4 带宽 |

容量决定“能不能放下”和“能并发多少”；带宽决定“每轮多久能读完”。容量增加还可能间接减少 tensor parallel degree，从而少做 collective。带宽增加则主要缩短每次数据流动，不会让放不下的模型突然放下。

因此不能用 `GB × TB/s` 之类自创乘积比较 GPU。两个轴要分别进入容量模型和时间模型。

来源：[NVIDIA HGX Components](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)与 [AMD CDNA Architecture](https://www.amd.com/en/technologies/cdna.html)。

## 4. “4 bit 权重占 35GB”为什么只是一阶近似

裸权重大小是：

```text
W_raw = parameter_count × payload_bits / 8
```

70B 参数在纯 4 bit payload 下确实是 `70×10^9×4/8 = 35GB`。但可计算格式需要 scale、分块、对齐，有时还有 zero point。

### 4.1 MXFP4 的有效位宽

OCP MXFP4 使用 E2M1 四位元素，每 32 个元素共享一个 8-bit E8M0 scale。忽略对齐：

```text
effective bits/value = 4 + 8/32 = 4.25 bit
70B 参数 ≈ 70e9 × 4.25/8 = 37.19GB
```

E8M0 scale 是 2 的幂，解码便宜、格式规整，但 scale 没有尾数，粒度相对粗。

### 4.2 NVFP4 的有效位宽

NVFP4 使用 E2M1 元素，每 16 个元素共享一个 E4M3 FP8 scale，另外有 tensor-level FP32 scale：

```text
effective bits/value ≈ 4 + 8/16 = 4.5 bit
70B 参数 ≈ 70e9 × 4.5/8 = 39.38GB
```

更小的 block 降低离群值污染范围，E4M3 scale 也能表达非 2 次幂缩放；代价是 scale 开销更高。每 tensor 的 FP32 scale 对大张量可忽略，但小张量、padding 与存储布局不能忽略。

所以 NVFP4 的优势不能写成“比 4 bit 还小”。它用略高的有效位宽换取更细的缩放与质量；NVIDIA 所说约 3.5× 相对 FP16 的节省，与 `16/4.5≈3.56` 正好一致。

来源：[OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)与 [NVIDIA NVFP4 技术说明](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)。

## 5. 量化误差为什么不能只看 bit 数

对一个 block，量化过程可抽象为：

```text
q_i = round_to_format(x_i / s)
x̂_i = q_i × s
```

误差来源至少有：

1. **clip error**：`x_i/s` 超过格式动态范围；
2. **rounding error**：相邻可表示值间距太大；
3. **scale error**：共享 scale 由 block 的统计量决定；
4. **accumulation error**：乘积虽低精度，累加路径仍有自己的精度；
5. **outlier propagation**：激活离群值让整个 block 的普通值失去分辨率。

block 越小，离群值影响越局部，但 scale 元数据与缩放操作越多；block 越大，吞吐更容易做高，量化误差通常更难控制。这是精度格式设计的核心权衡。

因此比较 FP4 实现要同时写明：

```text
element format + scale format + block size + rounding
+ accumulation precision + dense/sparse + quality metric
```

只写“支持 FP4”无法复现实验。

## 6. KV Cache：容量公式只是开始

对 GQA/MQA Transformer，不计 padding，一 token 的 KV 大小近似：

```text
KV_bytes/token
  = 2 × n_layers × n_kv_heads × head_dim × bytes/element
```

其中 2 代表 K 和 V。若模型有 80 层、8 个 KV head、head dimension 128，KV 使用 BF16：

```text
2 × 80 × 8 × 128 × 2 = 327,680 bytes/token = 320KiB/token
```

于是：

```text
128Ki tokens/request × 320KiB/token = 40GiB/request
32 concurrent requests              = 1.25TiB
```

### 6.1 工程容量还要乘三个系数

实际预留可写成：

```text
KV_reserved = KV_live
              × block_padding_factor
              × scheduler_headroom
              + metadata
```

- PagedAttention 以固定 block 分配，最后一个 block 会产生内部碎片；
- 请求长度分布和抢占策略要求调度余量；
- block table、引用计数和 prefix cache 索引也占空间；
- beam search、speculative decoding 会改变同时存活的分支数；
- tensor/context parallel 会改变每 rank 实际持有的 head 或 sequence 分片。

不能用“平均上下文 × 平均并发”做峰值容量。调度器遇到长尾请求时，平均数会掩盖 OOM。

### 6.2 KV 不只是占容量，也消耗每 token 带宽

标准 attention 在 decode 第 `L` 个 token 时，要读取此前约 `L` 个 token 的 K/V。单层读取量随上下文长度线性增长：

```text
KV_read_per_new_token ≈ L × KV_bytes/token
```

以上述模型、`L=128Ki` 为例，完整层集合的 KV 存量是 40GiB。实际 kernel 会分层读取，且 GQA、分片和 cache 命中改变路径，但总量说明长上下文 decode 可能从“读取权重受限”进一步变成“权重 + KV 共同带宽受限”。

KV 量化因此同时带来容量和带宽收益，但必须评估 attention 输出误差随上下文和任务的累积，不能只看短 benchmark。

## 7. FlashAttention 解决的是 IO 复杂度，不是让 HBM 消失

朴素 attention 会显式物化 `N×N` score matrix，再写回和读出 HBM。FlashAttention 把 Q/K/V 分块装入片上 SRAM，在线维护 softmax 的局部最大值和归一化项，从而避免完整 `N×N` 中间矩阵落到 HBM。

关键不是 FLOP 变少，而是 HBM IO 显著减少：

```text
naive:  materialize S = QKᵀ, P = softmax(S), then PV
tiled:  stream K/V blocks, keep partial softmax state on chip
```

它仍要读取 Q/K/V 和写出结果；decode 时历史 KV 仍需被访问。FlashAttention 抬高了有效算术强度，却没有取消长上下文的线性 KV 流量。

原论文：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)。

## 8. 用 HBM 带宽推导 decode 的绝对上界

若忽略 KV、激活、通信和一切开销，并假设每生成一个 token 至少从 HBM 流式读取一次权重：

```text
tokens/s ≤ sustainable_HBM_bandwidth / effective_weight_bytes
```

对 70B NVFP4，按 4.5 bit 约 39.38GB：

| 峰值 HBM 带宽 | 使用峰值直接相除的上限 |
|---:|---:|
| 8TB/s | 约 203 token/s |
| 22TB/s | 约 559 token/s |
| 23.3TB/s | 约 592 token/s |

以前用 35GB 裸 payload 会得到 229、629、666 token/s，高估约 12.5%。再考虑实际 HBM 利用率 `η`：

```text
tokens/s ≤ η × BW_peak / W_effective,  0 < η < 1
```

端到端还要减去 KV 读取、非 GEMM 算子、kernel launch、同步和 tensor-parallel collective。这个公式的价值不是预测最终结果，而是给出不可违反的物理上界：实测若高于它，说明权重被 cache、跨多 GPU 分摊、batch 复用，或统计口径不同。

### 8.1 多 GPU 不会凭空打破上限

若权重均匀切到 `p` 张 GPU，每张只读 `W/p`，HBM 聚合带宽也近似变为 `p×BW`，理想 token/s 可随 `p` 增长；但每层会引入 collective：

```text
T_token ≳ max(W / (p × BW_effective), T_compute)
          + T_collective_exposed
```

当权重读取时间被压低到和 collective 同量级，继续加 TP rank 收益会快速递减。这就是 scale-up 带宽必须随 HBM 一起增长的原因。

## 9. FP4 training 为什么一定是混合精度配方

训练中存在权重、激活、梯度、优化器一阶/二阶矩、master weight 等不同数值角色。它们对范围、舍入偏差和噪声容忍度完全不同。

NVIDIA 公布的 NVFP4 训练方法本身就说明“FP4 training”不是把所有张量强制成 4 bit：

- 元素是 16-value block 的 E2M1；
- scale 使用 E4M3，并保留 tensor-level scale；
- weight-gradient 路径使用随机 Hadamard 变换削弱 outlier；
- 权重可采用二维 16×16 scaling；
- 部分路径使用 stochastic rounding；
- 累加、归一化和优化器状态保留更高精度。

参考：[Training with NVFP4 on Blackwell](https://developer.nvidia.com/blog/train-models-faster-with-jax-and-maxtext-using-nvfp4-on-nvidia-blackwell/)。

训练显存也不能按 `P×2 byte` 估算。一个未分片的 Adam 混合精度粗略模型可能包含：

```text
BF16 parameters       2P bytes
BF16 gradients        2P bytes
FP32 master weights   4P bytes
FP32 first moment     4P bytes
FP32 second moment    4P bytes
--------------------------------
model states         16P bytes  （示意，取决于实现）
```

还没包括激活、临时 buffer 与通信 workspace。ZeRO/FSDP 把部分状态除以 data-parallel degree，但增加通信；activation checkpointing 降低激活容量，却用重算增加 FLOP。容量优化永远在时间模型里留下痕迹。

## 10. 一份可复现的低精度性能报告应写什么

至少公开以下信息：

1. 模型版本、权重格式、激活格式、KV 格式和累加精度；
2. scale 格式、block size、校准/训练方法；
3. dense 还是 sparse，是否计入结构化稀疏倍数；
4. 输入/输出长度分布，而不只是单个平均值；
5. request rate、continuous batch、并发和调度策略；
6. TTFT、ITL、TPOT、吞吐与 P50/P95/P99；
7. tensor/context/pipeline/expert parallel 配置；
8. 质量门槛，包括长上下文、代码、数学与目标业务集；
9. 实际 HBM 占用、带宽利用率和 power cap；
10. 软件、驱动、kernel 与编译参数版本。

低精度的意义不是把规格表上的 FLOPS 放大，而是以可接受的数值误差换取更低的容量和 IO 成本。只有把有效位宽、数据复用、KV 流量与并行通信放进同一模型，才能知道省下的字节最后有没有变成用户看到的 token/s。

## 参考资料

- [Roofline: An Insightful Visual Performance Model](https://dl.acm.org/doi/10.1145/1498765.1498785)
- [OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [NVIDIA NVFP4 for Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [NVIDIA NVFP4 Training](https://developer.nvidia.com/blog/train-models-faster-with-jax-and-maxtext-using-nvfp4-on-nvidia-blackwell/)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [AMD CDNA Architecture](https://www.amd.com/en/technologies/cdna.html)
