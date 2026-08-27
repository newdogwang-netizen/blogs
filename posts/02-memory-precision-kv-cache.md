# GPU 进阶笔记（六）：HBM4、FP4 与 KV Cache——算力数字为什么经常骗人

> 发布建议日期：2026-08-27  
> 关键词：HBM3e、HBM4、Roofline、FP8、FP4、MXFP4、KV Cache、推理

同一张 GPU 的宣传页上，FP4 峰值可能比 BF16 高很多；但把模型精度从 BF16 改成 FP4，端到端吞吐并不会自动按相同比例增长。

原因是 GPU 性能至少受三个上限约束：计算、显存带宽、通信。2026 年尤其需要理解后两者。

## 1. 从一个最小 Roofline 模型开始

定义算术强度：

```text
Arithmetic Intensity = 运算量 FLOPs / 从 HBM 搬运的字节数 Bytes
```

可达到的性能上限近似为：

```text
Achievable FLOPS <= min(
  峰值计算吞吐,
  HBM 带宽 × Arithmetic Intensity
)
```

- 大 batch 的矩阵乘法复用率高，更容易接近计算上限。
- 单 token decode 需要反复读取大量权重，算术强度低，常常受 HBM 带宽限制。
- MoE 的专家路由可能受 GPU 间互联限制。

所以，“FP4 算力翻倍”只说明计算屋顶变高；若工作负载已经撞在内存屋顶或网络屋顶上，实际速度不会翻倍。

## 2. HBM 代际：容量和带宽同样重要

| 代表产品 | 每 GPU HBM | 每 GPU 峰值 HBM 带宽 | 状态口径 |
|---|---:|---:|---|
| H200 | 141GB HBM3e | 4.8TB/s | 已量产 |
| B200 | 180GB HBM3e | 最高 8TB/s | 已量产 |
| B300 / MI355X | 288GB HBM3e | 最高 8TB/s | 已量产 |
| Rubin | 288GB HBM4 | 最高 22TB/s | 2026 初步规格 |
| MI455X | 432GB HBM4 | 23.3TB/s | 2026 已发布规格 |

NVIDIA 三代数据见 [HGX 参考架构](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)，Rubin 见 [架构技术文章](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)，AMD 见 [MI400 产品页](https://www.amd.com/en/products/accelerators/instinct/mi400.html)。

这里有两个趋势：

1. HBM4 不只是更快，也允许更大的单卡容量；
2. 容量增长可以减少模型切分，带宽增长可以加快每次读取，两者解决的问题不同。

## 3. 先算权重能否放下

最粗略的推理权重容量：

```text
权重字节数 ≈ 参数量 × 每参数字节数
```

以 70B 参数为例：

| 权重精度 | 理论权重大小 |
|---|---:|
| BF16 / FP16 | 140GB |
| FP8 / INT8 | 70GB |
| 4 bit | 35GB |

这只是裸权重。实际部署还需要：

- 量化 scale、zero point 和对齐填充；
- KV Cache；
- CUDA/ROCm context、通信 buffer 和算子 workspace；
- 激活与临时张量；
- 碎片和安全余量。

因此，“70B BF16 是 140GB，所以 141GB H200 刚好单卡能跑”是错误的工程结论。容量规划需要预留余量，通常还要根据推理引擎实测。

## 4. KV Cache 为什么在长上下文时代变成主角

对采用 GQA/MQA 的 Transformer，一层、一个 token 的 K/V 大小可近似为：

```text
KV bytes/token
  = 2 × layers × kv_heads × head_dim × bytes_per_element
```

其中前面的 2 表示 Key 和 Value。

假设一个模型有：80 层、8 个 KV heads、head_dim=128，KV 使用 BF16：

```text
2 × 80 × 8 × 128 × 2 bytes
= 327,680 bytes/token
= 320 KiB/token
```

单条 128K 上下文的 KV Cache：

```text
320 KiB × 131,072 ≈ 40 GiB
```

若有 32 条同样长度的并发请求，未压缩 KV 理论上约为 1.25TiB。这还没有计入权重和运行时空间。

这解释了为什么 2026 年推理平台同时追求：

- 更大的 HBM；
- 低精度 KV Cache（具体位宽与支持情况取决于模型和推理引擎）；
- PagedAttention，减少碎片；
- prefix caching，复用公共前缀；
- disaggregated prefill/decode，把不同阶段放到不同资源池；
- 更快的 scale-up fabric，让 KV 或专家可以跨 GPU 分布。

## 5. FP8、FP6、FP4 不是一个简单的“位数开关”

### 5.1 为什么低精度需要 scale

位数越少，可表示的动态范围和精度越有限。若整块张量只共享一个 scale，离群值会浪费大量表示范围。因此 2026 年常见的方向是 microscaling：把张量切成小 block，每个 block 使用自己的 scale。

OCP 的 MX 规范定义了 MXFP8、MXFP6、MXFP4 和 MXINT8 等格式。它规定交换格式和基本操作，但没有规定所有训练/推理软件必须使用同一种量化算法。参见 [OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。

### 5.2 相同“FP4”标签也未必可直接比较

- NVIDIA 常见 NVFP4；
- AMD CDNA 4/5 支持 OCP MXFP4 等格式；
- 不同实现可能使用不同 block size、scale 表示、累加精度和稀疏性口径；
- 峰值表格可能包含结构化稀疏，也可能是 dense。

因此跨厂商比较时至少要对齐：

```text
数据格式 + dense/sparse + 输入形状 + 累加精度
+ 准确率目标 + 功耗 + 软件版本
```

如果这些条件没写全，“40 PFLOPS > 35 PFLOPS”几乎没有采购意义。

## 6. 训练和推理对精度的容忍度不同

### 6.1 推理

权重量化到 8 bit 或 4 bit 已很常见，但能否保持质量取决于模型、层类型、校准数据和量化方法。通常：

- 权重容易降精度；
- 激活对离群值更敏感；
- KV Cache 可以独立选择精度；
- softmax、归一化、累加等环节常保留更高精度。

### 6.2 训练

训练不仅要存权重，还要存梯度、优化器状态、master weights 和激活。一个常见的 Adam 混合精度粗估是每参数十几字节，而不是 2 字节；具体值取决于是否使用 BF16 master weight、优化器实现、ZeRO/FSDP 分片和激活重计算。

低精度训练也通常是混合精度：矩阵乘使用 FP8/FP6/FP4，累加和部分敏感算子保留 BF16/FP32。不能把“支持 FP4 training”理解成所有张量始终只有 4 bit。

## 7. 用带宽估算 decode 的理论上限

做一个故意简化的下界估算：假设 70B 模型以 4 bit 保存，裸权重约 35GB；每生成一个 token，至少把权重从 HBM 读一遍；忽略 KV、激活、算子开销和带宽利用率。

```text
tokens/s 上限 ≈ HBM GB/s / 权重 GB
```

| GPU HBM 带宽 | 极理想单流上限 |
|---:|---:|
| 8TB/s | 约 229 token/s |
| 22TB/s | 约 629 token/s |

现实值会低得多，因为：

- 峰值 HBM 带宽不等于可持续应用带宽；
- 每层还有 KV、激活和元数据访问；
- kernel launch、同步和非 GEMM 算子占时间；
- 张量并行时增加集合通信；
- 低 batch 下计算单元利用率不足。

但这个估算仍然有用：它说明 decode 场景里 HBM4 的价值可能比更高的 Tensor FLOPS 更直接。

## 8. 一个不容易误导自己的检查表

看到任何“新 GPU 快 N 倍”的说法，依次检查：

1. 是峰值、microbenchmark 还是端到端？
2. 是 dense 还是 sparse？
3. 精度和质量目标是否相同？
4. 比较的是单卡、单机、整架还是整个集群？
5. 是 prefill 吞吐、decode 吞吐、首 token 延迟还是 token 间延迟？
6. batch、输入长度和输出长度是否相同？
7. 功耗和成本是否纳入？

2026 年真正稀缺的不是 FLOPS，而是在给定质量、延迟和功耗约束下，把数据持续喂给计算单元的能力。

## 参考资料

- [OCP MX Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [NVIDIA Rubin GPU Architecture](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/)
- [AMD CDNA 5 Architecture](https://www.amd.com/en/technologies/cdna.html)
- [Google Ironwood codesigned stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack/)
