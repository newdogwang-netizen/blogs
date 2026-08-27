---
layout: post
title: "GPU 进阶笔记（八）：训练与推理选型——从容量估算、通信模型到验收测试"
description: "一套可执行的 GPU 选型与验收方法：先算容量，再看通信，最后用业务基准测试做决定。"
date: 2026-08-27 12:00:00 +0800
categories: [gpu]
tags: [GPU 选型, MLPerf, NCCL, RCCL, 推理, 训练]
series: "2026 GPU 进阶笔记"
series_part: 4
reading_time: "18 分钟"
---

> 发布建议日期：2026-08-27  
> 关键词：GPU 选型、容量规划、MLPerf、NCCL、RCCL、推理吞吐、验收

前面三篇讲了硬件和互联，这一篇把问题落到工程流程：给定一个模型和服务目标，怎样估算 GPU 数，怎样避免买错，机器到货后又怎样验证。

## 1. 先写 SLO，再看 GPU

推理至少要明确：

- 输入/输出长度分布，而不是只有最大 context；
- 首 token 延迟（TTFT）；
- token 间延迟（ITL/TPOT）；
- 每秒请求数或每秒输出 token 数；
- 并发数；
- 目标模型质量与允许的量化方式；
- 可用性和成本目标。

训练至少要明确：

- 参数量、dense/MoE、激活参数量；
- token 总量、目标训练天数；
- 序列长度与 global batch；
- 精度、优化器、激活重计算；
- 数据并行、张量并行、流水线并行、专家并行；
- checkpoint 频率与恢复时间目标。

没有这些输入，“需要多少 GPU”没有可靠答案。

## 2. 第一道门：HBM 容量预算

### 2.1 推理

```text
HBM_required ≈ weights
             + KV_cache
             + activations/workspace
             + communication_buffers
             + runtime_and_fragmentation_margin
```

建议把每一项单独列出来，不要只用“显存利用率 90%”倒推。KV Cache 应按长度分布和并发计算；峰值上下文若极少出现，可用 admission control 或单独资源池处理。

### 2.2 训练

```text
HBM_required ≈ parameters
             + gradients
             + optimizer_states
             + master_weights
             + activations
             + temporary_buffers
```

ZeRO/FSDP 可以分片参数、梯度和优化器状态；tensor/pipeline parallel 会改变激活和通信；activation checkpointing 用额外计算换显存。必须用目标框架配置做小规模 dry run，再外推。

## 3. 第二道门：判断受什么限制

一个简化决策树：

```text
模型放不下？
  ├─ 是：先增加 HBM / 降低精度 / 分片
  └─ 否：性能是否随 batch 明显提升？
       ├─ 是：可能计算利用率不足，尝试 batching / fusion
       └─ 否：检查 HBM 带宽与通信
            ├─ HBM 接近饱和：memory-bound
            ├─ NVLink/NIC 接近饱和：communication-bound
            └─ 都不高：检查 CPU、数据、kernel gap、同步与软件
```

不要只看 `GPU-Util=100%`。它表示采样窗口内有 kernel 在运行，不表示 Tensor Core、HBM 或网络达到了理想效率。

## 4. 三个典型场景

### 场景 A：70B 级模型，长上下文在线推理

主要约束往往是 HBM 容量、HBM 带宽和尾延迟。

合理步骤：

1. 分别计算 BF16/FP8/4bit 权重；
2. 按 P50/P95 输入长度和并发估算 KV；
3. 验证 FP8/FP4 KV 对质量的影响；
4. 若模型单卡可放下，优先避免不必要的 tensor parallel；
5. 若必须切分，把通信留在 NVLink/Infinity Fabric/UALink domain；
6. 分别测 prefill 和 decode，不用一个“tokens/s”掩盖两者。

大 HBM 的 B300、MI355X、Rubin、MI455X 有利于减少切分；但最终选择还取决于目标推理引擎和内核成熟度。

### 场景 B：数百 B 参数 MoE 训练

主要约束是 expert all-to-all、集合通信和故障恢复。

合理步骤：

1. 把 expert parallel 尽量放进高带宽 scale-up domain；
2. 按 top-k 路由和负载不均衡估算最坏通信量；
3. 测 all-to-all，而不只测 all-reduce；
4. 验证计算与通信重叠；
5. 测一个 GPU/NIC/链路故障后的恢复行为；
6. 把 checkpoint 写入速度纳入 step time。

MLPerf Training v6.0 已加入 20B 和 671B MoE 预训练项目，说明公开 benchmark 也开始覆盖稀疏路由工作负载。参见 [MLPerf Training v6.0](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/)和 [参考实现](https://github.com/mlcommons/training)。

### 场景 C：企业微调、RAG 和中小模型混部

主要约束可能变成成本、隔离和运维，而不是最大 scale-up domain。

这类场景往往更适合 2/4/8 卡 PCIe 服务器或云实例。RTX PRO 6000 Blackwell Server Edition 提供 96GB GDDR7 和约 1.6TB/s 带宽，但没有 HBM/NVLink 级卡间能力；其优势是通用形态与较低部署门槛。参见 [NVIDIA RTX PRO 6000 Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)。

不要为了偶发的大任务，把所有常驻业务都部署到 72 卡液冷机架。

## 5. 到货验收：从静态拓扑到端到端

### 5.1 静态检查

NVIDIA 平台常用：

```bash
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi nvlink --status
nvidia-smi -q
```

AMD 平台常用：

```bash
amd-smi list
amd-smi static
amd-smi metric
rocminfo
```

检查：

- GPU、NIC、NVSwitch 数量与料单一致；
- GPU-NIC 的 NUMA/PCIe 亲和性符合设计；
- 链路宽度和速率没有降级；
- HBM 容量、ECC、固件和驱动版本正确；
- 功耗上限与散热配置一致。

### 5.2 单 GPU 基线

- HBM 带宽 microbenchmark；
- GEMM：覆盖 BF16、FP8、FP4 和业务真实 shape；
- 关键算子：attention、MoE、normalization；
- 30–60 分钟持续负载，检查频率、温度、功率与错误计数。

只测最大方阵 GEMM 会高估真实 Transformer 性能。

### 5.3 单机/单 rack 集合通信

NVIDIA 用 NCCL tests，AMD 用 RCCL tests，至少测：

- all-reduce；
- reduce-scatter + all-gather；
- all-to-all；
- 不同消息大小；
- 1、2、4、8、72 GPU 等关键规模。

记录算法带宽和 bus bandwidth，并与同型号健康集群的基线比较。不要把理论链路线速直接当作 NCCL/RCCL 应达到的数值。

### 5.4 跨机架网络

- 单流与多流 RDMA；
- 同 rail 与跨 rail；
- 空载与背景流量下的吞吐和 P99 延迟；
- ECN/PFC、重传、丢包、拥塞标记；
- 任一 leaf/spine 链路降级后的性能。

### 5.5 端到端工作负载

最后必须跑业务模型：

- 固定软件镜像、模型 commit、数据集和随机种子；
- 报告准确率/损失，不只报告速度；
- 同时报告吞吐、延迟、功耗和失败率；
- 预热后运行足够长时间；
- 保存完整配置，保证可复现。

## 6. 如何正确使用 MLPerf

MLPerf Training 测量达到目标质量所需时间；Inference 则区分 Offline、Server 等场景。它比厂商自选 demo 更可比，但仍不能替代你的业务测试。

使用公开结果时注意：

1. Closed 与 Open division 不应混为一谈；
2. 系统规模、软件版本和功耗模式要一致；
3. 不能用 72 GPU 总成绩除以 72 就断言单卡性能；
4. 大规模结果包含网络和软件优化，不只反映芯片；
5. benchmark 模型、序列长度和你的业务可能不同。

MLCommons 的结果规则要求多次独立运行并达到目标质量，比单次最好成绩更可靠。参见 [MLPerf Training 规则](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc)与 [Training v6.0 结果](https://mlcommons.org/benchmarks/training/)。

## 7. TCO：GPU 价格只是第一行

```text
TCO = accelerators + servers + network + optics/cables
    + power + cooling + floor_space
    + software/support + operations
    + downtime + stranded_capacity
```

机架级系统尤其要关注：

- 峰值功率与平均功率；
- 液冷 CDU、管路和漏液检测；
- 光模块功耗与更换成本；
- 备件和维修是否需要整 tray 下线；
- 调度碎片：只有大任务才能利用 72 卡 domain 吗？
- 降级运行：坏一张卡时是否整架不可用？

例如 GB300 NVL72 官方参考架构写明整架最高约 142kW；这样的系统不能像普通 8 卡服务器一样直接塞进传统机房。

## 8. 一份可直接使用的采购/验收表

| 类别 | 必填项 |
|---|---|
| 模型 | 参数量、架构、精度、上下文、质量目标 |
| 服务 | TTFT、TPOT、吞吐、并发、可用性 |
| GPU | 型号、形态、HBM 容量/带宽、功耗 |
| Scale-up | domain 大小、每 GPU 带宽、拓扑、降级能力 |
| Scale-out | 每 GPU NIC、速率、拓扑、超售比、拥塞控制 |
| 主机 | CPU、NUMA、PCIe 代际/宽度、内存、NVMe |
| 软件 | 驱动、固件、CUDA/ROCm、通信库、框架、推理引擎 |
| 设施 | 机架功率、液冷参数、网络/存储端口、消防与监控 |
| 验收 | 单卡、collective、网络、端到端、72 小时稳定性 |
| 商务 | 交付、保修、备件、现场服务、升级与退出成本 |

最终原则：**用容量模型淘汰放不下的方案，用通信模型淘汰扩不动的方案，用业务 benchmark 在剩余方案中做选择。** 这比比较一页峰值规格表可靠得多。

## 参考资料

- [MLPerf Training](https://mlcommons.org/benchmarks/training/)
- [MLPerf Training v6.0 结果公告](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/)
- [AMD MI355X Customer Acceptance Guide](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi355x.html)
- [NVIDIA GB300 NVL72 Reference Architecture](https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/index.html)
