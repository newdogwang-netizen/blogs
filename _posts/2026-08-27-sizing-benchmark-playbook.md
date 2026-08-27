---
layout: post
title: "GPU 进阶笔记（八）：从 SLO 反推 GPU 集群——容量、并行、排队与验收"
description: "一套面向生产的反向设计方法：从请求分布和训练期限出发，推导 HBM、并行度、网络预算、尾延迟和故障恢复。"
date: 2026-08-27 12:00:00 +0800
categories: [ai-gpu]
category_name: "AI 与 GPU"
tags: [容量规划, GPU 集群, NCCL, MLPerf, 推理 SLO, TCO]
series: "2026 GPU 进阶笔记"
series_part: 4
reading_time: "34 分钟"
---

> 资料状态：2026-08-27。本文不提供一个脱离 workload 的“GPU 数量答案”，而提供得到答案的方法和验收条件。

采购流程若从厂商 SKU 开始，团队会被迫用业务去解释已经选定的硬件。更可靠的顺序是反过来：先把质量、延迟、吞吐和完成期限写成约束，再用容量模型、时间模型和排队模型排除不可能方案，最后用业务 benchmark 决定。

```text
workload distribution + quality target + SLO
  → memory feasibility
  → single-GPU service time
  → parallelism & communication
  → queueing/headroom
  → cluster/facility/TCO
  → acceptance test
```

## 1. 把需求写成分布，不是一个“典型值”

### 1.1 在线推理输入

至少固定：

- 模型 commit、tokenizer、量化与质量门槛；
- prompt tokens 和 output tokens 的 P50/P90/P95/P99；
- 到达率的分钟级/秒级分布与 burst；
- TTFT、TPOT/ITL、端到端 latency 的 percentile SLO；
- streaming/non-streaming、priority、取消率；
- prefix 重复率、工具调用和多轮会话比例；
- 可用性目标与允许的 admission control。

一个“平均 4K context”可能同时代表所有请求约 4K，也可能是 99% 为 1K、1% 为 301K。两者的 KV 峰值、调度和尾延迟完全不同。

### 1.2 训练输入

至少固定：

- dense/MoE 参数、层数、hidden size、heads/experts、top-k；
- sequence length、global/micro batch；
- 训练 token 总量与 deadline；
- 目标 loss/质量，不只是固定 step 数；
- optimizer、精度、activation checkpointing；
- checkpoint 周期、大小、恢复时间目标；
- 可容忍的失败率和抢占策略。

“一万张 GPU 训练多久”仍没有答案，因为 step 里实际有多少 token、每 GPU 的 microbatch 是否高效、并行通信是否暴露都未知。

## 2. 第一张表：HBM 可行性预算

### 2.1 推理预算

对每个 replica：

```text
M_total = M_weights
        + M_KV_live
        + M_activations
        + M_workspace
        + M_collective
        + M_runtime
        + M_fragmentation/headroom
```

不能把最后几项合并成拍脑袋的“10%”。它们会随 engine、CUDA graph、kernel、max batch、TP degree 和 allocator 改变。做法是：公式算权重/KV，目标软件 dry run 测其余项，再保留已解释的 headroom。

对 GQA/MQA：

```text
KV/token = 2 × layers × kv_heads × head_dim × element_bytes
KV_live  = Σ(active_request_tokens × KV/token)
```

还要考虑 block padding、prefix cache 引用、speculative branch 和 scheduler reserve。

### 2.2 训练预算

可写成每个并行 rank 的分项：

```text
M_rank = M_model_states(TP, DP_shard)
       + M_activations(micro_batch, seq, PP, CP)
       + M_temp(shape, kernels)
       + M_communication(bucket_size)
```

ZeRO/FSDP 不是免费除法：

- shard optimizer/gradient/parameter 可以降低 `M_model_states`；
- 参数 AllGather 和梯度 ReduceScatter 增加通信；
- 更小 bucket 降低峰值内存但增加固定调用次数；
- activation checkpointing 降低保存量但增加重算；
- sequence/context parallel 降低每 rank 激活，却引入额外 collective。

容量方案必须同时进入 step-time 模型，否则很容易得到“放得下但慢到不可用”的配置。

## 3. 第二张表：单请求/单 step 时间分解

推理可拆为：

```text
TTFT ≈ T_queue + T_tokenize + T_prefill + T_first_collective
TPOT ≈ max(T_weight/HBM, T_compute, T_KV, T_collective)
       + T_nonoverlapped
```

训练则是：

```text
T_step = T_forward + T_backward + T_optimizer
       + T_exposed_communication
       + T_input/checkpoint
       + T_bubbles
```

`T_exposed_communication` 只计算不能被 kernel 隐藏的部分。看到网络计数器忙不代表它拖慢 step；反过来，网络利用率不高也可能因许多小 collective 的 `α` 延迟让 GPU 等待。

### 3.1 不要用 GPU Utilization 代替时间线

`GPU-Util=100%` 只说明采样窗口内有 kernel。可能是：

- 小 kernel 串行发射，Tensor Core 利用率很低；
- memory stall 很高；
- 一个 rank 在工作，其余 rank 等 barrier；
- 通信 kernel 正在 spin；
- shape 太窄，occupancy/SM efficiency 很差。

需要 profiler 时间线同时看：SM/Tensor activity、HBM bytes、kernel gaps、NCCL/RCCL、CPU launch、NIC counters 和最慢 rank。

## 4. 从单 replica 吞吐推导集群规模

若实测单 replica 在满足延迟 SLO 的条件下可稳定处理 `μ` requests/s，峰值目标到达率是 `λ_peak`，不要简单取 `ceil(λ/μ)`。同步服务的尾延迟在利用率接近 100% 时会非线性恶化。

先定义安全利用率 `ρ_target`：

```text
replicas_traffic ≥ ceil(λ_peak / (μ × ρ_target))
```

再同时满足容量、可用性和维护约束：

```text
replicas_final = max(
  replicas_traffic,
  replicas_KV_capacity,
  replicas_failure_domain,
  replicas_maintenance
)
```

`ρ_target` 不能套固定 70%。服务时间方差、burst、batch scheduler、priority 和 SLO 越严格，所需余量越大。应使用生产 trace 回放或离散事件模拟求得。

### 4.1 Continuous batching 的本质

增加 batch 可提高权重复用和 decode 算术强度，但也会：

- 等待凑批，提高 queue/TTFT；
- 让长请求占据 decode slot；
- 增大同时存活 KV；
- 在 TP collective 中产生更大的同步组；
- 让 P99 被少数长序列拖慢。

所以应画出 Pareto front：横轴 token throughput，纵轴 TTFT/TPOT P99，而不是只找吞吐最大点。

## 5. 并行度不是乘法表，而是拓扑映射问题

总 GPU 数通常写为：

```text
world_size = DP × TP × PP × CP × EP
```

这个乘式只保证 rank 数相乘正确，不保证放置合理。原则是把通信最频繁、最怕延迟的维度放进最快的物理域：

| 并行维度 | 典型通信 | 优先物理域 | 主要风险 |
|---|---|---|---|
| TP | 每层 AllReduce/AllGather | NVLink/UALink scale-up | GEMM 变窄、小消息延迟 |
| EP | dispatch/combine All-to-All | 大 scale-up 域 | 路由倾斜、热点 |
| CP/SP | K/V 或 activation exchange | scale-up，必要时 scale-out | 随序列增长 |
| PP | stage 间 activation P2P | 同架或相邻 rail | pipeline bubble |
| DP | gradient ReduceScatter/AllReduce | 跨架 scale-out | 大消息、可重叠性 |

机架级 72-GPU fabric 的核心用途，是让 TP/EP/CP group 不必在 8 rank 后立刻进入 HCA 网络。它并不意味着最优 job 一定使用 72 作为某个维度。

### 5.1 Pipeline bubble 的一阶检查

若有 `p` 个 pipeline stages、`m` 个 microbatches，最简单 schedule 的 bubble fraction 大致随：

```text
bubble ≈ (p - 1) / (m + p - 1)
```

增大 `m` 可摊薄 bubble，却增加激活存活和调度复杂度。stage 计算不均衡时，真正瓶颈是最慢 stage，不是平均 FLOPS。

### 5.2 MoE 不能只用激活参数量估算

MoE 每 token 只激活少数专家，计算 FLOP 低于同总参数 dense 模型，但所有专家权重仍占容量，dispatch/combine 又产生 All-to-All。需要额外记录：

- tokens per expert 的分布和最大/均值；
- capacity factor 与 dropped tokens；
- 本地/远端 expert 命中比例；
- expert GEMM 的 batch size；
- All-to-All 暴露时间。

平均负载平衡不能保证 P99 step；一个热点 expert 足以让所有 rank 等待。

## 6. 训练规模怎样从 deadline 反推

若总训练 token 为 `D`，目标天数 `T_days`，所需持续 global token throughput：

```text
throughput_required = D / (T_days × 86400)
```

若每 GPU 在目标并行配置下的有效吞吐是 `q` token/s，系统效率是 `η_scale`，初始 GPU 数：

```text
N ≈ throughput_required / (q × η_scale)
```

`η_scale` 不是常数。它包含通信、pipeline bubble、straggler、checkpoint、故障恢复和调度空洞，并随 N 增大而下降。

还要加入可用训练时间：

```text
effective_progress = wall_time
                   × scheduler_availability
                   × job_uptime
                   × useful_step_fraction
```

若一个 10,000-GPU 作业每隔数小时失败一次，而 checkpoint/重启耗时很长，峰值吞吐再高也可能输给规模更小但稳定的方案。

## 7. 验收必须像剥洋葱一样逐层建立基线

### 7.1 Layer 0：料单、固件与拓扑

验证：

- GPU/CPU/NVSwitch/HCA/SSD 数量和型号；
- PCIe generation/width、NVLink lane、HCA port rate；
- GPU↔CPU↔NIC NUMA affinity 与 rail cabling；
- firmware、driver、CUDA/ROCm、collective library 的兼容矩阵；
- power cap、温度、冷却流量、ECC 和历史错误计数；
- 时间同步与 telemetry timestamp。

只要这一层不一致，后面的性能差异就无法归因。

### 7.2 Layer 1：单 GPU roof

测：

- HBM read/write/copy 的持续带宽和 NUMA 路径；
- 业务真实 M/N/K shape 的 BF16/FP8/FP4 GEMM；
- attention（prefill/decode、不同 seq/head）；
- MoE grouped GEMM、normalization、embedding；
- 1 小时稳态 power/thermal/frequency/error。

最大方阵 GEMM 只证明峰值路径正常，不能代表 Transformer shape。

### 7.3 Layer 2：P2P 与 scale-up fabric

扫描：

- 每个 GPU pair 的单向/双向 read/write；
- 近端/远端、同节点/跨 OS 域；
- 8B 到数 GiB 的 latency-bandwidth curve；
- 同时多个 pair 的 bisection 与热点流量；
- IMEX/Fabric Manager 等控制面重启与恢复。

pairwise matrix 能发现一条降级 lane 或错误映射；只测 aggregate 会把坏点平均掉。

### 7.4 Layer 3：collectives

NCCL/RCCL tests 至少覆盖：

- AllReduce、AllGather、ReduceScatter、All-to-All；
- 1/2/4/8/整架/跨架规模；
- latency size 与 bandwidth size；
- 单 job 与多 job、空载与背景流量；
- `time`、`algbw`、`busbw`、P95/P99/max rank。

硬件 reduction/hierarchical 算法下，`busbw` 不再是物理线速。验收门槛应以健康同构系统、固定算法和 operation time 为基线。

### 7.5 Layer 4：网络与故障

把 RDMA 测试与交换机/NIC telemetry 对齐，验证：

- 同 rail、跨 rail、单流、多流；
- incast、all-to-all、background traffic；
- queue depth、ECN、PFC pause、retransmit/replay、FEC error；
- leaf/spine/HCA link failure 后的收敛、带宽与 packet ordering；
- 一条慢链路能否被快速定位。

### 7.6 Layer 5：业务模型

推理报告必须同时包含：

- quality、TTFT/TPOT/E2E P50/P95/P99；
- request/s、input/output token/s；
- context/output 分布、并发、batch policy；
- HBM/KV 使用、功率、失败/拒绝率。

训练报告必须包含：

- samples/tokens per second 与 time-to-quality；
- step-time breakdown 和 rank variance；
- achieved FLOPS、HBM、collective overlap；
- checkpoint time、故障恢复、有效训练占比；
- 软件镜像、模型/数据 commit 与随机种子。

## 8. 故障注入比 72 小时空跑更有信息量

长时间 soak test 能发现热、供电和偶发错误，但不能证明恢复路径。应在可控环境注入：

- 单 GPU Xid/ECC 隔离；
- 一个 HCA/rail/交换端口 down；
- compute tray/node 重启；
- Fabric Manager/IMEX 控制面重启；
- checkpoint 写入抖动或存储不可用；
- scheduler kill/preempt 与作业弹性恢复。

对每项记录：检测时间、隔离范围、作业表现、重试语义、数据一致性、恢复时间和人工步骤。目标不是证明“永不故障”，而是证明故障影响面与恢复成本已知。

## 9. 怎样使用 MLPerf，而不是被榜单使用

MLPerf Training 的核心指标是达到既定质量所需时间；Inference 区分 Offline、Server 等场景并有准确率约束。它提供比厂商自选 demo 更好的可比性，但不能直接替代业务 trace。

读结果时对齐：

1. Closed/Open division 与 benchmark 版本；
2. 系统 GPU 数、功耗模式、软件版本；
3. 模型/sequence/batch 与精度质量；
4. availability：Preview、Available 还是 cloud；
5. 是否是完整系统结果，不能把 72-GPU 成绩直接除以 72；
6. 结果反映芯片、网络、软件共同优化，不是芯片 IPC。

正确用法是：用 MLPerf 校验供应商方案是否具备成熟软件路径，再用你的输入分布和 SLO 做最终选择。[MLCommons Training](https://mlcommons.org/benchmarks/training/)与 [Training Rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc)。

## 10. TCO 要除以 useful work，不是除以 GPU 数

年度成本：

```text
TCO = amortized_hardware
    + network/optics/storage
    + energy/cooling/facility
    + software/support/people
    + downtime/recovery
    + stranded_capacity
```

更有意义的单位成本：

```text
inference: cost / accepted output token at target SLO & quality
training:  cost / useful training token, or cost to target quality
```

`useful` 会扣掉：超 SLO 请求、质量不达标结果、失败后回滚的 step、空闲碎片和重复计算。

例如 72-GPU rack 若调度器只能接纳 72-rank 大 job，小任务会造成整架碎片；若能安全切分为多个 fabric partition，又要验证租户隔离、带宽争用和故障影响面。硬件峰值利用率与业务有效利用率不是同一个指标。

## 11. 一份可以签字的验收矩阵

| 层级 | 指标 | 条件 | 门槛来源 | 失败动作 |
|---|---|---|---|---|
| 物理 | lane/port/ECC/thermal | 全量、稳态 | 厂商规格 + golden rack | 更换/重训/降级 |
| P2P | pair bandwidth/latency | 全 pair、全 size | 同构健康基线 | 定位链路/NUMA |
| Collective | time/algbw/tail | 多 scale、多 op | golden image | 算法/拓扑诊断 |
| Network | throughput/P99/ECN | 空载+拥塞+故障 | 网络设计 SLO | 调整路由/拥塞控制 |
| Inference | TTFT/TPOT/quality | 生产 trace | 业务 SLO | 调整 engine/capacity |
| Training | time-to-quality/uptime | 固定 runbook | 项目 deadline | 重做并行/恢复策略 |
| Facility | power/cooling/leak | 峰值与瞬态 | 设施设计 | 限功率/整改 |

每个门槛都应包含软件/固件版本、重复次数、统计方法和原始数据留存位置。没有版本和统计口径的“通过”无法在下一次升级后复验。

## 结语：先证明因果，再购买峰值

一个可用的 GPU 集群，不是所有 microbenchmark 都跑到最大值，而是能解释业务 SLO 如何由 HBM、计算、collective、排队和故障共同决定。

容量模型淘汰放不下的方案；时间模型淘汰单请求太慢的方案；通信与拓扑模型淘汰扩不动的方案；排队模型给出生产余量；故障注入和业务 benchmark 最后证明它能持续完成 useful work。到这一步，GPU 型号只是答案的一部分，而不是问题的起点。

## 参考资料

- [NVIDIA GB300 NVL72 Reference Architecture](https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/index.html)
- [NCCL Tests Performance Guide](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- [AMD MI355X System Acceptance](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi355x.html)
- [MLPerf Training](https://mlcommons.org/benchmarks/training/)
- [Megatron-LM Efficient Large-Scale Training](https://arxiv.org/abs/2104.04473)
