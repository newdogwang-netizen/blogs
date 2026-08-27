---
layout: post
title: "GPU 进阶笔记（二）：从4节点SU到18台HGX B300集群"
description: "依据NVIDIA HGX AI Factory参考架构，推导18节点、144 GPU的双平面Spectrum-X、rail、存储、管理与验收设计。"
date: 2026-08-27 14:00:00 +0800
categories: [ai-gpu]
category_name: "AI 与 GPU"
tags: [HGX B300, Scalable Unit, Spectrum-X, Rail, 集群设计]
series: "2026 GPU 进阶笔记"
series_part: 2
reading_time: "34 分钟"
---

> 资料状态：2026-08-27。本文假设已有18台完整、通过认证的HGX B300服务器，目标是训练、微调和分布式推理，计算网络采用Spectrum-X Ethernet双平面。具体OEM机型、交换机SKU、光模块和固件矩阵仍需供应商最终确认。

NVIDIA公开的HGX AI Factory参考架构没有“18节点标准BOM”。官方定义的最小模块是4台服务器组成一个Scalable Unit（SU），完整列出的设计点是32、64和128节点。

这不表示18台不能组网。正确方法是先保留官方的节点、SU、rail和fabric边界，再决定是为18台定制网络，还是把官方32节点设计少插14台。本文选择后者作为主方案：成本较高，但最接近已经验证的架构，也给未来扩到32节点留下空间。

为了避免把推导写成官方承诺，全文使用三个标签：

- **官方明确**：NVIDIA参考架构直接给出的规格或拓扑；
- **本文推导**：对官方端口数做算术或针对18节点的映射；
- **厂商确认**：必须结合交换机/OEM SKU完成的最终设计。

## 1. 先确定我们正在搭建什么

单台HGX B300采用`2-8-9-800`参考配置：

```text
2   × x86 CPU
8   × B300 GPU
8   × ConnectX-8 SuperNIC（GPU Compute E/W）
1   × BlueField-3 DPU（Converged N/S）
800 × Gb/s average network bandwidth per GPU
```

18台合计：

| 资源 | 单节点 | 18节点 |
|---|---:|---:|
| B300 GPU | 8 | 144 |
| GPU HBM3e | 2.304TB | 41.472TB |
| ConnectX-8 | 8 | 144 |
| BlueField-3 | 1 | 18 |
| 节点内NVLink domain | 1×8 GPU | 18个独立domain |
| 节点内NVLink聚合 | 14.4TB/s | 259.2TB/s，仅数学合计 |

最后一行尤其容易误读。18台机器的NVLink没有跨服务器相连，所以不存在一个259.2TB/s的144-GPU scale-up fabric。真正结构是：

```text
18 × [8-GPU NVSwitch domain]
              │
       ConnectX-8 / RoCE
              │
      Spectrum-X scale-out
```

跨节点的tensor、gradient和MoE token必须离开本机NVSwitch，进入ConnectX-8和Ethernet fabric。

官方节点说明：[HGX B300 Components](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)。

## 2. 官方4节点SU：它是网络积木，不是新的共享域

**官方明确**：一个SU由4台HGX B300 compute node组成。按每台8张GPU、8张双400G ConnectX-8和一张双400G BlueField-3计算：

```text
Compute:
4 nodes × 8 CX-8 × 2 × 400Gb/s
= 64 × 400Gb/s
= 25.6Tb/s aggregate port rate

Converged N/S:
4 nodes × 1 BF3 × 2 × 400Gb/s
= 8 × 400Gb/s
= 3.2Tb/s aggregate port rate

OOB:
4 nodes × 6 × 1Gb/s
= 24 × 1Gb/s management endpoints
```

<figure class="architecture-figure">
  <a href="{{ '/assets/images/hgx-b300-4-node-su.svg' | relative_url }}">
    <img src="{{ '/assets/images/hgx-b300-4-node-su.svg' | relative_url }}" alt="四台HGX B300组成的Scalable Unit，分别接入两个GPU计算平面、南北向网络和带外管理网络">
  </a>
  <figcaption>图1：依据NVIDIA 4-node SU端口说明重绘。点击查看SVG原图。</figcaption>
</figure>

SU解决的是模块化复制：交换端口、线缆、机架、电源和管理容量都可以按4节点成组规划。它**不会**把4台服务器变成32-GPU NVLink域，也不要求作业必须以32 GPU为单位调度。

为什么选择4而不是8或16？原因来自网络工程，而不是模型数学：

- 4节点产生64条400G compute连接，适合高radix交换机的端口切分；
- 4节点产生8条400G N/S连接，可以形成规则的冗余块；
- SU可以在不同机架之间复制，rail编号保持不变；
- 故障、布线和验收可以先以SU为范围完成，再扩到全fabric。

官方定义与算术见：[Enterprise RA Scalable Unit](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/networking-logical-architecture.html#enterprise-ra-scalable-unit-su)。

## 3. 18节点怎样映射到SU

**本文推导**：

```text
18 nodes = 4 × full SU + 2 nodes
```

可以形成：

| 逻辑单元 | 节点 | GPU | 状态 |
|---|---|---:|---|
| SU 01 | 01–04 | 32 | 完整SU |
| SU 02 | 05–08 | 32 | 完整SU |
| SU 03 | 09–12 | 32 | 完整SU |
| SU 04 | 13–16 | 32 | 完整SU |
| Partial SU | 17–18 | 16 | 非官方标准SU |

节点17–18不会因为“少两台”而无法工作，只是最后一个容量块不完整。建议在地址、交换端口、电源、机架和调度标签中给节点19–20预留位置，使下一次扩容能先补成第5个完整SU。

需要区分三种边界：

```text
NVLink boundary     = 8 GPU / node
SU planning block   = 4 nodes / 32 GPU
job boundary        = 由DP/TP/PP/EP策略决定，可跨多个SU
```

调度器不应机械地把一个SU视为不可切分的作业单元。

## 4. 推荐方案：缩配官方32节点双平面架构

**官方明确**：32节点参考设计的GPU compute fabric总计8台leaf、4台spine；每个plane分别为4 leaf + 2 spine。每张ConnectX-8的800G接口拆成两个400G，一个接Plane A，一个接Plane B。官方把rail配对为`1+5、2+6、3+7、4+8`。

**本文推导**：保留这12台compute switch和全部spine能力，只接18节点。其余服务器端口留作19–32节点扩容。

<figure class="architecture-figure">
  <a href="{{ '/assets/images/hgx-b300-18-node-cluster.svg' | relative_url }}">
    <img src="{{ '/assets/images/hgx-b300-18-node-cluster.svg' | relative_url }}" alt="18台HGX B300通过两个独立的四叶两脊Spectrum-X计算平面连接，同时连接南北向和带外管理网络">
  </a>
  <figcaption>图2：18节点建议拓扑。交换层沿用官方32节点逻辑，节点层为4个完整SU加1个partial SU。点击查看SVG原图。</figcaption>
</figure>

总体连接关系：

```text
                     ┌─ Compute Plane A: 4 Leaf + 2 Spine
18 × HGX B300 ───────┤
每台8×CX-8 2×400G    └─ Compute Plane B: 4 Leaf + 2 Spine

每台BlueField-3 ───── 2 × Converged N/S Switch

BMC/NIC mgmt ───────── OOB management switches
```

该方案的价值不只是扩容余量：

- Plane A与B拥有独立交换机和故障路径；
- NCCL可以同时利用两侧带宽；
- 一侧故障后仍保留GPU间连接，带宽按受损路径下降；
- 保留官方rail mapping，减少自定义拓扑带来的软件适配；
- 将来扩到32节点时不需要重建核心fabric。

代价是交换机、光模块和空闲端口较多。如果未来确定永远停在18节点，可以让NVIDIA网络合作伙伴给出定制Clos；但那不再是官方公开表格中的完整设计点。

## 5. 18节点计算网络的端口账

每台8张ConnectX-8，每张CX-8拆为两个400G：

```text
Plane A = 18 × 8 × 400Gb/s = 144×400G = 57.6Tb/s
Plane B = 18 × 8 × 400Gb/s = 144×400G = 57.6Tb/s
Both    = 288×400G          = 115.2Tb/s
```

换成byte/s：

```text
57.6Tb/s ÷ 8 = 7.2TB/s per plane line-rate aggregate
两个plane数学合计 = 14.4TB/s
```

这与单节点14.4TB/s NVLink聚合数字相同纯属口径巧合：前者是18节点的Ethernet端口合计，后者是单节点8 GPU的NVLink双向端口合计。协议、方向、延迟和可用语义完全不同。

建议建立正式的cabling matrix，至少包含：

| Node | GPU/CX index | Plane A port | Plane B port | Rail | NUMA |
|---|---:|---|---|---:|---:|
| hgx01 | 0 | leaf-a1/pXX | leaf-b1/pXX | 1 | OEM实测 |
| hgx01 | 1 | leaf-a2/pXX | leaf-b2/pXX | 2 | OEM实测 |
| … | … | … | … | … | … |
| hgx18 | 7 | leaf-a4/pXX | leaf-b4/pXX | 8 | OEM实测 |

端口号不能仅凭GPU编号猜测，要从OEM PCIe topology、NIC BDF和实际线缆反查。

## 6. Rail为什么配成1+5、2+6、3+7、4+8

每个local GPU index在所有节点中形成一条rail：

```text
Rail 1: hgx01.gpu0/nic0 ... hgx18.gpu0/nic0
Rail 2: hgx01.gpu1/nic1 ... hgx18.gpu1/nic1
...
Rail 8: hgx01.gpu7/nic7 ... hgx18.gpu7/nic7
```

官方32节点设计让一台leaf承载两条相隔4的rail：

```text
Leaf 1 → Rail 1 + 5
Leaf 2 → Rail 2 + 6
Leaf 3 → Rail 3 + 7
Leaf 4 → Rail 4 + 8
```

分层AllReduce可以先在每个8-GPU节点内经NVSwitch做ReduceScatter，再让8张NIC分别跨节点交换对应shard，最后回到节点内AllGather。

这样设计的目的不是禁止跨rail流量，而是让主通信路径有确定的GPU↔NIC亲和性，减少leaf/spine上的横向洗牌。需要同时满足：

- 容器中的GPU/NIC编号与宿主拓扑一致；
- Slurm/Kubernetes device plugin不随意重排设备；
- NCCL能读取正确的拓扑和多NIC信息；
- 两个plane的软件配置、MTU和固件版本一致；
- 一个rail降级时，告警能关联到具体GPU index。

## 7. 交换机BOM：哪些可以确定，哪些不能

| 角色 | 规划数量 | 依据 | 确定性 |
|---|---:|---|---|
| Compute leaf | 8 | 32-node参考架构，两个plane合计 | 官方数量，18节点缩配使用 |
| Compute spine | 4 | 32-node参考架构，两个plane合计 | 官方数量，18节点缩配使用 |
| N/S leaf | 2起 | 32-node参考设计为双交换机 | 需加入存储、控制、上游端口 |
| OOB leaf | 3台数学下限，建议4台 | 约108个node management endpoints | OEM端口数量确认后决定 |
| Control plane | 4–7台 | BCM/Slurm/K8s是否同时部署 | 软件方案决定 |
| Optics/cables | 待定 | 距离、交换机SKU、single/dual-port optic | 必须由厂商出表 |

当前参考架构逻辑表列出SN5600 128×400G交换机；网络硬件章节也介绍更新的800G交换平台。两类设备的OSFP、breakout和spine uplink计数不能照抄互换。

此外，官方32节点表中的uplink和transceiver列混合了400G逻辑接口与双400G物理模块。采购前需要NVIDIA/OEM提供以下签字件：

1. 每台switch的front-panel port map；
2. 400G logical lane到800G OSFP cage的映射；
3. leaf-spine oversubscription和故障后比例；
4. optic/DAC/AOC的part number和长度；
5. Spectrum-X、Cumulus、NIC firmware、DOCA-OFED兼容矩阵。

不要直接把网页表格转换成采购订单。

## 8. Converged North/South：不是“管理小网”

每台参考节点的一张BlueField-3 B3240提供两个400G端口，分别连接两台N/S交换机：

```text
18 nodes × 2 × 400G = 36×400G node-facing ports
```

它承载：

- 训练数据和checkpoint；
- 用户入口、镜像、软件包和作业控制；
- in-band管理；
- 分布式推理的模型/KV存储流量；
- 客户网络和服务网络上联。

这张网络应与GPU Compute E/W物理或至少严格逻辑隔离，避免checkpoint burst阻塞NCCL collective。

### 8.1 存储带宽

**官方明确**：NVIDIA-Certified Storage建议约12.5Gb/s/GPU。

**本文推导**：

```text
144 GPU × 12.5Gb/s = 1.8Tb/s = 225GB/s
```

225GB/s是全局顺序吞吐规划起点，不代表任意存储系统达到这个数字就合格。还要验证：

- 多客户端并发读取；
- checkpoint大文件写入与flush；
- dataset小文件和metadata；
- 故障/重建状态下的最低吞吐；
- GPUDirect Storage路径；
- P99 latency和作业step抖动。

官方指导：[NVIDIA-Certified Storage](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/nvidia-certified-storage.html)。

## 9. OOB和控制面：数据面坏了以后仍要能登录

官方SU按每节点约6个1G management endpoint估算，18节点约108个。三台48口交换机只有144口，扣除交换机管理、上联、控制节点、存储和预留后可能很紧。因此本文建议4台OOB leaf，并以双25G/100G上联到管理core。

OOB至少连接：

- server BMC；
- BlueField-3管理口；
- ConnectX/SuperNIC管理端点（以OEM实现为准）；
- compute/N-S/OOB switch管理口；
- PDU、CDU和环境传感器；
- 存储controller管理口。

控制节点的官方示例最多使用7台：2台Base Command Manager HA、2台Slurm head、3台Kubernetes control plane。若只运行Slurm，可缩减Kubernetes部分，但DNS、NTP、DHCP/PXE、registry、login和telemetry仍需冗余。

最重要的原则是：compute plane、N/S或host OS失效时，OOB仍能完成电源控制、console、固件检查和日志收集。

## 10. 机架、电力与冷却

HGX B300是平台，最终U数和冷却取决于OEM。若以DGX B300公开的10RU、最大14.5kW作为设施估算：

```text
18 × 10RU    = 180RU
18 × 14.5kW  = 261kW IT peak（仅compute nodes）
```

纯按42RU空间至少需要5个机架；但4台/架会达到58kW且重量很高。更保守的初步布局是6个compute rack、每架3台：

```text
6 racks × 3 nodes × 14.5kW = 43.5kW/rack
```

交换机适合end-of-row集中部署，让rail拓扑不依赖服务器实际在哪一架。设计还要加入：

- A/B独立供电和PDU容量；
- switch、storage和control power；
- 启动/负载瞬态；
- 机架承重与地板路径；
- 风冷CFM或OEM液冷参数；
- 维护空间、线缆弯曲半径和光纤长度。

261kW不能直接作为变压器或冷却容量，必须加入设施冗余和设计余量。

## 11. 推荐实施顺序

### Phase 0：冻结事实

- 收集18台OEM型号、序列号、CPU、RAM、CX-8/BF3 PCI BDF；
- 固定GPU/NVSwitch/NIC/DPU/BMC firmware矩阵；
- 确认每台是`2-8-9-800`还是`2-8-10-800`变体；
- 确认训练、推理、MoE和未来32节点扩容需求。

### Phase 1：机架、供电与OOB

- 上架后先只连接OOB和电源；
- 配置DNS、NTP、BMC、Redfish、PDU和资产编号；
- 用Base Command Manager完成BIOS、firmware和OS基线；
- 验证远程断电、console和日志收集。

### Phase 2：North/South与存储

- 连接每台BlueField-3到两台不同N/S交换机；
- 部署镜像仓库、login、scheduler和telemetry；
- 验证单节点和18节点聚合存储性能；
- checkpoint流量不得经过compute plane。

### Phase 3：先建Plane A

- 按rail表连接8×400G/node；
- 配置Spectrum-X validated RoCE、ECN/PFC、MTU和adaptive routing；
- 做1、2、4、8、16、18节点NCCL扫描；
- 在只有A侧的条件下保存性能基线。

### Phase 4：再建Plane B

- 复制但物理隔离A侧配置；
- 检查两侧firmware、MTU和routing一致；
- 验证NCCL同时使用两侧；
- 分别关闭A/B，确认单plane仍可通信。

### Phase 5：接入调度器

- Slurm建立GPU、node、SU和rack topology标签；
- 不把partial SU误标为4节点；
- Kubernetes配置GPU/NIC device plugin和拓扑感知调度；
- 保存容器内外GPU/NIC编号映射。

## 12. 验收矩阵

| 层级 | 必测项目 | 通过标准来源 |
|---|---|---|
| 单GPU | HBM、GEMM、attention、功耗/温度 | OEM golden node |
| 单节点 | 8×8 P2P、NVLink、NVSwitch、NCCL | 同型号健康节点 |
| 单plane | 2/4/8/16/18 node collectives | Plane A/B互相对称 |
| 双plane | AllReduce/AllGather/RS/All-to-All | 供应商验收基线 |
| 网络 | ECN、PFC、queue、FEC、replay、P99 | Spectrum-X design |
| 存储 | ≥225GB/s规划目标、metadata、checkpoint | 业务与认证存储基线 |
| 故障 | port、leaf uplink、plane、node失效 | 已定义的降级SLO |
| 业务 | step time、TTFT/TPOT、quality、uptime | 生产SLO |

NCCL tests需要从小消息扫到数GiB，并同时记录operation time、`algbw`、`busbw`和最慢rank。采用hierarchical或硬件加速算法时，`busbw`不一定代表物理线上字节率，不能直接与400G/800G线速比较。

故障注入至少包括：

1. 关闭一张CX-8的一个400G port；
2. 关闭一条leaf-spine uplink；
3. 停止整个Plane A；
4. 重启一台HGX node；
5. 制造checkpoint写入拥塞；
6. 验证作业失败、重试、恢复和数据一致性。

## 结论：18节点可以建，但应把“不规则”留在接入层

18台HGX B300最稳妥的结构是：4个完整SU加2节点partial SU，节点接入保留18台的实际数量，核心compute fabric沿用官方32节点双平面架构。这样不改变rail、plane和故障边界，只留下部分空端口。

真正不能从网页直接决定的是交换机具体SKU、leaf-spine uplink、光模块和oversubscription。它们必须由NVIDIA/OEM根据18台服务器的准确型号、线缆距离和未来扩容目标出具最终Low-Level Design与BOM。

## 参考资料

- [NVIDIA HGX AI Factory](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/)
- [HGX B300 Components](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)
- [Networking Physical Topologies](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/networking-physical-topologies.html)
- [Networking Logical Architecture](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/networking-logical-architecture.html)
- [Appendix: Node Configurations](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/appendix-node-configurations.html)
- [NVIDIA-Certified Storage](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/nvidia-certified-storage.html)
