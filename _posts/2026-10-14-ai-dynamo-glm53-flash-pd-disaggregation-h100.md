---
layout: post
title: "ai-Dynamo PD 分离部署学习笔记：Dynamo 组件、NIXL KV 传输与 H100 上的 GLM-5.3-Flash"
description: "沿一次 16×H100 上的真实部署，理解 Dynamo 三组件的分工、DGD/DCD 的滚动语义、NIXL 如何把 KV cache 搬过 InfiniBand，以及混合注意力模型在跨硬件移植时的 backend 约束。"
date: 2026-10-14 20:00:00 +0800
categories: [cloud-native-ai-infra]
category_name: "云原生与 AI 基础设施"
tags: [ai-Dynamo, vLLM, PD分离, GLM-5.3-Flash, H100, InfiniBand, Kubernetes]
reading_time: "20–25 分钟"
technical_series: false
toc: true
topic_url: /cloud-native-ai-infra/
---

PD 分离（Prefill-Decode Disaggregation）的论文和架构图看多了容易产生一种错觉：prefill 放一堆机器、decode 放一堆机器、中间传 KV，就结束了。真在托管 Kubernetes 集群上从零部署一次会发现，每一步背后都有一组值得展开的机制：operator 把什么渲染成了什么、KV 在物理上怎么跨节点搬、为什么某个 attention kernel 在一种卡上存在而在另一种上不存在。

这篇是学习笔记：沿一次真实部署（2 节点 × 8 H100 80GB SXM，Nebius 托管 K8s v1.36，模型 GLM-5.3-Flash 320B-A18B FP8，ai-Dynamo 1.5.0 + vLLM）把这些问题逐个过一遍。所有日志和报错都来自实际执行，不是转述。

## 一、PD 分离到底在拆什么

先把"为什么分离"说准确，因为它决定了后面所有组件的形状。

Prefill 和 decode 两个阶段的资源画像是对立的：

- **Prefill**（处理输入 prompt）是 compute-bound：一次前向算完整个序列的 KV，GPU 算力打满，显存带宽不是瓶颈
- **Decode**（逐 token 生成）是 memory-bound：每步只算一个 token，但要把整个 KV cache 从显存读一遍，算力大量闲置

传统合池部署里这两个阶段互相干扰：一个长 prompt 进来，decode 批次的 inter-token latency 立刻抖动。PD 分离的思路是按阶段独立扩缩容、独立调参（prefill 要大 batch 算力，decode 要大显存带宽），代价是 prefill 算出来的 KV cache 必须**搬到** decode 侧——这个"搬"就是整条链路里最工程化的部分，也是后面 NIXL 一节的主角。

GLM-5.3-Flash 这个模型让这个 trade-off 更有意思：它是 GLA/KDA 混合注意力架构（一部分层是线性注意力，带小的 recurrent 状态；一部分层是稀疏 full attention）。线性注意力层的"KV"不是逐 token 的 key/value 缓存而是固定大小的 conv/ssm 状态，这让它的 KV 传输量和传统 MHA 模型不同——vLLM 为此有专门的 hybrid KV cache manager（我们的配置里 `--no-disable-hybrid-kv-cache-manager` 就是开启它）。

## 二、Dynamo 平台组件：operator、etcd、DGD 各自是什么

装 ai-Dynamo 之前值得先分清三样东西，因为"装了 operator 但没装 etcd"是我们踩到的第一个坑——不理解分工的话，这个坑看起来毫无道理。

**Operator（控制面）**：管理一组 CRD（`DynamoGraphDeployment`、`DynamoComponentDeployment` 等）。你写一个 DGD 描述"frontend + prefill worker + decode worker"的图，operator 把每个 component 渲染成独立的 DCD（再由 DCD 渲染成 Deployment/Pod）。关键理解：**DGD 不是 Deployment 的薄封装，它有自己的滚动语义**——这个第六节会变成一个真正的坑。

**etcd（状态面）**：worker 启动后把自己注册进来（component 名、endpoint、transport 地址），frontend 的 kv-aware router 从这里发现 worker 并感知容量。没有 etcd，worker 起来了 frontend 也找不到它。我们的教训：helm chart 里 `global.etcd.install` **默认是 false**——第一次 `helm upgrade --install` 后 operator Running、CRD 注册齐全，看起来一切正常，但 etcd 根本不存在。判断信号很简单：`kubectl get pods -n dynamo` 里没有 `etcd-0`。

**可选组件（Grove/Kai/NATS）**：多节点编排和高级调度，单集群两节点的规模不需要，values 里全关。这里也有个版本迁移的坑：v1.5.0 把 `nats.enabled`/`grove.enabled` 顶层键挪到了 `global.<name>.install`，写旧键会被 chart 的 validation template 直接拒绝，报错信息里会给出正确写法。

最终的最小安装：

```bash
helm upgrade --install dynamo-platform . -n dynamo --create-namespace \
    -f values.yaml --set global.etcd.install=true
```

```
dynamo-platform-dynamo-operator-controller-manager   1/1   Running
dynamo-platform-etcd-0                                1/1   Running
```

## 三、存储：为什么官方 recipe 假设 RWX，而我们用了两个 RWO

官方 recipe 的 PVC 是 ReadWriteMany——一份模型缓存（GLM-5.3-Flash FP8 是 328GB）同时挂给 prefill 和 decode 两个 worker。这在有共享文件系统的环境（Filestore、EFS、CephFS）是最优解：下载一次，处处挂载。

我们的环境（Nebius 网络 SSD）不支持 RWX，apply 直接被 provisioner 拒绝：`Volume capabilities not supported`。退路是两个 RWO PVC、每个节点一份、模型下载两遍。这个"退路"本身没问题，但撞出了第二个学习点：**云的磁盘配额是按项目计的，跟节点剩余空间无关**。我按直觉开了 2 × 2Ti，第二个盘秒拒：

```
Quota limit exceeded. Exceeded limit for container tenant-xxx,
quota compute.disk.size.network-ssd
```

查配额服务（`nebius quotas quota-allowance list`）才看到全貌：项目 network-ssd 上限约 4 TiB，已用 3.87 TiB——各 VM 的 boot 盘和既有 PVC 都计入同一配额池。正确做法是开量前先查 quota 用量，而不是估算"节点好像还有空间"。改两个 500Gi（官方 recipe 本来也是这个量级，2Ti 是我过度申请）后顺利 Bound。

下载 Job 的一个细节值得记：用 pod anti-affinity 把两个下载 pod 钉到两个 GPU 节点，保证每份 PVC attach 到将来用它做推理的那个节点（RWO 的 attach 是节点级的）。`huggingface_hub[hf_xet]` 拉模型，实测 23 分钟/份 306GB——hf_xet 对对象存储的分块下载吞吐远好于直觉预估。

## 四、DGD 适配：把 manifest 的每一行分成两类

从官方 H200 disaggregated recipe 到 H100，最重要的不是"改了哪些参数"，而是建立一个分类框架：**manifest 里的配置行，一类是架构必需，一类是环境相关**。前者跨硬件动不得，后者必须逐行审视。

**架构必需（原样保留）**：

- `VLLM_SSM_CONV_STATE_LAYOUT=DS`、`VLLM_KV_CACHE_LAYOUT=HND` — GLM 混合注意力模型的状态/KV 布局要求
- `--kv-transfer-config` NixlConnector 的 `kv_producer`（prefill）/`kv_consumer`（decode）配对 — KV 传输的声明
- `--speculative-config '{"method":"mtp","num_speculative_tokens":7}'` — MTP 投机解码（decode 吞吐关键项）
- `--dyn-reasoning-parser glm45 --dyn-tool-call-parser glm47` — Dynamo frontend 的结构化输出解析器
- `--no-disable-hybrid-kv-cache-manager` — 混合注意力的 KV 管理器（见第一节）
- `runtimeVersionOverride: 1.4.1` — 镜像 tag 非语义版本时 admission 的要求

**环境相关（必须改）**：

- 节点选择：recipe 用 `nvidia.com/gpu.product: NVIDIA-H200` 标签，我们改 `kubernetes.io/hostname` 钉节点
- RuntimeClass：没有 GPU Operator 就没有 `runtimeClassName: nvidia` 和 `rdma/ib` resource。替代方案见下一节
- 显存预算：H100 80GB vs H200 141GB

显存预算的推导值得写细。FP8 权重 328GB ÷ TP8 ≈ 每卡 41GB，H100 80GB 上剩余 ~39GB 要装 KV cache + CUDA graph + activation。recipe 的 1M 上下文在这里不可能，降到 64K；`gpu-memory-utilization` 从 0.80/0.85 反而升到 0.90/0.93——因为 H100 上权重占比更高，不提高利用率 KV 就太小；decode 的 `max-num-seqs` 从 128 降到 64，等比例压缩并发下的 KV 占用。

| 参数 | H200 recipe | H100 适配 | 理由 |
|---|---|---|---|
| max-model-len | 1,048,576 | 65536 | KV 预算 |
| gpu-memory-utilization (P/D) | 0.80 / 0.85 | 0.90 / 0.93 | 权重占比更高 |
| max-num-seqs (decode) | 128 | 64 | 并发 KV 占用 |
| shm emptyDir | 128Gi | 64Gi | host 内存 |

这个表的意义不是这些具体数字，而是方法：**显存预算 = 卡上容量 − 权重 − 运行时开销，长上下文和并发数都是从这个余量里分配的**，换卡就重算，没有可以照抄的数。

## 五、InfiniBand：三种把 IB 交给容器的方式

集群的 IB fabric 是真实存在的：两个 GPU 节点在同一个 GPU cluster（fabric-4），拓扑显示共享前两级交换机。但"fabric 存在"和"容器里的 NCCL 能用"是两回事。把 IB 设备交给容器，K8s 上有三种路径，值得都认识一遍：

1. **DRA/RDMA 资源插件**（Nebius 的 dranet、或 gpu-operator 的 nvidia-network-operator）：设备以 `rdma/ib` resource 形式声明，调度器感知。最"正规"，但依赖 operator 生态。我们集群里 dranet desired=0（节点没打 `dranet-rdma-capable` 标签），等于没启用
2. **GPUDirect RDMA 设备注入**（Kubeflow/volcano 生态常见）：job 级配置自动挂 `/dev/infiniband`
3. **privileged + hostPath 直通**：容器 privileged，volume hostPath `/dev/infiniband`。最简单粗暴，牺牲多租户隔离（任何拿到这个 pod 的人都能碰节点 IB 设备）

同集群一个先前的 PyTorchJob 用的是方式 3，且 NCCL 日志确认了实际传输路径——这条证据链值得单独记，因为它同时回答了"IB 通不通"和"怎么验证"两个问题：

```
NET/IB : Using [0]mlx5_0:1/IB/SHARP ... [7]mlx5_7:1/IB/SHARP; OOB eth0:10.1.126.12
Channel 06/0 : 6[6] -> 14[6] [send] via NET/IBext_v10/2/GDRDMA
```

三行信息：每节点 8 张 HCA 全启用、带 SHARP（交换机内聚合，AllReduce 在交换机上完成而非 GPU 间两两传）、跨节点通道走 GDRDMA（GPU 显存直接进网卡，零拷贝，rank 0-7 在本节点、8-15 在对端）。我们的 worker 沿用方式 3，加两个 NCCL 环境变量：`NCCL_IB_HCA=mlx5`（指定 HCA 前缀）和 `NCCL_SOCKET_IFNAME=eth0`（带外通信走以太网）。

## 六、两个深坑：backend 兼容性与 operator 滚动死锁

### 坑一：attention backend 的硬件耦合

第一版 manifest 保留了 recipe 的 `--attention-backend FLASHMLA_SPARSE`，两个 worker CrashLoop。排 vLLM crash 的第一动作永远是看 `kubectl logs --previous`（别等 probe 超时，根因在启动早期）：

```
ValueError: Selected backend AttentionBackendEnum.FLASHMLA_SPARSE is not valid
for this configuration. Reason: ['head_size not supported']
```

直觉反应是退到非 sparse 的 `FLASHMLA`——换了个报错：

```
ValueError: Selected backend AttentionBackendEnum.FLASHMLA is not valid
for this configuration. Reason: ['sparse not supported']
```

两个报错放在一起才构成完整的认知：GLM-5.3-Flash 的混合注意力**必须**有 sparse attention 支持；`FLASHMLA_SPARSE` 的 sparse kernel 不支持此模型在 H100 上的 head_size 组合（它是面向 H200/GB200 调优的），而 `FLASHMLA` 压根不带 sparse。死结不在"选哪个 backend"，在"显式指定"这个动作本身。

修复：**删掉 `--attention-backend`，交给 vLLM platform 层自动选择**。cuda platform 的 `get_attn_backend_cls` 按模型架构 + 硬件能力选兼容组合，删掉后一次通过。这个坑的教学价值在于：显式 backend 指定永远是"某套硬件上的调优项"，不是可移植配置——跨硬件移植时它应该进第四节那个"环境相关"清单。

### 坑二：GPU 独占节点上的滚动死锁

修好 backend 后 `kubectl apply` 更新 DGD，operator 为每个 component 渲染新 DCD 并滚动更新——然后卡住不动。诊断信号：`kubectl get dynamocomponentdeployments` 显示新旧两个 DCD 并存，DGD status 里 `replicas: 2`，新 pod Pending、旧 pod CrashLoop。

死锁机制值得拆开看：operator 的滚动策略是**新副本 Ready 后才缩旧副本**。而新 pod Pending 的原因是 8 卡 GPU 全被旧 pod 占用；旧 pod 在 CrashLoop，但 CrashLoop 的 pod 不释放设备。于是：新等旧退出 → 旧等新 Ready → 永久循环。集群如果节点有余量，这个策略是优雅的；GPU 独占节点（每节点恰好一组 TP8 worker）上它是死锁。

我第一次的修复尝试是删旧 ReplicaSet——无效且值得记录为什么：DCD 控制器立即按旧模板重建了 RS（新 pod 还是旧配置）。**正确操作是删旧 DCD 对象本身**：

```
kubectl delete dynamocomponentdeployment \
    glm53-flash-disagg-vllmprefillworker-<old-hash> \
    glm53-flash-disagg-vllmdecodeworker-<old-hash> -n glm53
```

operator 对 component 的期望状态只认 DGD spec，删 DCD 不会引起回滚，新版本立即获得 GPU。泛化教训：**自定义 operator 的滚动语义 ≠ 原生 Deployment 的 maxSurge/maxUnavailable**，在独占资源的节点池上，"先起后删"类策略天然有死锁面——变更前主动把旧组件 scale 0（或删 DCD）是更稳的路径。

## 七、NIXL：KV 到底怎么过去

部署通了之后值得回头补的一课：KV 传输层 NIXL 在做什么。

NixlConnector 的两端配置是对称的：prefill 侧 `kv_role: kv_producer`、decode 侧 `kv_consumer`，都指定 `kv_buffer_device: cuda`（KV 传输缓冲在显存，配合 GDRDMA 走零拷贝路径；H200 recipe 里有 RDMA resource，我们裸 IB 设备上 UCX 自己探测）。加上 `kv-events-config`（zmq publisher）后 prefill 会发布 KV 事件流，router 据此做 **kv-aware 调度**——同一个会话的续写请求优先路由到还持有那段 KV 的 decode 实例，省掉重传。

一个有意思的验证视角：我们没做任何专门的 KV 传输测试，但 chat 请求能返回这件事本身就是证明——decode 侧在收到 prefill 传来的 KV 之前不可能产出第一个 token。TTFT 里就包含了这段跨节点 IB 传输的耗时。

## 八、验证清单与当前状态

冒烟测试四项（frontend 经 port-forward）：

```
GET /v1/models  → zai-org/GLM-5.3-Flash, context_window 65536
POST /v1/chat/completions          → reasoning_content 正常（glm45 parser）
POST ... stream=true               → 87 SSE chunks + [DONE]
POST ... tools=[get_weather]       → tool_calls 正确解析（glm47 parser）
```

`/health` 返回的 etcd 注册表是结构理解的最好材料——它把 DGD 的逻辑图落成了具体地址：

```
dyn://…backend.generate        tcp://10.1.126.16:38649/…   (decode worker)
dyn://…prefill.generate        tcp://10.1.50.155:35585/…   (prefill worker)
dyn://…prefill.worker_kv_query_source_…                    (KV 查询端点)
```

最终拓扑：prefill 8×H100（mem_util 0.90，max_len 64K，chunked prefill 8192 token）+ decode 8×H100（mem_util 0.93，MTP7，max_seqs 64），frontend 1 副本 kv-aware 路由。

## 九、还没做的与想继续学的

- PD 配比：1P1D 只是起点，prefill/decode 的算力配比应该是 workload（prompt 长度分布 × 输出长度分布）的函数，还没建模
- MTP7 接受率：投机解码的实际收益取决于 draft 接受率，需要 per-request spec decode metrics 打开看
- KV 传输吞吐对 TTFT 的分解：目前只知道"能通"，没量化 IB 带宽利用
- ePDL / Mooncake 等其他 KV 传输层的对比（vLLM 生态里 NixlConnector 不是唯一选项）

部署当天完成，以上是当天能沉淀的全部认知；有 benchmark 数据后继续写。
