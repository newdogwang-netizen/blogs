---
layout: post
title: "推理副本不再是一个 Pod：沿 LWS 源码理解成组调度与 PD 更新"
description: "一组 Pod 怎样成为一个可用副本？从 LWS v0.10.0 的启动、Volcano 集成到 DisaggregatedSet 的跨角色更新，分清对象、容量与请求三种就绪。"
date: 2026-09-19 09:01:00 +0800
categories: [cloud-native-ai-infra]
category_name: "云原生与 AI 基础设施"
tags: [LWS, DisaggregatedSet, Volcano, 推理编排]
series: "2026 上海大会技术研究笔记"
series_part: 1
track: main
reading_time: "18–22 分钟"
technical_series: true
toc: true
topic_url: /cloud-native-ai-infra/
---

假设一个模型需要两台机器共同提供一个推理副本。你希望启动两个副本，集群也恰好有四台空闲机器。四个 Pod 都进入了调度队列，最后却只有两个 Pod 获得资源，而且分别属于不同副本。GPU 已经被占用，两个模型都不能服务。

这不是“资源总量不足”，而是分配单位与可用性单位错位。对 Kubernetes 来说，两个运行中的 Pod 是进展；对推理引擎来说，两个不完整的通信组可能什么也做不了。

大会的 [Beyond Model Sharding：LWS 成组调度与 DisaggregatedSet 分享](https://www.lfopensource.cn/kubecon-cloudnativecon-openinfra-summit-pytorch-conference-china/program/schedule/?id=1222146)讨论的正是这类问题。下面沿 **LWS v0.10.0，提交 `2161f38`** 阅读实现；后续版本可能已有变化。图按源码关系重绘，文中未做 GPU 集群实测。

## 一、先确定“一个副本”究竟是什么

Deployment 的常见心智模型是：增加一个 Pod，就增加一个能独立响应请求的服务实例。模型分片打破了这个条件。若一个实例由 leader 与多个 worker 共同构成，外部看到一个入口，内部却有多个必须协调启动、通信、恢复的进程。

LWS 把复制单位提升为 **leader-worker group**。`replicas` 是组数，`leaderWorkerTemplate.size` 是每组的 Pod 数，包含 leader。它们都不是 GPU 数，也不是分布式运行时的 rank 数：一个 Pod 可以申请多张 GPU，容器入口又可能在 Pod 内启动多个 rank。这个映射必须由部署模板与推理引擎共同约定。[v0.10.0 API 定义](https://github.com/kubernetes-sigs/lws/blob/2161f38ff333f9e24d3c44adef7cd5d9787cac00/api/leaderworkerset/v1/leaderworkerset_types.go)

举一个纯算例：`replicas=3`、`size=2`，每个 Pod 使用 8 张 GPU，那么目标是 3 个模型副本、6 个 Pod、48 张 GPU。若引擎在每张 GPU 上放一个 rank，才进一步得到每组 16 个 rank；这是引擎配置的结果，而不是 LWS 字段自带的语义。

{% include infra-figure.html file="lws-object-model" alt="LWS 管理 leader StatefulSet，每个 leader 对应自己的 worker StatefulSet；服务副本是一个组而不是一个 Pod" caption="图 1 · 对象关系重绘。实线表示控制/归属，不表示推理请求数据流。示例 size=2、replicas=2。" %}

控制器复用了 StatefulSet 的稳定身份与更新机制：一个 leader StatefulSet 管理多个 leader；每个 leader 再对应自己的 worker StatefulSet。有一个例外：`size=1` 时，v0.10.0 的 Pod controller 直接跳过 worker StatefulSet 的创建，此时并不存在一个副本数为零的 worker 对象。[Pod controller](https://github.com/kubernetes-sigs/lws/blob/2161f38ff333f9e24d3c44adef7cd5d9787cac00/pkg/controllers/pod_controller.go)

这个抽象解决的是“哪些 Pod 属于同一个副本，以及怎样一起管理”。它没有自行完成通信初始化，也没有证明模型已经加载成功。因此，Pod readiness 的定义仍然重要：如果探针只检查进程存活，组的 Ready 状态就可能早于实际服务能力。

## 二、成组创建，不等于成组调度

创建四个有共同标签的 Pod，调度器仍可能逐个放置它们。Gang scheduling 的价值是让调度决策考虑一个组的最低启动条件，减少多个半成品相互占住资源的情况。

LWS v0.10.0 默认不开启这项集成，需要配置 scheduler provider，并部署相应的调度器与权限。该版本提供的是 Volcano provider；设计文档中提及的其他调度器不在这一版本的适配范围内。[Volcano provider 实现](https://github.com/kubernetes-sigs/lws/blob/2161f38ff333f9e24d3c44adef7cd5d9787cac00/pkg/schedulerprovider/volcano_provider.go)、[该版本示例](https://github.com/kubernetes-sigs/lws/blob/2161f38ff333f9e24d3c44adef7cd5d9787cac00/docs/examples/sample/gang-scheduling/README.md)

这里还有三层容易混在一起的条件：

| 条件 | 回答的问题 | 不自动保证什么 |
| --- | --- | --- |
| 组资源准入 | 能否给一个完整组留出足够资源？ | 不保证所有 Pod 立刻开始运行 |
| 拓扑放置 | 这些设备是否在允许的网络/拓扑范围内？ | 不保证 RDMA 配置和通信库正确 |
| 应用就绪 | 全部 rank 是否完成 rendezvous、权重加载和自检？ | 不保证高负载下仍满足延迟目标 |

所以“总共还有 16 张卡”只是第一条信息。如果它们被拆在两个不满足放置条件的区域，资源总量为真，组可运行仍然为假。

## 三、启动顺序也会改变调度条件

`LeaderReady` 策略希望先让 leader Ready，再创建 worker。它适合 leader 能独立准备控制服务、配置或发现信息的应用。如果 leader 的 readiness 又依赖 worker 到齐，就会形成闭环：worker 等 leader，leader 等 worker。

这不是增加重试次数能修好的问题。应该先画依赖图，明确 leader 的“启动期 Ready”究竟代表控制入口可用，还是整个模型可用；必要时用不同探针与对外服务门禁表达它们。

Volcano provider 对这种启动顺序有特定处理：通常 `minMember` 对应组大小；`LeaderReady` 下它会使用 1，而资源需求仍按整组计算。这说明 PodGroup 不能只按“最少几个 Pod”一个字段理解，还要结合资源、创建顺序与 scheduler 行为。[provider 源码](https://github.com/kubernetes-sigs/lws/blob/2161f38ff333f9e24d3c44adef7cd5d9787cac00/pkg/schedulerprovider/volcano_provider.go)

此外，独占拓扑路径还可能等待 leader 获得节点，再依据它的位置构建 worker 约束。`LeaderReady`、gang 与独占拓扑分别合理，不代表任意组合都天然无死锁。上线前应验证实际版本、探针与 scheduler 的组合，而不是分别通过三个单项测试就宣布完成集成。

## 四、PD 分离之后，复制单位又发生了一次变化

Prefill 和 decode 对算力、显存与请求生命周期的需求不同。拆分之后，一个在线服务往往不再是一种 LWS 的横向复制，而是多个角色的组合：P 角色若干组、D 角色若干组，以及应用自己的路由与 KV 传输路径。

DisaggregatedSet（下文简称 DS）给多角色编排增加了共同的期望状态。理解它可以抓住三个维度：**role 决定职责，revision 决定版本，slice 决定一份角色组合的编排分区**。不能把 slice 直接翻译为一个 GPU 分片，也不能假定它等于某个固定的机架范围。[DS API](https://github.com/kubernetes-sigs/lws/blob/2161f38ff333f9e24d3c44adef7cd5d9787cac00/api/disaggregatedset/v1/disaggregatedset_types.go)

DS 管理各角色的 LWS，并组织放置、伸缩与滚动更新。这层对象不会替推理引擎发明 KV 数据协议：两端用什么布局、传哪些缓存块、如何鉴权、请求失败后是否重算，仍属于数据面。

因此，用一张图把“DS → P/D → 用户请求”连起来还不够。控制器能让 P/D 出现，并不等于一个 P 已经知道应该把某个请求的 KV 发给哪个 D。

## 五、更新最难的不是拉起新 Pod，而是新旧版本共存

设旧版本有 P=2、D=4，新版本也希望达到同样配置。若先替换所有 P，旧 D 是否能读取新 P 的 KV？若先替换 D，旧 P 是否会把请求发到新 D？版本标签是控制面身份，KV 格式兼容才是数据面契约，两者不能互相替代。

v0.10.0 的 DS 使用分角色、分 revision 的子对象推进更新；planner 根据 surge、unavailable 等预算决定下一步增减，executor 等待新 revision 的角色状态稳定后继续。这里的比例推进有整数取整与阶段约束，不应被描述为每个瞬间都严格保持某个浮点 P:D 比例。[planner](https://github.com/kubernetes-sigs/lws/blob/2161f38ff333f9e24d3c44adef7cd5d9787cac00/pkg/controllers/disaggregatedset/planner.go)、[executor](https://github.com/kubernetes-sigs/lws/blob/2161f38ff333f9e24d3c44adef7cd5d9787cac00/pkg/controllers/disaggregatedset/executor.go)

{% include infra-figure.html file="lws-rollout-gates" alt="DS 更新包含对象存在、各角色至少一个组就绪、当前目标副本就绪和应用排空四种不同门禁" caption="图 2 · 更新门禁示意。前三项对应不同控制状态；最下方的请求排空与 KV 兼容需要数据面另行实现和验证。" %}

这里有两处容易混淆的检查：用于服务发现的 `revisionReadyOnAllRoles` 只要求各角色至少有一个 Ready 副本；executor 推进下一步前，则会检查新 revision 各角色在当前期望副本数下是否稳定。也就是说，一个版本可以已经出现在服务发现中，但尚未达到这一阶段要求的容量。[service manager](https://github.com/kubernetes-sigs/lws/blob/2161f38ff333f9e24d3c44adef7cd5d9787cac00/pkg/controllers/disaggregatedset/service_manager.go)

每个 revision 的 headless Service 可以提供对应成员的发现边界，但它不是理解 KV 协议的七层路由器。planner 中的 drain 也不能直接翻译为“自动等所有生成请求结束”：从副本计划里减少旧容量，与确认每条在途请求已经安全结束，是不同层的动作。

这带来一个具体的工程约束：更新预算要算上**可以配对工作的容量**。假设尚有 8 个 D，却只有一个吞吐受限的 P，监控面板上的 D Ready 数再漂亮，也不能按 8 个 D 的理想吞吐接流量。应验证完整请求链路的容量，再决定流量切换幅度。

## 六、扩容字段也必须有唯一的写入者

当 HPA、DS controller 和人工运维同时修改一个角色的副本数，问题不是“谁反应更快”，而是最后一次写入会覆盖谁的意图。平台需要明确所有权：角色由 DS 内联规格管理，还是由外部伸缩器管理。

该版本通过 RoleScaler 为外部伸缩提供稳定入口，避免让 HPA 追逐每次更新都会变化的 revision 子对象。对 External 模式，应沿 RoleScaler 这条路径理解期望副本，而不是继续把角色内联 `replicas` 当作唯一事实。[scaler manager](https://github.com/kubernetes-sigs/lws/blob/2161f38ff333f9e24d3c44adef7cd5d9787cac00/pkg/controllers/disaggregatedset/scaler_manager.go)

这里必须保留发布版边界：**v0.10.0 的 webhook 禁止 External scaling 与多于一个 slice 组合**；使用 DS placement 时，还限制与 LWS 的独占拓扑注解叠加。它们是具体实现阶段的约束，不是可以靠“理论上合理”绕开的配置细节。[DS webhook](https://github.com/kubernetes-sigs/lws/blob/2161f38ff333f9e24d3c44adef7cd5d9787cac00/pkg/webhooks/disaggregatedset/disaggregatedset_webhook.go)

## 七、一个 worker 出错，为什么可能重建整个组

如果 rank 之间持有共享通信状态，单独重启一个 worker，不一定能加入其他 rank 正在使用的 communicator。组级重建用更大的恢复范围，换取更容易解释的一致启动边界。

LWS 的组重建策略会结合容器重启、Pod 删除、组是否已经开始以及对象归属等条件处理事件。阅读代码时不能把它缩成“任何 Pod 有变化就删整个组”：旧 revision 的迟到事件不应误伤新组，尚在启动的组也有特定策略。组恢复最终仍意味着一段不可用时间，并可能丢失进程内 KV 与请求状态。[重启处理代码](https://github.com/kubernetes-sigs/lws/blob/2161f38ff333f9e24d3c44adef7cd5d9787cac00/pkg/controllers/pod_controller.go)

代价可以用一个工程量描述：`损失 ≈ 恢复期间缺失的有效容量 + 失败请求重试成本 + 缓存重建成本`。这不是性能公式，而是提醒我们，恢复快慢不能只看 Pod 从创建到 Running 的秒数。

## 八、怎样验收这套抽象

建议先在测试环境验证五个断点，每次同时记录对象事件和端到端请求结果，而非只保存 `kubectl get pods`：

| 注入条件 | 应该观察什么 |
| --- | --- |
| 资源只够半个组 | 是否产生长期占卡但不能服务的半成品，最终由谁回收/重试 |
| leader 探针依赖 worker | 是否形成启动循环，改探针后能否解除 |
| 新 P Ready、新 D 故障 | 更新停在哪一道门禁，旧版本还能否服务 |
| 更新时触发外部扩容 | 谁拥有期望副本，新旧 revision 的容量是否越过预算 |
| 删除一个正在服务的 worker | 恢复范围、在途请求结果、KV 重建和尾延迟 |

若模型本来就在一个 Pod 内独立服务，Deployment 加清晰的探针、排空和伸缩策略可能更简单。LWS/DS 值得引入的信号，不是模型“足够大”，而是你已经需要把**多 Pod 的共同命运、多角色的版本关系与组级资源条件**变成控制面可表达的契约。

部署前可以对照三张图检查：对象归属图里的一组 Pod，是否就是启动时互相等待的那组进程？路由器认定可用的 P/D，是否已经建立 KV 传输关系？这些对应关系需要由部署配置、探针和引擎共同保证，LWS/DS 只负责其中的编排部分。
