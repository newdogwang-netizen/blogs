---
layout: post
title: "Harbor 轨迹解析：从一次任务到 ATIF"
description: "用一个创建文件的例子，理解 Agent 日志怎样变成 ATIF，再沿源码看转换、校验与保存。"
date: 2026-11-03 16:00:00 +0800
categories: [cloud-native-ai-infra]
category_name: "云原生与 AI 基础设施"
topic_url: /cloud-native-ai-infra/
tags: [Harbor, Agent, Benchmark, Trajectory, ATIF, Claude Code, Codex]
reading_time: "主线约 15 分钟 · 深入约 30 分钟"
technical_series: true
guided_reading: true
toc: true
---

跑完一个 benchmark，你知道 Claude Code 成功了，Codex 失败了。接下来往往还想知道：失败前做了什么？哪个工具出了错？两者完成任务的过程有什么不同？

回答这些问题，需要保留 **Agent 完成任务的过程记录，也就是 trajectory（轨迹）**。

Harbor 的做法可以先记成一句话：**每个 Agent 按自己的方式记日志，Harbor 的适配器把日志整理成统一格式 ATIF，再保存和展示。**

<aside class="reading-guide" aria-label="本文阅读路线">
  <p><strong>按这条顺序读，不必一开始就记函数名。</strong></p>
  <ol>
    <li><a href="#one-task">先看一个任务</a>：轨迹里到底有什么，ATIF 为什么这样组织。</li>
    <li><a href="#different-agents">再看三种 Agent</a>：同一段过程，怎样从不同日志中整理出来。</li>
    <li><a href="#agent-adapter">最后进入源码</a>：适配接口、转换函数、校验与保存。可展开的补充说明留给深入阅读。</li>
  </ol>
</aside>

## 一、先看一个任务：创建文件，再查看目录 {#one-task}

我们先用一个教学任务贯穿全文：

> 创建 `hello.txt`，并告诉我当前工作目录。

Agent 需要调用终端工具，执行 `touch hello.txt` 和 `pwd`，然后回答用户。假设模型一次给出了两个工具调用，原始记录可能拆成下面 7 条事件。

| 顺序 | 发生了什么 |
| --- | --- |
| 1 | 用户提出任务 |
| 2 | 模型回答：“我来创建文件并检查目录。” |
| 3 | 模型发起工具调用 **c1**：`touch hello.txt` |
| 4 | 同一次模型输出发起工具调用 **c2**：`pwd` |
| 5 | c2 的结果先返回：`/app` |
| 6 | c1 的结果随后返回：退出码 0 |
| 7 | 模型读完结果，回答：“已完成，当前目录是 /app。” |

这里的 c1、c2 是调用编号，用来区分两次工具执行。结果返回的先后顺序，不一定与调用顺序一致。

**这些消息、动作和返回结果合在一起，就是这次任务的轨迹。** benchmark 评分告诉我们是否完成任务，轨迹帮助我们理解完成或失败的过程。

Harbor 把“某个 Agent 对某个任务的一次尝试”叫作 **Trial**。同一个任务跑 5 次，就是 5 个 trial，各有自己的记录与结果。[Trial 定义与结果][trial-result]

## 二、ATIF 把这段过程整理成什么样 {#what-is-atif}

不同 Agent 的日志长得不一样。Claude Code 有自己的会话文件，Codex 有另一种事件结构，OpenCode 则可以输出一串 JSON 事件。

如果查看器分别理解这些格式，每增加一个 Agent，就要再写一套显示和分析逻辑。Harbor 因此引入 **ATIF（Agent Trajectory Interchange Format，轨迹交换格式）**：让不同来源的记录，最后使用共同的 JSON 结构。[ATIF 规范][rfc]

ATIF 的主要单位叫 **Step**。在这个例子中，7 条原始事件可以整理成 3 个 steps：

<figure class="step-flow" aria-label="这个任务对应的三个 ATIF 步骤">
  <ol>
    <li><span class="step-flow-label">Step 1<br>用户</span><div><strong>提出任务</strong><p>创建 hello.txt，并查看当前目录。</p></div></li>
    <li><span class="step-flow-label">Step 2<br>Agent</span><div><strong>模型第一次输出，并调用工具</strong><p>“我来创建文件并检查目录。”</p><dl><dt>发起动作</dt><dd>c1：创建文件<br>c2：查看目录</dd><dt>收到反馈</dt><dd>c2 → /app<br>c1 → 退出码 0</dd></dl></div></li>
    <li><span class="step-flow-label">Step 3<br>Agent</span><div><strong>模型读取结果，再回答</strong><p>“已完成，当前目录是 /app。”</p></div></li>
  </ol>
  <figcaption>7 条原始事件，整理成 3 个步骤；两个工具结果都属于 Step 2。</figcaption>
</figure>

为什么工具结果也在 Step 2 里？因为它们是 **这一次模型输出所发起动作的反馈**。即使结果在原始日志的后面才出现，转换器也能根据调用编号，把它放回对应的 step。

> 一个 Step 可以包含多个工具调用。不要把 Step 数直接当成日志行数或命令数。

底层能够区分模型调用时，协议建议尽量一次模型调用对应一个 agent step；用户消息则单独形成 user step。[Step 语义与模型][step-model]

先认识 4 个字段就够了：

| 字段 | 用普通话理解 |
| --- | --- |
| `source` | 谁在说话：用户、Agent 或系统 |
| `message` | 这一轮说了什么 |
| `tool_calls` | 这一轮发起了哪些工具调用 |
| `observation.results` | 这些动作得到了什么反馈 |

模型的思考内容、token 用量、时间等信息也有位置可以记录，后面再看。ATIF 在这里是一套记录规则；Agent 如何调用模型、如何执行工具，仍由它自己的运行逻辑决定。

## 三、转换函数最重要的两件事：分组与配对 {#group-and-match}

所谓“转换”，最重要的工作是找回原始事件之间的关系。

### 先分组：哪些内容来自同一次模型输出

在我们的例子里，第 2 条文本和第 3、4 条工具调用属于同一次模型输出，应该进入同一个 step。第 7 条最终回答则属于下一次模型调用。

转换器需要找到一个可靠的分组依据，例如消息 ID，或明确的“这一轮开始／结束”事件。**不同 Agent 的主要差异，就在于原始日志提供了怎样的分组依据。**

### 再配对：每个结果属于哪个调用

看到 c2 的结果 `/app` 时，转换器必须找到此前编号为 c2 的调用。这个结果不能因为“先返回”，就被错放到 c1 上。

ATIF 用两个字段完成关联：

```text
调用：tool_call_id = "c2"
结果：source_call_id = "c2"
```

它们分别出现在 `tool_calls` 和 `observation.results` 中，并位于同一个 step。[工具结果模型][observation-result]、[引用校验][trajectory-model]

理解这个过程后，再看转换器里的字典就容易一些：

```python
turns[模型调用编号] = 这一轮的记录
calls[工具调用编号] = 发起这个调用的那一轮
```

第一张表负责分组，第二张表负责把晚到、乱序到达的结果放回去。真实实现还要处理重复事件、坏行和中断，但核心关系就是这两层。

<details class="source-detail" markdown="1">
<summary>展开：看一段 ATIF JSON，核对字段位置</summary>

下面只展示我们例子中的 Step 2。它是 step 片段，不是一份完整的根轨迹：

```json
{
  "step_id": 2,
  "source": "agent",
  "message": "Creating the file and checking the directory.",
  "tool_calls": [
    {
      "tool_call_id": "c1",
      "function_name": "shell",
      "arguments": {"command": "touch hello.txt"}
    },
    {
      "tool_call_id": "c2",
      "function_name": "shell",
      "arguments": {"command": "pwd"}
    }
  ],
  "observation": {
    "results": [
      {"source_call_id": "c2", "content": "/app"},
      {"source_call_id": "c1", "content": "exit code 0"}
    ]
  }
}
```

完整文件在它外面还有版本、Agent 身份和 `steps` 数组。可以直接打开本文的 [完整示例输出]({{ '/assets/examples/harbor-atif/trajectory.synthetic.json' | relative_url }}) 对照。

</details>

## 四、同一个例子，三种 Agent 分别怎样转换 {#different-agents}

下面都在回答同一个问题：**怎样判断哪些原始事件应该进入同一个 Step？**

### Claude Code：把同一个 message.id 的内容放在一起

Claude Code 会保存原生会话 JSONL。JSONL 可以理解为“每一行都是一个 JSON 事件”；一份文件里按顺序记着许多事件。

一次模型输出，可能拆成文本、工具块和 usage 更新等多条记录。Harbor 利用这些记录里的 **`message.id`** 识别同一次输出，再把它们合到同一轮里。[Claude 转换入口][claude-convert]

套回我们的例子：如果文本、c1 和 c2 都属于消息 m1，它们就进入 m1 这轮；后面的最终回答属于 m2，于是另起一轮。

```text
message.id = m1
  文本 + c1 + c2 → Step 2

message.id = m2
  最终回答      → Step 3
```

配对工具结果时，适配器维护 `pending_calls`，可以理解为“已经发起、还在等结果的调用”。看到 tool result，就用其 `tool_use_id` 找到之前的调用，把 output 补进去。[工具结果处理][claude-normalize]

这段思路可以压缩成伪代码：

```python
turn = 找到或创建这一条消息的记录(message_id)
turn.工具调用.append(call)
pending_calls[call.id] = call

# 稍后，工具结果到达
pending_calls[result.call_id].output = result.content
```

随后 `_convert_event_to_step()` 把整理好的记录构造成 `Step`、`ToolCall` 和 `ObservationResult`，再生成完整轨迹。[Step 构建函数][claude-step]

<details class="source-detail" markdown="1">
<summary>源码补充：Claude 的日志位置、去重、用量与中断处理</summary>

运行时，适配器把 `CLAUDE_CONFIG_DIR` 指向 `/logs/agent/sessions`，使原生 session 进入可收集目录。CLI 的 stream-json 输出另存为 `claude-code.txt`；轨迹正文主要来自 session JSONL，stdout 还可提供最终成本。[运行入口][claude-run]

`_convert_events_to_trajectory()` 先读取主 session 和 `subagents/*.jsonl`，跳过坏行，按事件 UUID 去重，再按时间排序。子 Agent 的来源信息可以保存在 `extra` 中。

同一 `message.id` 的多次 usage 更新，取最后的快照，避免重复相加。成本优先采用原生最终报告值；缺失时可估算，并记录估算来源。[汇总与保存][claude-save]

已经写进轮次的 tool use，即使没有收到结果，也能保留为没有 observation 的工具调用。孤立 tool result 则有降级或跳过路径，不能保证所有不完整事件都保留。

</details>

### Codex：用模型调用的边界分组

Codex 也保存原生 session JSONL，不过事件结构不同。Harbor 需要从消息、工具调用、工具结果和 token 统计中恢复模型调用的分界。[Codex 事件转换][codex-convert]

这份实现为每次模型调用分配内部编号 `api_call_id`。有实际模型输出后，遇到相应的 `token_count` 事件，便记录这一轮的用量并推进编号。

于是，同一轮的文本、c1 和 c2 可以带着相同的 `api_call_id`，之后交给 **`_group_events_by_api_call_id()`** 合并：

```text
文本、c1、c2 + 相同 api_call_id
                 ↓
             一个 agent Step
```

工具调用与 output 仍按 `call_id` 配对；分组函数负责把属于同一次模型调用的内容汇到一起。用户消息会触发已有分组的输出，不会无条件混进模型回答。[分组函数][codex-group]

> Claude 主要借助原生消息 ID；Codex 在这份实现中恢复模型调用编号。两者最终都要回答“这一轮模型做了什么”。

<details class="source-detail" markdown="1">
<summary>源码补充：Codex 事件类型、session 选择和不完整调用</summary>

常见映射如下。先读清主线，再查这张表即可。

| 原生事件 | 主要用途 |
| --- | --- |
| `session_meta` | session ID、CLI 版本等身份信息 |
| `turn_context` | model 与原生 turn 信息 |
| `event_msg / token_count` | 单次和累计 token 用量，以及内部调用分组 |
| `response_item / message` | 用户或 assistant 文本 |
| `function_call`、`custom_tool_call` | 工具名、参数与调用编号 |
| 对应的 `*_output` | 按编号补全工具结果 |

运行结束时，适配器尝试把 `CODEX_HOME` 下的 sessions 复制到日志目录。转换器选择最近的日期目录与其中排序最后的 JSONL，并不通用地合并所有 session。[日志复制][codex-run]、[session 选择][codex-session]

该快照的普通 function call 先放入 pending 表，等 output 到达才加入待输出的事件；末尾没有统一导出剩余 pending calls。由此可以推断，截断在结果返回前的调用可能未进入 ATIF，应连同原始日志检查。[转换数据流][codex-convert]

每步使用单次调用的 `last_token_usage`；总量可读取最后一个累计 `total_token_usage`。不能把每条累计值重新相加。原生未报告成本时，还可能结合价格表估算。[统计与保存][codex-save]

</details>

### OpenCode：在开始和结束事件之间收集内容

OpenCode 可以直接输出 JSON 事件流，Harbor 把它保存为 `opencode.txt`，再逐行读取。[读取函数][opencode-parse]、[运行入口][opencode-run]

它的分组依据更直观：看到 `step_start` 就开始收集，看到 `step_finish` 就结束这一轮。

```text
step_start
  ├─ 文本
  ├─ 思考内容（如果输出里有）
  ├─ 工具调用 c1 及结果
  └─ 工具调用 c2 及结果
step_finish
         ↓
     一个 agent Step
```

`_convert_events_to_trajectory()` 将区间内的内容组合起来，工具的 `callID` 用来关联调用与结果，区间结束事件还提供 token 和 cost 信息。[转换函数][opencode-convert]

这也带来一个清楚的边界：当前实现只输出收到 `step_finish` 的区间。即使文件尾部已有部分文本或工具记录，未结束的轮次也不会自动成为 ATIF step。

<details class="source-detail" markdown="1">
<summary>源码补充：OpenCode 的用户消息与其他 Agent 的记录来源</summary>

OpenCode 优先取流中的 user 事件；较旧版本缺失时，适配器用 `run()` 保存的实际 instruction 补上用户输入，然后连续重排 step ID。

其他 Agent 也可以复用相同的目标结构，但采集材料各不相同：

| Agent | 先取得什么材料 |
| --- | --- |
| Gemini CLI | 原生 session JSON / JSONL |
| mini-swe-agent | 显式输出的原生 trajectory 文件 |
| OpenHands | session events，或原始模型 completions |
| Terminus-2 | 自己执行循环中的模型响应和终端反馈 |

对应实现：[Gemini][gemini]、[mini-swe-agent][mini]、[OpenHands][openhands]、[Terminus-2][terminus]。Terminus 直接构建 ATIF，不必再从另一套 CLI 日志恢复关系。

</details>

## 五、看懂转换后，再看 Harbor 怎样接入 Agent {#agent-adapter}

到这里，转换函数的职责已经明确了：读原始记录，分组、配对，生成 ATIF。

但在 benchmark 里，还需要有人负责安装 Agent、启动它、取回文件。**Agent 适配器就是把这些环节接起来的代码。**

对能安装到任务环境里的 CLI，Harbor 常用 `BaseInstalledAgent`。先认识三个主要入口：

| 入口 | 负责什么 |
| --- | --- |
| `install()` | 安装程序和依赖 |
| `run()` | 启动 Agent，让原生日志写到指定位置 |
| `populate_context_post_run()` | 日志收集后，转换轨迹并提取用量 |

实际还有 `configure()`，用于每次运行都需要的准备。它与 `install()` 的区别放在下面的补充说明中。[安装与准备接口][installed-base]

再把它们放进一次 Trial，就能看到完整顺序：

```text
① 准备任务环境，安装和配置 Agent
② 调用 run()，Agent 执行任务并记录日志
③ Trial 把日志同步到宿主机
④ 适配器转换日志，写 agent/trajectory.json
⑤ Verifier 给任务评分，Trial 保存结果
```

`Verifier` 是任务的评分程序。它通常检查 Agent 留下的文件或环境结果；评分与过程记录分别保存。[单步骤执行][single-step]

{% include infra-figure.html file="harbor-trajectory-pipeline" alt="不同 Agent 的原始记录经各自转换器变成统一 ATIF，再供查看器和 Harbor Hub 使用" caption="全流程示意 · 把前面的转换函数放回执行链：Agent 记录，Trial 收集，适配器整理，下游读取。" %}

可以沿这条函数链定位源码：

```text
SingleStepTrial._run_agent()
  → agent.run()
  → Trial._sync_agent_output()
  → agent.populate_context_post_run()
```

日志同步在收尾的 `finally` 路径中，因此运行超时或非零退出，也会尝试保存已经产生的记录。[Trial 收尾逻辑][trial]

<details class="source-detail" markdown="1">
<summary>接入细节：目录、configure、能力声明与 context 条件</summary>

`self.logs_dir` 是宿主机的 trial 日志目录；`environment_logs_dir` 是沙箱里的对应目录，普通 Agent 常用 `/logs/agent`。已挂载时直接准备读取，否则通过 environment 接口下载。[BaseAgent][agent-base]、[环境接口][environment]

`configure()` 在 `install()` 之后执行，跳过安装时也会运行。上传本次 runner、注册 MCP 等工作适合放在这里。`MODEL_CONNECTION` 则可声明模型 provider、API 格式与连接输入的映射。[连接解析][auth-resolution]

`AgentCapabilities(atif=True)` 声明集成支持生成 ATIF；它不会代替写文件。输出、恢复原生 session、加载 ATIF 是不同能力。[能力定义][capabilities]

这个快照仅在 `AgentContext` 存在且为空时调用 post-run hook。若在 `run()` 里先填了 context，就不要再假定框架一定会通过该 hook 生成轨迹。Terminus 自己填 context、自己保存轨迹，属于另一条实现路径。

内置名称由 `AgentFactory` 映射到类；本地开发也能用 `-a my_package.my_agent:MyAgent` 指定实现。[Factory][factory]

</details>

## 六、ATIF 在代码中怎样约束这些记录 {#atif-models}

前面讲的是格式的含义。到了代码里，Harbor 用一组 **Pydantic 数据模型**来表达它。可以把它理解为：创建记录时，同时检查字段类型和必须满足的规则。

先看最外面的两层：[模型目录][model-dir]

```text
Trajectory：整份轨迹
  ├─ agent：谁执行了任务
  └─ steps：按顺序排列的步骤
       ├─ message：说了什么
       ├─ tool_calls：做了什么
       ├─ observation：得到了什么反馈
       └─ metrics：用了多少 token、花费多少
```

在转换器整理完事件之后，便可以创建这些对象。下面是一个可独立构造的最小例子：

```python
trajectory = Trajectory(
    schema_version="ATIF-v1.7",
    agent=Agent(name="teaching-agent", version="1.0"),
    steps=[
        Step(step_id=1, source="user", message="Create hello.txt."),
        Step(step_id=2, source="agent", message="Done."),
    ],
)
```

真实转换器会把工具、结果、统计等一并填进去。最后调用 `to_json_dict()` 导出，再写成 JSON 文件。[根模型与导出方法][trajectory-model]

### 哪些错误会被拦下来

最容易理解的是下面三种：

- 步骤编号是 1、2、5，中间有空洞：拒绝。
- 某个结果写着来自 c9，但同一步里没有 c9 调用：拒绝。
- 在 user step 上填入仅适用于 Agent 的工具调用或模型用量：拒绝。

第一、二种由 `Trajectory` 的校验方法处理；第三种由 `Step` 处理。自定义信息应放进允许的 `extra` 字典，随意增加顶层字段也会被拒绝。[Step 校验][step-model]、[Trajectory 校验][trajectory-model]

> 校验通过，表示记录满足格式规则。它不会证明工具真的执行了，也不会证明日志已经完整。

例如，工具返回了错误，只要调用编号和内容结构正确，这仍然是一份有效记录。反过来，任务成功了，轨迹的字段关联也可能写错。

想检查文件，可以运行官方校验器：

```bash
python -m harbor.utils.trajectory_validator path/to/trajectory.json
```

它复用上述模型，还能在文件输入下检查引用的本地媒体路径。[校验器][validator]

<details class="source-detail" markdown="1">
<summary>协议补充：完整校验位置、Step 计数与版本边界</summary>

| 方法 | 规则 |
| --- | --- |
| `Step.validate_timestamp()` | 时间能解析为 ISO 8601 |
| `Step.validate_agent_only_fields()` | Agent 专属字段放在 agent step |
| `Step.validate_llm_call_count_zero_fields()` | 没有模型调用的确定性调度，不带 LLM 用量或 reasoning |
| `Trajectory.validate_step_ids()` | 编号从 1 连续递增 |
| `Trajectory.validate_tool_call_references()` | 结果引用同一步中的调用 |
| `Trajectory.validate_embedded_subagent_trajectory_ids()` | 嵌入子轨迹的文档 ID 存在且不重复 |

`message` 必需，但可以是空字符串，模型只有工具调用时不必编造回答。原生角色 `assistant` 要映射成 ATIF 的 `agent`。

`llm_call_count` 表达 step 代表几次模型调用。0 表示确定性调度；1 表示一次；大于 1 表示聚合；未填则表示生产者没有追踪它。

该快照的根模型接受 v1.0 至 v1.8，默认 v1.8；Claude Code、Codex、OpenCode 等转换器仍显式输出 v1.7。消费者应读取文件自己的 `schema_version`。当前实现也不会仅凭版本字符串，为每个历史版本切换一整套独立的严格模型。

</details>

## 七、动手看：7 条事件怎样变成 3 个 Steps {#try-converter}

仓库附了一个 [教学转换器]({{ '/assets/examples/harbor-atif/convert_demo.py' | relative_url }})。它接收 [合成 JSONL 输入]({{ '/assets/examples/harbor-atif/native.synthetic.jsonl' | relative_url }})，使用 Harbor 的真实 ATIF 模型生成结果。

输入就是开头的 7 条事件，使用教学字段 `turn_id` 和 `call_id`。它不模拟某个厂商的全部日志协议，也不执行记录里的命令。

打开 `convert_events()`，主要看两张表：

```python
turns = {}   # turn_id → 一轮模型输出
calls = {}   # call_id → 发起调用的那一轮
```

收到工具调用时，先把调用加入所属轮次，再记下它的编号。收到结果时，按编号取回那一轮，将结果放进去。

下面是简化的核心逻辑：

```python
# 发起调用
turn = get_turn(event["turn_id"])
turn.setdefault("tool_calls", []).append(tool_call)
calls[event["call_id"]] = turn

# 结果到达
owner = calls[event["call_id"]]
owner.setdefault("observation", {"results": []})["results"].append(result)
```

因此 c2 先返回也没有关系：它仍会通过 `calls["c2"]` 回到正确轮次。所有事件处理完后，再按顺序分配 step ID、构造 `Trajectory` 并写文件。

在安装了本文对应 Harbor 版本的 Python 3.12+ 环境中，从博客仓库根目录执行：

```bash
python assets/examples/harbor-atif/convert_demo.py \
  assets/examples/harbor-atif/native.synthetic.jsonl \
  /tmp/harbor-atif-demo/trajectory.json

python -m harbor.utils.trajectory_validator \
  /tmp/harbor-atif-demo/trajectory.json
```

预期结果是 **7 条事件变成 3 个 Steps**。可与仓库中的 [示例输出]({{ '/assets/examples/harbor-atif/trajectory.synthetic.json' | relative_url }}) 对照。

<details class="source-detail" markdown="1">
<summary>接入示例：把纯转换函数放进 post-run hook</summary>

下面只展示 hook，省略安装、配置和运行入口。假设 `run()` 已让 Agent 把教学格式日志写入 `native.jsonl`，Trial 也完成了同步：

```python
class MyAgent(BaseInstalledAgent):
    capabilities = AgentCapabilities(atif=True)

    def populate_context_post_run(self, context):
        native = self.logs_dir / "native.jsonl"
        if not native.exists():
            return

        events = [json.loads(line) for line in native.read_text().splitlines()
                  if line.strip()]
        trajectory = convert_events(events)
        target = self.logs_dir / "trajectory.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(trajectory.to_json_dict()))
        temporary.replace(target)

        metrics = trajectory.final_metrics
        if metrics is not None:
            context.n_input_tokens = metrics.total_prompt_tokens
            context.n_output_tokens = metrics.total_completion_tokens
            context.n_cache_tokens = metrics.total_cached_tokens
```

真正适配原生 Agent 时，需要换成它的日志解析器，补齐安装、版本检测和模型连接配置。

教学脚本还会拒绝孤立结果与重复 call/result、保留未返回的调用、对重复 usage 快照取最后值，并在用量不完整时保留未知值。这些都是明确的实现选择。

</details>

## 八、文件保存在哪里，Hub 又保存什么 {#save-and-upload}

单步骤任务完成后，先找这三个位置即可：[目录定义][paths]

```text
jobs/<job-name>/<trial-name>/
├─ config.json               本次用了什么 Agent、模型与任务
├─ result.json               评分、异常、用量等运行结果
└─ agent/
   ├─ trajectory.json        转换后的 ATIF
   └─ ...                    原生会话、stdout、媒体文件
```

目录名未必包含 Agent 或模型名；要确认对应关系，应结合配置、trial 身份和结果。也不要默认 ATIF 的 `session_id` 就是 trial UUID。[Trial 命名][trial-config]

保留原始日志也很有用：转换器升级后可以重新解析；遇到中断或缺失字段，可以核对转换前到底留下了什么。

上传到 Hub 时，公开客户端会保存 trial 归档，并可单独保存根部的规范轨迹：

```text
trials/<trial-uuid>/trial.tar.gz
trials/<trial-uuid>/trajectory.json
```

数据库再关联 job、trial、Agent、模型和评分，文件路径也记录在相应条目里。[上传实现][uploader]、[存储客户端][storage]、[路径提交][upload-db]

需要分清两个开关：

- **`--stream`**：运行中预览轨迹。它定期收集 session，复用转换器刷新预览，有 Agent 和环境能力限制。
- **`--upload`**：trial 完成后上传结果。job 运行中可以陆续出现已完成的 trials，但不表示每个 token 都在持续上传。

实时预览会先写临时文件，再替换正式 JSON；最终收尾另做正式转换。[预览实现][stream-source]

多步骤任务另有 `steps/<step-name>/agent/trajectory.json`。该快照的独立轨迹快捷上传只检查 trial 根部文件，检查多步骤结果时应进入归档里的 steps。[多步骤归档][multi-step]

## 九、进阶：子 Agent、上下文压缩与多模态 {#advanced}

前面的主线已经可以解释普通任务。出现下面这些场景时，再展开相应部分即可。

<details class="source-detail" markdown="1">
<summary>一个 Agent 又调用了子 Agent，怎样关联轨迹？</summary>

先分清三种身份：Trial UUID 对应一次 benchmark 尝试；`session_id` 对应逻辑运行；`trajectory_id` 对应一份轨迹文档。

同一次运行可以有多个子 Agent、多个文档，因此 session ID 可以共享。嵌入引用应以 `trajectory_id` 匹配父文档中的 `subagent_trajectories`；外部文件引用则使用 `trajectory_path`。仅提供 session ID 不能唯一定位子文档。[子轨迹引用模型][subagent-ref]

协议支持嵌入子轨迹，不代表所有转换器都会这样输出。Claude 当前也会合并子 Agent 事件，保留 sidechain 身份；读取时要看实际文件采用的是合并时间线、嵌入文档还是外部引用。

</details>

<details class="source-detail" markdown="1">
<summary>上下文压缩后，为什么会有多个 trajectory 文件？</summary>

长任务压缩历史后，可能开始新的轨迹段。Terminus 的 `linear_history` 可生成 `trajectory.json`、`trajectory.cont-1.json` 等文件，并用 `continued_trajectory_ref` 连接。[连续段保存][terminus-cont]

旧内容如果被复制到新段作为上下文，使用 `is_copied_context=True` 标记。它不是本段新发生的交互；构造监督微调（SFT）数据时需要过滤，避免把同一段历史反复算作新样本。

</details>

<details class="source-detail" markdown="1">
<summary>轨迹里有图片或音频，需要一起保存哪些文件？</summary>

消息与 observation 可以从字符串变成 `ContentPart` 数组，里面混合文本与媒体引用。图片、音频另存文件，JSON 记录媒体类型和路径；v1.8 增加音频及可选时长。[内容模型][content-model]

因此，复制 JSON 时也应带上它引用的 `images/`、`audio/` 等文件。单独一个轨迹 JSON，可能不足以完整查看记录。

</details>

<details class="source-detail" markdown="1">
<summary>怎样避免重复计算 token、成本和训练数据？</summary>

先判断 usage 是单次值、更新快照还是累计值。同一消息的更新快照不应重复求和；累计 token 也不能再按每条事件相加。

ATIF 的 `cached_tokens` 是 `prompt_tokens` 的子集。教学样例总输入 75、其中缓存命中 30，因此总输入仍是 75。成本也应区分原生报告值与价格表估算值。

轨迹中可以记录可见 reasoning、token IDs 和 logprobs，但这些信息取决于原始来源是否提供。有一份 ATIF，并不自动意味着有完整逐 token 训练数据。确定性调度与复制上下文也需要根据相应字段筛选。

</details>

## 十、检查一个新适配器时，从这几件事开始 {#verify-adapter}

先用小任务检查关联是否正确，再扩大到完整 benchmark：

1. **一轮发出两个调用**：是否仍是一轮模型记录，用量有没有重复计算。
2. **让结果乱序返回**：是否按 call ID 配对，而不是按到达顺序猜测。
3. **在工具返回前中断**：原始日志与 ATIF 各保留了多少，差异能否解释。
4. **检查用量和附件**：未知用量是否被误写成 0，媒体与子轨迹是否随文件保存。

最后再跑格式校验。它能检查结构，但不代替对上述行为的核对。

文章主线可以带回到开头的那个任务：**用模型调用确定 Step，用调用编号关联结果，再由适配器把不同 Agent 的日志整理成这套结构。** 看到轨迹缺失或统计不对时，先沿这两种关系查，比从整个框架的类名开始查更容易定位。

<details class="source-detail" markdown="1">
<summary>本文参考版本、验证范围与源码阅读入口</summary>

源码固定为 Harbor **`f6cf2e2e86a102a9d2980ce8888f4bda251aa998`**，提交日期 2026-11-03，项目版本 `0.25.0`。图和伪代码按这一快照整理；例子是合成数据。[源码快照][revision]

研究中，所选 Codex、OpenCode、mini-swe-agent 的同步转换与落盘测试共 96 项通过；另用合成 Claude 原生日志验证了真实 post-run 转换。教学样例验证了分组、乱序结果关联、中断记录、缺失用量和格式约束。没有执行真实模型 benchmark 或向 Hub 上传。

按阅读顺序查源码，可以从这些入口开始：

- Claude：`_convert_events_to_trajectory()` → `_convert_event_to_step()`。
- Codex：`_convert_events_to_trajectory()` → `_group_events_by_api_call_id()`。
- OpenCode：`_parse_stdout()` → `_convert_events_to_trajectory()`。
- 框架：`SingleStepTrial._run_agent()` → `Trial._sync_agent_output()`。
- 协议：`Step` → `Trajectory` → `TrajectoryValidator`。

本文描述的是公开客户端可核实的行为，没有据此推断 Hosted Harbor 服务端内部的额外调度与索引实现。

</details>

[revision]: https://github.com/harbor-framework/harbor/commit/f6cf2e2e86a102a9d2980ce8888f4bda251aa998
[factory]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/factory.py
[environment]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/environments/base.py
[trial]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/trial/trial.py#L1160
[trial-config]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/models/trial/config.py#L580
[trial-result]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/models/trial/result.py
[paths]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/models/trial/paths.py
[agent-base]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/base.py
[installed-base]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/base.py#L1149
[capabilities]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/capabilities.py
[auth-resolution]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/agent_auth_resolution.py
[single-step]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/trial/single_step.py#L73
[multi-step]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/trial/multi_step.py#L337
[claude-run]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/claude_code.py#L2104
[claude-convert]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/claude_code.py#L1309
[claude-normalize]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/claude_code.py#L1427
[claude-step]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/claude_code.py#L815
[claude-save]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/claude_code.py#L1751
[codex-run]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/codex.py#L1911
[codex-session]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/codex.py#L818
[codex-convert]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/codex.py#L1357
[codex-group]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/codex.py#L1017
[codex-save]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/codex.py#L1673
[opencode-parse]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/opencode.py#L535
[opencode-convert]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/opencode.py#L567
[opencode-run]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/opencode.py#L889
[gemini]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/gemini_cli.py
[mini]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/mini_swe_agent.py
[openhands]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/installed/openhands.py#L833
[terminus]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/terminus_2/terminus_2.py#L1717
[terminus-cont]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/agents/terminus_2/terminus_2.py#L2088
[rfc]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/rfcs/0001-trajectory-format.md
[model-dir]: https://github.com/harbor-framework/harbor/tree/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/models/trajectories
[trajectory-model]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/models/trajectories/trajectory.py
[step-model]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/models/trajectories/step.py
[observation-result]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/models/trajectories/observation_result.py
[subagent-ref]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/models/trajectories/subagent_trajectory_ref.py
[content-model]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/models/trajectories/content.py
[validator]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/utils/trajectory_validator.py
[stream-source]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/trial/sync_trajectory.py
[uploader]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/upload/uploader.py#L540
[storage]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/upload/storage.py
[upload-db]: https://github.com/harbor-framework/harbor/blob/f6cf2e2e86a102a9d2980ce8888f4bda251aa998/src/harbor/upload/db_client.py#L569
