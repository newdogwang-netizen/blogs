# newdogwang's blog

一个记录技术实践、系统原理、AI 基础设施与持续研究的个人博客。

## 在线阅读

<https://newdogwang-netizen.github.io/blogs/>

## 分类与专题

- **AI 与 GPU**：GPU 架构、互联、内存精度，以及 HGX B300 八卡节点与 18 台服务器集群设计。
- **NLP 与语言模型**：SLP3 阅读笔记，独立于 GPU 分类。
- **云原生与 AI 基础设施**：[专题目录](https://newdogwang-netizen.github.io/blogs/cloud-native-ai-infra/)，收录 2026 上海联合大会相关的 14 篇技术研究笔记（8 篇主线 + 6 篇延伸）。

新系列配有 28 张原创 SVG 图，讨论实现细节、故障处理和测试方法。各篇注明资料来源和参考版本，区分已有能力与架构推导；文中算例不是 GPU 实测结果。

## 2026 GPU 进阶笔记

1. [2026 数据中心 GPU：为什么从 8 卡服务器进入 72 卡机架](_posts/2026-08-27-2026-rack-scale-gpu.md)
2. [HBM4、FP4 与 KV Cache：算力数字为什么经常骗人](_posts/2026-08-27-memory-precision-kv-cache.md)
3. [Scale-up 与 Scale-out：NVLink、UALink、PCIe 和 800G 网络](_posts/2026-08-27-scale-up-scale-out-network.md)
4. [训练与推理选型：从容量估算、通信模型到验收测试](_posts/2026-08-27-sizing-benchmark-playbook.md)

网站使用 GitHub Pages 与 Jekyll 发布，支持分类、标签和专题系列。向 `_posts/` 添加带有 Front Matter 的 Markdown 文件并推送到 `main`，即可发布新文章。发布前请检查文章内的版本和引用、内部链接、SVG 可读性及移动端排版。

研究大纲与本地验收记录在 `outputs/`，旧方向草稿保留在 `work/superseded-cloud-native/`；两者均被排除在建站之外。工作区写作与线上发布是两个步骤，未推送的修改不会改变线上网站。
