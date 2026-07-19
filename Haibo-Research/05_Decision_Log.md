# 决策日志

记录所有重大决策。每条决策带来源、理由和影响。
按时间顺序排列。未来回头看时，这里的每一条都应该能回答"当时为什么这么决定"。

---

## Decision 2026-07-18-001：暂停 H20 开发

**来源：** PI 审稿委员会（AI Reviewer + Cognitive Scientist + Control Scientist 三方审阅）

**决定：** 暂停 H20（常识层扩展、编译层、节点生长机制）的开发。启动控制有效性分析和反馈回路设计。

**理由：**
1. 海波在交互层是开环系统（缺少 LLM 输出到图状态的反馈回路）
2. 双语者注入的耦合强度不足以覆盖 LLM 的 RLHF 偏好（MAX_INJECT=4 截断 99% 图信息）
3. 扩散收敛性无谱半径保证（入度权重列和未归一化）
4. 认知状态空间未形式化定义
5. H20 的三个目标不解决上述任何一个问题

**证据来源：**
- Cognitive Scientist review_001（7 条审稿意见）
- Control Scientist review_001（5 条控制论分析）
- H11-H19 实验数据

**影响：**
- 停止：编译层开发、元认知节点设计、节点生长机制实验
- 启动：控制有效性分析、反馈回路设计
- 允许：现有模块维护、测试补充

---

## Decision 2026-07-18-002：路径 C 触发——注入路线证伪

**来源：** 互信息实验（Pilot 1 "大海" + Pilot 2 "桌子"，共 40 次生成）

**决定：** 确认图→LLM 的 embedding 注入控制在当前配置下被证伪。触发路径 C。

**证据：**

| 指标 | Pilot 1（大海） | Pilot 2（桌子） | 退出阈值 |
|------|---------------|----------------|----------|
| 类间/类内 embedding 距离比 | **0.93** | **0.99** | < 1.2 |
| 关键词分类 | FEAR 10/10 awe | 全部 neutral | — |
| 综合判断 | 不可区分 | 不可区分 | 终止 |

**科学含义：**
1. FEAR 注入与基线的 LLM 输出来自同一分布
2. 4 个虚拟 token 的输入层注入信号被 Qwen 1.5B 的自回归生成循环完全淹没
3. 这是 Control Scientist 预测的"1 瓦特控制 1000 瓦特"的实验验证

**影响：**
- 废弃：BilingualInjector（embedding 注入路线关闭）
- 废弃：chat_with_haibo.py 旧版
- 写新脚本：talk_to_haibo.py（纯 prompt 通信，无注入）
- 启动：方向修正讨论

---

## Decision 2026-07-18-003：方向修正为 Haibo 2.0

**来源：** 阅读 BriLLM 论文 + PI 方向反思记录

**决定：** 项目从"图控制 LLM"转向"语义涌现（Semantic Emergence）"研究。新核心问题：**概念是否能够从 Token 关系中自然形成？** 不再试图控制任何 LLM。

**理由：**
1. BriLLM（上海交大）的 Token Graph 路线与 Haibo 关注的问题不同——词图 vs 语义图
2. 不冲突：一个是"Token 如何传播"，一个是"Semantic 如何形成"
3. Haibo 1.0 试图一步完成"认知+语义+意图+语言+推理+控制"，跨度过大
4. 新方向从最底层开始：验证语义信号是否存在于词共现中

**新架构假设链：**
```
Input Text → Token Graph → Semantic Emergence → Semantic Graph → Context → Intent → Language Generation
```

**影响：**
- 关闭：Haibo 1.0 所有依赖注入的路线
- 存档：BilingualInjector、SteeringInjector、chat_with_haibo.py
- 启动：Phase 0 最小实验（词袋 + PCA 验证语义信号存在性）
- 启动：Haibo 2.0 核心假设文档（H2.0-1~4，链式依赖）

---

## Decision 2026-07-18-004：Phase 0 实验设计

**来源：** AI Reviewer（基于新方向反思）

**决定：** Phase 0 不建任何新架构。先用 TF-IDF + PCA 检验在同一语料中，"苹果科技"和"苹果水果"两类句子是否自然可分。

**实验：** `06_Experiment/p0_bow_apple.py`

**通过标准：** 类间/类内余弦距离比 > 1.2

**不通过则：** 语义不在词的共现中，Token Graph 方向没有数据基础。需重新审视。

**状态：** 已写代码，待执行者运行。
