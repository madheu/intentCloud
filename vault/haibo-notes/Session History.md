---
tags: [haibo, 会话, 历史]
---

# Haibo 项目会话记录（Hermes Agent 视角）

> 时间线：从 Haibo 1.0 到当前语义图研究
> 最后更新：2026-07-22

---

## Phase 0：Haibo 1.0（2026-07-18 之前）

### 背景
- intentCloud 原型基于 GPT-2 124M → 迁移到 Qwen2.5-0.5B（2026-07-17）
- 核心机制：**语义图 → Embedding 注入 → LLM 生成控制**
- 已完成 H01-H19 全部测试（~101/102 通过）
- 里程碑：H12 双海波拓扑演化、H19 GraphInterpreter+人格锚定

### 关键发现
- Qwen hidden state vs input embedding 量级差 33 倍，必须归一化
- VIRTUAL_REPEAT=3 增强注入信号
- 恐惧是唯一"窄路"路径（GPT-2 时代发现）
- H16e 认知测试 16/16 通过

### 相关文件
- `E:\intentCloud\logs\chat_*.md` — 自动对话日志
- `E:\intentCloud\.hermes\session_context.md` — Haibo 1.0 上下文

---

## Phase 1：方向修正（2026-07-18）

### 事件
PI 审查委员会三方审阅，发现五个核心问题：
1. 海波在交互层是**开环系统**（无 LLM 输出→图状态反馈）
2. 注入耦合强度不足以覆盖 RLHF 偏好（MAX_INJECT=4 截断 99%）
3. 扩散无谱半径保证
4. 认知状态空间未形式化
5. H20 无法解决上述问题

### 互信息实验（Pilot 1 "大海" + Pilot 2 "桌子"）
- **结果**：类间/类内 embedding 距离比 0.93 和 0.99（阈值 < 1.2 = 不可区分）
- **MI 测量**：I(intent_cloud; y_LLM) ≈ 0
- **结论**：FEAR 注入与基线的 LLM 输出来自同一分布。4 个虚拟 token 的注入信号被自回归循环完全淹没

### 决策记录
- Decision 2026-07-18-001：暂停 H20 开发
- Decision 2026-07-18-002：路径 C 触发——注入路线证伪
- Decision 2026-07-18-003：方向修正为 Haibo 2.0

### 角色
- PI（首席研究员）+ AI Reviewer + Cognitive Scientist + Control Scientist 三方审阅
- 后续引入 Research Executor 角色

---

## Phase 2：语义消歧——P_Resolve 系列（2026-07-18 晚）

### 核心发现
**P0.8（Token Graph）**
- 节点 = 词 Token，边 = 共现关系
- 所有测试结果：**不确定** ❌
- 根源：错把"词"当语义节点

**P0.9（概念节点）**
- 节点 = 语义概念（科技类、水果类等），分层结构
- 10/11 正确 ✅
- 核心教训：**节点类型定义是语义图设计的第一决策**

### 用户设计贡献
- 用户提出"每次只比 4 个概念，超出就拓层"的树状层级原则
- P_Resolve-10 验证通过
- 形成布尔 4 轴设计思路：天/地/人/心 → 物/事/质/序

---

## Phase 3：P0-A 系列——布尔轴 + 词向量（2026-07-19）

### 进展
- 14 轮实验，6 个歧义词（花/光/行/口等），38 句
- **最佳正确率：50% 封顶**（从未超过 MFS 基线 52.6%）
- jieba 整词匹配 → 破坏单字词
- 子串匹配更鲁棒

### 布尔 4 轴演化
| 版本 | 轴 | 来源 | 状态 |
|------|-----|------|------|
| 早期 | 天/地/人/心 | 用户自创 | 废弃 |
| P0-A | 物/事/质/序 (Entity/Event/Property/Relation) | 上层本体论 | 50%封顶 |
| 转向 | HowNet 义原 + Frame Semantics | 已有理论 | 当前方向 |

### 关键确认（ChatGPT 外部审稿）
- 布尔作为**约束 prior 合理**，作为最终 theory 有问题
- 海波创新应在**计算机制上**，不在语言学上

---

## Phase 4：P0-R 系列——评测修复（2026-07-20 至 07-21）

### 问题
- P0-A 的金标不一致（混合 WSD/话题分类/语素分类）
- 目标污染（全句扫描歧义词）
- 词袋丢失复合词证据

### 修复
- 5 层基线框架：MFS → HowNet 第一义项 → 复合词优先 → 词性过滤 → 义原重叠 → 级联
- predict.py / score.py 分离
- 字符 span 定位（非关键词扫描）
- 各层多样性自检（单标签=终止）
- MFS 从 dev 算，绝不泄露进 blind

### 最终结果（2026-07-21 盲测）
| 层 | 正确率 |
|-----|--------|
| MFS | 55% |
| L2-L5（语义层） | 46-47% |
| **级联 L3→L4→L5→MFS** | **58.3% ✅** |

- 级联通过 53.3% 线 ✅
- 但逐层递增趋势不成立 ❌
- 语义消歧有信号但弱于语料偏差

### 外部顾问
- ChatGPT / Codex / DeepSeek 参与审稿
- ChatGPT 指出"MFS 基线未超"并建议修正评测基础
- 两次占位实现被顾问指出无效

---

## Phase 5：P0-B 方向确立（2026-07-21 后）

### 当前方向
- 接入 HowNet 义原
- LLM 生成语境数据（已生成 2100 句：`p0_b2_llm_contexts.json`）
- 核心转向：不发明语义分类，直接采用已有语言学理论

### BriLLM 关系界定
- BriLLM（上海交大）已覆盖：Token 节点化、动态信号传播、基于信号能量的 next-token prediction
- Haibo 不做重复
- **创新点**：Token → Semantic Node 跃迁

### 新假设链（H-SG-1~5）
- H-SG-1：语义分区——不同上下文的 Token 活动可无监督形成稳定分区
- H-SG-2~5：关系提取、语义角色、图学习、语言映射

---

## Phase 6：记忆迁移 + Obsidian 集成（2026-07-22）

### 事件
- Hermes Memory 从 89%（7,249 字符/45 条）降到 5%（475 字符/4 条）
- User Profile 从 60%（3,045 字符）降到 9%（494 字符）
- 项目数据全部迁入 Obsidian Vault

### Vault 内容
```
C:\...\Obsidian Vault\
├── Haibo Research.md               # 索引
├── Haibo Research Overview.md      # 项目哲学+假设链+1.0结论
├── P0 Series Experiments.md        # 全部实验数据
├── Haibo Design Decisions.md       # 设计决策+原则+审委会
├── Haibo Config.md                 # 环境+技能+发布配置
├── Haibo User Workflow.md          # 工作流偏好+方法论
└── Welcome.md                      # 入门
```

---

## 外部 AI 协作记录

### ChatGPT
- 2026-07-19：布尔轴审稿（先验合理/最终理论有问题）
- 2026-07-20：P0-R 评测问题诊断
- 存档：`E:\intentCloud\Haibo-Research\03_notes\2026-07-19_chatgpt_feedback_boolean.md`
- 存档：`E:\intentCloud\docs\chatgpt_advice.md`

### DeepSeek
- 用于非结构化概念讨论
- 协助设计决策和方向判断

### Codex
- P0-R 实验数据生成的执行顾问

---

## 哲学与设计原则（持续生效）

- **辩证唯物主义**：普遍联系、矛盾驱动、量变质变（贯穿所有设计决策）
- **语言不是思维本身**：语言是思维在低维空间的投影
- **人是社会关系的总和**：AI的"自我"是交互中积累的蓝图/锚点/边权重/记忆
- **Boolean 4-dim 原则**：每超过 4 个维度向外拓一层，每次只比 4 个
- **诚实面对失败**：P0 系列记录了 7 个 falsification 链
- **不重新发明语义学**：用已有理论，创新在计算机制
