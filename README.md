# intentCloud
当前LLM本质仍是基于token的自回归生成，依赖逐个预测下一token，虽含隐式语义，却未显式建模“语言的组织过程”。人类生成语言时，通常先形成概念意图（如天空、散射、蓝光的关系），再逻辑规划，最后才转化为语句。  为此，我提出新范式：在语言输出前，引入可持续更新的“语义状态空间”（意图空间），存储概念、关系、目标与约束。生成不再依赖token历史，而是依赖该结构化状态，并由其产生文本。该状态应随对话动态演化，实现跨轮次一致性与延续性，而非每轮从上下文临时推断。这并非优化token预测，而是重构生成结构：从“序列生成”转向“语义组织 → 语言表达”的双阶段系统。
# Intent Cloud — 前额叶意图云

意图驱动型生成系统原型。实现"语言生成"与"意图状态"的解耦。

## 快速开始

```bash
pip install numpy requests
ollama serve          # 确保本地模型可用
python frontal_lobe_prototype.py
```

## 项目结构

```
intentCloud/
├── frontal_lobe_prototype.py   # 主程序（单文件）
├── intent_cloud.jsonl           # 自动生成：持久化意图记录
├── README.md
└── docs/
    └── analysis_v0.1.md         # 对话实跑分析报告
```

## 架构

```
User Input → Intent Extractor → IntentPacket → IntentCloud.store()
                                                     ↓
                                             divergent_retrieve() → 余弦相似度
                                                     ↓
                                             compose() → 聚合意图蓝图
                                                     ↓
                                             Generator → 最终回答
```

## v0.1 已知问题

| 问题 | 影响 | 状态 |
|------|------|------|
| ID#001 幽灵激活（阈值 0.02 过低） | 首轮无关意图持续污染后续对话 | ✅ 已修复 |
| constraints 不跨轮持久 | 用户约束下一轮丢失 | ✅ 已修复 |
| identity 字段 23/24 为"未知" | 模型未从对话提取用户身份 | ✅ 已修复 |
| 模型固有身份覆盖蓝图 | "我是 Qwythos"压倒角色设定 | ✅ 已修复 |
| 无遗忘机制 | 意图云无限增长，旧记录变噪音 | ✅ 已修复 |
| 无量化指标/对照实验 | 无法测量意图云的效果 | ✅ 已修复 |

## 实跑数据

24 轮对话分析见 `docs/analysis_v0.1.md`
