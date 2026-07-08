# Intent-Space-Prototype 开发进度同步文档

## 一、项目概述

**项目名称**：Intent-Space-Prototype  
**核心目标**：验证「语义组织（意图云）→ 语言表达」双阶段生成范式  
**技术路线**：激活引导（Activation Steering）——从内部驱动 LLM，而非 API 调用  

## 二、当前状态

| 维度 | 状态 |
|------|------|
| 代码仓库 | https://github.com/madheu/intentCloud (main 分支) |
| 测试覆盖 | 67/67 通过 |
| MVP 演示 | 可用（自包含模式） |

## 三、已完成的核心模块

### 3.1 意图提取器（Intent Extractor）
- **文件**：`core/intent_extractor.py`
- **功能**：将用户输入转化为结构化蓝图 JSON
- **蓝图字段**：`identity`(固定) / `core_task` / `deep_goal` / `constraints` / `concepts`
- **状态**：✓ 完成

### 3.2 意图云（Intent Cloud）v2
- **文件**：`core/intent_cloud.py`
- **核心 API**：
  - `add(blueprint)` — 存储蓝图到内存
  - `retrieve(blueprint, top_k)` — Jaccard n-gram 相似度检索
  - `enhance_blueprint(blueprint)` — 合并历史概念/约束
- **后端**：`InMemoryIntentStorage`（可替换为向量数据库）
- **状态**：✓ 完成

### 3.3 激活引导器（Activation Steering）
- **文件**：`core/steering.py`
- **核心组件**：
  - `SteeringVector` — 引导向量（归一化方向向量）
  - `SteeringVectorExtractor` — 从蓝图/对比对提取向量
  - `SteeringInjector` — 通过 forward hook 注入指定层
- **注入方式**：`hidden_states[last_token] += steering_vector × strength`
- **状态**：✓ 完成

### 3.4 MVP 演示脚本
- **文件**：`mvp_steering_demo.py`
- **功能**：走通完整数据流并打印中间产物
- **运行方式**：
  ```bash
  python mvp_steering_demo.py                    # 自包含模式（无外部依赖）
  python mvp_steering_demo.py --model qwythos    # 加载本地 Qwythos 模型
  ```
- **状态**：✓ 可用（自包含模式）

### 3.5 辅助模块
| 模块 | 文件 | 功能 |
|------|------|------|
| 注意力路由 | `core/attention.py` | 基于相似度/显著性的注意力竞争 |
| 配置加载 | `core/config_loader.py` | 从 config.yaml 加载内核配置 |
| 硬约束执行 | `core/constraint_executor.py` | 规则过滤（非安全导向） |
| 降级处理 | `core/fallback.py` | 异常捕获与优雅降级 |
| 三层缓存记忆 | `core/memory.py` | GPU热层→RAM温层→Disk冷层 |
| 元认知监控 | `core/metacognition.py` | 监控→评估→规划三段式 |
| 全局工作空间 | `core/workspace.py` | 异步事件循环 + 注意力广播 |

## 四、数据流架构

```
用户输入
  → 意图提取器（生成蓝图 JSON）
  → 意图云.add()（存储）
  → 意图云.retrieve()（检索相似历史）
  → 意图云.enhance_blueprint()（蓝图增强）
  → 蓝图 → steering_vector（映射）
  → SteeringInjector.inject()（注入 LLM 中间层）
  → LLM 生成（受引导向量影响）
  → 输出验证
  → 返回用户
```

## 五、待办事项

### P0 — 优先级最高

1. **修复 CharTokenizer**：添加 `__call__` 方法，使 `tokenizer(prompt, return_tensors="pt")` 可调用
   - 文件：`mvp_steering_demo.py`
   - 当前状态：报错 `TypeError: 'CharTokenizer' object is not callable`

2. **加载本地 Qwythos 模型**：确保 `--model qwythos` 参数能正确加载本地模型文件
   - 当前状态：报错无法连接 huggingface.co（本地模型路径可能不正确）
   - 需要确认：Qwythos 模型的本地存储路径

### P1 — 接下来

3. **蓝图→引导向量映射优化**：当前使用伪随机哈希，需替换为基于语义的映射
4. **chromadb 集成**：将 `InMemoryIntentStorage` 替换为向量数据库
5. **OpenAI LLM 客户端**：添加真实 LLM 调用（带超时/重试/回退）
6. **输出验证模块**：检查生成结果是否符合蓝图约束

### P2 — 未来规划

7. **对比对引导向量提取**：使用 `mean(act(正例)) - mean(act(反例))` 方法
8. **多层注入策略**：根据蓝图内容动态选择注入层和强度
9. **性能优化**：GPU 加速、批量处理
10. **完整测试套件扩展**：端到端集成测试

## 六、技术栈

| 维度 | 要求 |
|------|------|
| 语言 | Python 3.10+ |
| 核心库 | torch, transformers, pydantic |
| 模型 | Qwythos（本地）或自包含 TinyTransformer |
| 测试 | pytest（67/67 通过） |
| 配置 | config.yaml |

## 七、运行方式

```bash
# 克隆仓库
git clone https://github.com/madheu/intentCloud.git
cd intentCloud/IntCog-R

# 安装依赖
pip install torch transformers pydantic pyyaml structlog pytest

# 运行 MVP（自包含模式）
python mvp_steering_demo.py

# 运行测试
python -m pytest tests/ -v

# 运行基准测试
python benchmarks/benchmark.py
```

## 八、代码结构

```
IntCog-R/
├── core/                    # 核心模块
│   ├── intent_extractor.py  # 意图提取器
│   ├── intent_cloud.py      # 意图云 v2
│   ├── steering.py          # 激活引导器
│   ├── memory.py            # 三层缓存记忆
│   ├── metacognition.py     # 元认知监控
│   ├── workspace.py         # 全局工作空间
│   ├── attention.py         # 注意力路由
│   ├── config_loader.py     # 配置加载
│   ├── constraint_executor.py # 硬约束执行
│   ├── fallback.py          # 降级处理
│   ├── generator.py         # 生成器
│   ├── internal_speech.py   # 内部言语
│   ├── json_utils.py        # JSON 工具
│   └── models.py            # 数据模型
├── tests/                   # 测试套件
│   └── unit/                # 单元测试（13个文件）
├── benchmarks/              # 基准测试
├── mvp_steering_demo.py     # MVP 演示脚本
├── main.py                  # 主入口
├── config.yaml              # 配置文件
└── pyproject.toml           # 项目配置
```

## 九、关键设计决策

1. **identity 固定**：身份从系统配置读取，不由 LLM 提取，保证身份稳定性
2. **双阶段范式**：先组织语义（蓝图），再生成语言，而非直接调用 LLM
3. **激活引导**：从内部层注入引导向量，而非在输出端控制
4. **接口可替换**：`InMemoryIntentStorage` 可替换为 chromadb，不影响上层逻辑

## 十、已知问题

| 问题 | 严重程度 | 状态 |
|------|----------|------|
| CharTokenizer 不可调用 | 高 | 待修复 |
| Qwythos 本地模型路径未配置 | 中 | 待确认 |
| GPU 版 torch 安装问题（Python 3.14 兼容性） | 低 | 已绕过（CPU 版可用） |

---

**最后更新**：2026-07-08  
**分支**：main  
**提交**：a229517