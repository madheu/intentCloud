# Intent-Space-Prototype 开发进度同步文档

## 一、项目概述

**项目名称**：Intent-Space-Prototype  
**核心目标**：验证「语义组织（意图云）→ 语言表达」双阶段生成范式  
**技术路线**：激活引导（Activation Steering）——从内部驱动 LLM，而非 API 调用  

## 二、当前状态

| 维度 | 状态 |
|------|------|
| 代码仓库 | https://github.com/madheu/intentCloud (main 分支) |
| 测试覆盖 | 160/160 通过 |
| MVP 演示 | 可用（自包含模式） |
| 意图云重构 | Harness 6-9 全部完成 |

## 三、已完成的核心模块

### 3.1 意图提取器（Intent Extractor）
- **文件**：`core/intent_extractor.py`
- **功能**：将用户输入转化为结构化蓝图 JSON
- **蓝图字段**：`identity`(固定) / `core_task` / `deep_goal` / `constraints` / `concepts`
- **状态**：✓ 完成

### 3.2 意图云（Intent Cloud）v3 — 双层时间尺度重构
- **文件**：`core/intent_cloud.py`、`core/intent_cloud_dynamics.py`、`core/intent_cloud_config.py`、`core/anchor_system.py`
- **核心架构**：
  - **快子系统（激活扩散）**：固定权重下迭代扩散，`Δa[i] = α × (input[i] + Σw[j→i]×a[j] - a[i])`
  - **慢子系统（权重更新）**：激活收敛后才更新，综合更新律 `w_new = Proj(w_old + ηa_i·a_j) - γ(w_old - w_ref) - K_d(w_old - w_prev)`
  - **三类锚点**：概念锚点（固定激活）/ 拓扑锚点（固定边权重）/ 结构锚点（全局限幅）
- **v3 API**：
  - `process_interaction(input_signals)` — 完整交互：先扩散，收敛后再更新权重
  - `diffuse_activation(input_signals)` — 快子系统：激活扩散到收敛
  - `update_weights_after_diffusion(activations)` — 慢子系统：更新非锚点边权重
  - `add(blueprint)` / `retrieve(blueprint, top_k)` / `enhance_blueprint(blueprint)` — v2 API 保留
- **后端**：`InMemoryIntentStorage`（可替换为向量数据库）
- **状态**：✓ 完成（Harness 6-9）

### 3.3 意图到向量映射层（Intent-to-Vector Mapping）
- **文件**：`core/intent_to_vector.py`
- **三步映射**：
  1. **找路径**：选权重最高的非锚点边作为活跃意图路径
  2. **对比对生成**：根据边类型（refines/contrasts/evokes/constrains）从配置读取模板，填充 `{concept}`/`{target}` 生成正反例
  3. **向量提取 + 强度**：复用 `SteeringVectorExtractor`，强度 = `base_strength × edge_weight × trust_score`
- **降级策略**：无边→None，无 extractor→零向量，未知边类型→回退 connects 模板
- **状态**：✓ 完成（Harness 9）

### 3.4 激活引导器（Activation Steering）
- **文件**：`core/steering.py`
- **核心组件**：
  - `SteeringVector` — 引导向量（归一化方向向量）
  - `SteeringVectorExtractor` — 从蓝图/对比对提取向量
  - `SteeringInjector` — 通过 forward hook 注入指定层
- **注入方式**：`hidden_states[last_token] += steering_vector × strength`
- **状态**：✓ 完成

### 3.5 MVP 演示脚本
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

### 4.1 总览

```
用户输入
  → 意图提取器（生成蓝图 JSON）
  → 意图云 v3：
      ├─ 快子系统：激活扩散 → 收敛
      │   └─ 概念锚点激活值固定
      └─ 慢子系统：权重更新（收敛后触发）
          └─ 拓扑锚点边权重固定
  → 意图到向量映射层：
      ├─ 找路径（权重最高的非锚点边）
      ├─ 生成对比对（按边类型模板）
      └─ 提取 steering_vector + 计算强度
  → SteeringInjector.inject()（注入 LLM 中间层）
  → LLM 生成（受引导向量影响）
  → 输出验证
  → 返回用户
```

### 4.2 意图云 v3 双层时间尺度

```
process_interaction(input_signals)
    │
    ├─ Phase 1 — Fast: diffuse_activation()
    │   └─ 固定权重下迭代：
    │        Δa[i] = α × (input[i] + Σw[j→i]×a[j] - a[i])
    │        clamp(a[i], 0, a_max)
    │        收敛条件：max(|Δa|) < ε 或达到 max_iter
    │
    └─ Phase 2 — Slow: update_weights_after_diffusion()
        └─ 遍历所有非锚点边：
             w_new = Proj(w_old + ηa_i·a_j)
                   - γ(w_old - w_ref)
                   - K_d(w_old - w_prev)
             死区：|a_i·a_j| < δ 时跳过赫布项
             投影：clamp(w_new, w_min, w_max)
```

## 五、待办事项

### P0 — 优先级最高

1. **LLM 模型选择**：确认使用哪种 HF 格式模型运行 Activation Steering（Qwythos GGUF 不支持 PyTorch forward hook）
   - Qwythos 9B → 通过 llama-cpp-python 直推，仅支持 logits-level steering
   - GPT-2 124M → 支持 forward hook，但太小效果微弱
   - 建议：寻找 HF 格式的中型模型（如 Qwen-1.8B / TinyLlama-1.1B）

2. **意图云 v3 端到端集成**：将 `process_interaction()` 接入主管线

### P1 — 接下来

3. **蓝图→意图云节点映射**：将 IntentBlueprint 的 fields 映射到意图云的外壳节点和边类型
4. **Qwythos 9B + intent_to_vector 集成**：用 Qwythos 做 backbone，走通「蓝图→对比对→steering vector→生成」链路
5. **输出验证模块**：检查生成结果是否符合蓝图约束

### P2 — 未来规划

6. **多层注入策略**：根据意图路径动态选择注入层和强度
7. **侧抑制机制**：全局抑制信号，防止激活扩散失控
8. **性能优化**：GPU 加速、批量处理
9. **chromadb 集成**：替换 InMemoryIntentStorage
10. **端到端集成测试**：完整数据流测试套件

## 六、技术栈

| 维度 | 要求 |
|------|------|
| 语言 | Python 3.10+ |
| 核心库 | torch, transformers, pydantic, llama-cpp-python |
| 模型 | Qwythos（GGUF, 9B, via llama-cpp-python）或 GPT-2（HF 格式） |
| 测试 | pytest（160/160 通过） |
| 配置 | config.yaml |

## 七、运行方式

```bash
# 克隆仓库
git clone https://github.com/madheu/intentCloud.git
cd intentCloud/IntCog-R

# 安装依赖
pip install torch transformers pydantic pyyaml structlog pytest httpx llama-cpp-python

# 运行 MVP（自包含模式）
python mvp_steering_demo.py

# 运行 Qwythos 完整管线（需安装 llama-cpp-python + GGUF 模型）
python mvp_qwythos_demo.py

# 运行真实 Activation Steering 演示（GPT-2）
python mvp_real_steering.py

# 运行测试
python -m pytest tests/ -v

# 运行基准测试
python benchmarks/benchmark.py
```

## 八、代码结构

```
IntCog-R/
├── core/                         # 核心模块
│   ├── intent_extractor.py       # 意图提取器
│   ├── intent_cloud.py           # 意图云 v3（核心类 + CloudEdge + update_weight）
│   ├── intent_cloud_config.py    # 拓扑演化配置
│   ├── intent_cloud_dynamics.py  # 快慢分离动力学
│   ├── intent_to_vector.py       # 意图→向量映射层
│   ├── anchor_system.py          # 锚点系统
│   ├── steering.py               # 激活引导器
│   ├── ollama_client.py          # [NEW] Ollama HTTP 客户端
│   ├── qwythos_client.py         # [NEW] Qwythos GGUF 直推客户端
│   ├── json_utils.py             # JSON 工具
│   ├── memory.py                 # 三层缓存记忆
│   ├── metacognition.py          # 元认知监控
│   ├── workspace.py              # 全局工作空间
│   ├── attention.py              # 注意力路由
│   ├── config_loader.py          # 配置加载
│   ├── constraint_executor.py    # 硬约束执行
│   ├── fallback.py               # 降级处理
│   ├── generator.py              # 生成器
│   ├── internal_speech.py        # 内部言语
│   └── models.py                 # 数据模型
├── tests/                        # 测试套件
│   └── unit/                     # 单元测试（17个文件）
├── benchmarks/                   # 基准测试
├── mvp_steering_demo.py          # MVP 演示脚本（TinyTransformer / 本地 HF）
├── mvp_ollama_steering.py        # [NEW] Ollama 版演示
├── mvp_real_steering.py          # [NEW] 真实 Activation Steering 演示（GPT-2）
├── mvp_qwythos_demo.py           # [NEW] Qwythos 9B 完整管线演示
├── main.py                       # 主入口
├── config.yaml                   # 配置文件
└── pyproject.toml                # 项目配置
```

## 九、关键设计决策

1. **identity 固定**：身份从系统配置读取，不由 LLM 提取，保证身份稳定性
2. **双阶段范式**：先组织语义（蓝图），再生成语言，而非直接调用 LLM
3. **激活引导**：从内部层注入引导向量，而非在输出端控制
4. **接口可替换**：`InMemoryIntentStorage` 可替换为 chromadb，不影响上层逻辑
5. **双层时间尺度**：快子系统（激活扩散）收敛后才触发慢子系统（权重更新），保证演化稳定
6. **综合更新律**：赫布学习 + 参考模型拉回 + 动量阻尼，三项耦合抑制漂移和振荡
7. **三类锚点**：概念锚点（固定激活）/ 拓扑锚点（固定边权重）/ 结构锚点（全局限幅），构成不可变骨架
8. **配置驱动演化**：所有可调参数集中在 `IntentCloudConfig`，模板从 `config.yaml` 加载
9. **GGUF 限制确认**：Qwythos GGUF 无法用 PyTorch forward hook，需 HF 格式模型才能做真实 Activation Steering

## 十、已知问题

| 问题 | 严重程度 | 状态 |
|------|----------|------|
| Qwythos GGUF 不支持 PyTorch forward hook | 中 | 已确认，需 HF 格式模型替代 |
| GPT-2 124M 对 Steering 不敏感 | 低 | 已验证，机制正确但效果微弱 |
| Qwythos 输出含 `<think>` 标签（Qwen 家族通病） | 低 | 已修复（json_utils.py 自动剥离） |
| GPU 版 torch 安装问题（Python 3.14 兼容性） | 低 | 已绕过（CPU 版可用） |

## 十一、Harness 里程碑

| Harness | 主题 | 状态 | 交付物 |
|---------|------|------|--------|
| 1-5 | 基础架构（提取器/意图云v2/steering/MVP） | ✓ 完成 | 67 测试通过 |
| 6 | 综合权重更新律（赫布+参考拉回+动量阻尼） | ✓ 完成 | `update_weight()` + 25 测试 |
| 7 | 锚点系统（概念/拓扑/结构三类锚点） | ✓ 完成 | `AnchorSystem` + 37 测试 |
| 8 | 快慢分离动力学（双层时间尺度） | ✓ 完成 | `diffuse_activation()` + 15 测试 |
| 9 | 意图→向量映射层 | ✓ 完成 | `map_intent_to_vector()` + 16 测试 |

---

**最后更新**：2026-07-09
**分支**：main
**提交**：1e92c2a

## 十二、本轮新增文件清单（2026-07-08）

### 意图云 v3 重构（来自远端 Harness 6-9）

| 文件 | 说明 |
|------|------|
| `core/anchor_system.py` | 三类锚点系统（概念/拓扑/结构） |
| `core/intent_cloud_config.py` | 拓扑演化配置参数 |
| `core/intent_cloud_dynamics.py` | 快慢分离动力学（双层时间尺度） |
| `core/intent_to_vector.py` | 意图→向量映射层（对比对生成 + 向量提取） |
| `tests/unit/test_anchor_system.py` | 锚点系统测试（+37） |
| `tests/unit/test_dynamics_separation.py` | 快慢分离测试（+15） |
| `tests/unit/test_intent_cloud_update.py` | 权重更新测试（+25） |
| `tests/unit/test_intent_to_vector.py` | 映射层测试（+16） |

### 模型接入与演示脚本（本地新增，文件头部标注 [NEW]）

| 文件 | 说明 |
|------|------|
| `core/ollama_client.py` | Ollama HTTP 客户端（httpx，raw 模式，logit_bias） |
| `core/qwythos_client.py` | Qwythos GGUF 直推客户端（llama-cpp-python，LogitsProcessor steering） |
| `mvp_ollama_steering.py` | Ollama 版演示（Prompt Engineering 方式） |
| `mvp_real_steering.py` | 真实 Activation Steering 演示（GPT-2 + 对比对向量提取） |
| `mvp_qwythos_demo.py` | Qwythos 9B 完整管线演示（意图提取 + 安全生成） |

### 修复文件

| 文件 | 说明 |
|------|------|
| `core/json_utils.py` | 增加 `<think>` 标签剥离 + 非贪婪 JSON 提取（修复 Qwythos 多 JSON 输出问题） |
