# intentCloud

当前LLM本质仍是基于token的自回归生成，依赖逐个预测下一token，虽含隐式语义，却未显式建模"语言的组织过程"。人类生成语言时，通常先形成概念意图（如天空、散射、蓝光的关系），再逻辑规划，最后才转化为语句。为此，我提出新范式：在语言输出前，引入可持续更新的"语义状态空间"（意图空间），存储概念、关系、目标与约束。生成不再依赖token历史，而是依赖该结构化状态，并由其产生文本。该状态应随对话动态演化，实现跨轮次一致性与延续性，而非每轮从上下文临时推断。这并非优化token预测，而是重构生成结构：从"序列生成"转向"语义组织 → 语言表达"的双阶段系统。

---

## Intent-Space-Prototype — 意图驱动型生成系统原型

实现"语言生成"与"意图状态"的解耦，通过激活引导（Activation Steering）从内部驱动 LLM。

## 快速开始

```bash
git clone https://github.com/madheu/intentCloud.git
cd intentCloud/IntCog-R

# 安装依赖
pip install torch transformers pydantic pyyaml structlog pytest

# 运行 MVP（自包含模式，无需外部模型）
python mvp_steering_demo.py

# 运行 MVP（本地模型模式）
python mvp_steering_demo.py --model qwythos

# 运行全量测试
python -m pytest tests/ -v
```

## 项目结构

```
IntCog-R/
├── core/                         # 核心模块
│   ├── intent_extractor.py       # 意图提取器
│   ├── intent_cloud.py           # 意图云 v3（核心类 + CloudEdge + update_weight）
│   ├── intent_cloud_config.py    # 拓扑演化配置（η/γ/K_d/δ + α/ε/max_iter）
│   ├── intent_cloud_dynamics.py  # 快慢分离动力学
│   ├── intent_to_vector.py       # 意图→向量映射层
│   ├── anchor_system.py          # 锚点系统（概念/拓扑/结构三类锚点）
│   ├── steering.py               # 激活引导器
│   ├── memory.py                 # 三层缓存记忆
│   ├── metacognition.py          # 元认知监控
│   ├── workspace.py              # 全局工作空间
│   ├── attention.py              # 注意力路由
│   ├── config_loader.py          # 配置加载
│   ├── constraint_executor.py    # 硬约束执行
│   ├── fallback.py               # 降级处理
│   ├── generator.py              # 生成器
│   ├── internal_speech.py        # 内部言语
│   ├── json_utils.py             # JSON 工具
│   └── models.py                 # 数据模型
├── tests/                        # 测试套件（160/160 通过）
│   └── unit/
├── benchmarks/                   # 基准测试
├── mvp_steering_demo.py          # MVP 演示脚本
├── main.py                       # 主入口
├── config.yaml                   # 配置文件
└── pyproject.toml                # 项目配置
```

## 数据流架构

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

### 意图云 v3 双层时间尺度

```
process_interaction(input_signals)
    │
    ├─ Phase 1 — Fast: diffuse_activation()
    │   └─ 固定权重下迭代：
    │        Δa[i] = α × (input[i] + Σw[j→i]×a[j] - a[i])
    │        收敛条件：max(|Δa|) < ε 或达到 max_iter
    │
    └─ Phase 2 — Slow: update_weights_after_diffusion()
        └─ 遍历所有非锚点边：
             w_new = Proj(w_old + η·a_i·a_j)
                   - γ·(w_old - w_ref)
                   - K_d·(w_old - w_prev)
             死区：|a_i·a_j| < δ 时跳过赫布项
             投影：clamp(w_new, w_min, w_max)
```

## 开发里程碑

| Harness | 主题 | 状态 | 核心交付 |
|---------|------|------|----------|
| 1-5 | 基础架构（提取器/意图云v2/steering/MVP） | ✓ 完成 | 67 测试 |
| 6 | 综合权重更新律（赫布+参考拉回+动量阻尼） | ✓ 完成 | `update_weight()` + 25 测试 |
| 7 | 锚点系统（概念/拓扑/结构三类锚点） | ✓ 完成 | `AnchorSystem` + 37 测试 |
| 8 | 快慢分离动力学（双层时间尺度） | ✓ 完成 | `diffuse_activation()` + 15 测试 |
| 9 | 意图→向量映射层 | ✓ 完成 | `map_intent_to_vector()` + 16 测试 |

## 关键设计决策

1. **identity 固定**：身份从系统配置读取，不由 LLM 提取，保证身份稳定性
2. **双阶段范式**：先组织语义（蓝图），再生成语言，而非直接调用 LLM
3. **激活引导**：从内部层注入引导向量，而非在输出端控制
4. **双层时间尺度**：快子系统（激活扩散）收敛后才触发慢子系统（权重更新），保证演化稳定
5. **综合更新律**：赫布学习 + 参考模型拉回 + 动量阻尼，三项耦合抑制漂移和振荡
6. **三类锚点**：概念锚点（固定激活）/ 拓扑锚点（固定边权重）/ 结构锚点（全局限幅）
7. **配置驱动演化**：所有可调参数集中在 `IntentCloudConfig`，模板从 `config.yaml` 加载

## 已知问题

| 问题 | 严重程度 | 状态 |
|------|----------|------|
| CharTokenizer 不可调用 | 高 | 远端已修复，本地待同步 |
| Qwythos 本地模型路径未配置 | 中 | 待确认 |
| GPU 版 torch 安装问题（Python 3.14 兼容性） | 低 | 已绕过（CPU 版可用） |

## 待办事项

### P0
- 修复 CharTokenizer `__call__` 方法（本地同步）
- 确认 Qwythos 模型本地路径

### P1
- 意图云 v3 端到端集成（`process_interaction()` 接入主管线）
- 蓝图→意图云节点映射
- chromadb 集成
- OpenAI LLM 客户端
- 输出验证模块

### P2
- 多层注入策略
- 侧抑制机制
- 性能优化（GPU 加速、批量处理）
- 端到端集成测试

## v0.1 已知问题（历史）

| 问题 | 状态 |
|------|------|
| ID#001 幽灵激活（阈值 0.02 过低） | ✅ 已修复 |
| constraints 不跨轮持久 | ✅ 已修复 |
| identity 字段 23/24 为"未知" | ✅ 已修复 |
| 模型固有身份覆盖蓝图 | ✅ 已修复 |
| 无遗忘机制 | ✅ 已修复 |
| 无量化指标/对照实验 | ✅ 已修复 |
