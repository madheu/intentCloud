# intentCloud — 意图驱动型生成系统原型

当前 LLM 本质仍是基于 token 的自回归生成，依赖逐个预测下一个 token，
虽含隐式语义，却未显式建模范畴组织过程。
本项目在语言输出前，引入可持续更新的语义状态空间（意图空间），
存储概念、关系、目标与约束。
生成不再依赖 token 历史，而是依赖该结构化状态，并由其产生文本。
该状态应随对话动态演化，实现跨轮次一致性与延续性。

**从"序列生成"转向"语义组织 → 语言表达"的双阶段系统。**

---

## 项目状态（2026-07-14）

```
✅ H10: 常识加载（36节点 / 42边）
✅ H11: 端到端链路贯通（Qwen2.5-7B 4-bit，不OOM）
✅ H12: 双海波拓扑演化 — 核心验证通过
✅ H13: 控制曲线校准
✅ H14: 引导弹性地图 + 自适应强度调节器
✅ H15: 视角转换器（认知重评机制）
✅ H16: 多轮对话上下文管理器 + 内化门控
⬜ H17: 理解前端（输入→映射→内化）
```

---

## 核心发现

| 发现 | 说明 |
|------|------|
| **短语 > 单字词** | 短词中层embedding坍缩(cos_sim 0.9998)，短语修复后有效(0.40) |
| **海波驱动LLM** | 双海波对照：拓扑演化决定输出方向 |
| **恐惧是唯一窄路** | 恐惧[0.01,0.05] vs 敬畏[1.0,10.0] — 死亡焦虑假设 |
| **生命对冲死亡** | 恐惧链视角转换：恐惧40%+生命25%+壮丽20%+敬畏15% |
| **安全三原则** | 会话清空 / 暂存不内化 / 清理不改权重 |

---

## 架构

```
用户输入
  → DialogContext（话题追踪 + 情绪趋势）
    → 意图云（激活扩散 → 权重演化）
      → StrengthRegulator（弹性地图查询 → clamp强度）
        → PerspectiveConverter（视角混合 → 多维向量）
          → BilingualInjector（虚拟token prepend）
            → Qwen 生成（输出文本）
```

---

## 快速开始

```bash
git clone https://github.com/madheu/intentCloud.git
cd intentCloud/IntCog-R

# 安装依赖
pip install torch transformers accelerate bitsandbytes pydantic pyyaml

# 端到端验证（需本地Qwen2.5-7B）
python scripts/e2e_test.py --model /path/to/Qwen2.5-7B-Instruct

# 完整实验日志
cat experiment_log.md

# 单元测试
python -m pytest tests/ -v
```

---

## 项目结构

```
IntCog-R/
├── core/                          # 核心模块
│   ├── intent_cloud.py            # 意图云（激活扩散 + 权重更新）
│   ├── bilingual_injector.py      # 双语者注入（embedding层）
│   ├── steering.py                # 中间层引导注入
│   ├── strength_regulator.py      # 自适应强度调节器（弹性地图查询）
│   ├── perspective_converter.py   # 视角转换器（认知重评）
│   ├── dialog_context.py          # 多轮对话上下文（内化门控）
│   ├── anchor_system.py           # 锚点系统
│   ├── intent_cloud_config.py     # 拓扑演化配置
│   ├── intent_cloud_dynamics.py   # 快慢分离动力学
│   ├── node_embedding.py          # 节点embedding查询
│   ├── intent_to_vector.py        # 意图→向量映射层
│   ├── common_sense_loader.py     # 常识数据加载
│   └── models.py                  # 数据模型
├── scripts/                       # 实验脚本
│   ├── e2e_test.py                # 端到端首跑
│   ├── e2e_diagnose.py            # embedding区分度诊断
│   ├── e2e_direction_test.py      # 方向性注入测试
│   ├── e2e_swap_test.py           # swap因果验证
│   ├── e2e_injection_compare.py   # 注入方式对比
│   ├── e2e_dual_cloud_evolution.py # 双海波拓扑演化
│   ├── calibrate_injection_strength.py # 控制曲线
│   ├── build_elasticity_map.py    # 弹性地图构建
│   ├── test_perspective_converter.py # 视角转换验证
│   ├── test_multiturn_dialog.py   # 多轮对话验证
│   └── inject_common_sense.py     # 常识生成
├── data/
│   ├── common_sense.json          # 常识知识图谱（36节点/42边）
│   └── elasticity_map.json        # 8条路径弹性地图
├── tests/                         # 测试（38个单元测试）
│   ├── unit/
│   └── integration/
├── experiment_log.md              # 完整实验日志
├── experiment_summary.md          # 实验摘要
├── injection_strength_curve.csv   # 控制曲线数据
└── hypergraph_summary.md          # 英文总结报告
```

---

## 验证链路（全部跑通）

| 阶段 | 实验 | 状态 |
|------|------|:----:|
| 常识加载 + 基础链路 | H10 | ✅ |
| 端到端首跑 | H11a | ✅ |
| 方向注入验证 + 短语修复 | H11b-e | ✅ |
| **双海波拓扑演化驱动LLM** | **H12** | **✅ 核心里程碑** |
| 控制曲线校准 | H13 | ✅ |
| 弹性地图 + 强度调节器 | H14 | ✅ |
| 视角转换器 | H15 | ✅ |
| 多轮对话 + 内化门控 | H16 | ✅ |
| 多维语义理解层（5模块/26测试） | H17 | ✅ |
| 集成验证（20/21通过） | H18 | ✅ |

---

## 测试结果

| 测试套件 | 通过/总数 | 状态 |
|----------|:---------:|:----:|
| 强度调节器（H14b） | 10/10 | ✅ |
| 对话上下文（H16a） | 8/8 | ✅ |
| 反馈捕获（H16b） | 10/10 | ✅ |
| 暴力测试（H16d） | 12/12 | ✅ |
| 认知测试（H16e） | 16/16 | ✅ |
| 指代消解（H17a） | 6/6 | ✅ |
| 纠正共情（H17b） | 7/7 | ✅ |
| 矛盾处理（H17c） | 4/4 | ✅ |
| 情绪球（H17d） | 5/5 | ✅ |
| 结构化记忆（H17e） | 4/4 | ✅ |
| 集成验证（H18） | 20/21 | ✅ |

---

### 仓库状态

```bash
732bf4e H14-H16: 弹性地图 + 强度调节器 + 视角转换器 + 多轮对话
9e79bb5 添加实验日志文件
d9b7e8b H10-H13: 端到端验证 + 双海波演化 + 控制曲线校准
```

推送命令：
```bash
git push origin main
```

---

## 论文 / 参考文献

详见 `docs/references/` 目录。

## 许可证

Apache 2.0 License
