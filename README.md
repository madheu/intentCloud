# intentCloud — 意图驱动型生成系统原型

当前 LLM 本质仍是基于 token 的自回归生成，依赖逐个预测下一个 token，虽含隐式语义，却未显式建模范畴组织过程。本项目在语言输出前，引入可持续更新的语义状态空间（意图空间），存储概念、关系、目标与约束。生成不再依赖 token 历史，而是依赖该结构化状态，并由其产生文本。该状态应随对话动态演化，实现跨轮次一致性与延续性。

**从"序列生成"转向"语义组织 → 语言表达"的双阶段系统。**

---

## 当前状态

- **模型**: Qwen2.5-1.5B Instruct（float16, CUDA）
- **测试**: 226 passed / 43 skipped
- **许可证**: Apache 2.0

---

## 核心发现

| 发现 | 说明 |
|------|------|
| **短语 > 单字词** | 短词中层embedding坍缩(cos_sim 0.9998)，短语修复后有效(0.40) |
| **海波驱动LLM** | 双海波对照：拓扑演化决定输出方向 |
| **恐惧是唯一窄路** | 恐惧[0.01,0.05] vs 敬畏[1.0,10.0] — 死亡焦虑假设 |
| **生命对冲死亡** | 恐惧链视角转换：恐惧40%+生命25%+壮丽20%+敬畏15% |
| **安全三原则** | 会话清空 / 暂存不内化 / 清理不改权重 |
| **注入改变 logits** | 大海节点注入后 Top-1 token 概率从 5.7%→40.7%，cos=0.90 |
| **虚拟 token 临界点** | >4 个虚拟 token 导致模型哑火，MAX_INJECT=4 修复后 100% 输出率 |

---

## 架构

```
用户输入 → 话题检测 → IntentCloud(激活扩散) → 双语者注入(embedding层) → LLM生成
            ↓              ↓                        ↓
        关键词匹配     22+节点扩散              top-4节点→虚拟token
                        ↓
                  GraphInterpreter
                        ↓
                    Decision
                        ↓
                  SystemPrompt
```

图状态可见的交互：

```
[你] 大海好美，但有时候也让人害怕

触发: 大海(0.5), 恐惧(0.5)
扩散: 海洋↑0.33, 紧张↑0.33, 壮丽↑0.22, 悲伤↑0.19, 生命↑0.19, 愤怒↑0.18, 喜悦↑0.16, 敬畏↑0.16
活跃: 23节点 → 注入: 4tok
输出: 而平静时，则展现出一种深邃与宁静，仿佛在告诉我们生命的美好。
```

---

## 关键数据

### 虚拟 token 数量 → 输出率

![token vs output](docs/charts/chart1_token_vs_output.png)

### 0.5B vs 1.5B 模型对比

![model comparison](docs/charts/chart2_model_comparison.png)

### Step 5: 注入确实改变了 logits

![logits change](docs/charts/chart3_logits_change.png)

### Graph → LLM 信号流

![signal flow](docs/charts/chart4_signal_flow.png)

### VIRTUAL_REPEAT=1 反而信号更强

![repeat effect](docs/charts/chart5_repeat_effect.png)

---

## 验证链路

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
| GraphInterpreter + 人格锚定 | H19 | ✅ |
| **模型切换 + 扩散接入注入 + 注入器重构** | **H20** | **✅** |

### H20 关键变更

| 变更 | 说明 |
|------|------|
| 模型 GPT-2→Qwen2.5-1.5B | 告别训练数据泄露和重复退化 |
| spread→injection 接入 | 扩散结果不再被丢弃，全图激活传给注入器 |
| input embedding 归一化 | hidden state→input space 缩放，std 比 33x→0.8x |
| VIRTUAL_REPEAT 3→1 | 信号强度反而提升（大海 24.2%→40.7%） |
| MAX_INJECT=4 | 防止过多虚拟 token 导致模型哑火 |
| 图状态可见交互模式 | 用户看到 Graph 的认知过程 |
| `chat_with_haibo.py` | 中文交互 + Qwen模板 + 扩散显示 |

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
| 双语者注入器（H20） | 21/21 | ✅ |
| **总计** | **226/269** | **✅** |

---

## 快速开始

```bash
git clone https://github.com/madheu/intentCloud.git
cd intentCloud/IntCog-R

# 安装依赖
pip install torch transformers pydantic pyyaml

# 交互式聊天（需先下载 Qwen2.5-1.5B）
python scripts/chat_with_haibo.py

# 单元测试
python -m pytest tests/ -v
```

---

## 项目结构

```
IntCog-R/
├── core/                          # 核心模块
│   ├── intent_cloud.py            # 意图云（激活扩散 + 权重更新）
│   ├── bilingual_injector.py      # 双语者注入（embedding层，归一化+top4截断）
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
│   ├── graph_interpreter.py       # Graph→Decision翻译
│   ├── prompt_builder.py          # Decision→Prompt构建
│   └── models.py                  # 数据模型
├── scripts/                       # 实验脚本
│   ├── chat_with_haibo.py         # 交互式聊天（图状态可见）
│   ├── e2e_test.py                # 端到端首跑
│   ├── e2e_diagnose.py            # embedding区分度诊断
│   ├── e2e_direction_test.py      # 方向性注入测试
│   ├── e2e_swap_test.py           # swap因果验证
│   ├── e2e_injection_compare.py   # 注入方式对比
│   ├── e2e_dual_cloud_evolution.py # 双海波拓扑演化
│   ├── calibrate_injection_strength.py # 控制曲线
│   ├── build_elasticity_map.py    # 弹性地图构建
│   ├── test_perspective_converter.py # 视角转换验证
│   └── test_multiturn_dialog.py   # 多轮对话验证
├── data/
│   ├── common_sense.json          # 常识知识图谱（36节点/42边，含LLM embedding）
│   └── elasticity_map.json        # 8条路径弹性地图
├── docs/
│   ├── charts/                    # H19/H20诊断图表
│   ├── intent-space/              # 意图空间理论文档
│   ├── decisions/                 # 决策注册表
│   └── references/                # 文献参考库
├── logs/                          # 实验与诊断日志
├── tests/                         # 测试
│   ├── unit/
│   └── integration/
└── experiment_log.md              # 完整实验日志
```

---

## 论文 / 参考文献

详见 `docs/references/` 目录。

## 许可证

Apache 2.0 License
