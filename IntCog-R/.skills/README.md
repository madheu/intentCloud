# intent-cloud-skills — 意图云项目技能索引

> 基于 addyosmani/agent-skills + mattpocock/skills 转写
> 适配：IntCog-R Python 项目，harness 驱动开发工作流

---

## 技能速查

| 技能 | 文件 | 触发词 | 用途 |
|------|------|--------|------|
| **code-review** | `code-review.md` | "review"、"审查" | 双轴审查（标准+规格） |
| **diagnose** | `diagnose.md` | "debug"、"排查"、"报错" | 六阶段 bug 诊断 |
| **tdd** | `tdd.md` | "TDD"、"先写测试" | 红-绿-重构循环 |
| **codebase-design** | `codebase-design.md` | "设计接口"、"深模块" | 深模块设计原则 |
| **grilling** | `grilling.md` | "压测设计"、"挑战方案" | 实现前压测设计 |
| **handoff** | `handoff.md` | "导出进度"、"交接" | 跨 agent 上下文传递 |

---

## 工作流：harness 驱动开发

```
Harness 提示词
    │
    ├─ grilling ── 压测设计，确认接口
    │
    ├─ tdd ── 红-绿-重构，垂直切片实现
    │
    ├─ diagnose ── 失败时排查
    │
    ├─ code-review ── 审查标准+规格
    │
    └─ handoff ── 导出进度给本地 agent
```

---

## 工程规范（始终生效）

以下规则在所有 skill 中自动生效，不需要显式触发：

1. **类型注解**：所有函数必须有参数和返回值类型
2. **模块 docstring**：每个 `.py` 文件顶部有模块级文档
3. **关键逻辑注释**：解释"为什么这样做"，不解释"做了什么"
4. **边界检查**：输入验证、空值处理、越界保护
5. **配置驱动**：可调参数放入 `IntentCloudConfig` 或 `config.yaml`
6. **测试覆盖**：每个新模块至少 3 个测试场景
7. **review 后推送**：写完代码 → review → 推送前确认