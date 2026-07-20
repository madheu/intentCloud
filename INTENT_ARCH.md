# Intent Architecture — 意图层系统架构

## 核心概念

在现有 AI 交互模型（Request → Response）之上，增加一层 **意图管理层**，
使得 AI 具备：
- 多意图并行
- 优先级调度
- 状态持久化
- 资源隔离
- 自主推进

## 三层架构

```
┌─────────────────────────────────────────────┐
│              Intent Layer                    │  ← 新增
│  意图调度器 (Scheduler)                      │
│  ├─ Intent Registry     (所有活跃意图)       │
│  ├─ Priority Queue      (优先级队列)         │
│  ├─ State Machine       (生命周期管理)       │
│  └─ Resource Allocator  (冲突解决)           │
├─────────────────────────────────────────────┤
│              Action Layer                    │  ← 现有
│  Skills | Todo | Process | MCP Tools        │
├─────────────────────────────────────────────┤
│              Execution Layer                 │  ← 现有
│  Terminal | File | Web | Vision | ...       │
└─────────────────────────────────────────────┘
```

## 意图生命周期

```
  ┌────────────┐
  │  PENDING   │  ← 刚注册，等待评估
  └─────┬──────┘
        │ 评估通过
        ▼
  ┌────────────┐
  │  ACTIVE    │  ← 正在执行/可调度
  └─────┬──────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
┌──────┐ ┌──────┐
│BLOCKED│ │READY │  ← 等待资源/依赖
└──┬───┘ └──┬───┘
   │        │
   └──┬─────┘
      ▼
┌───────────┐
│ COMPLETED │
├───────────┤
│ FAILED    │
├───────────┤
│ CANCELLED │
└───────────┘
```

## 意图结构体 (Intent Schema)

```json
{
  "id": "intent-001",
  "name": "修复MobianWiFi",
  "goal": "排查并修复Mi MIX 2S的WiFi Q6 DSP崩溃",
  "status": "active",
  "priority": 8,
  "created": "2026-07-10T10:00:00Z",
  "context": {
    "device": "polaris",
    "kernel": "6.x",
    "firmware": "q6"
  },
  "dependencies": ["intent-000"],
  "subtasks": [
    {"id": "st-1", "desc": "检查dmesg日志", "status": "done"},
    {"id": "st-2", "desc": "分析remoteproc状态", "status": "pending"}
  ],
  "resources": ["terminal", "file"],
  "result": null
}
```

## 调度策略

- **优先级抢占**：高优先级意图可以打断低优先级
- **时间片轮转**：同优先级按时间片执行
- **资源锁**：同一资源不被两个意图同时占用
- **依赖解析**：意图B依赖意图A完成才启动
- **超时驱逐**：超时未完成的意图自动降级/取消

## 与现有系统的集成

| 现有组件 | 意图层映射 |
|---------|-----------|
| Todo | 意图的子任务列表 |
| Process | 意图的执行线程 |
| Skill | 意图的执行策略库 |
| MCP | 意图可调用的外部能力 |
| Memory/Fact | 意图的持久化状态 |

## 持久化

意图状态存储在 `~/.hermes/intents/` 目录下：
- `registry.json` — 所有意图索引
- `intents/<id>.json` — 单个意图详情
- `logs/<id>.log` — 执行日志
