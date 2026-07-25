---
tags: [haibo, MCP, 记忆]
---

# Haibo Memory MCP 服务器

> 位置：`E:\intentCloud\IntCog-R-repo\IntCog-R\mcp\haibo_memory_server.py`
> 数据：`E:\intentCloud\IntCog-R-repo\IntCog-R\data\haibo_memory.json`
> 测试：`E:\intentCloud\IntCog-R-repo\IntCog-R\mcp\test_haibo_memory.py`
> 注册：`E:\intentCloud\IntCog-R-repo\IntCog-R\mcp\register_desktop_mcp.py`

---

## 是什么

Haibo Memory MCP 是海波项目独立的记忆系统，通过 MCP（Model Context Protocol）暴露给 Hermes Agent。它是一个**轻量级 TF-IDF 记忆库**，不需要 PyTorch/GPU 即可运行。

**注意：这和 Hermes 自己的 Memory 系统是两套东西。** Haibo Memory 是项目级记忆（存实验记录、理论发现），Hermes Memory 是 Agent 级别记忆（存用户偏好、工作流）。

## 架构

```
Hermes Agent
    │
    ├── MCP Tool: store_memory
    ├── MCP Tool: retrieve_memories
    ├── MCP Tool: list_recent_memories
    ├── MCP Tool: get_memory_stats
    ├── MCP Tool: search_by_emotion
    ├── MCP Tool: delete_oldest_memories
    └── MCP Tool: update_memory_config
            │
            ▼
    HaiboMemoryStore
        ├── TF-IDF Bag-of-Words Embedder（纯 Python，512 维）
        │   ├── CJK → 重叠字符二元组（"海波记忆"→[海波,波记,记忆]）
        │   └── ASCII → 小写单词（"MCPTest"→[mcptest]）
        ├── JSON 文件持久化（data/haibo_memory.json）
        ├── 时间衰减（默认 λ=0.01）
        └── 情感增强（默认 boost=0.1）
```

## 积分公式

```
score = (query_similarity + goal_similarity) × exp(-λ × Δt) × (1 + |valence| × boost)
```

| 因子 | 含义 | 默认值 |
|------|------|--------|
| query_similarity | 查询文本与记忆的余弦相似度 | — |
| goal_similarity | 意图目标与记忆的余弦相似度 | — |
| λ (time_decay_lambda) | 时间衰减因子（越大旧记忆衰减越快） | 0.01 |
| boost (emotional_boost) | 情感增强系数 | 0.3 |
| valence | 情感效价（-1 ~ 1） | — |

## 提供的工具（7 个）

| 工具 | 作用 | 关键参数 |
|------|------|----------|
| `store_memory` | 存一条记忆（带情感和元数据） | text, emotional_valence?, metadata? |
| `retrieve_memories` | 语义检索最相关记忆 | query, intent_goals?, top_k?, min_score? |
| `list_recent_memories` | 列出最近 N 条记忆 | n? |
| `get_memory_stats` | 统计：总量、情感分布、配置 | — |
| `search_by_emotion` | 按情感效价范围检索 | min_valence?, max_valence?, top_k? |
| `delete_oldest_memories` | 删除最旧 N 条（释放空间） | n |
| `update_memory_config` | 运行时更新参数 | time_decay_lambda?, emotional_boost? |

## 注册方式

### 桌面版注册
```bash
python E:\intentCloud\IntCog-R-repo\IntCog-R\mcp\register_desktop_mcp.py
```
这会写入 Hermes Desktop 的 `config.yaml` 的 `mcp_servers.haibo-memory` 段。

### 配置文件示例
```yaml
mcp_servers:
  haibo-memory:
    command: python
    args:
      - "E:\\intentCloud\\IntCog-R-repo\\IntCog-R\\mcp\\haibo_memory_server.py"
    env:
      HAIBO_MEMORY_PATH: "E:\\intentCloud\\IntCog-R-repo\\IntCog-R\\data\\haibo_memory.json"
      HAIBO_TIME_DECAY: "0.01"
      HAIBO_EMOTIONAL_BOOST: "0.3"
    timeout: 30
```

## 运行方式

MCP 服务器通过 stdio 协议运行。由 Hermes Agent 在启动时自动启动，无需手动管理。

**退出**：当 Hermes Agent 退出时自动终止。
**重启**：重启 Hermes 会话即可重启 MCP。

## 数据存储

`haibo_memory.json` 格式：
```json
{
  "entries": [
    {
      "text": "记忆内容",
      "timestamp": "2026-07-19T10:47:15+00:00",
      "emotional_valence": 0.5,
      "metadata": {"type": "experiment"}
    }
  ],
  "vectors": [[...], ...],  // 预计算向量
  "stats": {"total_entries": 0, ...},
  "updated_at": "..."
}
```

## 当前状态

- 服务器代码已完成（616 行）
- 测试脚本可用（test_haibo_memory.py）
- 数据文件已创建但为空（0 条记忆）
- 未在 Hermes Desktop 上实际注册（需要手动跑 register_desktop_mcp.py）

## 已知局限

1. TF-IDF Embedder 是轻量方案，对短文本/专业术语的语义区分力有限
2. 如果未来需要更高的语义精度，可替换为 sentence-transformers 或 LLM embedding
3. ~~哈希碰撞风险~~ → 已改用 SHA-256 确定性哈希
4. ~~update_memory_config 不落盘~~ → 已持久化
5. ~~search_by_emotion O(n) 全量扫描~~ → 已加情感分桶索引
6. ~~并发写不安全~~ → 已加 threading.Lock
7. 不支持记忆自动压缩/合并，只手动删除最旧条目

## 维护清单

- [ ] 首次使用：跑一次 `register_desktop_mcp.py`
- [ ] 定期检查：`get_memory_stats` 查看存储量
- [ ] 空间不足：`delete_oldest_memories(n=X)` 释放
- [ ] 如需调整：`update_memory_config` 改时间衰减或情感增强
- [ ] 代码修改后：跑 `test_haibo_memory.py` 验证
