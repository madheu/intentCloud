# 研究日志 (Research Log)

> 按时间顺序记录研究进展、发现、失败、转折。
> 不记录代码变更（代码在 git），只记录研究层面的决策和洞察。

---

## 2026-07-18 — 上午：审查启动

### 建立结构
- 审查委员会启动（AI Reviewer / Cognitive Scientist / Control Scientist / PI）
- 建立 Haibo-Research/ 结构化审查目录体系

### 审查结论
- **Control Scientist**：赫布学习与扩散收敛数学矛盾。反例构造（ρ(A)=1.099→发散）。推荐方案 D。
- **Cognitive Scientist**：闭环缺失。修正为 L1/L2/L3 三层反馈分析。推荐路径 C。
- **AI Reviewer**：文献定位（与 Representation Engineering 重叠）。设计互信息 I(a*; y_LLM) 测量实验。

### PI 决策
- D-001：暂停 H20
- D-003：注册三条路径（A/B/C）

---

## 2026-07-18 — 下午：实验 → 证伪 → 重新定位

### Pilot 1（"大海" prompt）
- FEAR 10/10 → awe，连续 embedding 类间/类内比 = 0.93
- 初步误判：prompt 的 awe 基座问题

### Pilot 2（"桌子" prompt）
- FEAR 10/10 neutral，NONE 10/10 neutral
- 类间/类内比 = 0.99
- **关键转折**：不是 prompt 问题，是注入接口问题

### 结论
- **H1（图驱动生成方向）**：实验证伪（两轮 pilot，I≈0）
- **H3（双层稳定性）**：数学证否（反例已构造）
- **路径 C 触发**：终止"图结构控制 LLM 生成"核心假设
- **BilingualInjector 废弃**：embedding 注入路线关闭

### 阅读 BriLLM 论文
- 上海交大 Token Graph 方向与 Haibo 不同（词图 vs 语义图）
- 引发方向修正：Haibo 2.0 → Semantic Emergence

### 重新定位
- Haibo 1.0 核心路线被证伪
- Haibo 2.0 方向：研究"Meaning 是如何形成的"——语义从 Token 关系中涌现
- 新架构：Token Graph → Semantic Emergence → Context → Intent → Language

### Phase 0 启动
- 最小实验：词袋 + PCA 检验"苹果科技"和"苹果水果"是否自然可分
- 代码：`06_Experiment/p0_bow_apple.py`
- 如果通过 → Token Graph 有数据基础
- 如果失败 → 更换研究方向
