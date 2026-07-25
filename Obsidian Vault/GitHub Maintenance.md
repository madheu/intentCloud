# GitHub 仓库维护

> 仓库：`https://github.com/madheu/intentCloud`
> 本地根目录：`E:\intentCloud\`

---

## 双仓库结构

```
E:\intentCloud\                  ← intentCloud（主仓库）
    ├── .gitignore               # 忽略 models/, IntCog-R-repo/, logs/ 等
    ├── README.md
    ├── ABOUT.md
    ├── Haibo-Research/          # 研究文档（追踪）
    ├── docs/                    # 理论文档（追踪）
    │
    └── IntCog-R-repo/
        └── IntCog-R/            ← 独立 git 仓库（被主 .gitignore 忽略）
            ├── core/            # 核心代码
            ├── tests/           # 测试
            ├── mcp/             # MCP 服务器
            └── data/            # 数据文件（.gitignore 中可能忽略）
```

**为什么是双仓库？**
- `intentCloud` 是研究项目（文档、实验、理论）
- `IntCog-R` 是独立代码库（实现代码），有自己的版本节奏
- 主 `.gitignore` 中 `IntCog-R-repo/` 避免嵌套 git 混乱

---

## intentCloud 主仓库

### 分支
| 分支 | 用途 |
|------|------|
| `main` | 主分支，所有发布 |
| `hermes/mvp` | 历史 MVP 版本分支 |

### 最近的提交（2026-07-21）
```
fb68614 P0-R complete: 60句盲测+5层级联58.3%, 超MFS基线, H-SG-1~5
befabbe add P0 series comprehensive experiment report
30f9141 research note: semantic direction rethink
89151d2 add ABOUT, 3 charts, update preview
86c1717 P0-A: 概念域→布尔4轴, 纸83%光50%; 更新README
```

### 提交习惯
1. **提交粒度**：按实验阶段组织（一个 P0 实验一提交）
2. **提交信息**：中文，含核心数字（正确率/基线对比）
3. **README/ABOUT 同步更新**：每次实验阶段结束都更新
4. **图表**：`ABOUT.md` 中的实验对比图（GitHub 预览页）

### `.gitignore` 内容摘要

| 模式 | 说明 |
|------|------|
| `models/` | 模型文件太大 |
| `__pycache__/` | Python 缓存 |
| `.hermes/`, `.reasonix/` | Hermes 配置 |
| `IntCog-R-repo/`, `IntCog-R-mvp/` | 独立子仓库 |
| `logs/`, `tools/` | 工具/日志 |
| `*.safetensors`, `.bin`, `.onnx` | 二进制模型 |
| `Haibo-Research/06_Experiment/*.py` | 临时实验脚本 |

### 未提交内容（当前工作区）
- `Haibo-Research/06_Experiment/` 下大量 P0-R 实验脚本和输出文件
- 这些是新实验的中间产物，阶段结束后选择性提交

### 提交流程
```powershell
# 查看状态
git -C E:\intentCloud status

# 添加研究文档
git -C E:\intentCloud add Haibo-Research/
git -C E:\intentCloud add docs/

# 提交（带实验关键数字）
git -C E:\intentCloud commit -m "描述：核心结果"

# 推送
git -C E:\intentCloud push origin main
```

---

## IntCog-R 子仓库

### 当前状态
- 独立 git 仓库，remote 指向同一个 intentCloud 仓库
- `data/` 和 `mcp/` 目录目前在 `.gitignore` 中未追踪
- 最后提交：`593debb merge: 意图云 v3 (Harness 6-9) + Ollama/Qwythos`

### 提交注意
```powershell
# 在 IntCog-R 子目录内操作
git -C E:\intentCloud\IntCog-R-repo\IntCog-R add -A
git -C E:\intentCloud\IntCog-R-repo\IntCog-R commit -m "..."
git -C E:\intentCloud\IntCog-R-repo\IntCog-R push origin main
```

---

## Git 操作速查

### 日常提交
```powershell
# 主仓库
git -C E:\intentCloud add -A
git -C E:\intentCloud commit -m "描述"
git -C E:\intentCloud push origin main

# 子仓库
git -C E:\intentCloud\IntCog-R-repo\IntCog-R add -A
git -C E:\intentCloud\IntCog-R-repo\IntCog-R commit -m "描述"
git -C E:\intentCloud\IntCog-R-repo\IntCog-R push origin main
```

### 查看日志
```powershell
git -C E:\intentCloud log --oneline -10
git -C E:\intentCloud\IntCog-R-repo\IntCog-R log --oneline -10
```

### 查看未提交
```powershell
git -C E:\intentCloud status --short
```

### 拉取最新
```powershell
git -C E:\intentCloud pull origin main
```

---

## GitHub Pages / 预览

`ABOUT.md` 作为 GitHub 仓库预览页，包含：
1. 项目一句话定位
2. P0 系列实验对比图（嵌入图片链接）
3. 布尔轴演化图
4. 当前阶段说明

更新预览页只需要：
1. 更新 `ABOUT.md`
2. 更新 `README.md`（含项目定位 + 快速链接）
3. `git add` + `git commit` + `git push`

---

## 注意事项

1. **不要提交模型文件**（`models/` 已忽略）
2. **不要提交 Hermes 配置**（`.hermes/` 已忽略）
3. **实验中间产物**（`06_Experiment/` 下的数据输出）—阶段结束后决定哪些保留
4. **提交信息带数字**（正确率/基线对比/通过标志），方便日后追溯
5. **研究文档和代码分开提交**：Haibo-Research/ 归主仓库，IntCog-R/ 归子仓库
6. **忘记推送没关系**—`git status` 和 `git log` 能看出本地 vs 远端差异
