# ── 归档 P0 实验并推送 GitHub ──

# 第一步：确认 .gitignore 生效（models/ 等不会被提交）
git status

# 第二步：暂存关键文件
git add .gitignore
git add README.md
git add Haibo-Research/
git add IntCog-R/scripts/talk_to_haibo.py

# 第三步：提交
git commit -m "Haibo 2.0 transition: archive P0 series, update README

Resolve experiments (P0.1-P0.7) complete.
Core conclusion: word-level granularity insufficient for semantic disambiguation.
Injection route (Haibo 1.0) closed — structural signal bottleneck confirmed.
New direction: Semantic Emergence from Token Graph (Haibo 2.0).

See Haibo-Research/ for full review documents, decision logs, and experiments."

# 第四步：推送到 GitHub
git push origin main
