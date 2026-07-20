"""multi_expert_review — 图结构多专家审稿系统。

基于 labelled directed graph 连接多个专家角色（PI、AI Reviewer、
Cognitive Scientist、Control Scientist 等），通过 publish/subscribe
workspace 实现并行审稿，并支持冲突检测、仲裁和决策追踪。

组件：
- models.py:   ExpertRole, ReviewEdge, ReviewGraph, Decision
- workspace.py: ReviewWorkspace, ExpertReviewer, DecisionAggregator
- demo.py:      Haibo 审稿委员会的四角色演示
"""
