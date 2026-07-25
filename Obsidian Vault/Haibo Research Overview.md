# Haibo (IntentCloud) Research Overview

## Project Location
- **Root:** `E:\intentCloud\`
- **Codebase:** `E:\intentCloud\IntCog-R-repo\IntCog-R\`
- **Docs:** `E:\intentCloud\docs\` (intent-space theory, decision registry, references)
- **Session Context:** `E:\intentCloud\.hermes\session_context.md`

## Core Philosophy
> **语言不是思维本身，是思维在低维空间的投影。**

## Current Direction (Haibo 2.0, 2026-07-21)

**Core Question:** 能否从 Token 活动路径中无监督形成稳定、可复用、持久的 Semantic Node？

### Key Distinction
- **BriLLM** — 已覆盖 Token 节点传播
- **Haibo innovation** — Token → Semantic Node 跃迁（不是 Token Graph）
- **Graph nodes = semantic concepts** (科技类/水果类), **not** word tokens
- Word → concept is a belonging relationship

### Hypothesis Chain (H-SG-1~5)
Replaces old Haibo 2.0 assumptions.

## Haibo 1.0 (Closed)
- Path C: Embedding injection → LLM control
- **Falsified:** I(intent_cloud; y_LLM) ≈ 0
- Static embedding signals overwhelmed by Transformer's dynamic computation
- **DEPRECATED:** Do not continue optimizing embedding injection unless new evidence appears

## What Remains Valuable from Haibo 1.0
- Semantic graph representation
- Graph diffusion mechanism
- GraphInterpreter
- Semantic Decision extraction
- Prompt-based semantic guidance

## Core Principles
1. Don't reinvent semantics — adopt existing theories (Frame Semantics, HowNet, Ontology, Semantic Role)
2. Haibo's innovation is in **computation mechanism** (how to organize, learn, evolve graph structure), not linguistics
3. Validable hypothesis → minimum implementation → controlled experiment → measurement → falsification or support
