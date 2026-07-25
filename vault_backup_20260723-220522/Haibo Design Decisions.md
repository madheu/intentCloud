---
tags: [haibo, 决策, 设计]
---

# Haibo — Design Decisions & Principles

## Node Type: The First Design Decision
- **Rule:** Graph nodes = **concepts** (科技类, 水果类), **NOT** word tokens
- **Proof:** P0.8 (word nodes) → ALL uncertain; P0.9 (concept nodes) → 10/11 correct
- Word → Concept is a **belonging relationship**, not identity

## Semantic Axes Evolution

| Version | Axes | Source | Status |
|---------|------|--------|--------|
| Early | 天/地/人/心 (Heaven/Earth/Human/Heart) | Self-invented | Replaced |
| P0-A | 物/事/质/序 (Entity/Event/Property/Relation) | Upper ontology | Abandoned (50% ceiling) |
| Current | HowNet sememes + Frame Semantics | Existing theory | In development |

**Key principle:** Do NOT invent semantic taxonomy. Draw from:
- HowNet (sememes)
- FrameNet (Frame Semantics)
- Semantic Role theory
- Ontology/Description Logic
- Conceptual Spaces

## Boolean 4-Axis Role
- Boolean vectors are **diffusion constraints (prior)**, NOT final semantic representation
- User design principle: "every 4+ dimensions, expand outward one layer, compare only 4 at a time"
- Validated by P_Resolve-10

## Research Design Principles

### Knowledge Acquisition Pipeline
1. **HowNet Bootstrap** — existing structured semantic data
2. **LLM Expansion** — generate context sentences (2100 sentences done)
3. **TF-IDF Calibration** — statistical weighting
4. **User Interaction Evolution** — incremental learning

### Experiment Methodology
1. **Pilot first** (1 condition × 10 seeds) to validate tooling
2. **Control group** must be **same source** as experimental group
3. Always run **MFS baseline** first (from dev set, never blind)
4. **predict.py** and **score.py** must be separate
5. Blind gold labels → score.py only
6. Each layer: **diversity self-check** (if single label → abort)
7. Report **honest failures**, not optimistic spins

### Validation Priority
> Data before theory. Decisions based on quantitative results (MI / class ratio / confidence / accuracy).
> Honest falsification is more valuable than positive conclusions.

## Review Committee Architecture
| Role | Scope |
|------|-------|
| PI (首席研究员) | Overall direction, final judgment |
| Cognitive Scientist | Cognitive plausibility, learning mechanisms |
| Control Scientist | Non-linear control, dynamic systems |
| AI System Reviewer | Computational feasibility, existing solutions |
| Research Executor | Run experiments, report results |

**Workflow:** PI directs → Executor runs → AI Reviewer checks → Cognitive/Control review → PI integrates → Record in decision log

### Important: Role-based output routing
- Each review must specify recipient role and scope
- Do NOT self-route; if AI Reviewer suggests "find a cognitive architect", route to the right role
- Records are accumulated, NOT overwritten

## Decision Log (05_Decision_Log.md)
> Records WHY decisions were made, not just WHAT was done.
> Location: `E:\intentCloud\Haibo-Research\05_Decision_Log.md`

## Comprehensive Research Tree
```
Cognitive Linguistics
├── Frame Semantics (FrameNet)
├── Prototype Theory
├── Conceptual Metaphor
└── Cognitive Grammar
NLP Semantic Theory
├── Semantic Role Labeling
├── Event Semantics
└── Distributional Semantics
Knowledge Representation
├── Ontology / Description Logic
├── HowNet (Sememes)
├── Knowledge Graphs
└── Conceptual Spaces
```

**Check this tree before inventing a new semantic theory.**
