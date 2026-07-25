# P0 Series Experiments — Full Record

## P0 (Word Embedding Verification — CLOSED)
- 14 rounds, 6 ambiguous words, 38 sentences
- Best accuracy: 50% (capped)
- Never exceeded MFS baseline (52.6%)
- Key lesson: Jieba whole-word segmentation is destructive for single-character ambiguous words (花/光/行/口)
- Substring matching is the only reliable approach

## P0-A (Semantic Axis / Boolean / Auto-expansion — CLOSED)
- Boolean 4-axis evolution: 天/地/人/心 → 物/事/质/序 (Entity/Event/Property/Relation)
- Based on upper ontology, not self-invented
- Boolean vector role: **diffusion constraint (prior)**, NOT final semantic representation
- Best accuracy: 50% (same ceiling as P0)
- Word list coverage was the bottleneck, not the algorithm
- Jieba uncontrollable → manual word list approach abandoned

### Key Validation
- ChatGPT review confirmed: Boolean as constraint prior is reasonable, as final theory is problematic
- P0-A series concluded: go to HowNet/Linguistics theory

## P_Resolve (Graph-based Disambiguation)
- **P0.8 (word nodes + co-occurrence edges):** ALL uncertain ❌
- **P0.9 (concept nodes + hierarchy):** 10/11 correct ✅
- **Root cause of P_Resolve-08 failure:** misidentified node type — used word tokens instead of concept nodes
- **Core lesson:** node type definition is the FIRST design decision of any semantic graph

## P0-R (Task & Measurement Repair — COMPLETED 2026-07-21)
- Dev set: 38 sentences ; Blind test: 60 sentences
- Final pipeline: cascade L3→L4→L5→MFS fallback

### Results
| Layer | Accuracy |
|-------|----------|
| MFS (strongest single) | 55% |
| L2-L5 (semantic) | 46-47% (all below MFS) |
| Cascade (L3→L4→L5→MFS) | 58.3% ✅ (passed 53.3% line) |

### Key Lessons
- **MFS baseline must be computed from dev set first, before any experiment**
- Layer-by-layer increasing trend NOT established ❌
- Semantic disambiguation has signal but weaker than corpus bias
- predict.py and score.py must be **separate scripts**
- Target span identification: use character span, not keyword scanning
- Each layer must self-check diversity (single label → abort)

### P0-R Pipeline Constraints (Final Version)
1. MFS computed from dev set only, never leaked into blind
2. Each layer: diversity self-check (single label = abort)
3. target_span: character-based, not keyword scan
4. predict.py + score.py separated
5. Input JSON: {id, sentence, target_word, target_occurrence, candidate_senses}
6. Blind gold labels only read by score.py

## Semantic Graph Design Evolution
| Version | Node Type | Edge | Result |
|---------|-----------|------|--------|
| P0.8 | Word token | Co-occurrence | ALL uncertain ❌ |
| P0.9 | Concept | Hierarchy + belonging | 10/11 correct ✅ |

**P0-B (Current):** HowNet sememes + LLM-generated context sentences (2100 sentences in p0_b2_llm_contexts.json)
