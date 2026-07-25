# Haibo — Environment & Configuration

## Hardware
- **GPU:** RTX 4060 Laptop, 8GB VRAM
- **OS:** Windows 11
- **Shell:** PowerShell 5.1 (Hermes terminal)
- **Python:** 3.14.4

## ComfyUI
- **Install:** ComfyUI Desktop
- **Shared Models:** `E:\Comfy-Desktop\ComfyUI-Shared\models\`
- **Flux GGUF workflow:** saved as "公众号生图工作流.json" in `user/default/workflows/`
- **Flux GGUF on 4060 8GB:**
  - Recommended quantization: Q4_K_M or Q5_K_S
  - 1024×768 takes ~5 minutes
  - Model: `flux1-dev-Q5_K_S.gguf`
- **Workflow nodes:** UnetLoaderGGUF + DualCLIPLoaderGGUF + CLIPTextEncodeFlux + ae_flux VAE
- **extra_model_paths.yaml:** configured to point to `E:\Comfy-Desktop\ComfyUI-Shared\models\`

## WeChat — 零号词云公众号
- **Account:** 零号词云 (AppID: wx78788e395b33ff9e, AppSecret in settings)
- **Type:** 订阅号 (cannot auto-publish via API, need manual post)
- **Articles:** `E:\Diviner\wechat_articles\`
- **Assets:** `assets\` subfolder
- **Scripts:** `make_images_template.py`, `experiment_charts_template.py`
- **Cover style:** Dark terminal (dark background + monospace + ASCII elements)

### Content Strategy
| Ratio | Type | Description |
|-------|------|-------------|
| 3:2:1 | Practical : Deep : Paper | AI tips + theory + paper readings |
| Frequency | Every 3 days | |

### Publishing Constraints
- Titles: ≤11 Chinese chars / 33 chars
- No `author` field (unverified account will error)
- IP whitelist needed for API calls
- Use `json.dumps(p, ensure_ascii=False)` + UTF-8 header (NOT `requests.post(json=p)`)
- Write `.py` file first, then run `python file.py` (NOT `python -c` — PS eats {})
- HTML formatting: colored blocks/bg sections, not plain text

## Data & Resources
- Tencent 200-dim word vectors (local)
- HowNet / OpenHowNet (via Python library)
- LLM-generated training data: 2100 sentences in `p0_b2_llm_contexts.json`

## Skill Inventory (Relevant)
- `chinese-ai-content-creation` — content for 零号词云 (created 2026-07-12)
- `research-project-review` — structured multi-role review
- `semantics-graph-design` — semantic graph design for disambiguation
- `cognitive-architecture-prototyping` — IntCog-R related
