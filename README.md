# ComfyUI-H3-VisionPromptor

**Local, API-key-free MiniMax H3 prompt enhancer for ComfyUI.**

Feed an idea (plus optional reference images) and get a production-ready
[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3) prompt — with shots,
camera vocabulary, dialogue tags, soundscape, and music fields — generated
entirely by a **local VLM running on ComfyUI's native text-generation stack**
(`comfy.sd.load_clip` + `CLIP.tokenize/generate/decode`). No API keys, no
cloud calls, no extra pip packages.

---
<img width="1388" height="778" alt="Screenshot 2026-08-10 060901" src="https://github.com/user-attachments/assets/19badf35-e50a-4a3d-a2db-eed62163ca39" />

<img width="1511" height="1003" alt="Screenshot 2026-08-10 062800" src="https://github.com/user-attachments/assets/ef923be7-4a10-4428-b047-5c631aa00275" />

## Features

- **H3 Vision Promptor (Local VLM)** — idea (+ up to 4 images) → finished H3
  prompt (T2VA / I2VA / FL2VA / L2VA / Ref2VA), with a two-stage pipeline:
  vision analysis of the reference image(s) → H3 prompt-writing generation →
  deterministic post-processing (alignment lines, missing-field repair,
  7000-char smart trim, validation warnings).
- **H3 Vision Analyzer (Local VLM)** — standalone batched image description,
  useful on its own or to feed other nodes.
- Ships the complete H3 prompt-writer system prompt (distilled from the
  official MiniMax-H3 `h3-prompt-writing` guides) in
  `prompts/h3_system_base.txt` — hot-reloaded from disk on every run, so you
  can edit it without restarting ComfyUI.
- Zero external dependencies. Works with any generative VLM ComfyUI can load
  as a text encoder: **Qwen3-VL, Gemma-3/4-Vision** instruct repacks, etc.
  (Qwen2.5-VL repacks are vision-tower-only on current master — see
  *Model setup*.)

## Requirements

- A **recent ComfyUI** with the V3 node API (`comfy_api.latest`) and the
  native text-generation CLIP stack (the same stack used by the core
  *Generate Text* node in `comfy_extras/nodes_textgen.py`). Update ComfyUI
  if in doubt.
- A generative VLM text-encoder checkpoint (see below).
- Python 3.10+. No `pip install` needed — `requirements.txt` is comments only.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/benjiyaya/ComfyUI-H3-VisionPromptor.git
# or: unzip the release archive into ComfyUI/custom_nodes/
```

Restart ComfyUI. The nodes appear under **H3/Promptor**.

## Model setup

You have two equivalent ways to give the nodes a VLM — the `clip` input wins
when both are present.

**Recommended models (verified generative on current ComfyUI master):**

1. **Qwen3-VL instruct text-encoder repacks** (e.g. *Qwen3-VL-4B-Instruct* or
   *Qwen3-VL-8B-Instruct*) — the recommended, verified-generative family on
   current master. This is the safest choice for both nodes.
2. **Gemma-3 vision instruct repacks (≤12B)** (e.g. *Gemma-3-4B-it*) — second
   recommendation; fully supported via the Gemma chat template
   (`<start_of_turn>…<image_soft_token>`). Note: Gemma-3 consumes all
   connected images as a **single batch** bound to one `<image_soft_token>`,
   so multi-image prompts are written from one combined view.
3. **Gemma-4 vision repacks** — supported via the dedicated Gemma-4 template
   (`<|turn>user\n…` with one `<|image><|image|>` media pair per image placed
   after the text).

> **Warning — Qwen2.5-VL repacks:** on current ComfyUI master, Qwen2.5-VL
> text-encoder repacks expose only the **vision tower** and are **not
> generatable** — generation with them raises the "does not support native
> text generation" (`not generatable`) error. Use a Qwen3-VL repack instead.

**Option A — dropdown (simplest).** Place a supported VLM text-encoder
`.safetensors` repack into `ComfyUI/models/text_encoders/`, e.g. a
*Qwen3-VL-4B/8B-Instruct* or *Gemma-3-4B-it* text-encoder repack (the same
files the native **CLIPLoader** node uses). Refresh the page; the file appears
in the `text_encoder` dropdown.

**Option B — CLIP input.** Add the native **CLIPLoader** node, select your VLM
there, and connect its `CLIP` output to the `clip` input of this node. Leave
the dropdown on `<none - use CLIP input>`.

> If the selected checkpoint's architecture doesn't support ComfyUI's native
> generation, the node raises a clear `RuntimeError` telling you what to do.

## Node reference

### H3 Vision Promptor (Local VLM) — `H3VisionPromptor`

| Input | Type | Default | Notes |
|---|---|---|---|
| `clip` | CLIP (optional) | — | Connect from CLIPLoader; overrides the dropdown |
| `text_encoder` | combo | `<none - use CLIP input>` | Files in `models/text_encoders/` |
| `user_idea` | string (multiline) | `""` | Your rough idea; may be empty if images carry the concept |
| `images` | IMAGE ×0–4 (growable) | — | Click *add input* for more sockets (`image_0…image_3`) |
| `task_type` | combo | `Auto` | Auto: 0 img→T2VA, 1→I2VA, 2→FL2VA, ≥3→Ref2VA. Pick **L2VA** manually for last-frame tasks |
| `duration` | float | `8.0` | 4.0–15.0 s, drives word budget and timestamp validation |
| `vision_mode` | combo | `detailed_subject_scene` | Presets from `prompts/vision_presets.json`, or `skip (idea only)` |
| `extra_instructions` | string (optional) | `""` | Dialogue lines, mood, camera wishes |
| `custom_system_prompt` | string (optional, advanced) | `""` | Fully replaces the built-in H3 system prompt |
| `seed` | int | `0` | `control_after_generate: randomize` |
| `temperature` | float | `0.7` | 0 = greedy |
| `top_p` / `top_k` | float / int | `0.95` / `64` | Sampling |
| `max_tokens` | int | `2048` | Generation length cap |
| `variants` | int | `1` | 1–4 prompt variants (seeds `seed+i`), separated by `===== VARIANT n =====` |
| `use_default_template` | bool (advanced) | `False` | Use the model's built-in chat template instead of ours |
| `keep_model_loaded` | bool (advanced) | `True` | Cache the CLIP between runs |

**Outputs:** `prompt` (STRING), `vision_context` (STRING), `debug` (STRING
JSON: clip source, detected task, timings, warnings).

### H3 Vision Analyzer (Local VLM) — `H3VisionAnalyzer`

| Input | Type | Default | Notes |
|---|---|---|---|
| `clip` / `text_encoder` | as above | — | Same resolution logic |
| `images` | IMAGE ×1–4 (growable) | — | At least one required |
| `vision_mode` | combo | `detailed_subject_scene` | Preset keys + `custom` |
| `custom_prompt` | string (optional) | `""` | Used when `vision_mode = custom` |
| `seed` / `temperature` / `max_tokens` | — | `0` / `0.2` / `2048` | Low default temperature for factual descriptions |
| `keep_model_loaded` | bool (advanced) | `True` | — |

**Output:** `description` (STRING). Batched single pass; if the batch fails
with >1 images it automatically retries per image and joins the results as
`<Picture N>: ...` lines.

## Task-mode cheat sheet

| Mode | Images | Alignment line | Structure |
|---|---|---|---|
| **T2VA** | 0 | none | 3 fields, style+composition at `[Shot 1]` |
| **I2VA** | 1 (first frame) | `For the target video, at 0.00 seconds… <Picture 1> (from [Shot 1]) is fully referenced.` | anchor → onset → development → result |
| **FL2VA** | 2 (first+last) | `How the reference pictures align…` both pictures, end = duration | single shot favored, lands exactly on Picture 2 |
| **L2VA** | 1 (last frame) | `How the reference pictures align…` end = duration | converge onto Picture 1 only in the final shot |
| **Ref2VA** | ≥1 (references, not frames) | none | six sections: `subject_definitions` / `summary` / `retention_analysis` / `detailed_description` / `overall_soundscape` / `non_diegetic_music` |

## Example workflow

`example_workflows/h3_vision_promptor_i2va.json` — **CLIPLoader →
H3VisionPromptor ← LoadImage → ShowText**. Drag it onto the canvas (or use
*Open*), pick your VLM in CLIPLoader, load a first-frame image, and run. The
I2VA prompt appears in the text display. (`ShowText|pysssss` comes from
ComfyUI-Custom-Scripts; if you don't have it, connect the `prompt` output to
any other STRING consumer instead.)

## Troubleshooting

- **`[ComfyUI-H3-VisionPromptor] requires a recent ComfyUI …`** — your ComfyUI
  predates the V3 API (`comfy_api.latest`). Update ComfyUI (the native
  *Generate Text* node must exist in your build).
- **`The CLIP … does not support native text generation`** — the checkpoint is
  not a generative VLM (e.g. a plain SD/SDXL CLIP — or a **Qwen2.5-VL repack,
  which is vision-tower-only on current master**) or your ComfyUI build can't
  generate with that architecture. Use a Qwen3-VL or Gemma-3/4 vision instruct
  repack and/or update ComfyUI.
- **`clip.tokenize/generate rejected the native … arguments`** — ComfyUI is
  too old for the native generation signature. Update ComfyUI.
- **VRAM/RAM notes** — a 4–7B VLM typically needs ~5–16 GB depending on dtype.
  Set `keep_model_loaded = False` to reload per run instead of caching
  (`CLIP_CACHE` holds the model by checkpoint path when enabled).
- **How templates work** — by default the node builds an H3-aware chat
  template per model family (Qwen `<|im_start|>…` with
  `<|vision_start|><|image_pad|><|vision_end|>` per image; Gemma-3
  `<start_of_turn>…` with the system prompt folded into the user turn and a
  single `<image_soft_token>` — Gemma-3 binds only the first image token and
  consumes the connected images as one batch; Gemma-4 `<|turn>user\n…` with
  one `<|image><|image|>` pair per image placed after the text) and passes
  `skip_template=True` to `clip.tokenize`. Turn on `use_default_template`
  (advanced) to let the model's own template apply instead.
- **Multi-image fallback** — the vision pass is one batched call; if it fails
  with several images connected, the node retries image-by-image and labels
  the results `<Picture N>:`.
- **Autogrow note** — the image inputs use the V3
  `io.Autogrow` + `TemplatePrefix` pattern (growable `image_0…image_3`
  sockets), mirroring proven usage in the wild. On older frontends without
  Autogrow support, update ComfyUI/frontend.

## Credits & differences vs 1038lab/ComfyUI-MiniMax-H3-Promptor

This project is inspired by
[1038lab/ComfyUI-MiniMax-H3-Promptor](https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor)
and encodes the same official MiniMax H3 prompt-writing guides
(`MiniMax-AI/MiniMax-H3`, `skills/h3-prompt-writing/references/base-en.txt` +
`ref-en.txt`). Key differences:

| | 1038lab H3-Promptor | this package |
|---|---|---|
| LLM backend | cloud APIs (OpenAI/Gemini/Claude) or Ollama | **ComfyUI-native local VLM** (CLIPLoader / text_encoders) |
| API keys | required for cloud providers | **none** |
| External pip deps | provider SDKs | **zero** |
| Vision payload | base64 images to a chat API | native `clip.tokenize(..., image=tensor)` |

License: MIT (see `LICENSE`). Model checkpoints are subject to their own
licenses.
