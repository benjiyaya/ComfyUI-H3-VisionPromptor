"""Vision analysis pass: describe reference images with the local VLM.

Presets live in prompts/vision_presets.json and are re-read on EVERY call so
users can edit them without restarting ComfyUI.
"""

from __future__ import annotations

import json
import os

from . import engine

_PRESETS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "prompts", "vision_presets.json")

VISION_SYSTEM_PROMPT = (
    "You are an expert film director and visual analyst assisting a video-generation "
    "prompt writer. Describe only what is actually visible; never speculate beyond the "
    "frame. Output plain descriptive text only — no JSON, no markdown fences, no "
    "commentary about yourself."
)


def load_vision_presets() -> dict:
    """Read prompts/vision_presets.json EVERY call (no import-time caching)."""
    try:
        with open(_PRESETS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data:
            return data
    except Exception as e:
        print(f"[H3 VisionPromptor] Failed to load vision_presets.json: {e}")
    # Minimal built-in fallback so the node never breaks.
    return {
        "detailed_subject_scene": "Describe this image for a video-generation prompt writer, covering subjects, scene, lighting, style, and framing. One paragraph per image.",
    }


def _n_images(image_tensor) -> int:
    try:
        return int(image_tensor.shape[0])
    except Exception:
        return 1 if image_tensor is not None else 0


def analyze_images(clip, image_tensor, preset_key: str, custom_prompt: str | None,
                   seed, temperature, max_tokens) -> str:
    """Run one batched vision pass over image_tensor and return plain text.

    If the batched call raises and there is more than one image, fall back to
    per-image passes joined as "<Picture N>: ..." lines.
    """
    if image_tensor is None or _n_images(image_tensor) == 0:
        return ""

    instruction = (custom_prompt or "").strip()
    if preset_key == "custom" and not instruction:
        raise RuntimeError("vision_mode='custom' requires a non-empty custom_prompt")
    if not instruction:
        presets = load_vision_presets()
        instruction = presets.get(preset_key)
        if instruction is None:
            # preset_key may itself be a custom/free-form key; fall back to the
            # first preset or a generic instruction.
            instruction = next(iter(presets.values()),
                               "Describe this image precisely and factually for a video-generation prompt writer.")

    n = _n_images(image_tensor)
    plural = f" There are {n} images; write one paragraph per image in order." if n > 1 else ""
    user = instruction + plural

    try:
        return engine.generate_text(
            clip, VISION_SYSTEM_PROMPT, user, image_tensor=image_tensor,
            seed=seed, temperature=temperature, top_p=0.95, top_k=64,
            max_tokens=max_tokens,
        ).strip()
    except Exception:
        if n <= 1:
            raise

    # Per-image fallback.
    import torch  # only available inside ComfyUI runtime

    parts = []
    for i in range(n):
        single = image_tensor[i:i + 1]
        desc = engine.generate_text(
            clip, VISION_SYSTEM_PROMPT, instruction, image_tensor=single,
            seed=int(seed) + i, temperature=temperature, top_p=0.95, top_k=64,
            max_tokens=max_tokens,
        ).strip()
        parts.append(f"<Picture {i + 1}>: {desc}")
    return "\n\n".join(parts)
