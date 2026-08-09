"""ComfyUI V3 node definitions for H3 Vision Promptor.

Mirrors the native TextGenerate node (comfy_extras/nodes_textgen.py) for the
tokenize/generate/decode call pattern, and the proven io.Autogrow usage with
io.Autogrow.TemplatePrefix for multi-image inputs.

Top-level comfy imports (folder_paths, comfy_api.latest) are acceptable here —
this module is only imported inside ComfyUI (or by tests that stub them).
"""

from __future__ import annotations

import json
import time

import folder_paths
from comfy_api.latest import io

from . import engine, vision, prompt_builder, post_processor

CATEGORY = "H3/Promptor"
NONE_OPTION = "<none - use CLIP input>"
SKIP_VISION = "skip (idea only)"
CUSTOM_VISION = "custom"
TASK_OPTIONS = ["Auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"]
MAX_SEED = 0xFFFFFFFF


def _text_encoder_options() -> list[str]:
    try:
        return [NONE_OPTION] + folder_paths.get_filename_list("text_encoders")
    except Exception:
        return [NONE_OPTION]


def _vision_preset_keys() -> list[str]:
    return list(vision.load_vision_presets().keys())


def _images_autogrow(min_images: int) -> io.Autogrow.Input:
    """Multi-image input as a growable group of IMAGE sockets (image_0..image_3),
    following the Autogrow.TemplatePrefix pattern used in the wild."""
    return io.Autogrow.Input(
        "images",
        optional=True,
        tooltip="Reference image(s). Add up to 4. 1 image = I2VA/Ref2VA, 2 = FL2VA, 3+ = Ref2VA.",
        template=io.Autogrow.TemplatePrefix(
            input=io.Image.Input("image", tooltip="Reference image"),
            prefix="image_",
            min=min_images,
            max=4,
        ),
    )


def _collect_images(images) -> "object | None":
    """Normalize the Autogrow payload ({'image_0': tensor, ...} / list / tensor)
    into a single batched IMAGE tensor, or None when nothing is connected."""
    if images is None:
        return None
    values = []
    if isinstance(images, dict):
        # sorted() is correct here because the Autogrow keys are image_0..image_3
        # (max 4 < 10), so lexicographic order == numeric order.
        for key in sorted(images.keys()):
            values.append(images[key])
    elif isinstance(images, (list, tuple)):
        values = list(images)
    else:
        values = [images]
    values = [v for v in values if v is not None]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    import torch
    # Each Autogrow socket yields [1, H, W, C] and H/W may differ between
    # sockets, which makes torch.cat(dim=0) raise. Mirror ComfyUI core's
    # ImageBatch: center-crop every tensor (equally from both sides) to the
    # minimum (H, W) across the batch before concatenating.
    min_h = min(int(v.shape[1]) for v in values)
    min_w = min(int(v.shape[2]) for v in values)
    cropped = []
    for v in values:
        h, w = int(v.shape[1]), int(v.shape[2])
        top = (h - min_h) // 2
        left = (w - min_w) // 2
        cropped.append(v[:, top:top + min_h, left:left + min_w, :])
    return torch.cat(cropped, dim=0)


def _n_images(image_tensor) -> int:
    if image_tensor is None:
        return 0
    try:
        return int(image_tensor.shape[0])
    except Exception:
        return 1


class H3VisionPromptor(io.ComfyNode):
    """Local, API-key-free MiniMax H3 prompt enhancer powered by a native VLM
    text encoder (Qwen2.5-VL / Qwen3-VL / Gemma vision)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3VisionPromptor",
            display_name="H3 Vision Promptor (Local VLM)",
            category=CATEGORY,
            search_aliases=["MiniMax", "H3", "prompt", "VLM", "vision prompt"],
            description=(
                "Turns an idea (+ optional reference images) into a production-ready MiniMax H3 "
                "prompt using a LOCAL VLM running on ComfyUI's native text-generation stack. "
                "No API keys, no external services."
            ),
            inputs=[
                io.Clip.Input("clip", optional=True,
                              tooltip="Optional: connect CLIP from the native CLIPLoader node. "
                                      "Overrides the text_encoder dropdown."),
                io.Combo.Input("text_encoder", options=_text_encoder_options(),
                               tooltip="VLM checkpoint from models/text_encoders/ "
                                       "(e.g. a Qwen3-VL or Gemma-3 instruct repack)."),
                io.String.Input("user_idea", multiline=True, default="",
                                placeholder="A young woman on a night train reads a letter, then smiles..."),
                _images_autogrow(min_images=0),
                io.Combo.Input("task_type", options=TASK_OPTIONS, default="Auto",
                               tooltip="Auto: 0 images=T2VA, 1=I2VA, 2=FL2VA, 3+=Ref2VA. "
                                       "Pick L2VA explicitly for last-frame tasks."),
                io.Float.Input("duration", default=8.0, min=4.0, max=15.0, step=0.5,
                               tooltip="Target video length in seconds."),
                io.Combo.Input("vision_mode", options=_vision_preset_keys() + [SKIP_VISION],
                               tooltip="How the VLM describes the reference image(s) before writing. "
                                       "'skip (idea only)' writes from the idea alone."),
                io.String.Input("extra_instructions", multiline=True, default="", optional=True,
                                placeholder="Optional extras: dialogue lines, mood, camera wishes..."),
                io.String.Input("custom_system_prompt", multiline=True, default="", optional=True,
                                advanced=True,
                                tooltip="Advanced: fully replace the built-in H3 system prompt."),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,
                             control_after_generate="randomize"),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05),
                io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=64, min=0, max=500),
                io.Int.Input("max_tokens", default=2048, min=256, max=8192, step=64),
                io.Int.Input("variants", default=1, min=1, max=4,
                             tooltip="Generate N prompt variants (seed, seed+1, ...) in one run."),
                io.Boolean.Input("use_default_template", default=False, advanced=True,
                                 tooltip="Let the model's built-in chat template wrap the prompt "
                                         "instead of this node's H3-aware template."),
                io.Boolean.Input("keep_model_loaded", default=True, advanced=True,
                                 tooltip="Cache the text-encoder CLIP between runs (faster, uses VRAM/RAM)."),
            ],
            outputs=[
                io.String.Output("prompt"),
                io.String.Output("vision_context"),
                io.String.Output("debug"),
            ],
        )

    @classmethod
    def execute(cls, text_encoder, user_idea, task_type, duration, vision_mode,
                seed, temperature, top_p, top_k, max_tokens, variants,
                use_default_template, keep_model_loaded,
                clip=None, images=None, extra_instructions="",
                custom_system_prompt="") -> io.NodeOutput:
        t_start = time.time()

        clip_obj, source_desc = engine.resolve_clip(clip, text_encoder, keep_model_loaded)
        image_tensor = _collect_images(images)
        n_images = _n_images(image_tensor)
        task = prompt_builder.detect_task(task_type, n_images)

        # --- vision pass -------------------------------------------------
        vision_context = ""
        vision_seconds = 0.0
        vision_error = None
        if image_tensor is not None and vision_mode != SKIP_VISION:
            t_vis = time.time()
            try:
                vision_context = vision.analyze_images(
                    clip_obj, image_tensor, vision_mode, None,
                    seed=seed, temperature=temperature, max_tokens=max_tokens,
                )
            except Exception as e:
                vision_error = str(e)
                print(f"[H3 VisionPromptor] Vision pass failed, continuing without it: {e}")
            vision_seconds = time.time() - t_vis

        # --- prompt-writing generation -----------------------------------
        system, user = prompt_builder.build_messages(
            task, duration, user_idea, vision_context, extra_instructions,
            "English", custom_system_prompt,
        )

        results: list[str] = []
        warnings: list[str] = []
        t_gen = time.time()
        for i in range(int(variants)):
            raw = engine.generate_text(
                clip_obj, system, user,
                image_tensor=image_tensor,
                seed=int(seed) + i,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                use_default_template=use_default_template,
            )
            clean, warns = post_processor.finalize(raw, task, duration)
            results.append(clean)
            warnings.extend(f"[variant {i + 1}] {w}" for w in warns)
        gen_seconds = time.time() - t_gen

        if vision_error:
            warnings.append(f"vision pass failed: {vision_error}")

        prompt = results[0]
        for i in range(1, len(results)):
            prompt += f"\n\n===== VARIANT {i + 1} =====\n\n" + results[i]

        debug = json.dumps({
            "clip_source": source_desc,
            "model_family": engine.detect_family(clip_obj),
            "detected_task": task,
            "n_images": n_images,
            "duration_seconds": duration,
            "word_budget": prompt_builder.word_budget(task, duration),
            "vision_mode": vision_mode if image_tensor is not None else f"{vision_mode} (no images)",
            "vision_seconds": round(vision_seconds, 2),
            "generation_seconds": round(gen_seconds, 2),
            "total_seconds": round(time.time() - t_start, 2),
            "variants": int(variants),
            "warnings": warnings,
        }, indent=2, ensure_ascii=False)

        return io.NodeOutput(prompt, vision_context, debug)


class H3VisionAnalyzer(io.ComfyNode):
    """Standalone VLM image analysis (one batched vision pass)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3VisionAnalyzer",
            display_name="H3 Vision Analyzer (Local VLM)",
            category=CATEGORY,
            search_aliases=["MiniMax", "H3", "vision", "describe", "VLM"],
            description=(
                "Describes reference image(s) with a LOCAL VLM (native text generation). "
                "Feed the description into the H3 Vision Promptor or use it standalone."
            ),
            inputs=[
                io.Clip.Input("clip", optional=True,
                              tooltip="Optional: connect CLIP from the native CLIPLoader node."),
                io.Combo.Input("text_encoder", options=_text_encoder_options(),
                               tooltip="VLM checkpoint from models/text_encoders/."),
                _images_autogrow(min_images=1),
                io.Combo.Input("vision_mode", options=_vision_preset_keys() + [CUSTOM_VISION],
                               tooltip="Analysis preset, or 'custom' to use custom_prompt."),
                io.String.Input("custom_prompt", multiline=True, default="", optional=True,
                                placeholder="Used when vision_mode = custom."),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,
                             control_after_generate="randomize"),
                io.Float.Input("temperature", default=0.2, min=0.0, max=2.0, step=0.05),
                io.Int.Input("max_tokens", default=2048, min=256, max=8192, step=64),
                io.Boolean.Input("keep_model_loaded", default=True, advanced=True,
                                 tooltip="Cache the text-encoder CLIP between runs."),
            ],
            outputs=[
                io.String.Output("description"),
            ],
        )

    @classmethod
    def execute(cls, text_encoder, vision_mode, seed, temperature, max_tokens,
                keep_model_loaded, clip=None, images=None, custom_prompt="") -> io.NodeOutput:
        clip_obj, _source = engine.resolve_clip(clip, text_encoder, keep_model_loaded)
        image_tensor = _collect_images(images)
        if image_tensor is None:
            raise RuntimeError(
                "[H3 VisionPromptor] H3VisionAnalyzer needs at least one image: "
                "connect an IMAGE output to the node's image input(s)."
            )
        prompt_override = (custom_prompt or "").strip() if vision_mode == CUSTOM_VISION else None
        if vision_mode == CUSTOM_VISION and not prompt_override:
            raise RuntimeError(
                "[H3 VisionPromptor] vision_mode='custom' requires a non-empty custom_prompt: "
                "type your analysis instruction into the custom_prompt input."
            )
        description = vision.analyze_images(
            clip_obj, image_tensor, vision_mode, prompt_override,
            seed=seed, temperature=temperature, max_tokens=max_tokens,
        )
        return io.NodeOutput(description)
