"""Engine: CLIP resolution, model-family detection, chat templating, and native
text generation via ComfyUI's built-in CLIP.generate stack.

This module mirrors the call pattern of ComfyUI's native TextGenerate node
(comfy_extras/nodes_textgen.py @ master):

    tokens = clip.tokenize(prompt, image=image, skip_template=not use_default_template, min_length=1)
    generated_ids = clip.generate(tokens, do_sample=..., max_length=..., temperature=...,
                                  top_k=..., top_p=..., min_p=..., repetition_penalty=...,
                                  presence_penalty=..., seed=...)
    generated_text = clip.decode(generated_ids)

All comfy / folder_paths imports are done lazily inside functions so this file
compiles and imports cleanly outside of ComfyUI.
"""

from __future__ import annotations

# Cache of loaded CLIP objects keyed by absolute checkpoint path.
CLIP_CACHE: dict[str, object] = {}


# ---------------------------------------------------------------------------
# CLIP resolution
# ---------------------------------------------------------------------------

def _clip_is_generatable(clip) -> bool:
    return clip is not None and hasattr(clip, "generate") and hasattr(clip, "tokenize") and hasattr(clip, "decode")


def _assert_generatable(clip, source_desc: str):
    if not _clip_is_generatable(clip):
        raise RuntimeError(
            f"[H3 VisionPromptor] The CLIP from {source_desc} does not support native text generation "
            "(missing tokenize/generate/decode). Use a generative VLM text encoder "
            "(e.g. Qwen3-VL / Gemma-3/4-Vision instruct repacks — note Qwen2.5-VL repacks are "
            "vision-tower-only and not generatable) loaded via the native "
            "CLIPLoader node or from models/text_encoders/, and update ComfyUI to a version whose "
            "comfy.sd CLIP implements .generate()."
        )


def resolve_clip(clip_input, text_encoder_name: str, keep_model_loaded: bool = True):
    """Return (clip_obj, source_desc).

    Prefers ``clip_input`` when provided; otherwise loads ``text_encoder_name``
    from ComfyUI's ``models/text_encoders/`` folder (cached by full path when
    ``keep_model_loaded`` is True). Raises RuntimeError with actionable guidance
    when neither source is usable or the resolved arch lacks generate().
    """
    if clip_input is not None:
        _assert_generatable(clip_input, "the connected CLIP input")
        return clip_input, "connected CLIP input"

    none_markers = {"", "<none - use CLIP input>", "none", "None", None}
    if text_encoder_name in none_markers:
        raise RuntimeError(
            "[H3 VisionPromptor] No CLIP source: connect a CLIP output (from the native CLIPLoader "
            "node) or pick a model in the 'text_encoder' dropdown. Supported: generative VLM text "
            "encoders such as Qwen3-VL / Gemma-3/4-Vision instruct repacks placed in "
            "ComfyUI/models/text_encoders/."
        )

    try:
        import folder_paths
        import comfy.sd
    except ImportError as e:  # pragma: no cover - only inside ComfyUI
        raise RuntimeError(
            "[H3 VisionPromptor] Could not import ComfyUI modules (folder_paths / comfy.sd). "
            "This node must run inside a recent ComfyUI."
        ) from e

    full_path = folder_paths.get_full_path_or_raise("text_encoders", text_encoder_name)
    if keep_model_loaded and full_path in CLIP_CACHE:
        return CLIP_CACHE[full_path], f"text_encoders/{text_encoder_name} (cached)"

    clip = comfy.sd.load_clip(
        ckpt_paths=[full_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
    )
    _assert_generatable(clip, f"text_encoders/{text_encoder_name}")
    if keep_model_loaded:
        CLIP_CACHE[full_path] = clip
    return clip, f"text_encoders/{text_encoder_name}"


# ---------------------------------------------------------------------------
# Model-family detection and chat templating
# ---------------------------------------------------------------------------

def detect_family(clip) -> str:
    """Sniff the CLIP's transformer / tokenizer class names to classify the model.

    Returns "qwen" | "gemma4" | "gemma" | "generic" (case-insensitive matching).
    "gemma4" is checked before "gemma" because Gemma-4 class names also contain
    the substring "gemma".
    """
    names: list[str] = []
    try:
        transformer = getattr(getattr(clip, "cond_stage_model", None), "transformer", None)
        if transformer is not None:
            names.append(type(transformer).__name__)
    except Exception:
        pass
    try:
        tokenizer = getattr(clip, "tokenizer", None)
        if tokenizer is not None:
            names.append(type(tokenizer).__name__)
            inner = getattr(tokenizer, "tokenizer", None)
            if inner is not None:
                names.append(type(inner).__name__)
    except Exception:
        pass
    haystack = " ".join(names).lower()
    if "qwen" in haystack:
        return "qwen"
    if "gemma4" in haystack:
        return "gemma4"
    if "gemma" in haystack:
        return "gemma"
    return "generic"


def _vision_block(family: str, n_images: int) -> str:
    """Per-family repeated vision token block for n_images images."""
    if n_images <= 0:
        return ""
    if family == "qwen":
        return "<|vision_start|><|image_pad|><|vision_end|>\n" * n_images
    if family == "gemma4":
        # Gemma-4: one <|image><|image|> pair per image, placed AFTER the text.
        return "<|image><|image|>" * n_images
    if family == "gemma":
        # Gemma-3 binds only the first image token and treats the whole batch as
        # a single images entry: emit exactly ONE soft token regardless of count.
        return "<image_soft_token>\n"
    return ""


def build_chat_text(family: str, system: str, user: str, n_images: int, template_override: str | None) -> str:
    """Build the raw chat-formatted prompt text for the given model family."""
    if template_override:
        images = _vision_block(family, n_images)
        return (
            template_override
            .replace("{system}", system)
            .replace("{user}", user)
            .replace("{images}", images)
        )

    if family == "qwen":
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n"
            + _vision_block(family, n_images)
            + f"{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    if family == "gemma4":
        # Gemma-4 (comfy/text_encoders/gemma4.py): <|turn> markers, no system
        # role (fold system into the user turn), media block AFTER the text.
        text = f"<|turn>user\n{system}\n\n{user}"
        if n_images > 0:
            text += "\n" + _vision_block(family, n_images)
        return text + "\n<|turn>model\n"
    if family == "gemma":
        # Gemma has no system role: fold the system prompt into the user turn.
        return (
            "<start_of_turn>user\n"
            + _vision_block(family, n_images)
            + f"{system}\n\n{user}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    return f"{system}\n\n{user}"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _strip_echo(decoded: str, prompt: str, family: str) -> str:
    """Strip an echoed prompt prefix from decoded text if the model returned it."""
    text = decoded
    if text.startswith(prompt):
        text = text[len(prompt):]
    # Some decoders return the full conversation; cut everything up to and
    # including the assistant turn marker when it appears verbatim.
    for marker in ("<|im_start|>assistant\n", "<start_of_turn>model\n", "<|turn>model\n"):
        if marker in text:
            text = text.split(marker, 1)[1]
    # Remove trailing turn terminators.
    for stop in ("<|im_end|>", "<end_of_turn>", "<|im_start|>", "<start_of_turn>", "<|turn>"):
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def generate_text(clip, system: str, user: str, image_tensor=None, seed: int = 0, temperature: float = 0.7,
                  top_p: float = 0.95, top_k: int = 64, max_tokens: int = 2048, min_p: float = 0.0,
                  repetition_penalty: float = 1.0, use_default_template: bool = False,
                  template_override: str | None = None) -> str:
    """Full chat-format generation using ComfyUI's native CLIP generation stack.

    Builds the prompt text via build_chat_text(detect_family(clip), ...), tokenizes
    with image=image_tensor when provided, generates, decodes, strips any echoed
    prompt prefix, and returns the clean string.
    """
    _assert_generatable(clip, "the resolved CLIP")
    family = detect_family(clip)
    n_images = 0
    if image_tensor is not None:
        try:
            n_images = int(image_tensor.shape[0])
        except Exception:
            n_images = 1
    prompt = build_chat_text(family, system, user, n_images, template_override)

    try:
        tokens = clip.tokenize(prompt, image=image_tensor, skip_template=not use_default_template, min_length=1)
    except TypeError:
        # Older / variant tokenize signatures: fall back to the minimal call.
        try:
            tokens = clip.tokenize(prompt, image=image_tensor)
        except TypeError as e:
            raise RuntimeError(
                "[H3 VisionPromptor] clip.tokenize(...) rejected the native keyword arguments "
                f"({e}). Update ComfyUI to a recent version whose CLIP implements the native "
                "text-generation tokenize signature (prompt, image=..., skip_template=..., min_length=1), "
                "and make sure the connected CLIP comes from CLIPLoader with a generative VLM."
            ) from e
    except Exception as e:
        raise RuntimeError(f"[H3 VisionPromptor] clip.tokenize failed: {e}") from e

    try:
        generated_ids = clip.generate(
            tokens,
            do_sample=temperature > 0.0,
            max_length=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=0.0,
            seed=seed,
        )
    except TypeError as e:
        raise RuntimeError(
            "[H3 VisionPromptor] clip.generate(...) rejected the native sampling arguments "
            f"({e}). Update ComfyUI and use a CLIP whose architecture supports native generation "
            "(connect the CLIP output of CLIPLoader or pick a text_encoder from models/text_encoders/)."
        ) from e
    except Exception as e:
        raise RuntimeError(f"[H3 VisionPromptor] clip.generate failed: {e}") from e

    decoded = clip.decode(generated_ids)
    return _strip_echo(decoded, prompt, family)
