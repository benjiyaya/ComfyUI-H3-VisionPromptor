#!/usr/bin/env python3
"""Smoke test for ComfyUI-H3-VisionPromptor — runs OUTSIDE ComfyUI.

Stubs comfy_api.latest / folder_paths / comfy.sd with fakes, then verifies:
  1. the extension package imports and exposes comfy_entrypoint
  2. both node schemas define with the expected ids / inputs / outputs
  3. post_processor.finalize cleans a messy sample (fences + think block +
     preamble + missing audio fields + bad timestamps)
  4. prompt_builder.detect_task matrix (Auto mapping + explicit overrides)
  5. engine.generate_text end-to-end against a fake generative CLIP
  6. gemma4 family detection + template shape (media block AFTER the text)
  7. gemma (Gemma-3) emits exactly ONE <image_soft_token> regardless of n_images
  8. _collect_images center-crops mixed-resolution batches to min (H, W)
  9. vision_mode='custom' with an empty custom_prompt raises RuntimeError
  10. finalize trims BEFORE appending audio fields (appended fields survive)
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

PKG_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Stubs for ComfyUI modules
# ---------------------------------------------------------------------------

def _make_io_type(name):
    class Input:
        def __init__(self, id=None, **kwargs):
            self.id = id
            self.kwargs = kwargs

    class Output:
        def __init__(self, id=None, **kwargs):
            self.id = id
            self.kwargs = kwargs

    Input.__qualname__ = f"{name}.Input"
    Output.__qualname__ = f"{name}.Output"
    return type(name, (), {"Input": Input, "Output": Output, "Type": object})


class _Schema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _NodeOutput:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @property
    def result(self):
        return self.args or None


class _ComfyNode:
    @classmethod
    def define_schema(cls):
        raise NotImplementedError


class _ComfyExtension:
    async def get_node_list(self):
        return []


def _install_stubs():
    io = types.ModuleType("comfy_api.latest.io")
    for name in ("Clip", "Combo", "String", "Float", "Int", "Boolean", "Image",
                 "Audio", "Video", "Mask", "Latent", "Conditioning"):
        setattr(io, name, _make_io_type(name))

    class _AutogrowTemplate:
        def __init__(self, input=None, **kwargs):
            self.input = input
            self.kwargs = kwargs

    class _AutogrowInput:
        def __init__(self, id=None, template=None, **kwargs):
            self.id = id
            self.template = template
            self.kwargs = kwargs

    io.Autogrow = type("Autogrow", (), {
        "Input": _AutogrowInput,
        "TemplatePrefix": _AutogrowTemplate,
        "TemplateNames": _AutogrowTemplate,
        "Type": dict,
    })
    io.Schema = _Schema
    io.NodeOutput = _NodeOutput
    io.ComfyNode = _ComfyNode

    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ComfyExtension = _ComfyExtension

    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_filename_list = lambda folder: ["fake_qwen_vl.safetensors"] if folder == "text_encoders" else []
    folder_paths.get_full_path_or_raise = lambda folder, name: f"/fake/models/{folder}/{name}"
    folder_paths.get_folder_paths = lambda folder: [f"/fake/models/{folder}"]

    class _FakeTransformer:
        pass

    _FakeTransformer.__name__ = "Qwen2_5_VLTransformer"

    class FakeTokenizer:
        pass

    FakeTokenizer.__name__ = "Qwen2Tokenizer"

    class FakeClip:
        """Mimics the native generative CLIP interface."""
        def __init__(self):
            self.cond_stage_model = types.SimpleNamespace(transformer=_FakeTransformer())
            self.tokenizer = FakeTokenizer()
            self.calls = []

        def tokenize(self, prompt, image=None, skip_template=False, min_length=1):
            self.calls.append(("tokenize", prompt, image, skip_template, min_length))
            return {"prompt": prompt, "image": image}

        def generate(self, tokens, do_sample=False, max_length=512, temperature=1.0,
                     top_k=50, top_p=1.0, min_p=0.0, repetition_penalty=1.0,
                     presence_penalty=0.0, seed=None):
            self.calls.append(("generate", max_length, seed))
            return [1, 2, 3]

        def decode(self, ids):
            return ("integrated_multimodal_description: [Shot 1] Cinematic, a wide shot "
                    "frames a quiet platform.\n\noverall_soundscape: Distant rail hum.\n\n"
                    "non_diegetic_music: N/A")

    sd = types.ModuleType("comfy.sd")
    sd.load_clip = lambda ckpt_paths, embedding_directory=None, clip_type=None: FakeClip()
    sd.CLIPType = types.SimpleNamespace(STABLE_DIFFUSION="stable_diffusion")

    comfy = types.ModuleType("comfy")
    comfy.sd = sd

    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest
    sys.modules["comfy_api.latest.io"] = io
    sys.modules["folder_paths"] = folder_paths
    sys.modules["comfy"] = comfy
    sys.modules["comfy.sd"] = sd
    return FakeClip


def _load_package():
    spec = importlib.util.spec_from_file_location(
        "h3_vision_promptor_pkg", PKG_DIR / "__init__.py",
        submodule_search_locations=[str(PKG_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def main():
    FakeClip = _install_stubs()
    pkg = _load_package()

    # 1. extension import + entrypoint
    check("extension exposes comfy_entrypoint", hasattr(pkg, "comfy_entrypoint"))
    ext = asyncio.run(pkg.comfy_entrypoint())
    node_list = asyncio.run(ext.get_node_list())
    check("extension node list has both nodes", len(node_list) == 2)

    from h3_vision_promptor_pkg.py import nodes, post_processor, prompt_builder, engine, vision

    # 2. schemas
    s1 = nodes.H3VisionPromptor.define_schema()
    check("H3VisionPromptor node_id", s1.node_id == "H3VisionPromptor")
    check("H3VisionPromptor display_name", s1.display_name == "H3 Vision Promptor (Local VLM)")
    check("H3VisionPromptor category", s1.category == "H3/Promptor")
    expected_inputs = ["clip", "text_encoder", "user_idea", "images", "task_type",
                       "duration", "vision_mode", "extra_instructions",
                       "custom_system_prompt", "seed", "temperature", "top_p",
                       "top_k", "max_tokens", "variants", "use_default_template",
                       "keep_model_loaded"]
    got_inputs = [i.id for i in s1.inputs]
    check("H3VisionPromptor input order/names", got_inputs == expected_inputs, f"got {got_inputs}")
    check("H3VisionPromptor outputs", [o.id for o in s1.outputs] == ["prompt", "vision_context", "debug"])
    seed_in = s1.inputs[9]
    check("seed control_after_generate=randomize",
          seed_in.kwargs.get("control_after_generate") == "randomize")
    imgs_in = s1.inputs[3]
    check("images autogrow template prefix image_", imgs_in.template.kwargs.get("prefix") == "image_")
    check("images autogrow min=0 max=4",
          imgs_in.template.kwargs.get("min") == 0 and imgs_in.template.kwargs.get("max") == 4)

    s2 = nodes.H3VisionAnalyzer.define_schema()
    check("H3VisionAnalyzer node_id", s2.node_id == "H3VisionAnalyzer")
    check("H3VisionAnalyzer display_name", s2.display_name == "H3 Vision Analyzer (Local VLM)")
    check("H3VisionAnalyzer output", [o.id for o in s2.outputs] == ["description"])
    a_inputs = [i.id for i in s2.inputs]
    check("H3VisionAnalyzer inputs",
          a_inputs == ["clip", "text_encoder", "images", "vision_mode", "custom_prompt",
                       "seed", "temperature", "max_tokens", "keep_model_loaded"],
          f"got {a_inputs}")
    check("H3VisionAnalyzer images min=1 max=4",
          s2.inputs[2].template.kwargs.get("min") == 1 and s2.inputs[2].template.kwargs.get("max") == 4)

    # 3. post_processor.finalize on a messy sample
    messy = (
        "Sure! Here is your rewritten prompt:\n\n"
        "```text\n"
        "<think>let me plan the shots carefully</think>\n"
        "integrated_multimodal_description: [Shot 1] Cinematic, a wide shot frames a quiet "
        "station platform at dawn. [Shot 2] At 00:09.000, the camera cuts to a close-up of a "
        "torn ticket. [Shot 3] At 00:04.000, the shot cuts to a departing train.\n"
        "```\n"
        "Hope this helps!"
    )
    clean, warns = post_processor.finalize(messy, "T2VA", 6.0)
    check("finalize strips preamble/fences/think",
          clean.startswith("integrated_multimodal_description:"))
    check("finalize removes trailing fence text", "Hope this helps" not in clean and "```" not in clean)
    check("finalize auto-appends overall_soundscape N/A", "overall_soundscape: N/A" in clean)
    check("finalize auto-appends non_diegetic_music N/A", "non_diegetic_music: N/A" in clean)
    check("finalize warns auto-append soundscape",
          any("auto-appended missing field: overall_soundscape:" in w for w in warns))
    check("finalize warns timestamps not increasing",
          any("not strictly increasing" in w for w in warns))
    check("finalize warns timestamp exceeds duration",
          any("exceeds video duration" in w for w in warns))

    # ensure_alignment
    i2va_clean, _ = post_processor.finalize(
        "integrated_multimodal_description: [Shot 1] Cinematic...\n\n"
        "overall_soundscape: Wind.\n\nnon_diegetic_music: N/A", "I2VA", 8.0)
    check("ensure_alignment prepends I2VA line",
          i2va_clean.startswith("For the target video, at 0.00 seconds into the target video, "
                                "<Picture 1> (from [Shot 1]) is fully referenced."))
    fl2va_clean, _ = post_processor.finalize(
        "integrated_multimodal_description: [Shot 1] Cinematic...\n\n"
        "overall_soundscape: Wind.\n\nnon_diegetic_music: N/A", "FL2VA", 8.0)
    check("ensure_alignment prepends FL2VA line with duration %.2f",
          fl2va_clean.startswith("How the reference pictures align with the target video")
          and "8.00-second mark" in fl2va_clean)

    # smart trim
    long_text = "integrated_multimodal_description: " + ("word\n" * 2000) + \
                "\n\noverall_soundscape: x\n\nnon_diegetic_music: N/A"
    trimmed, twarns = post_processor.finalize(long_text, "T2VA", 8.0)
    check("finalize trims to <=7000 chars", len(trimmed) <= 7000)

    # m1: trim happens BEFORE appending missing audio fields, so the appended
    # fields always survive (finalize only warns about the final length).
    long_missing_audio = "integrated_multimodal_description: " + ("word\n" * 2000)
    trimmed2, twarns2 = post_processor.finalize(long_missing_audio, "T2VA", 8.0)
    check("finalize keeps appended audio fields after trim",
          trimmed2.rstrip().endswith("non_diegetic_music: N/A")
          and "overall_soundscape: N/A" in trimmed2)
    check("finalize reports auto-append after trim",
          any("auto-appended missing field: overall_soundscape:" in w for w in twarns2)
          and any("auto-appended missing field: non_diegetic_music:" in w for w in twarns2))
    check("finalize warns about trim without stale length claim",
          any("trimmed at the last line boundary" in w for w in twarns2))

    # 4. task detection matrix
    matrix = {(0, "Auto"): "T2VA", (1, "Auto"): "I2VA", (2, "Auto"): "FL2VA",
              (3, "Auto"): "Ref2VA", (4, "Auto"): "Ref2VA",
              (1, "L2VA"): "L2VA", (0, "t2va"): "T2VA", (2, "Ref2VA"): "Ref2VA"}
    ok = all(prompt_builder.detect_task(t, n) == want for (n, t), want in matrix.items())
    check("detect_task matrix", ok)
    try:
        prompt_builder.detect_task("BOGUS", 1)
        check("detect_task rejects unknown", False)
    except ValueError:
        check("detect_task rejects unknown", True)

    # word budgets
    check("word_budget base = duration*25", prompt_builder.word_budget("I2VA", 8.0) == 200)
    check("word_budget Ref2VA capped 500", prompt_builder.word_budget("Ref2VA", 20.0) == 500)

    # build_messages
    system, user = prompt_builder.build_messages("I2VA", 8.0, "a cat naps", "A grey cat on a sofa.",
                                                 "keep it calm", "English", "")
    check("build_messages loads base system prompt", "You are an H3 Prompt Writer" in system)
    check("build_messages includes task addendum", "I2VA" in system and "fully referenced" in system)
    check("build_messages embeds duration constraint",
          "Constraint: The video will be 8.0 seconds long (approx. 192 frames)." in user)
    check("build_messages embeds vision context + idea",
          "A grey cat on a sofa." in user and "a cat naps" in user)

    # 5. engine.generate_text with a fake generative CLIP
    clip = FakeClip()
    out = engine.generate_text(clip, "SYS", "USER", image_tensor=None, seed=42,
                               temperature=0.7, max_tokens=256)
    check("generate_text returns decoded text", out.startswith("integrated_multimodal_description:"))
    tok_call = clip.calls[0]
    check("tokenize called with native kwargs",
          tok_call[0] == "tokenize" and tok_call[3] is True and tok_call[4] == 1)
    check("family detected as qwen", engine.detect_family(clip) == "qwen")
    qtext = engine.build_chat_text("qwen", "SYS", "USER", 2, None)
    check("qwen chat template",
          qtext.startswith("<|im_start|>system\nSYS<|im_end|>")
          and qtext.count("<|vision_start|><|image_pad|><|vision_end|>") == 2
          and qtext.endswith("<|im_start|>assistant\n"))
    gtext = engine.build_chat_text("gemma", "SYS", "USER", 1, None)
    check("gemma chat template folds system into user turn",
          gtext.startswith("<start_of_turn>user\n<image_soft_token>\nSYS\n\nUSER")
          and gtext.endswith("<start_of_turn>model\n"))
    # m5: Gemma-3 binds only the first image token — exactly ONE soft token
    # no matter how many images are connected.
    gtext3 = engine.build_chat_text("gemma", "SYS", "USER", 3, None)
    check("gemma emits exactly ONE <image_soft_token> for n_images=3",
          gtext3.count("<image_soft_token>") == 1,
          f"got {gtext3.count('<image_soft_token>')}")
    gtext0 = engine.build_chat_text("gemma", "SYS", "USER", 0, None)
    check("gemma emits no soft token for n_images=0",
          "<image_soft_token>" not in gtext0)

    # M2: gemma4 family — detection (gemma4 must win over the 'gemma' substring)
    # and template shape (media block AFTER the text, one pair per image).
    class _G4Transformer:
        pass

    _G4Transformer.__name__ = "Gemma4Transformer"

    class _G4Tokenizer:
        pass

    _G4Tokenizer.__name__ = "Gemma4UnifiedTokenizer"

    g4clip = types.SimpleNamespace(
        cond_stage_model=types.SimpleNamespace(transformer=_G4Transformer()),
        tokenizer=_G4Tokenizer(),
    )
    check("detect_family returns gemma4 for Gemma-4 class names",
          engine.detect_family(g4clip) == "gemma4")
    g4text = engine.build_chat_text("gemma4", "SYS", "USER", 2, None)
    check("gemma4 template media after text, one pair per image",
          g4text == "<|turn>user\nSYS\n\nUSER\n<|image><|image|><|image><|image|>\n<|turn>model\n",
          f"got {g4text!r}")
    g4text0 = engine.build_chat_text("gemma4", "SYS", "USER", 0, None)
    check("gemma4 template without images has no media block",
          g4text0 == "<|turn>user\nSYS\n\nUSER\n<|turn>model\n", f"got {g4text0!r}")
    check("_strip_echo handles <|turn> terminator",
          engine._strip_echo("<|turn>user\nSYS\n\nUSER\n<|turn>model\nANSWER<|turn>", g4text0, "gemma4")
          == "ANSWER")
    otext = engine.build_chat_text("generic", "SYS", "USER", 0, "{system} || {user} || {images}")
    check("template override placeholders", otext == "SYS || USER || ")

    # resolve_clip prefers connected input and caches loader path
    c2, desc = engine.resolve_clip(clip, "<none - use CLIP input>")
    check("resolve_clip prefers clip input", c2 is clip and "connected" in desc)
    c3, desc3 = engine.resolve_clip(None, "fake_qwen_vl.safetensors")
    c4, _ = engine.resolve_clip(None, "fake_qwen_vl.safetensors")
    check("resolve_clip loads + caches text encoder", c3 is c4 and "text_encoders/" in desc3)

    # vision presets load
    presets = vision.load_vision_presets()
    check("vision presets loaded", "detailed_subject_scene" in presets and "scene_only" in presets)

    # m2: vision_mode='custom' with an empty/whitespace custom_prompt must raise.
    class FakeImage:
        shape = (1, 64, 64, 3)

    for bad_prompt in (None, "", "   \n  "):
        try:
            vision.analyze_images(clip, FakeImage(), "custom", bad_prompt,
                                  seed=1, temperature=0.2, max_tokens=64)
            check(f"custom vision mode rejects empty custom_prompt ({bad_prompt!r})", False)
        except RuntimeError as e:
            check(f"custom vision mode rejects empty custom_prompt ({bad_prompt!r})",
                  "requires a non-empty custom_prompt" in str(e), str(e))
    # analyzer node path raises the same clear error before touching the VLM
    try:
        nodes.H3VisionAnalyzer.execute(
            text_encoder="fake_qwen_vl.safetensors", vision_mode="custom",
            seed=1, temperature=0.2, max_tokens=256, keep_model_loaded=True,
            images={"image_0": FakeImage()}, custom_prompt="  ",
        )
        check("analyzer custom mode empty prompt raises", False)
    except RuntimeError as e:
        check("analyzer custom mode empty prompt raises",
              "requires a non-empty custom_prompt" in str(e), str(e))

    # M1: _collect_images center-crops mixed-resolution batches to min (H, W).
    if torch is None:
        print("[SKIP] torch not available — _collect_images mixed-resolution checks skipped")
    else:
        big = torch.zeros(1, 64, 96, 3)
        big[0, 8:56, 24:72, :] = 1.0  # centered 48x48 patch
        small = torch.full((1, 48, 48, 3), 2.0)
        batched = nodes._collect_images({"image_0": big, "image_1": small})
        check("_collect_images crops to min (H, W)",
              tuple(batched.shape) == (2, 48, 48, 3), f"got {tuple(batched.shape)}")
        check("_collect_images center-crops (equal both sides)",
              bool((batched[0] == 1.0).all()) and bool((batched[1] == 2.0).all()))
        odd = torch.zeros(1, 5, 7, 3)
        odd[0, 1:4, 2:5, :] = 3.0  # centered 3x3 patch (odd dims: extra pixel bottom/right)
        even = torch.full((1, 3, 3, 3), 4.0)
        b2 = nodes._collect_images([odd, even])
        check("_collect_images handles odd crop offsets",
              tuple(b2.shape) == (2, 3, 3, 3) and bool((b2[0] == 3.0).all()))
        single = torch.zeros(1, 10, 12, 3)
        check("_collect_images passes single image through untouched",
              nodes._collect_images({"image_0": single}) is single)
        check("_collect_images returns None for empty payload",
              nodes._collect_images(None) is None and nodes._collect_images({}) is None)

    # analyzer execute path with dict autogrow payload (needs torch -> skip image tensor;
    # single image object works without torch)
    analyzer_out = nodes.H3VisionAnalyzer.execute(
        text_encoder="fake_qwen_vl.safetensors", vision_mode="detailed_subject_scene",
        seed=1, temperature=0.2, max_tokens=256, keep_model_loaded=True,
        images={"image_0": FakeImage()},
    )
    check("analyzer execute returns NodeOutput with description",
          isinstance(analyzer_out, _NodeOutput) and len(analyzer_out.args) == 1
          and isinstance(analyzer_out.args[0], str) and analyzer_out.args[0])

    print()
    if FAILURES:
        print(f"SMOKE TEST FAILED: {len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("SMOKE TEST PASSED: all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
