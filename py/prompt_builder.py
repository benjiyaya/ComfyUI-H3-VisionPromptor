"""Prompt assembly: task detection, system-prompt loading, word budgets, and
(system, user) message construction for the H3 prompt-writing generation.
"""

from __future__ import annotations

import os

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")

TASK_TYPES = ("T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA")


def detect_task(task_type: str, n_images: int) -> str:
    """Resolve the effective task. Auto: 0 images -> T2VA, 1 -> I2VA, 2 -> FL2VA, >=3 -> Ref2VA."""
    t = (task_type or "Auto").strip()
    if t.lower() != "auto":
        for known in TASK_TYPES:
            if t.lower() == known.lower():
                return known
        raise ValueError(f"[H3 VisionPromptor] Unknown task_type '{task_type}'. "
                         f"Expected one of Auto, {', '.join(TASK_TYPES)}.")
    if n_images <= 0:
        return "T2VA"
    if n_images == 1:
        return "I2VA"
    if n_images == 2:
        return "FL2VA"
    return "Ref2VA"


def _read_prompt_file(*parts: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, *parts), "r", encoding="utf-8") as f:
        return f.read().strip()


def load_system_prompt(task: str) -> str:
    """Base H3 system prompt + task addendum, re-read from disk on every run."""
    base = _read_prompt_file("h3_system_base.txt")
    addendum = _read_prompt_file("tasks", f"{task.lower()}.txt")
    return base + "\n\n" + addendum


def word_budget(task: str, duration: float) -> int:
    """Target body length in words. Base modes: duration*25. Ref2VA's
    detailed_description targets 350-500 words -> duration*35 capped at 500."""
    if task == "Ref2VA":
        return min(int(duration * 35), 500)
    return int(duration * 25)


def build_messages(task, duration, user_idea, vision_context, extra_instructions,
                   output_language, custom_system_prompt) -> tuple[str, str]:
    """Build the (system, user) pair for the prompt-writing generation."""
    system = (custom_system_prompt or "").strip() or load_system_prompt(task)

    budget = word_budget(task, duration)
    frames = int(duration * 24)

    user_parts: list[str] = []
    if vision_context and vision_context.strip():
        user_parts.append(
            "Reference image analysis (facts about the provided picture(s); "
            "<Picture N> refers to the Nth provided image):\n" + vision_context.strip()
        )

    idea = (user_idea or "").strip()
    if idea:
        user_parts.append("User idea:\n" + idea)
    else:
        user_parts.append("User idea: (none provided — invent a coherent, concrete scene "
                          "consistent with the reference image analysis, if any.)")

    user_parts.append(f"Constraint: The video will be {duration} seconds long (approx. {frames} frames).")

    if task == "Ref2VA":
        user_parts.append(f"Write the detailed_description section at roughly {budget} words "
                          "(target range 350-500 words; prioritize a complete spoken timeline "
                          "over mechanically hitting the count).")
    else:
        user_parts.append(f"Write the integrated_multimodal_description body at roughly {budget} words.")

    if extra_instructions and extra_instructions.strip():
        user_parts.append("Additional instructions from the user (respect them unless they "
                          "conflict with the H3 format rules):\n" + extra_instructions.strip())

    if (output_language or "English").strip().lower().startswith("chinese"):
        user_parts.append("Language: write the prompt body in Simplified Chinese (简体中文), "
                          "but keep all field names, shot labels, reference labels, speaker IDs, "
                          "and dialogue/lyrics in their original language exactly as provided.")
    else:
        user_parts.append("Language: write the prompt body in English. Dialogue, lyrics, and "
                          "on-screen text keep their original language verbatim.")

    return system, "\n\n".join(user_parts)
