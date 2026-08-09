"""Post-processing of raw VLM output into a clean H3 prompt:

sanitize -> ensure_alignment -> smart trim -> auto-append missing audio fields
-> final length re-check (warn only), with a validate() that reports
human-readable warnings.
"""

from __future__ import annotations

import re

MAX_PROMPT_CHARS = 7000

# Headers that mark the real start of an H3 prompt (used to cut preambles).
RECOGNIZED_HEADERS = (
    "subject_definitions:",
    "integrated_multimodal_description:",
    "For the target video",
    "How the reference pictures",
)

I2VA_ALIGNMENT = ("For the target video, at 0.00 seconds into the target video, "
                  "<Picture 1> (from [Shot 1]) is fully referenced.")

_FL2VA_ALIGNMENT_TMPL = ("How the reference pictures align with the target video — "
                         "<Picture 1> (from [Shot 1]) aligns with the 0.00-second mark of the "
                         "target video; <Picture 2> (from [Shot {n}]) aligns with the "
                         "{duration:.2f}-second mark of the target video.")

_L2VA_ALIGNMENT_TMPL = ("How the reference pictures align with the target video — "
                        "<Picture 1> (from [Shot {n}]) aligns with the {duration:.2f}-second "
                        "mark of the target video.")

_TIMESTAMP_RE = re.compile(r"\bAt (\d{1,2}):(\d{2})\.(\d{3})\b")
_SHOT_RE = re.compile(r"\[Shot (\d+)\]")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*$", re.MULTILINE)

AUDIO_FIELDS = ("overall_soundscape:", "non_diegetic_music:")


_FENCED_BLOCK_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)^\s*```\s*$", re.DOTALL | re.MULTILINE)


def sanitize(text: str) -> str:
    """Strip ``` fences, <think>...</think> blocks, and any preamble before the
    first recognized H3 header."""
    if not text:
        return ""
    out = _THINK_RE.sub("", text)
    # If the model wrapped the prompt in a fenced block, prefer the fenced
    # content that actually holds an H3 header (drops chatter around the fence).
    for m in _FENCED_BLOCK_RE.finditer(out):
        if any(h in m.group(1) for h in RECOGNIZED_HEADERS):
            out = m.group(1)
            break
    out = _FENCE_RE.sub("", out)
    # Cut everything before the first recognized header.
    first = None
    for header in RECOGNIZED_HEADERS:
        idx = out.find(header)
        if idx != -1 and (first is None or idx < first):
            first = idx
    if first is not None:
        out = out[first:]
    return out.strip()


def _last_shot_index(text: str) -> int:
    shots = [int(m) for m in _SHOT_RE.findall(text)]
    return max(shots) if shots else 1


def ensure_alignment(text: str, task: str, duration: float) -> str:
    """Prepend the mode's mandatory alignment instruction line when absent."""
    if task == "I2VA":
        if "For the target video, at 0.00 seconds" not in text:
            text = I2VA_ALIGNMENT + "\n\n" + text
    elif task == "FL2VA":
        if "How the reference pictures align" not in text:
            line = _FL2VA_ALIGNMENT_TMPL.format(n=_last_shot_index(text), duration=duration)
            text = line + "\n\n" + text
    elif task == "L2VA":
        if "How the reference pictures align" not in text:
            line = _L2VA_ALIGNMENT_TMPL.format(n=_last_shot_index(text), duration=duration)
            text = line + "\n\n" + text
    return text


def _timestamps(text: str) -> list[float]:
    return [int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1000.0
            for m in _TIMESTAMP_RE.finditer(text)]


def validate(text: str, task: str, duration: float) -> list[str]:
    """Return a list of human-readable warnings about the prompt."""
    warnings: list[str] = []

    if task == "Ref2VA":
        required = ("subject_definitions:", "summary:", "retention_analysis:",
                    "detailed_description:") + AUDIO_FIELDS
    else:
        required = ("integrated_multimodal_description:",) + AUDIO_FIELDS
    for field in required:
        if field not in text:
            warnings.append(f"missing required field: {field}")

    if task == "I2VA" and "For the target video, at 0.00 seconds" not in text:
        warnings.append("missing I2VA alignment instruction line")
    if task in ("FL2VA", "L2VA") and "How the reference pictures align" not in text:
        warnings.append(f"missing {task} alignment instruction line")

    stamps = _timestamps(text)
    for a, b in zip(stamps, stamps[1:]):
        if b <= a:
            warnings.append(f"timestamps not strictly increasing: {a:.3f}s followed by {b:.3f}s")
            break
    for s in stamps:
        if s > duration:
            warnings.append(f"timestamp {s:.3f}s exceeds video duration {duration:.2f}s")
            break

    if len(text) > MAX_PROMPT_CHARS:
        warnings.append(f"prompt length {len(text)} chars exceeds {MAX_PROMPT_CHARS}")

    return warnings


def _smart_trim(text: str) -> str:
    """Trim to <= MAX_PROMPT_CHARS at the last line boundary before the limit."""
    if len(text) <= MAX_PROMPT_CHARS:
        return text
    cut = text.rfind("\n", 0, MAX_PROMPT_CHARS)
    if cut == -1:
        cut = MAX_PROMPT_CHARS
    return text[:cut].rstrip()


def finalize(text: str, task: str, duration: float) -> tuple[str, list[str]]:
    """Full cleanup pipeline. Returns (clean_text, warnings)."""
    out = sanitize(text)
    out = ensure_alignment(out, task, duration)

    warnings = validate(out, task, duration)

    # Smart-trim FIRST so the auto-appended audio fields below can never be
    # trimmed away. Replace the stale pre-trim length warning with an accurate
    # "was trimmed" note.
    if len(out) > MAX_PROMPT_CHARS:
        out = _smart_trim(out)
        warnings = [w for w in warnings if not w.startswith("prompt length")]
        warnings.append(
            f"prompt exceeded {MAX_PROMPT_CHARS} chars; trimmed at the last line boundary"
        )

    # Auto-append missing audio fields as N/A and reword their warnings.
    fixed_warnings: list[str] = []
    for w in warnings:
        if w in (f"missing required field: {AUDIO_FIELDS[0]}",
                 f"missing required field: {AUDIO_FIELDS[1]}"):
            field = w.split(": ", 1)[1]
            out = out.rstrip() + f"\n\n{field} N/A"
            fixed_warnings.append(f"auto-appended missing field: {field} N/A")
        else:
            fixed_warnings.append(w)

    # Re-check length AFTER appending: warn only — never trim away the fields
    # we just appended.
    if len(out) > MAX_PROMPT_CHARS:
        fixed_warnings.append(
            f"prompt length {len(out)} chars still exceeds {MAX_PROMPT_CHARS} after "
            "cleanup; left untrimmed to preserve the appended audio fields"
        )
    return out, fixed_warnings
