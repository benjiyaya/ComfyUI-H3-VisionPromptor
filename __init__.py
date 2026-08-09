"""ComfyUI-H3-VisionPromptor — local VLM prompt enhancer for MiniMax H3.

Registers the V3 extension when a recent ComfyUI (comfy_api.latest + native
text generation) is available; otherwise prints an actionable message instead
of crashing the whole custom-node load.
"""

try:
    from comfy_api.latest import ComfyExtension, io  # noqa: F401
    from .py.nodes import H3VisionPromptor, H3VisionAnalyzer

    class H3VisionPromptorExtension(ComfyExtension):
        async def get_node_list(self):
            return [H3VisionPromptor, H3VisionAnalyzer]

    async def comfy_entrypoint():
        return H3VisionPromptorExtension()

except ImportError as e:  # old ComfyUI — don't crash the whole custom-node load
    print(f"[ComfyUI-H3-VisionPromptor] requires a recent ComfyUI "
          f"(comfy_api.latest + native text generation). Error: {e}")
