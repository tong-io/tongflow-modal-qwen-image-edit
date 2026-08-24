"""Modal deploy entry for Qwen-Image-Edit.

Deploy:
  modal deploy deploy.py

Design constraints:
  - Keep this file mostly self-contained because Modal remote imports may mount
    only the entry file.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, List

import modal
from tongflow import deploy
from tongflow.models.image_edit import ImageEditInput, ImageEditOutput
from tongflow.models.image_fusion import ImageFusionInput, ImageFusionOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.slots import node_slot


_cfg: dict[str, Any] = {}
_hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
# 2511 over 2509: same size and same pipeline, better subject consistency and
# less drift between the input and the edit. Ungated, unlike the FLUX.2 line.
REPO_ID = str(_hf.get("repoId") or "Qwen/Qwen-Image-Edit-2511")
MODEL_DIR = f"/models/{REPO_ID}"

# Sampling defaults from the model card — plugin-internal, not ABI fields.
DEFAULT_NUM_INFERENCE_STEPS = 40
DEFAULT_TRUE_CFG_SCALE = 4.0
DEFAULT_GUIDANCE_SCALE = 1.0
# The card passes a single space rather than "": the pipeline builds a negative
# branch either way, and an empty string trips its prompt-length check.
DEFAULT_NEGATIVE_PROMPT = " "

volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)


# ── app ──────────────────────────────────────────────────────────────────────

APP_NAME = Path(__file__).resolve().parent.name
app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .pip_install(
        "tongflow==0.2.21",
        "fastapi[standard]",
        "diffusers==0.40.0",
        "transformers==5.4.0",
        "safetensors==0.7.0",
        "pillow==12.1.1",
        "accelerate==1.13.0",
        "huggingface_hub==1.6.0",
        "sentencepiece==0.2.1",
    )
)

with image.imports():
    import torch
    from diffusers import QwenImageEditPlusPipeline


# 80GB, not L40S: the transformer is 40.9GB in bf16 and the Qwen2.5-VL text
# encoder another 16.6GB, so the weights alone overflow a 48GB card. Offloading
# would swap most of that over PCIe on every call — slower per image, and Modal
# bills by the second either way.
@deploy
@app.cls(
    scaledown_window=2,
    image=image,
    gpu="H100",
    volumes={"/models": volume},
    timeout=1800,
)
class Inference:
    @modal.enter()
    def load(self):
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def _png_bytes(
        self,
        prompt: str,
        images: List[Any],
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> bytes:
        # height/width left as None makes the pipeline follow the input image,
        # which is what an edit should do unless the node asks otherwise.
        kwargs: dict[str, Any] = {
            "image": images,
            "prompt": prompt,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "true_cfg_scale": DEFAULT_TRUE_CFG_SCALE,
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
            "num_images_per_prompt": 1,
        }
        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height
        if seed is not None:
            kwargs["generator"] = torch.Generator(device="cuda").manual_seed(int(seed))

        with torch.inference_mode():
            result = self.pipe(**kwargs)

        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _to_pil(raw: bytes) -> Any:
        from PIL import Image

        return Image.open(io.BytesIO(raw)).convert("RGB")

    @modal.method()
    @node_slot(NodeSlots.IMAGE_EDIT)
    def image_edit(self, input: ImageEditInput) -> ImageEditOutput:
        if input.image is None:
            return ImageEditOutput(success=False, error="Missing image")
        text = (input.text or "").strip()
        if not text:
            return ImageEditOutput(success=False, error="Missing edit instruction")

        # match_input_size is the product's way of saying "don't resize me";
        # honouring it means passing no explicit size at all.
        keep_size = input.match_input_size if input.match_input_size is not None else True
        raw = self._png_bytes(
            text,
            [self._to_pil(prompt_media_to_bytes(input.image))],
            seed=input.seed,
            width=None if keep_size else input.width,
            height=None if keep_size else input.height,
        )
        return ImageEditOutput(success=True, image=asset(raw, mime="image/png"))

    @modal.method()
    @node_slot(NodeSlots.IMAGE_FUSION)
    def image_fusion(self, input: ImageFusionInput) -> ImageFusionOutput:
        imgs = input.images or []
        if len(imgs) < 2:
            return ImageFusionOutput(success=False, error="Need at least 2 images")
        text = (input.text or "").strip()
        if not text:
            return ImageFusionOutput(success=False, error="Missing fusion instruction")

        raw = self._png_bytes(
            text,
            [self._to_pil(prompt_media_to_bytes(x)) for x in imgs],
            seed=input.seed,
            width=input.width,
            height=input.height,
        )
        return ImageFusionOutput(success=True, image=asset(raw, mime="image/png"))

    # Cloud single-node self-serve: ONE container, direct browser stream. The
    # browser's EventSource is 302'd here with taskId/token/origin; serve_stream
    # _from_spec (SDK) fetches the run spec from the Worker, runs the slot
    # in-container, and streams progress + result. Streaming dodges the 150s
    # cap. Label is uniform (`<app>-serve`) so the Worker derives the URL.
    @modal.fastapi_endpoint(method="GET", label=f"{APP_NAME}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin,
                taskId,
                token,
                __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*",
            },
        )
