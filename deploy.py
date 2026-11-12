"""Modal deploy entry for Qwen-Image 2.1 (headless ComfyUI).

Deploy:
  modal deploy deploy.py

Qwen-Image-2.1 is one model for four nodes: text-to-image, instruction edit,
multi-image fusion (references are the same input path as an edit, just more of
them) and background removal, which the model answers with a genuinely
transparent PNG because the 2.1 VAE is RGBA.

ComfyUI rather than Diffusers on two counts: the quantised weights are
ComfyUI's own int8 `convrot` format, which Diffusers has no loader for, and
`QwenImage21Pipeline` is still unreleased on Diffusers' main branch, so the
Diffusers route would pin a moving commit.

Design constraints:
  - Keep this file mostly self-contained because Modal remote imports may mount
    only the entry file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

import modal
from tongflow import deploy
from tongflow.models.image_edit import ImageEditInput, ImageEditOutput
from tongflow.models.image_fusion import ImageFusionInput, ImageFusionOutput
from tongflow.models.image_gen import ImageGenInput, ImageGenOutput
from tongflow.models.image_matting import ImageMattingInput, ImageMattingOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.slots import current_params, node_slot


def _adv(name: str, default):
    """Advanced-section override (``TONGFLOW_SLOT_PARAMS``) or the plugin default."""
    v = current_params().get(name)
    if v is None:
        return default
    if isinstance(default, bool):
        return bool(v)
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


# Per-run knobs offered under the node's collapsed "Advanced" section.
# Pure literal (the platform scanner reads it by AST, never imports this
# module). Values reach the handlers via current_params(); an untouched
# control is absent there and falls back to the plugin default.
#
# `resolution` is the reference-image budget, not the output size: references
# are resized to about resolution x resolution pixels, aspect preserved, and 0
# means "leave each reference at its own size". An edit wants 0, so the result
# comes back framed exactly like the input; a fusion normalises instead,
# because ten full-size references are ten times the sequence to attend over.
TONGFLOW_SLOT_PARAMS = {
    "image-gen": {
        "steps": {"type": "integer", "default": 25, "min": 1, "max": 60, "label": "Steps"},
        "cfg": {"type": "number", "default": 1.0, "min": 1.0, "max": 10.0, "step": 0.5, "label": "CFG scale"},
        "transparent": {"type": "boolean", "default": False, "label": "Transparent background (RGBA)"},
    },
    "image-edit": {
        "steps": {"type": "integer", "default": 25, "min": 1, "max": 60, "label": "Steps"},
        "cfg": {"type": "number", "default": 1.0, "min": 1.0, "max": 10.0, "step": 0.5, "label": "CFG scale"},
        "resolution": {"type": "integer", "default": 0, "min": 0, "max": 2048, "step": 32, "label": "Reference resolution (0 = keep)"},
    },
    "image-fusion": {
        "steps": {"type": "integer", "default": 25, "min": 1, "max": 60, "label": "Steps"},
        "cfg": {"type": "number", "default": 1.0, "min": 1.0, "max": 10.0, "step": 0.5, "label": "CFG scale"},
        "resolution": {"type": "integer", "default": 1024, "min": 0, "max": 2048, "step": 32, "label": "Reference resolution (0 = keep)"},
    },
    "image-matting": {
        "steps": {"type": "integer", "default": 25, "min": 1, "max": 60, "label": "Steps"},
    },
}


COMFY = "/opt/ComfyUI"
# TextEncodeQwenImage21 / QwenImage21Cache land in v0.37.0; v0.36.0 has neither.
COMFY_TAG = "v0.37.0"
COMFY_MODELS = "/models/comfyui"
COMFY_LOG = "/tmp/comfy.log"

# Comfy-Org's own repacks, the files the official 2.1 templates name. The int8
# `convrot` variants are the templates' default and cut the resident weights
# from 32GB to 17GB at the quality the rotation-aware quantisation was tuned
# for; the bf16 pair is there if a comparison is ever wanted.
UNET = "qwen_image_2.1_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_8b_int8_convrot.safetensors"
VAE = "qwen_image_2.1_vae_bf16.safetensors"

# 2.1 has guidance distilled in, so cfg stays at 1.0 and there is no Lightning
# LoRA to fuse — 25 steps in a single pass is the official template's setting,
# and it is already cheaper than 2511's eight steps with CFG.
STEPS = int(os.environ.get("QIE_STEPS") or 25)
CFG = float(os.environ.get("QIE_CFG") or 1.0)
# Reference budget when a slot does not override it (see TONGFLOW_SLOT_PARAMS).
RESOLUTION = int(os.environ.get("QIE_RESOLUTION") or 1024)
# Output size for a text-to-image run that names none. One megapixel, as the
# official template ships; the model goes to 2048x2048 natively if asked.
DEFAULT_SIZE = int(os.environ.get("QIE_DEFAULT_SIZE") or 1024)

# Prefix KV cache: `auto` spends spare VRAM first and falls back to RAM, which
# is what makes a re-edit of the same references cheap.
CACHE_DEVICE = (os.environ.get("QIE_CACHE_DEVICE") or "auto").strip()
CACHE_DTYPE = (os.environ.get("QIE_CACHE_DTYPE") or "default").strip()

# 17GB of weights plus 2K activations over as many as ten references. An L40S
# (48GB) is a plausible fit now that the int8 pair replaced 2511's 30GB of fp8
# — QIE_GPU is there to try it — but 80GB is what this is known to run on, and
# the failure mode of guessing low is an OOM inside KSampler after the whole
# graph has already been paid for.
GPU = (os.environ.get("QIE_GPU") or "H100").strip()
# Spare flags for the ComfyUI server, e.g. `--lowvram` to trade speed for a
# smaller card. Left empty: ComfyUI's own memory management is the default.
COMFY_ARGS = [a for a in (os.environ.get("QIE_COMFY_ARGS") or "").split() if a]

# The model card promises up to ten references; the node exposes sixteen
# slots. Ten is the number we are willing to stand behind.
MAX_IMAGES = 10

# The official background-removal template's instruction, verbatim. The PNG it
# asks for is the point: the 2.1 VAE decodes four channels, so the alpha the
# model paints survives all the way to the file.
MATTING_PROMPT = "Remove the background, and output a PNG image"

# Transparent text-to-image is a prompt convention rather than a flag; this is
# the phrasing the model card documents.
RGBA_PREFIX = "This is an RGBA image with transparency. "
RGBA_SUFFIX = " The image has alpha channel and the background is transparent."

volume = modal.Volume.from_name("models", create_if_missing=True)

APP_NAME = Path(__file__).resolve().parent.name
app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        f"git clone --depth 1 --branch {COMFY_TAG} "
        f"https://github.com/comfyanonymous/ComfyUI.git {COMFY}",
        f"pip install -r {COMFY}/requirements.txt",
    )
    .pip_install("tongflow==1.0.0", "fastapi[standard]")
    .env({
        "PYTHONPATH": COMFY,
        "HF_HOME": "/models/hf",
        # A 2511 run once peaked at 24.9GiB of active memory while holding
        # 44.9GiB reserved — twenty of it lost to fragmentation, which is the
        # exact pattern expandable segments exist for.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

with image.imports():
    import json
    import random
    import subprocess
    import time
    import urllib.error
    import urllib.request


def _tail_log(n: int = 3000) -> str:
    """Tail of the ComfyUI server stdout — the per-node execution trace, which
    is where a failure actually explains itself."""
    try:
        with open(COMFY_LOG, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return "(no server log)"


def _graph(prompt: str, images: List[str], seed: int, steps: int, cfg: float,
           resolution: int, width: Optional[int] = None,
           height: Optional[int] = None) -> dict:
    """API-format graph, wired as the official 2.1 templates wire it.

    One graph covers all four slots: no references and an explicit size is
    text-to-image, references with no size is an edit that keeps the input's
    framing, and everything else is the same run with the latent swapped.

    TextEncodeQwenImage21 does the work the 2511 graph needed extra nodes for.
    It resizes each reference itself, encodes them through the VAE as
    `reference_latents`, and hands back an empty latent shaped to the first
    reference — which is why an edit that should not reframe simply uses that
    third output instead of an EmptyLatentImage.
    """
    g: dict[str, Any] = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "QwenImage21Cache",
              "inputs": {"model": ["1", 0], "device": CACHE_DEVICE,
                         "dtype": CACHE_DTYPE}},
        "5": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TEXT_ENCODER, "type": "qwen_image",
                         "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    }

    encode: dict[str, Any] = {
        "clip": ["5", 0],
        "vae": ["6", 0],
        "prompt": prompt,
        "negative_prompt": "",
        "resolution": int(resolution),
    }
    # Autogrow inputs are addressed by their finalized dotted ids — the same
    # `images.image_N` names the node's schema expands to.
    for i, name in enumerate(images):
        node_id = f"10{i}"
        g[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        encode[f"images.image_{i + 1}"] = [node_id, 0]
    g["7"] = {"class_type": "TextEncodeQwenImage21", "inputs": encode}

    if width and height:
        g["9"] = {"class_type": "EmptyLatentImage",
                  "inputs": {"width": int(width), "height": int(height),
                             "batch_size": 1}}
        latent: list = ["9", 0]
    else:
        latent = ["7", 2]

    g["12"] = {"class_type": "KSampler",
               "inputs": {"model": ["2", 0], "positive": ["7", 0],
                          "negative": ["7", 1], "latent_image": latent,
                          "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": "euler", "scheduler": "simple",
                          "denoise": 1.0}}
    g["13"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["12", 0], "vae": ["6", 0]}}
    # SaveImage writes whatever channels it is handed, so a four-channel decode
    # lands as an RGBA PNG without a separate alpha path.
    g["14"] = {"class_type": "SaveImage",
               "inputs": {"images": ["13", 0], "filename_prefix": "qi21"}}
    return g


def _submit(base: str, wf: dict) -> tuple[bool, Any]:
    """Queue a graph, poll to completion, return (True, png) or (False, error)."""
    t0 = time.monotonic()
    body = json.dumps({"prompt": wf}).encode()
    req = urllib.request.Request(
        f"{base}/prompt", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        return False, f"workflow rejected: {e.read().decode()[:1500]}"

    out, status = None, {}
    for _ in range(1800):
        time.sleep(1)
        with urllib.request.urlopen(f"{base}/history/{pid}", timeout=10) as r:
            hist = json.loads(r.read())
        if pid not in hist:
            continue
        h = hist[pid]
        status = h.get("status", {})
        if status.get("status_str") == "error":
            return False, ("comfy error: "
                           + json.dumps(status.get("messages", status))[:1500]
                           + "\n[server log]\n" + _tail_log())
        if h.get("outputs") and status.get("completed"):
            out = h["outputs"]
            break
    if not out:
        return False, "timed out\n[server log]\n" + _tail_log()

    print(f"[qi21] graph done in {time.monotonic() - t0:.0f}s", flush=True)
    for node_out in out.values():
        for item in node_out.get("images", []):
            fn, sub = item.get("filename"), item.get("subfolder", "")
            d = {"output": "output", "temp": "temp"}.get(item.get("type"), "output")
            path = os.path.join(COMFY, d, sub, fn or "")
            if fn and os.path.isfile(path):
                with open(path, "rb") as fh:
                    raw = fh.read()
                if raw:
                    return True, raw
    return False, ("no image output; outputs=" + json.dumps(out)[:600]
                   + "\n[server log]\n" + _tail_log())


@deploy
@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/models": volume},
    timeout=1800,
    scaledown_window=2,
)
class Inference:
    @modal.enter()
    def _boot(self) -> None:
        """Boot the ComfyUI server once; reused across calls (models stay warm)."""
        t0 = time.monotonic()
        os.makedirs(COMFY_MODELS, exist_ok=True)
        with open(os.path.join(COMFY, "extra_model_paths.yaml"), "w") as f:
            f.write(
                "qie_volume:\n"
                f"  base_path: {COMFY_MODELS}/\n"
                "  diffusion_models: diffusion_models\n"
                "  text_encoders: text_encoders\n"
                "  vae: vae\n"
            )
        for sub, name in (("diffusion_models", UNET), ("text_encoders", TEXT_ENCODER),
                          ("vae", VAE)):
            if not os.path.isfile(os.path.join(COMFY_MODELS, sub, name)):
                raise RuntimeError(
                    f"{sub}/{name} missing from the models volume — run "
                    "`modal run download.py::download` first"
                )

        self._logfh = open(COMFY_LOG, "wb")
        self.proc = subprocess.Popen(
            ["python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
             "--disable-auto-launch", *COMFY_ARGS],
            cwd=COMFY, stdout=self._logfh, stderr=subprocess.STDOUT,
        )
        self.base = "http://127.0.0.1:8188"
        info = None
        for _ in range(600):
            if self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited early: {self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"{self.base}/object_info", timeout=2) as r:
                    if r.status == 200:
                        info = json.loads(r.read())
                        break
            except Exception:
                time.sleep(1)
        if info is None:
            raise RuntimeError("ComfyUI server did not become ready")
        for cls in ("TextEncodeQwenImage21", "QwenImage21Cache"):
            if cls not in info:
                raise RuntimeError(f"{cls} missing from ComfyUI {COMFY_TAG} — bump COMFY_TAG")
        print(f"[qi21] comfy {COMFY_TAG} ready in {time.monotonic() - t0:.0f}s "
              f"(gpu={GPU} steps={STEPS} cfg={CFG:g} cache={CACHE_DEVICE}/{CACHE_DTYPE}"
              + (f" args={' '.join(COMFY_ARGS)}" if COMFY_ARGS else "")
              + ") — weights load lazily", flush=True)

    @modal.exit()
    def _shutdown(self) -> None:
        try:
            self.proc.terminate()
        except Exception:
            pass

    def _stage(self, blobs: List[bytes]) -> List[str]:
        os.makedirs(f"{COMFY}/input", exist_ok=True)
        names = []
        for i, raw in enumerate(blobs):
            name = f"qie_{os.getpid()}_{time.time_ns()}_{i}.png"
            with open(f"{COMFY}/input/{name}", "wb") as f:
                f.write(raw)
            names.append(name)
        return names

    def _run(self, text: str, blobs: List[bytes], seed: Optional[int],
             resolution: int, width: Optional[int] = None,
             height: Optional[int] = None) -> tuple[bool, Any]:
        s = int(seed) if seed is not None else random.randrange(2**31)
        return _submit(
            self.base,
            _graph(text, self._stage(blobs), s, _adv("steps", STEPS),
                   _adv("cfg", CFG), resolution, width, height),
        )

    @modal.method()
    @node_slot(NodeSlots.IMAGE_GEN)
    def image_gen(self, input: ImageGenInput) -> ImageGenOutput:
        text = (input.text or "").strip()
        if not text:
            return ImageGenOutput(success=False, error="Missing prompt")
        if _adv("transparent", False):
            text = RGBA_PREFIX + text + RGBA_SUFFIX
        ok, res = self._run(
            text, [], input.seed, RESOLUTION,
            input.width or DEFAULT_SIZE,
            input.height or DEFAULT_SIZE,
        )
        if not ok:
            return ImageGenOutput(success=False, error=str(res))
        return ImageGenOutput(success=True, image=asset(res, mime="image/png"))

    @modal.method()
    @node_slot(NodeSlots.IMAGE_EDIT)
    def image_edit(self, input: ImageEditInput) -> ImageEditOutput:
        if input.image is None:
            return ImageEditOutput(success=False, error="Missing image")
        text = (input.text or "").strip()
        if not text:
            return ImageEditOutput(success=False, error="Missing edit instruction")
        # match_input_size is the product's way of saying "don't resize me",
        # and it is the default: an edit that silently reframes is a bad edit.
        keep = input.match_input_size if input.match_input_size is not None else True
        ok, res = self._run(
            text,
            [prompt_media_to_bytes(input.image)],
            input.seed,
            _adv("resolution", 0),
            None if keep else input.width,
            None if keep else input.height,
        )
        if not ok:
            return ImageEditOutput(success=False, error=str(res))
        return ImageEditOutput(success=True, image=asset(res, mime="image/png"))

    @modal.method()
    @node_slot(NodeSlots.IMAGE_FUSION)
    def image_fusion(self, input: ImageFusionInput) -> ImageFusionOutput:
        imgs = input.images or []
        if len(imgs) < 2:
            return ImageFusionOutput(success=False, error="Need at least 2 images")
        # Refusing beats quietly dropping references the user wired up.
        if len(imgs) > MAX_IMAGES:
            return ImageFusionOutput(
                success=False,
                error=f"This model takes at most {MAX_IMAGES} images; got {len(imgs)}",
            )
        text = (input.text or "").strip()
        if not text:
            return ImageFusionOutput(success=False, error="Missing fusion instruction")
        ok, res = self._run(
            text,
            [prompt_media_to_bytes(x) for x in imgs],
            input.seed,
            _adv("resolution", RESOLUTION),
            input.width,
            input.height,
        )
        if not ok:
            return ImageFusionOutput(success=False, error=str(res))
        return ImageFusionOutput(success=True, image=asset(res, mime="image/png"))

    @modal.method()
    @node_slot(NodeSlots.IMAGE_MATTING)
    def image_matting(self, input: ImageMattingInput) -> ImageMattingOutput:
        if input.image is None:
            return ImageMattingOutput(success=False, error="Missing image")
        # Reference kept at its own size and the latent taken from it, so the
        # cut-out lines up pixel for pixel with what went in.
        ok, res = self._run(
            MATTING_PROMPT, [prompt_media_to_bytes(input.image)], None, 0
        )
        if not ok:
            return ImageMattingOutput(success=False, error=str(res))
        return ImageMattingOutput(success=True, image=asset(res, mime="image/png"))

    @modal.fastapi_endpoint(method="GET", label=f"{APP_NAME}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache",
                     "Access-Control-Allow-Origin": "*"},
        )
