"""Modal deploy entry for Qwen-Image-Edit (headless ComfyUI).

Deploy:
  modal deploy deploy.py

ComfyUI rather than Diffusers because the quantised weights are ComfyUI's own
format: fp8 tensors paired with `weight_scale` companions, which Diffusers has
no loader for. Reading them through ComfyUI keeps fp8 resident instead of
dequantising to bf16, which is what turns a 57.7GB / 80GB-card model into a
31GB / L40S one.

The graph mirrors the official `image_qwen_image_edit_2511` template, minus its
FluxKontextMultiReferenceLatentMethod pair — the template's own note says those
are unnecessary with Comfy-Org files, which is what this plugin downloads.

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
TONGFLOW_SLOT_PARAMS = {
    "image-edit": {
        "steps": {"type": "integer", "default": 8, "min": 1, "max": 50, "label": "Steps"},
        "cfg": {"type": "number", "default": 1.0, "min": 1.0, "max": 10.0, "step": 0.5, "label": "CFG scale"},
        "shift": {"type": "number", "default": 3.1, "min": 1.0, "max": 10.0, "step": 0.1, "label": "Shift"},
        "lora_strength": {"type": "number", "default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05, "label": "Lightning LoRA strength"},
    },
    "image-fusion": {
        "steps": {"type": "integer", "default": 8, "min": 1, "max": 50, "label": "Steps"},
        "cfg": {"type": "number", "default": 1.0, "min": 1.0, "max": 10.0, "step": 0.5, "label": "CFG scale"},
        "shift": {"type": "number", "default": 3.1, "min": 1.0, "max": 10.0, "step": 0.1, "label": "Shift"},
        "lora_strength": {"type": "number", "default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05, "label": "Lightning LoRA strength"},
    },
}


COMFY = "/opt/ComfyUI"
COMFY_TAG = "v0.33.4"
COMFY_MODELS = "/models/comfyui"
COMFY_LOG = "/tmp/comfy.log"

# Comfy-Org's own repacks, the files the official 2511 template names.
UNET = "qwen_image_edit_2511_fp8mixed.safetensors"
TEXT_ENCODER = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
# LightX2V distillation, fused at sampling time by LoraLoaderModelOnly. Eight
# steps rather than four: the same LoRA family, one notch back from the fastest
# setting, which is where the quality argument is easiest to win.
LORA = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
STEPS = int(os.environ.get("QIE_STEPS") or 8)
# A distilled model has classifier-free guidance baked out; the template's
# lightning branch drives cfg to 1.0 and the base branch to 4.0.
CFG = float(os.environ.get("QIE_CFG") or 1.0)
SHIFT = float(os.environ.get("QIE_SHIFT") or 3.1)
LORA_STRENGTH = float(os.environ.get("QIE_LORA_STRENGTH") or 1.0)

# An L40S ran the whole graph and then died allocating inside KSampler. The
# weights are not the problem — ComfyUI honours this checkpoint's own
# `comfy_quant` markers and keeps the transformer at its stored 20.5GB — but
# 48GB does not also cover a 9.4GB text encoder and this model's activations
# over two references. 80GB does. Nothing forces the choice: the Hopper ban
# that shaped the earlier plan belonged to Diffusers' Nunchaku loader, which
# this no longer goes through.
GPU = (os.environ.get("QIE_GPU") or "H100").strip()
# Spare flags for the ComfyUI server, e.g. `--lowvram` to trade speed for a
# smaller card. Left empty: ComfyUI's own memory management is the default.
COMFY_ARGS = [a for a in (os.environ.get("QIE_COMFY_ARGS") or "").split() if a]

# TextEncodeQwenImageEditPlus exposes image1/image2/image3 and nothing beyond.
MAX_IMAGES = 3

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
    .pip_install("tongflow==0.3.3", "fastapi[standard]")
    .env({
        "PYTHONPATH": COMFY,
        "HF_HOME": "/models/hf",
        # The failing run peaked at 24.9GiB of active memory while holding
        # 44.9GiB reserved — twenty of it lost to fragmentation, which is the
        # exact pattern expandable segments exist for.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

with image.imports():
    import io
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


def _graph(prompt: str, images: List[str], seed: int,
           width: Optional[int] = None, height: Optional[int] = None) -> dict:
    """API-format graph, wired as the official 2511 template wires it.

    `width`/`height` decide the output size, and they decide it through the
    latent's shape: with denoise at 1.0 the sampler replaces the latent's
    contents entirely, so the template's VAEEncode is really just a way of
    saying "same size as the input". Asking for a size swaps in the empty
    latent the official text-to-image template uses.
    """
    g: dict[str, Any] = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": ["1", 0], "shift": _adv("shift", SHIFT)}},
        "3": {"class_type": "CFGNorm",
              "inputs": {"model": ["2", 0], "strength": 1.0}},
        "4": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["3", 0], "lora_name": LORA,
                         "strength_model": _adv("lora_strength", LORA_STRENGTH)}},
        "5": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TEXT_ENCODER, "type": "qwen_image",
                         "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    }

    # One LoadImage + FluxKontextImageScale per reference; the first doubles as
    # the latent the sampler denoises, which is what makes this an edit rather
    # than a generation.
    scaled: list[list] = []
    for i, name in enumerate(images):
        load, scale = f"10{i}", f"11{i}"
        g[load] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g[scale] = {"class_type": "FluxKontextImageScale",
                    "inputs": {"image": [load, 0]}}
        scaled.append([scale, 0])

    def encode(text: str) -> dict:
        inputs: dict[str, Any] = {"clip": ["5", 0], "prompt": text, "vae": ["6", 0]}
        for i, ref in enumerate(scaled):
            inputs[f"image{i + 1}"] = ref
        return {"class_type": "TextEncodeQwenImageEditPlus", "inputs": inputs}

    g["7"] = encode(prompt)
    g["8"] = encode("")
    if width and height:
        g["9"] = {"class_type": "EmptySD3LatentImage",
                  "inputs": {"width": int(width), "height": int(height),
                             "batch_size": 1}}
    else:
        g["9"] = {"class_type": "VAEEncode",
                  "inputs": {"pixels": scaled[0], "vae": ["6", 0]}}
    g["12"] = {"class_type": "KSampler",
               "inputs": {"model": ["4", 0], "positive": ["7", 0],
                          "negative": ["8", 0], "latent_image": ["9", 0],
                          "seed": seed, "steps": _adv("steps", STEPS), "cfg": _adv("cfg", CFG),
                          "sampler_name": "euler", "scheduler": "simple",
                          "denoise": 1.0}}
    g["13"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["12", 0], "vae": ["6", 0]}}
    g["14"] = {"class_type": "SaveImage",
               "inputs": {"images": ["13", 0], "filename_prefix": "qie"}}
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

    print(f"[qie] graph done in {time.monotonic() - t0:.0f}s", flush=True)
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
                "  loras: loras\n"
            )
        for sub, name in (("diffusion_models", UNET), ("text_encoders", TEXT_ENCODER),
                          ("vae", VAE), ("loras", LORA)):
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
        for cls in ("TextEncodeQwenImageEditPlus", "FluxKontextImageScale", "CFGNorm"):
            if cls not in info:
                raise RuntimeError(f"{cls} missing from ComfyUI {COMFY_TAG} — bump COMFY_TAG")
        print(f"[qie] comfy {COMFY_TAG} ready in {time.monotonic() - t0:.0f}s "
              f"(gpu={GPU} steps={STEPS} cfg={CFG:g} shift={SHIFT:g}"
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
             width: Optional[int] = None,
             height: Optional[int] = None) -> tuple[bool, Any]:
        s = int(seed) if seed is not None else random.randrange(2**31)
        return _submit(
            self.base, _graph(text, self._stage(blobs), s, width, height)
        )

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
            input.width,
            input.height,
        )
        if not ok:
            return ImageFusionOutput(success=False, error=str(res))
        return ImageFusionOutput(success=True, image=asset(res, mime="image/png"))

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
