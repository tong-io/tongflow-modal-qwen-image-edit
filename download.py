"""Modal download entry for qwen-image-edit (ComfyUI layout).

Run:
  modal run download.py::download

Self-contained: Modal remote execution may mount only this file, so do not
import other local modules (e.g. `impl.py`, `config.py`).
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


COMFY_MODELS = "/models/comfyui"

# (repo, file in repo, subdirectory ComfyUI looks in). Comfy-Org's repacks are
# what the official 2511 template names; the LoRA is LightX2V's distillation.
FILES = [
    (
        "Comfy-Org/Qwen-Image-Edit_ComfyUI",
        "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
        "diffusion_models",
    ),
    (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "text_encoders",
    ),
    (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "split_files/vae/qwen_image_vae.safetensors",
        "vae",
    ),
    (
        "lightx2v/Qwen-Image-Edit-2511-Lightning",
        "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
        "loras",
    ),
]

volume = modal.Volume.from_name("models", create_if_missing=True)

model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "huggingface_hub==1.6.0",
    ),
    volumes={"/models": volume},
    # ~31GB over four files; the default hour is not enough on a cold volume.
    timeout=7200,
)
def _download() -> None:
    from huggingface_hub import hf_hub_download

    for repo, path, sub in FILES:
        dest_dir = os.path.join(COMFY_MODELS, sub)
        dest = os.path.join(dest_dir, os.path.basename(path))
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"Already present, skipping: {sub}/{os.path.basename(path)}")
            continue
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Downloading {repo}/{path} -> {dest}")
        # ComfyUI reads a flat directory per kind, so the repo's own nesting is
        # dropped: fetch into a scratch tree, then move the file into place.
        got = hf_hub_download(repo_id=repo, filename=path, local_dir="/tmp/hf")
        os.replace(got, dest)
        print(f"Done: {dest}")

    volume.commit()


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
