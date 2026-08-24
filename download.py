"""Modal download entry for qwen-image-edit.

Run:
  modal run download.py::download

Self-contained: Modal remote execution may mount only this file, so do not
import other local modules (e.g. `impl.py`, `config.py`).
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import modal



_cfg: dict[str, Any] = {}
_hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
REPO_ID = str(_hf.get("repoId") or "Qwen/Qwen-Image-Edit-2511")
REVISION = str(_hf.get("revision") or "")
MODEL_DIR = f"/models/{REPO_ID}"

# Must exist at module import time for Modal.
volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)

model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub==1.6.0",
    ),
    volumes={"/models": volume},
    # 57.7GB across ten shards; the default hour is not enough on a cold volume.
    timeout=7200,
)
def _download() -> None:
    # The repo is ungated, so no token is involved.
    from huggingface_hub import snapshot_download

    # The transformer shards are the bulk; treat their presence as the marker
    # so a half-finished pull resumes instead of being skipped.
    marker = os.path.join(MODEL_DIR, "model_index.json")
    if os.path.exists(marker) and os.path.isdir(os.path.join(MODEL_DIR, "transformer")):
        print(f"Model already exists at {MODEL_DIR}, skipping")
        return

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODEL_DIR,
        revision=REVISION or None,
    )
    volume.commit()
    print(f"Model downloaded to {MODEL_DIR}")


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
