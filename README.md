# tongflow-modal-qwen-image-edit

[Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) as a TongFlow plugin, served from your own [Modal](https://modal.com) account.

## Node slots

| Slot | Node | What it does |
| --- | --- | --- |
| `image-edit` | Image → Image | Edit one image from a written instruction. |
| `image-fusion` | Images → Image | Combine two or more images into one scene. |

Both run on the same `QwenImageEditPlusPipeline`, whose `image` argument takes a
list — the two slots differ only in how many images the node hands it.

## Weights

`Qwen/Qwen-Image-Edit-2511` — **57.7 GB** (a 40.9 GB transformer plus a 16.6 GB
Qwen2.5-VL text encoder), pulled once into the shared `models` Modal Volume:

```bash
modal run download.py::download
```

The repo is **not gated**, so no Hugging Face token is involved.

## Deploy

```bash
modal deploy deploy.py
```

Runs on an **H100**: the weights alone overflow a 48 GB card, and offloading
them would swap most of that over PCIe on every call.

## Defaults

Sampling follows the model card — 40 steps, `true_cfg_scale` 4.0,
`guidance_scale` 1.0. They are plugin-internal constants, not ABI fields.

An edit keeps the input image's size unless the node turns off "match input
size"; fusion uses the node's width/height when set, and the pipeline's own
derivation otherwise.
