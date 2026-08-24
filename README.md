# tongflow-modal-qwen-image-edit

Qwen-Image-Edit as a TongFlow plugin, served from your own [Modal](https://modal.com) account.

## Node slots

| Slot | Node | What it does |
| --- | --- | --- |
| `image-edit` | Image → Image | Edit one image from a written instruction. |
| `image-fusion` | Images → Image | Combine two or more images into one scene. |

Both run on the same `QwenImageEditPlusPipeline`, whose `image` argument takes a
list — the two slots differ only in how many images the node hands it.

## Weights

[`lite-infer/qwen-image-edit-2509-lightning-4steps-nunchaku-lite-int4_r32-bnb4-text-encoder`](https://huggingface.co/lite-infer/qwen-image-edit-2509-lightning-4steps-nunchaku-lite-int4_r32-bnb4-text-encoder)
— **18 GB**, pulled once into the shared `models` Modal Volume:

```bash
modal run download.py::download
```

Ungated, so no Hugging Face token is involved.

It is a Diffusers-native repack of `Qwen/Qwen-Image-Edit-2509` with three things
already done to it:

| | |
| --- | --- |
| Transformer | SVDQuant **int4** (rank 32), for the `nunchaku_lite` loader — 11.6 GB, from 40.9 GB |
| Text encoder | BitsAndBytes **4-bit NF4** — 6.2 GB, from 16.6 GB |
| Steps | Lightning **4-step** LoRA fused in — from 40 |

**2509 rather than 2511** only because no 2511 checkpoint is packaged for this
loader: every nunchaku 2511 repo on the Hub is a ComfyUI single file, which
Diffusers cannot read (it has no handling for the `weight_scale` companion
tensors those carry).

## Deploy

```bash
modal deploy deploy.py
```

Runs on an **L40S**, and not by preference — the Diffusers Nunchaku quantizer
refuses Hopper outright, so an H100 is not available to this checkpoint. It
wants Turing or newer for int4, which Ada satisfies. The upstream benchmark
peaks at 21 GiB, so 48 GB is roomy.

## Defaults

Four steps with `true_cfg_scale` 1.0: the distillation is fused into the
weights, and a distilled model has classifier-free guidance baked out already.
Plugin-internal constants, not ABI fields.

An edit keeps the input image's size unless the node turns off "match input
size"; fusion uses the node's width/height when set, and the pipeline's own
derivation otherwise.
