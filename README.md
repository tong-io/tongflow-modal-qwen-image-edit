# tongflow-modal-qwen-image-edit

[Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) as a TongFlow plugin, run through a headless [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on your own [Modal](https://modal.com) account.

## Node slots

| Slot | Node | What it does |
| --- | --- | --- |
| `image-edit` | Image → Image | Edit one image from a written instruction. |
| `image-fusion` | Images → Image | Combine **two or three** images into one scene. |

Three is a hard ceiling: `TextEncodeQwenImageEditPlus` exposes `image1`,
`image2`, `image3` and nothing further. A fourth reference is refused rather
than quietly dropped.

## Why ComfyUI and not Diffusers

The quantised weights are ComfyUI's own format — fp8 tensors paired with
`weight_scale` companions — and Diffusers has no loader for them. Reading them
through ComfyUI keeps fp8 resident instead of dequantising to bf16, which is
the whole difference between this and a 57.7 GB model on an 80 GB card:

| | Diffusers, unquantised | Here |
| --- | ---: | ---: |
| Download | 57.7 GB | **31 GB** |
| GPU | H100 | **L40S** |
| Steps | 40 | **8** |

## Weights

Comfy-Org's own repacks — the files the official 2511 template names — plus
LightX2V's distillation LoRA:

| File | Size | Into |
| --- | ---: | --- |
| `qwen_image_edit_2511_fp8mixed.safetensors` | 20.53 GB | `diffusion_models/` |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | 9.38 GB | `text_encoders/` |
| `qwen_image_vae.safetensors` | 0.25 GB | `vae/` |
| `Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors` | 0.85 GB | `loras/` |

```bash
modal run download.py::download
```

All ungated, so no Hugging Face token is involved.

## Deploy

```bash
modal deploy deploy.py
```

ComfyUI is pinned to `v0.33.4` and the server boots once per container, so
weights stay warm across calls.

## The graph

Mirrors the official `image_qwen_image_edit_2511` template:

```
UNETLoader → ModelSamplingAuraFlow(3.1) → CFGNorm → LoraLoaderModelOnly → KSampler
LoadImage → FluxKontextImageScale → TextEncodeQwenImageEditPlus.image1..3
                                  → VAEEncode → KSampler.latent_image
KSampler(euler, simple, 8 steps, cfg 1.0) → VAEDecode → SaveImage
```

Two things differ from the template. Its
`FluxKontextMultiReferenceLatentMethod` pair is dropped — the template's own
note says they are unnecessary with Comfy-Org files, which is what this
downloads. And the lightning branch is taken unconditionally rather than
through a switch: 8 steps at `cfg` 1.0, because a distilled model has
classifier-free guidance baked out already.

`QIE_STEPS`, `QIE_CFG`, `QIE_SHIFT` and `QIE_LORA_STRENGTH` override the
sampling constants without a code change.
