# tongflow-modal-qwen-image-edit

[Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) as a TongFlow plugin, run through a headless [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on your own [Modal](https://modal.com) account.

One model serves four nodes. The repository keeps its `qwen-image-edit` name from the 2511 release it used to run, so that installs and Modal deployments carry over.

> **Licence.** Qwen-Image-2.1 is released under the [Qwen Research Licence](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), which is **not** a commercial licence. Check it before putting output in front of customers.

## Node slots

| Slot | Node | What it does |
| --- | --- | --- |
| `image-gen` | Text → Image | Generate from a prompt, up to 2048×2048 natively. |
| `image-edit` | Image → Image | Edit one image from a written instruction, output framed like the input. |
| `image-fusion` | Images → Image | Combine **up to ten** references into one scene. |
| `image-matting` | Image → Image | Cut the subject out as a transparent PNG. |

Ten references is the model card's own promise; the node exposes sixteen slots.
An eleventh is refused rather than quietly dropped.

The alpha channel is real, not a post-process: the 2.1 VAE encodes and decodes
four channels, so `SaveImage` writes an RGBA PNG straight out of the sampler.
Text-to-image can do the same — turn on **Transparent background** under
Advanced, which wraps the prompt in the RGBA phrasing the model card documents.

## Why ComfyUI and not Diffusers

Two reasons. The quantised weights are ComfyUI's own int8 `convrot` format,
which Diffusers has no loader for; and `QwenImage21Pipeline` has not landed in
a Diffusers release, so that route would pin a moving `main`.

## Weights

Comfy-Org's repacks — the files the official 2.1 templates name:

| File | Size | Into |
| --- | ---: | --- |
| `qwen_image_2.1_int8_convrot.safetensors` | 7.26 GB | `diffusion_models/` |
| `qwen3vl_8b_int8_convrot.safetensors` | 9.35 GB | `text_encoders/` |
| `qwen_image_2.1_vae_bf16.safetensors` | 0.68 GB | `vae/` |

```bash
modal run download.py::download
```

All ungated, so no Hugging Face token is involved.

Upgrading from the 2511 build of this plugin leaves 31 GB of superseded files
on the shared `models` volume. Once every deployment reading them is gone:

```bash
modal run download.py::prune
```

## Deploy

```bash
modal deploy deploy.py
```

ComfyUI is pinned to `v0.37.0` — the first release carrying
`TextEncodeQwenImage21` and `QwenImage21Cache` — and the server boots once per
container, so weights stay warm across calls.

## The graph

One graph covers all four slots; what changes is the reference list and where
the empty latent comes from.

```
UNETLoader → QwenImage21Cache(auto) ────────────────┐
CLIPLoader ┐                                        ↓
LoadImage ×N → TextEncodeQwenImage21.images.image_N → KSampler → VAEDecode → SaveImage
VAELoader ─┘                            ↓ latent ───↗
                                 EmptyLatentImage (only when a size is asked for)
```

`TextEncodeQwenImage21` does the work 2511 needed extra nodes for: it resizes
each reference itself, encodes them through the VAE as `reference_latents`, and
returns an empty latent shaped to the first reference. An edit that must not
reframe simply takes that third output; asking for a width and height swaps in
an `EmptyLatentImage` instead. There is no `ModelSamplingAuraFlow`, no
`CFGNorm` and no Lightning LoRA — 2.1 has guidance distilled in, so the
official templates sample 25 steps at `cfg` 1.0 in a single pass, which is
already cheaper than 2511's eight steps with CFG.

`QwenImage21Cache` is what makes re-editing the same references cheap: the
prefix KV cache spends spare VRAM first and falls back to RAM.

## Knobs

Per-run, under the node's collapsed **Advanced** section: `steps`, `cfg`,
`resolution`, and `transparent` on text-to-image.

`resolution` is the *reference* budget, not the output size — each reference is
resized to about `resolution × resolution` pixels with its aspect preserved, and
`0` leaves it alone. Editing defaults to `0` so the result lines up with the
input; fusion defaults to `1024`, because ten full-size references are ten times
the sequence to attend over.

Deployment-wide, as environment variables read at deploy time: `QIE_STEPS`,
`QIE_CFG`, `QIE_RESOLUTION`, `QIE_DEFAULT_SIZE`, `QIE_CACHE_DEVICE`,
`QIE_CACHE_DTYPE`, `QIE_GPU`, `QIE_COMFY_ARGS`.

The default GPU is an H100. 17 GB of weights plus 2K activations may well fit
an L40S now that int8 replaced 2511's 30 GB of fp8 — `QIE_GPU=L40S` is there to
try it — but 80 GB is what this is known to run on.
