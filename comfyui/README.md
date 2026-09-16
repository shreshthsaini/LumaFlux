# LumaFlux in ComfyUI

SDR-to-HDR conversion inside ComfyUI, with the pipeline from this repository. LumaFlux turns 8-bit SDR (BT.709) images and frame batches into 10-bit HDR (PQ, BT.2020) with a frozen FLUX.1-dev diffusion transformer plus 71M adapter parameters. No prompt, 8 sampling steps, and the same image model handles video through shared noise and tone-curve smoothing.

The nodes call the reference `LumaFluxPipeline`, so the sampling loop, the video stabilization, and the tone decoder are the released ones. This folder adds loading from ComfyUI's model folders, batch-as-video handling, an SDR preview, and two save nodes that keep the HDR signal intact.

## Install

Through ComfyUI-Manager (search for LumaFlux), or by hand. The repository root is the node pack, so clone the whole repository:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/shreshthsaini/LumaFlux
pip install -r LumaFlux/requirements.txt
```

Restart ComfyUI. `requirements.txt` holds only the inference dependencies; the nodes import `lumaflux` straight from `src/`, so nothing else from this repository needs installing. If the `lumaflux` package is already installed in ComfyUI's environment, symlinking this `comfyui/` folder into `custom_nodes` works too.

## Models

| What | Where it comes from |
|---|---|
| LumaFlux adapters (`lumaflux-main.safetensors`, 285 MB, or `lumaflux-hdrtv1k.safetensors`) | Downloaded into `models/lumaflux/` on first use from [shreshthsaini/LumaFlux](https://huggingface.co/shreshthsaini/LumaFlux). Drop the file there yourself to skip the download. |
| FLUX.1-dev transformer and VAE | Either the gated diffusers repo `black-forest-labs/FLUX.1-dev` (accept its license, then `huggingface-cli login`), or the single-file checkpoints you already use in ComfyUI: pick `flux1-dev.safetensors` from `models/diffusion_models` and `ae.safetensors` from `models/vae`. Single-file loading needs no Hugging Face login. |
| SigLIP encoder `google/siglip-so400m-patch14-384` | Downloaded to the Hugging Face cache on first use (about 1.7 GB). |

`lumaflux-main` is the general model trained on 314k UGC and PGC pairs. `lumaflux-hdrtv1k` is trained in-domain on HDRTV1K, for comparisons on that benchmark.

The adapter weights function only on top of FLUX.1-dev and are distributed under the FLUX.1-dev Non-Commercial License. The code is Apache-2.0 like the rest of the repository.

## Nodes

All nodes live under the `LumaFlux` category.

**LumaFlux Loader** builds the pipeline: frozen transformer, VAE, SigLIP, plus one adapter checkpoint. It unloads whatever ComfyUI has resident first, since the bf16 backbone alone needs about 24 GB of GPU memory. Outputs a `LUMAFLUX_MODEL`.

**LumaFlux SDR to HDR** converts an `IMAGE` batch. Every frame of the batch is treated as one video sequence.

| Input | What it does |
|---|---|
| `steps` | Euler steps from t=1 to t=0. 8 in the paper; more costs time without improving quality |
| `seed` | Seeds the bridge noise. The same seed reproduces the output exactly |
| `shared_noise` | Reuse one noise realization for the whole batch (video stabilization, on in the paper) |
| `rqs_ema` | Weight of the previous frame's tone-curve parameters (0.8 in the paper). Ignored for a single image |
| `tone_strength` | Scales the learned tone expansion. 1.0 is the paper; 0 leaves the VAE decode as is |

Outputs: `hdr_pq`, the PQ/BT.2020 signal in [0, 1] mastered at 1000 nits, and `preview`, an SDR tone-mapped view for Preview Image.

**Save HDR PNG (16-bit PQ)** writes each frame as a 16-bit PNG holding the PQ code values (signal x 65535), tagged with a `cICP` chunk so HDR-aware viewers (Chrome, recent macOS) display it as HDR. The file layout matches the frames the CLI writes, so the evaluation scripts read them directly.

**Save HDR Video (HEVC 10-bit PQ)** encodes the batch as `yuv420p10le` HEVC tagged BT.2020 / SMPTE 2084 with 1000-nit mastering metadata, the paper's delivery format. It needs an `ffmpeg` with a 10-bit `libx265` on PATH and refuses to run with an 8-bit-only build rather than write a banded file.

**LumaFlux HDR Preview (tone map)** turns `hdr_pq` into an SDR image with an adjustable white point (203 nits by default, the BT.2408 reference) for viewing on an ordinary monitor.

Do not route `hdr_pq` into the stock Save Image node: it quantizes to 8 bits and drops the colour tagging, and the PQ signal will look flat and gray.

## Wiring

```
Load Image (or a frame batch) ──┐
LumaFlux Loader ────────────────┴─> LumaFlux SDR to HDR ──hdr_pq──> Save HDR PNG / Save HDR Video
                                                        └─preview─> Preview Image
```

`example_workflows/lumaflux_sdr_to_hdr.json` is this graph, ready to load.

## Requirements and speed

Inputs are padded to a multiple of 16 with edge replication and cropped back afterwards, so any size works. Expect roughly 5 seconds and 27 GB of GPU memory per 1080p frame at 8 steps in bf16, the same as the CLI. Loading the backbone once takes a few minutes and about 24 GB of system RAM. The nodes are exercised on CPU by `tests/test_comfyui_nodes.py` with the miniature test stack; the full model needs a CUDA GPU with more than 24 GB.
