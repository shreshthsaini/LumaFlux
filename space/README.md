---
title: LumaFlux SDR to HDR
emoji: 🌅
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Prompt-free SDR-to-HDR with a frozen FLUX diffusion transformer
models:
- shreshthsaini/LumaFlux
- black-forest-labs/FLUX.1-dev
---

# LumaFlux: SDR to HDR

Demo for [LumaFlux](https://arxiv.org/abs/2604.02787): physically-guided inverse tone mapping that lifts 8-bit SDR frames to 10-bit PQ / BT.2020 HDR with a frozen FLUX.1-dev backbone, prompt-free, in 8 transport steps. Outputs a tone-mapped preview, the 16-bit HDR PNG, and the predicted tone curve.

Code: https://github.com/shreshthsaini/LumaFlux · Weights: https://huggingface.co/shreshthsaini/LumaFlux · Project page: https://shreshthsaini.github.io/LumaFlux/

## Running it

The Space needs a GPU (about 27 GB for a 1080p frame) and an `HF_TOKEN` secret from an account that has accepted the FLUX.1-dev license. The adapter weights are pulled from `shreshthsaini/LumaFlux` on first start.
