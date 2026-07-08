# LumaFlux

**Lifting 8-Bit Worlds to HDR Reality with Physically-Guided Diffusion Transformers**

Shreshth Saini, Hakan Gedik, Neil Birkbeck, Yilin Wang, Balu Adsumilli, Alan C. Bovik
*The University of Texas at Austin · Google, Inc.* — [arXiv:2604.02787](https://arxiv.org/abs/2604.02787)

LumaFlux converts 8-bit SDR (BT.709) video into 10-bit HDR (PQ, BT.2020) by adapting a
**frozen** Flux MM-DiT backbone with four lightweight, physically interpretable modules:

| Module | Role | Paper |
|---|---|---|
| **PGA** (Physically-Guided Adaptation) | gated low-rank residual on attention value projections, driven by luminance / gradient / saturation maps and FFT band energies | §4.3 |
| **PCM** (Perceptual Cross-Modulation) | FiLM conditioning of hidden states on frozen SigLIP embeddings | §4.4 |
| **HDR Residual Coupler** | timestep/layer-gated fusion of physical + perceptual token residuals | §4.5 |
| **RQS Tone-Field Decoder** | monotone rational-quadratic spline that expands VAE-decoded luma into HDR | §4.6 |

All modules are scheduled by a shared timestep–layer conditioner Ψ(t, ℓ) (§4.2) and are
**zero-initialized** so the pretrained generative prior is exactly preserved at step 0.
The model is fully **prompt-free**: no T5/CLIP text encoders at train or test time.

---

## Installation

```bash
cd lumaflux
pip install -e ".[demo,dev]"
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.1, ffmpeg (for video I/O and codec
degradations). HDR video export additionally needs an ffmpeg whose libx265
supports 10-bit (`ffmpeg -h encoder=libx265` must list `yuv420p10le`) — the
writer refuses 8-bit-only builds rather than silently banding the PQ signal;
`pip install imageio-ffmpeg` ships a static 10-bit-capable binary if your
system build lacks it. The default backbone `black-forest-labs/FLUX.1-dev` is gated —
`huggingface-cli login` or set `HF_TOKEN`. An ungated fallback config using
FLUX.1-schnell is provided (`configs/model/flux_schnell.yaml`), and a fully
offline tiny stack (`configs/model/tiny_debug.yaml`) is used by the tests.

## Data: curating the SDR–HDR corpus (§5)

The pipeline normalizes heterogeneous HDR sources (HIDROVQA, CHUG, LIVE-TMHDR)
to PQ/BT.2020 at a 1,000-nit mastering peak (Eq. 20), then synthesizes SDR
variants with the composite degradation chain (Eq. 19):
8 tone-mapping operators × CRF {23, 31, 39} x264 round-trips.

```bash
# smoke-test corpus (procedural HDR, no downloads)
python scripts/make_synthetic_data.py --out-dir data/synthetic

# real corpus: describe your sources in a spec file, then
python scripts/prepare_data.py --spec my_sources.yaml   # see file header for the schema
```

The result is a `manifest.jsonl` of HDR frames (16-bit PNG, PQ/BT.2020) paired
with all SDR variants (8-bit PNG, BT.709). `lumaflux.data.SdrHdrPairs` samples
PGC:UGC 1:1 and a random variant per draw; `lumaflux.data.to_hf_dataset`
exports the manifest as a HuggingFace `datasets.Dataset`.

Implemented TMOs (`lumaflux/data/tmo.py`): `ocio_v2`, `bt2446c_gm`,
`hard_clip_gm`, `bt2446a`, `reinhard`, `youtube_logc`, `bt2390_eetf_gm`,
`gamma_clip` — plus pass-through of expert-graded SDR where available
(LIVE-TMHDR).

## Training

Experiment tracking uses **[trackio](https://github.com/gradio-app/trackio)**
(local-first, wandb-compatible API). Inspect runs with
`trackio show --project lumaflux`.

```bash
# single GPU (debug / small runs)
scripts/train.sh /path/to/manifest.jsonl

# TACC VISTA (multi-node, 1x H200 per node) — the paper setup is 4 nodes:
sbatch -N 4 --export=ALL,MANIFEST=/path/manifest.jsonl scripts/train_vista.sbatch
```

`scripts/train_vista.sbatch` wires `accelerate launch` to SLURM
(`num_machines = num_processes = SLURM_NNODES`, rendezvous on the first node).
Global batch size = `train.batch_size` × nodes × `train.grad_accum`; the
default config (`batch_size: 4`, 4 nodes) reproduces the paper's global 16.
Paper schedule: 200k steps, AdamW (lr 1e-4, β = 0.9/0.999), cosine annealing
with 5k warmup, bf16.

Only adapters train (the Flux backbone, VAE and SigLIP stay frozen);
checkpoints store **adapter weights only** (safetensors). Push them to the Hub
with `scripts/export_adapters.py`.

### Flow construction (implementation note)

Eq. 18 supervises the decoded HDR output; Algorithm 1 starts inference at
`z_1 = E_VAE(x_sdr)`. To keep train and test trajectories consistent we train
on the noisy linear bridge

```
z_t = (1 - t) z_hdr + t z_start ,   z_start = z_sdr + γ ε ,   v* = z_start - z_hdr
```

(`γ = train.bridge_noise`, default 0.05) with the velocity-matching loss, and
apply the Eq. 18 reconstruction terms (L1 on linear luminance + linear RGB via
inverse PQ, plus RQS knot-slope smoothness) every `train.recon_every` steps
through a one-step x̂₀ estimate decoded by the frozen VAE + RQS head.

## Inference

```bash
python -m lumaflux.inference.cli \
  --config configs/model/flux_dev.yaml \
  --adapters runs/lumaflux-dev/adapters_final.safetensors \
  --input input_sdr.mp4 --output output_hdr.mp4 \
  --num-steps 40 --strength 1.0
```

Prompt-free, 40 Euler ODE steps (paper setting). Video outputs are 10-bit
HEVC, `yuv420p10le`, BT.2020 primaries + SMPTE-2084 (PQ) transfer with
1,000-nit ST 2086 mastering metadata; non-video outputs are 16-bit PNG frames
holding the PQ signal. `--strength` is the tone-expansion control recommended
in the paper's broader-impact note.

## Evaluation (§6)

```bash
python -m lumaflux.evaluation.benchmark \
  --config configs/model/flux_dev.yaml --adapters adapters.safetensors \
  --manifest data/luma_eval/manifest.jsonl --output-dir eval_results
```

Reports per-TMO/per-CRF breakdowns (the layout of Tables 1–2) as CSV +
markdown. Metrics: PU21-PSNR, PU21-PSNR(Y), PU21-SSIM, ΔE_ITP (BT.2124),
HDR-LPIPS (LPIPS on PU21 encoding). HDR-VDP-3 has no Python port: set
`HDRVDP3_PATH` to a MATLAB/Octave toolbox checkout to enable the wrapper,
otherwise the column is skipped.

## Demo

```bash
python -m lumaflux.demo.app --config configs/model/flux_dev.yaml --adapters adapters.safetensors
```

Gradio app: upload an SDR frame → HDR 16-bit PNG (PQ/BT.2020) + BT.2446c SDR
preview + the predicted RQS tone curve, with strength/steps/seed controls.

## Tests

```bash
pytest tests/            # 57 tests, CPU-only, offline (tiny stack)
```

Covers: transfer-function round-trips and known anchors, gamut matrix
inverses, ICtCp/PU21 sanity, TMO monotonicity, RQS monotonicity +
identity-at-init, frozen-backbone audit, **adapted-vs-frozen identity at
initialization**, gradient flow to adapters, curation/degradation on synthetic
clips, a 4-step training run with trackio + checkpoint resume, and pipeline /
video-writer / demo smoke tests.

## Repository layout

```
lumaflux/
├── configs/model/        # flux_dev (paper), flux_schnell (ungated), tiny_debug (offline)
├── scripts/              # data prep, synthetic data, train.sh, train_vista.sbatch, export
├── src/lumaflux/
│   ├── color/            # ST-2084/BT.709/HLG transfer, 709<->2020 gamut, ICtCp, PU21
│   ├── data/             # TMOs, degradation chain, PQ normalization, curation, datasets
│   ├── models/           # PGA, PCM, coupler, Psi(t,l), RQS, block wrappers, factory
│   ├── training/         # losses (Eq. 18 + velocity), cosine schedule, accelerate loop
│   ├── inference/        # prompt-free ODE pipeline, HDR HEVC/PNG writers, CLI
│   ├── evaluation/       # PU21 metrics, dE_ITP, HDR-LPIPS, HDR-VDP3 hook, benchmark
│   └── demo/             # Gradio app
└── tests/
```

## Citation

```bibtex
@article{saini2026lumaflux,
  title   = {LumaFlux: Lifting 8-Bit Worlds to HDR Reality with Physically-Guided Diffusion Transformers},
  author  = {Saini, Shreshth and Gedik, Hakan and Birkbeck, Neil and Wang, Yilin and Adsumilli, Balu and Bovik, Alan C.},
  journal = {arXiv preprint arXiv:2604.02787},
  year    = {2026}
}
```

## License

MIT (see repository root). FLUX.1-dev weights are subject to the
Black Forest Labs non-commercial license; FLUX.1-schnell is Apache-2.0.
