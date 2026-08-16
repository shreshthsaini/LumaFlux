# Training

The registered main run uses four GH200 GPUs for about 36 hours, approximately 144 GPU-hours. Its fixed training settings are 100,000 optimizer steps, global batch 16, learning rate `2e-4`, 512-pixel crops, and bf16 precision. The HDRTV1K variant uses 50,000 steps on the HDRTV1K training split.

FLUX.1-dev and its VAE remain frozen, and the SigLIP encoder remains frozen. Training updates the LumaFlux adapter modules: PGA, PCM, the HDR Residual Coupler, and the RQS tone-field decoder. The resulting adapter has 71,315,893 trainable parameters, 0.57% of the FLUX.1-dev backbone.

Users must separately accept the gated FLUX.1-dev license on Hugging Face. Released adapter files are available from [the LumaFlux Hugging Face repository](https://huggingface.co/shreshthsaini/LumaFlux).

## Launch

Prepare `data/curated/manifest.jsonl` as described in `docs/DATA.md`, then launch the registered configuration:

```bash
accelerate launch --num_processes 4 scripts/train.py \
  --config configs/train/main.yaml \
  --output-dir runs/lumaflux-main
```

The per-process batch size is 2 and gradient accumulation is 2, giving global batch 16 with four processes. To train the in-domain variant, use `configs/train/hdrtv1k.yaml` after preparing `data/hdrtv1k/manifest.jsonl`.

## Resume

Pass a full `.pt` checkpoint to `--resume`:

```bash
accelerate launch --num_processes 4 scripts/train.py \
  --config configs/train/main.yaml \
  --output-dir runs/lumaflux-main \
  --resume runs/lumaflux-main/checkpoint_step0050000.pt
```

Resume restores adapter parameters, optimizer and scheduler state, mixed-precision scaler state when present, optimizer-step counters, data-loader state, and random-number-generator state. Periodic checkpoints follow `ckpt_every` and `ckpt_keep` in the selected configuration. The final adapter is written as `adapters_final.safetensors`.
