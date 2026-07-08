#!/usr/bin/env bash
# Single-node convenience launcher (1x H200): for quick local runs / debugging.
# For real training on TACC VISTA (multi-node), use scripts/train_vista.sbatch.
# Usage: scripts/train.sh <manifest.jsonl> [output_dir]
set -euo pipefail
MANIFEST=${1:?usage: train.sh <manifest.jsonl> [output_dir]}
OUTDIR=${2:-runs/lumaflux-dev}

accelerate launch --num_processes 1 --mixed_precision bf16 \
  -m lumaflux.training.train \
  --config configs/model/flux_dev.yaml \
  --manifest "$MANIFEST" \
  --output-dir "$OUTDIR"
