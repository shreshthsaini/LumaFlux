#!/usr/bin/env bash
#SBATCH -J lumaflux
#SBATCH -o logs/lumaflux.%j.out
#SBATCH -e logs/lumaflux.%j.err
#SBATCH -p gh                      # VISTA GPU partition (1 GPU per node)
#SBATCH -N 4                       # nodes; 1 process per node, 1 GPU each
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -A <your_allocation>

# Multi-node LumaFlux training on TACC VISTA: each node has a single H200,
# so num_processes == num_machines == SLURM_NNODES. Rendezvous goes through
# the first node of the allocation.
#
# Usage:
#   sbatch --export=ALL,MANIFEST=/path/manifest.jsonl[,OUTDIR=...,CONFIG=...] \
#       scripts/train_vista.sbatch
# Adjust -N to scale nodes; the global batch size is
#   train.batch_size (per process) x SLURM_NNODES x train.grad_accum.

set -euo pipefail
mkdir -p logs

MANIFEST=${MANIFEST:?set MANIFEST=/path/to/manifest.jsonl}
OUTDIR=${OUTDIR:-$SCRATCH/lumaflux/runs/$SLURM_JOB_ID}
CONFIG=${CONFIG:-configs/model/flux_dev.yaml}

# Activate your environment here (module load / conda / venv), e.g.:
#   source $WORK/lumaflux-venv/bin/activate
# and make sure HF_TOKEN + HF_HOME point at a shared cache:
#   export HF_HOME=$WORK/hf_cache

MAIN_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MAIN_PORT=${MAIN_PORT:-29500}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker0}

# trackio logs locally per run; sync/inspect from the login node with
#   trackio show --project lumaflux
export TRACKIO_DIR=${TRACKIO_DIR:-$OUTDIR/trackio}

srun bash -c "accelerate launch \
  --num_machines \$SLURM_NNODES \
  --num_processes \$SLURM_NNODES \
  --machine_rank \$SLURM_NODEID \
  --main_process_ip $MAIN_NODE \
  --main_process_port $MAIN_PORT \
  --mixed_precision bf16 \
  --dynamo_backend no \
  -m lumaflux.training.train \
  --config $CONFIG \
  --manifest $MANIFEST \
  --output-dir $OUTDIR"
