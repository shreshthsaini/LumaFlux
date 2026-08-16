"""LumaFlux trainer: accelerate loop with trackio experiment tracking.

Flow construction (documented design choice; see README "Training"):
the SDR and HDR frames are encoded with the frozen VAE and connected by a
noisy linear bridge

    z_t = (1 - t) z_hdr + t z_start,    z_start = z_sdr + gamma * eps

with constant target velocity v* = z_start - z_hdr. Inference uses the same
velocity sign and integrates from a configurable time s down to t=0.
Reconstruction losses (Eq. 18) are applied every ``recon_every`` optimizer
steps through a one-step x0 estimate z0_hat = z_t - t * v_pred decoded by the
frozen VAE + RQS head.
"""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader

from ..data.dataset import SdrHdrPairs, SdrHdrShards, collate_sdr_hdr, is_shard_manifest
from ..models.factory import build_model, load_config
from ..utils import count_parameters, seed_everything
from .losses import reconstruction_losses, velocity_loss
from .schedule import cosine_with_warmup


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train LumaFlux adapters")
    p.add_argument("--config", required=True, help="YAML config (model + train sections)")
    p.add_argument(
        "--manifest",
        default=None,
        help="curated manifest.jsonl; defaults to data.manifest in the config",
    )
    p.add_argument("--output-dir", default="runs/lumaflux")
    p.add_argument("--resume", default=None, help="full training checkpoint to resume from")
    p.add_argument("--max-steps", type=int, default=None, help="override train.max_steps")
    return p.parse_args(argv)


def _capture_rng_state(
    dataset: SdrHdrPairs | SdrHdrShards, loader_generator: torch.Generator
) -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "dataset": dataset.rng.getstate(),
        "loader_generator": loader_generator.get_state(),
    }


def _restore_rng_state(
    state: dict, dataset: SdrHdrPairs | SdrHdrShards, loader_generator: torch.Generator
) -> None:
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    dataset.rng.setstate(state["dataset"])
    loader_generator.set_state(state["loader_generator"])


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    global_step: int,
    dataset: SdrHdrPairs | SdrHdrShards,
    loader_generator: torch.Generator,
    scaler=None,
) -> None:
    state = {
        "format_version": 2,
        "adapters": model.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "completed_optimizer_steps": global_step,
        "global_step": global_step,
        "rng": _capture_rng_state(dataset, loader_generator),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)


def _load_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    dataset: SdrHdrPairs | SdrHdrShards,
    loader_generator: torch.Generator,
    scaler=None,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "adapters",
        "optimizer",
        "scheduler",
        "completed_optimizer_steps",
        "global_step",
        "rng",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"Incomplete training checkpoint, missing keys: {missing}")
    if checkpoint["completed_optimizer_steps"] != checkpoint["global_step"]:
        raise RuntimeError("Checkpoint optimizer-step counters disagree")

    model.load_trainable_state_dict(checkpoint["adapters"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    _restore_rng_state(checkpoint["rng"], dataset, loader_generator)
    return int(checkpoint["global_step"])


def _rotate_periodic_checkpoints(output_dir: Path, keep: int = 3) -> None:
    checkpoints = sorted(
        output_dir.glob("checkpoint_step*.pt"),
        key=lambda path: int(path.stem.removeprefix("checkpoint_step")),
    )
    for checkpoint in checkpoints[:-keep]:
        checkpoint.unlink()


def train(
    cfg: dict,
    manifest: str,
    output_dir: str,
    resume: str | None = None,
    max_steps_override: int | None = None,
    stop_at_step: int | None = None,
    metrics_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> dict:
    tcfg = cfg.get("train", {})
    seed_everything(tcfg.get("seed", 0))
    # Parameter usage differs between recon and non-recon steps (the RQS head
    # only runs every `recon_every` steps), so DDP must tolerate unused params.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=tcfg.get("grad_accum", 1),
        mixed_precision=tcfg.get("mixed_precision", "no"),
        kwargs_handlers=[ddp_kwargs],
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg, device="cpu")
    params = model.trainable_parameters()
    if accelerator.is_main_process:
        n_train = count_parameters(model)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)")

    shard_manifest = is_shard_manifest(manifest)
    dataset_class = SdrHdrShards if shard_manifest else SdrHdrPairs
    dataset = dataset_class(
        manifest,
        crop_size=tcfg.get("crop_size", 512),
        seed=tcfg.get("seed", 0),
    )
    loader_generator = torch.Generator().manual_seed(tcfg.get("seed", 0))
    loader = DataLoader(
        dataset,
        batch_size=tcfg.get("batch_size", 16),
        shuffle=True,
        num_workers=tcfg.get("num_workers", 4),
        drop_last=True,
        generator=loader_generator,
        collate_fn=collate_sdr_hdr if shard_manifest else None,
    )

    max_steps = max_steps_override or tcfg.get("max_steps", 200_000)
    optimizer = torch.optim.AdamW(
        params,
        lr=tcfg.get("lr", 1e-4),
        betas=tuple(tcfg.get("betas", (0.9, 0.999))),
        weight_decay=tcfg.get("weight_decay", 0.01),
    )
    schedule_name = tcfg.get("scheduler", "cosine")
    if schedule_name != "cosine":
        raise ValueError(f"Unsupported training scheduler: {schedule_name}")
    scheduler = cosine_with_warmup(optimizer, tcfg.get("warmup_steps", 5_000), max_steps)

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    raw_model = accelerator.unwrap_model(model)

    run = None
    if accelerator.is_main_process:
        import trackio

        run_name = tcfg.get("run_name", f"lumaflux-{int(time.time())}")
        run = trackio.init(
            project=tcfg.get("trackio_project", "lumaflux"),
            name=run_name,
            config={**cfg.get("model", {}), **tcfg},
        )

    global_step = 0
    if resume:
        global_step = _load_checkpoint(
            resume,
            raw_model,
            optimizer,
            scheduler,
            dataset,
            loader_generator,
            scaler=accelerator.scaler,
        )
    if global_step > max_steps:
        raise ValueError(
            f"checkpoint global_step {global_step} exceeds configured max_steps {max_steps}"
        )
    target_step = max_steps if stop_at_step is None else min(max_steps, stop_at_step)
    if target_step < global_step:
        raise ValueError(
            f"stop_at_step {target_step} is behind checkpoint global_step {global_step}"
        )

    bridge_noise = tcfg.get("bridge_noise", 0.05)
    recon_every = tcfg.get("recon_every", 4)
    log_every = tcfg.get("log_every", 50)
    ckpt_every = tcfg.get("ckpt_every", 1_000)
    w_velocity = tcfg.get("w_velocity", 1.0)
    lambdas = (tcfg.get("lambda1", 1.0), tcfg.get("lambda2", 0.5), tcfg.get("lambda3", 0.1))

    last_loss = float("nan")
    model.train()
    while global_step < target_step:
        for batch in loader:
            if global_step >= target_step:
                break
            with accelerator.accumulate(model):
                sdr, hdr = batch["sdr"], batch["hdr"]
                do_recon = bool(recon_every) and global_step % recon_every == 0
                # Call the *prepared* model so DDP/FSDP hooks see the full
                # gradient-producing computation and sync adapter gradients.
                fwd = model(sdr, hdr, bridge_noise=bridge_noise, do_recon=do_recon)

                loss = w_velocity * velocity_loss(fwd["v_pred"], fwd["v_target"])
                logs = {"loss/velocity": loss.item()}

                if do_recon:
                    decoded = fwd["decoded"]
                    rec = reconstruction_losses(
                        decoded["hdr"], hdr,
                        spline=(
                            {k: decoded[k] for k in ("widths", "heights", "derivs")}
                            if "widths" in decoded
                            else None
                        ),
                        lambda1=lambdas[0], lambda2=lambdas[1], lambda3=lambdas[2],
                        peak_nits=raw_model.peak_nits,
                    )
                    loss = loss + rec["recon_total"]
                    logs.update({f"loss/{k}": v.item() for k, v in rec.items()})

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, tcfg.get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            last_loss = loss.item()
            logs.update(
                {
                    "loss/total": last_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "train/step": global_step,
                }
            )
            if metrics_callback is not None and accelerator.is_main_process:
                metrics_callback(global_step, dict(logs))
            if run is not None and global_step % log_every == 0:
                # "step" is reserved by trackio; log under train/step instead.
                run.log(logs)
            if accelerator.is_main_process and ckpt_every and global_step % ckpt_every == 0:
                checkpoint_path = out / f"checkpoint_step{global_step:07d}.pt"
                _save_checkpoint(
                    checkpoint_path,
                    raw_model,
                    optimizer,
                    scheduler,
                    global_step,
                    dataset,
                    loader_generator,
                    scaler=accelerator.scaler,
                )
                keep = tcfg.get("ckpt_keep", 0)
                if keep:  # 0/absent = keep everything (user checkpoint-retention policy 2026-08-08)
                    _rotate_periodic_checkpoints(out, keep=keep)

    if accelerator.is_main_process:
        raw_model.save_adapters(out / "adapters_final.safetensors")
        final_checkpoint = out / "checkpoint_final.pt"
        _save_checkpoint(
            final_checkpoint,
            raw_model,
            optimizer,
            scheduler,
            global_step,
            dataset,
            loader_generator,
            scaler=accelerator.scaler,
        )
        if run is not None:
            final_logs = {"loss/total": last_loss, "train/step": global_step}
            run.log(final_logs)
            import trackio

            trackio.finish()
    accelerator.wait_for_everyone()
    return {
        "steps": global_step,
        "global_step": global_step,
        "completed_optimizer_steps": global_step,
        "lr": scheduler.get_last_lr()[0],
        "final_loss": last_loss,
        "output_dir": str(out),
        "checkpoint": str(out / "checkpoint_final.pt"),
    }


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    manifest = args.manifest or cfg.get("data", {}).get("manifest")
    if not manifest:
        raise SystemExit("--manifest is required when the config has no data.manifest")
    train(
        cfg,
        manifest,
        args.output_dir,
        resume=args.resume,
        max_steps_override=args.max_steps,
    )


if __name__ == "__main__":
    main()
