"""LumaFlux trainer: accelerate loop with trackio experiment tracking.

Flow construction (documented design choice; see README "Training"):
the SDR and HDR frames are encoded with the frozen VAE and connected by a
noisy linear bridge

    z_t = (1 - t) z_hdr + t z_start,    z_start = z_sdr + gamma * eps

with constant target velocity v* = z_start - z_hdr. Inference integrates
the learned ODE from z_1 = E(x_sdr) down to t=0 (Algorithm 1), so train
and test trajectories share the same anchor. Reconstruction losses (Eq. 18)
are applied every ``recon_every`` steps through a one-step x0 estimate
z0_hat = z_t - t * v_pred decoded by the frozen VAE + RQS head.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader

from ..data.dataset import SdrHdrPairs
from ..models.factory import build_model, load_config
from ..utils import count_parameters, seed_everything
from .losses import reconstruction_losses, velocity_loss
from .schedule import cosine_with_warmup


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train LumaFlux adapters")
    p.add_argument("--config", required=True, help="YAML config (model + train sections)")
    p.add_argument("--manifest", required=True, help="curated manifest.jsonl")
    p.add_argument("--output-dir", default="runs/lumaflux")
    p.add_argument("--resume", default=None, help="adapter checkpoint to resume from")
    p.add_argument("--max-steps", type=int, default=None, help="override train.max_steps")
    return p.parse_args(argv)


def train(cfg: dict, manifest: str, output_dir: str, resume: str | None = None,
          max_steps_override: int | None = None) -> dict:
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
    if resume:
        model.load_adapters(resume)
    params = model.trainable_parameters()
    if accelerator.is_main_process:
        n_train = count_parameters(model)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)")

    dataset = SdrHdrPairs(
        manifest,
        crop_size=tcfg.get("crop_size", 512),
        seed=tcfg.get("seed", 0),
    )
    loader = DataLoader(
        dataset,
        batch_size=tcfg.get("batch_size", 16),
        shuffle=True,
        num_workers=tcfg.get("num_workers", 4),
        drop_last=True,
    )

    max_steps = max_steps_override or tcfg.get("max_steps", 200_000)
    optimizer = torch.optim.AdamW(
        params,
        lr=tcfg.get("lr", 1e-4),
        betas=tuple(tcfg.get("betas", (0.9, 0.999))),
        weight_decay=tcfg.get("weight_decay", 0.01),
    )
    scheduler = cosine_with_warmup(optimizer, tcfg.get("warmup_steps", 5_000), max_steps)

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    raw_model = accelerator.unwrap_model(model)

    run = None
    if accelerator.is_main_process:
        import trackio

        run = trackio.init(
            project=tcfg.get("trackio_project", "lumaflux"),
            name=tcfg.get("run_name", f"lumaflux-{int(time.time())}"),
            config={**cfg.get("model", {}), **tcfg},
        )

    bridge_noise = tcfg.get("bridge_noise", 0.05)
    recon_every = tcfg.get("recon_every", 4)
    log_every = tcfg.get("log_every", 50)
    ckpt_every = tcfg.get("ckpt_every", 1_000)
    w_velocity = tcfg.get("w_velocity", 1.0)
    lambdas = (tcfg.get("lambda1", 1.0), tcfg.get("lambda2", 0.5), tcfg.get("lambda3", 0.1))

    step = 0
    last_loss = float("nan")
    model.train()
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            with accelerator.accumulate(model):
                sdr, hdr = batch["sdr"], batch["hdr"]
                do_recon = bool(recon_every) and step % recon_every == 0
                # Call the *prepared* model so DDP/FSDP hooks see the full
                # gradient-producing computation and sync adapter gradients.
                fwd = model(sdr, hdr, bridge_noise=bridge_noise, do_recon=do_recon)

                loss = w_velocity * velocity_loss(fwd["v_pred"], fwd["v_target"])
                logs = {"loss/velocity": loss.item()}

                if do_recon:
                    decoded = fwd["decoded"]
                    rec = reconstruction_losses(
                        decoded["hdr"], hdr,
                        spline={k: decoded[k] for k in ("widths", "heights", "derivs")},
                        lambda1=lambdas[0], lambda2=lambdas[1], lambda3=lambdas[2],
                    )
                    loss = loss + rec["recon_total"]
                    logs.update({f"loss/{k}": v.item() for k, v in rec.items()})

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, tcfg.get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            last_loss = loss.item()
            if run is not None and step % log_every == 0:
                # "step" is reserved by trackio; log under train/step instead.
                logs.update({"loss/total": last_loss, "lr": scheduler.get_last_lr()[0],
                             "train/step": step})
                run.log(logs)
            if accelerator.is_main_process and ckpt_every and step > 0 and step % ckpt_every == 0:
                raw_model.save_adapters(out / f"adapters_step{step:07d}.safetensors")
            step += 1

    if accelerator.is_main_process:
        raw_model.save_adapters(out / "adapters_final.safetensors")
        if run is not None:
            run.log({"loss/total": last_loss, "train/step": step})
            import trackio

            trackio.finish()
    accelerator.wait_for_everyone()
    return {"steps": step, "final_loss": last_loss, "output_dir": str(out)}


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    train(cfg, args.manifest, args.output_dir, resume=args.resume,
          max_steps_override=args.max_steps)


if __name__ == "__main__":
    main()
