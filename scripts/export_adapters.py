#!/usr/bin/env python3
"""Upload a trained adapter checkpoint to the Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="adapters .safetensors")
    p.add_argument(
        "--repo-id",
        default="shreshthsaini/LumaFlux",
        help="Hugging Face repository for the adapter file",
    )
    p.add_argument("--config", default=None, help="model config yaml to upload alongside")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, exist_ok=True, private=args.private)
    api.upload_file(
        path_or_fileobj=args.checkpoint,
        path_in_repo="adapters.safetensors",
        repo_id=args.repo_id,
    )
    if args.config:
        api.upload_file(
            path_or_fileobj=args.config,
            path_in_repo=Path(args.config).name,
            repo_id=args.repo_id,
        )
    print(f"uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
