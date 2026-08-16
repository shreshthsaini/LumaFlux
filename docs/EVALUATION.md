# Evaluation

Luma-Eval reports seven full-reference metrics:

| Metric | Direction |
|---|---|
| PU21-PSNR | Higher is better |
| PU21-PSNR-Y | Higher is better |
| PU21-SSIM | Higher is better |
| dE-ITP | Lower is better |
| HDR-LPIPS | Lower is better |
| HDR-VDP-3 | Higher is better |
| FR-HIDROVQA | Lower is better |

## Protocol regimes

Luma-Eval uses three tracks: synthetic tone-mapping plus codec degradations, expert-graded SDR from LIVE-TMHDR, and native paired SDR and HDR. Every method receives identical inputs and is scored with identical output encoding and metric implementations. Evaluation samples use 16-bit PQ BT.2020 at a common 1,000-nit mastering convention.

Published-benchmark evaluation retains the original benchmark protocol. The paper reports HDRTV1K results on 117 published test pairs. Published baseline values retain their authors' protocols and are not directly rankable against Luma-Eval results. The 33.34 dB in-domain result belongs to the HDRTV1K-trained checkpoint. The main mixed-corpus checkpoint is zero-shot on HDRTV1K and scores 27.52 dB.

## Score predictions

Place `predictions_manifest.jsonl` in a predictions directory. Each JSONL record must provide paths named `prediction` and `reference`, relative to the manifest, for paired 16-bit PQ BT.2020 PNG files. Video records should also provide a shared `video` or `video_id` and a `frame_index`.

```bash
python scripts/benchmark.py \
  --predictions-dir results/predictions \
  --output-dir results/luma-eval
```

The output includes per-frame, per-video, overall, and metric-availability tables. Use `--no-video-metrics` for still images or `--no-optional-metrics` when optional metric backends are not installed.

HDR-VDP-3 requires its toolbox and a MATLAB or Octave runtime. These are not bundled. Install them separately, set `HDRVDP3_PATH` to the toolbox directory, and optionally set `HDRVDP3_BACKEND` to the runtime executable.
