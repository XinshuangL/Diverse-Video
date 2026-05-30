# Models

All model weights used by `generate.py` and `eval.py` live here.

## Contents

| Path | Source | Size | Used by |
|---|---|---|---|
| `video_embed/model.pt` | **ours (shipped)** | 48 KB | `generate.py` (joint setups) — latent video-level embedder M_v |
| `video_embed/projector.pt` | **ours (shipped)** | 52 KB | `generate.py` — alignment matrix A_v |
| `frame_embed/model.pt` | **ours (shipped)** | 40 KB | `generate.py` (joint setups) — latent frame-level embedder M_f |
| `frame_embed/projector.pt` | **ours (shipped)** | 36 KB | `generate.py` — alignment matrix A_f |
| `latent_interpolation/model.pt` | **ours (shipped)** | 792 KB | `generate.py` — latent frame interpolator M_c |
| `eden.pt` | **third-party (download)** | 603 MB | `eval.py` (consistency / MSE) |
| `Wan2.1-T2V-1.3B/` | **third-party (download)** | ~17 GB | `generate.py` (base text-to-video model) |

## Third-party downloads

### Wan 2.1 t2v-1.3B
Base text-to-video flow-matching model. Place the full HuggingFace repo at
`models/Wan2.1-T2V-1.3B/`.

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir models/Wan2.1-T2V-1.3B
```

Or grab the files manually from
<https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B>. Required contents include
`diffusion_pytorch_model.safetensors`, `models_t5_umt5-xxl-enc-bf16.pth`,
`Wan2.1_VAE.pth`, `config.json`, `configuration.json`.

### EDEN
Used by `eval.py` for the temporal-consistency MSE metric (interpolates each
frame from its neighbors). Place the checkpoint file directly at
`models/eden.pt`.

Download from the EDEN authors' release at
<https://github.com/bbldcver/EDEN>. The inference config that consumes this
checkpoint is `EDEN/configs/eval_eden.yaml`.
