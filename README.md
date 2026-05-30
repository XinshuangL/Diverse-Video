# Consistency-Preserving Diverse Video Generation

Official implementation of the [paper](https://arxiv.org/abs/2602.15287). 

Two entry points:
`generate.py` produces videos; `eval.py` scores them and writes per-metric JSONs under `results/<setup>/`.

## 1. Environment

Tested with Python 3.10, torch 2.8.0 + CUDA 12.8:

```bash
conda create -n DiverseVideo python=3.10 -y
conda activate DiverseVideo

pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0

pip install \
    "transformers>=4.49,<5" tokenizers accelerate safetensors \
    diffusers easydict ftfy \
    scipy tqdm matplotlib \
    imageio imageio-ffmpeg opencv-python mediapy Pillow \
    PyYAML torchdiffeq einops

# eval.py diversity uses VideoPrism on JAX:
pip install "jax[cuda12]==0.6.2" flax
pip install "git+https://github.com/google-deepmind/videoprism.git"

# Built against the installed torch:
pip install flash-attn==2.8.3 --no-build-isolation
```

Download Wan 2.1 t2v-1.3B from <https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B> into `models/Wan2.1-T2V-1.3B/`. EDEN weights (needed for the consistency metric in `eval.py`) go to `models/eden.pt`; see [`models/README.md`](models/README.md).

For reproducibility, this repository provides preprocessed `data/` assets: cached Wan T5 conditioning under `data/processed_cond/<prompt>_processed.pt` and per-prompt projection embeds under `data/text_embeds/<prompt>/{text_embed.pt, video_model_text_embed.npy}`.

## 2. `generate.py`

```
python generate.py --setup <SETUP> [--ckpt_dir PATH] [--num_seeds N] [--batch_size B]
```

Defaults: `--num_seeds 50`, `--batch_size 4`, `--ckpt_dir models/Wan2.1-T2V-1.3B`. Resolution 832×480, 81 frames, 50 flow-matching steps. Inputs come from `data/processed_cond/` (cached Wan T5 conditioning) and `data/text_embeds/` (per-prompt projection embeds).

Writes `results/<setup>/<prompt>/<seed>_<batch_index:02d>.mp4` — 2000 mp4s per setup (10 prompts × 50 seeds × 4 batched videos). A `(prompt, seed)` is skipped if its last batch index already exists, so re-runs resume cleanly.

## 3. `eval.py`

```
python eval.py --setup <SETUP> [--metric {diversity,naturalness,consistency,all}]
               [--num_seeds N] [--batch_size B]
               [--eden_config PATH] [--eden_ckpt PATH]
```

`--metric all` (default) writes:

| File | Contents |
|---|---|
| `diversity.json`   | Vendi-v (VideoPrism-B) + Vendi-f (CLIP-B) |
| `naturalness.json` | CNI (Color Naturalness Index)             |
| `consistency.json` | MSE vs EDEN-interpolated neighbors        |
| `metrics.json`     | Headline mean(±ci) for vendi_v/vendi_f/cni/mse |

Each summary value is `{"mean": float, "std": float, "ci_half": float}` with 95% CI over the 50 seeds.

> **GPU memory note.** `--metric all` runs JAX (diversity) and EDEN (consistency) in the same process; JAX preallocates ~25 GB and never releases it before EDEN allocates, so the default OOMs on a 32 GB GPU. Comfortable on ≥40 GB GPUs. On smaller cards, invoke each metric in its own process: `--metric diversity`, then `--metric naturalness`, then `--metric consistency`.

## Acknowledgement

Thanks to the codebase from [Wan](https://github.com/Wan-Video/Wan2.1) and [EDEN](https://github.com/bbldcver/EDEN).

## Citation
If you find our code or paper helps, please consider citing:
```
@article{liu2026consistency,
  title={Consistency-Preserving Diverse Video Generation},
  author={Liu, Xinshuang and Li, Runfa Blark and Nguyen, Truong},
  journal={arXiv preprint arXiv:2602.15287},
  year={2026}
}
```
