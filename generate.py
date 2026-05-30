"""
Generate videos.

Reads:
  data/processed_cond/<prompt>_processed.pt    (Wan T5 conditioning)
  data/text_embeds/<prompt>/{text_embed.pt,    (per-prompt projection embeds)
                                video_model_text_embed.npy}
  models/{video,frame}_embed/{model.pt, projector.pt}  (trained M_v, M_f, A_v, A_f)
  models/latent_interpolation/model.pt            (trained M_c)
  models/Wan2.1-T2V-1.3B/                          (Wan 2.1 t2v-1.3B; download separately)

Writes:
  results/<setup>/<prompt>/<seed>_<batch_index:02d>.mp4

Usage:
  python generate.py --setup full
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent      # repo root
DATA = HERE / "data"
MODELS = HERE / "models"
RESULTS = HERE / "results"
WAN_DIR = HERE / "Wan2.1"

sys.path.insert(0, str(WAN_DIR))

import numpy as np
import torch
import torch.nn.functional as F

import wan
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import cache_video

from models import (
    FrameEmbedModel,
    LatentInterpolation,
    VideoEmbedModel,
    remove_sn_from_model,
)


PROMPTS = [
    "A vehicle moves through an open landscape.",
    "A pet plays on the floor.",
    "An insect eats on a leaf.",
    "An object emerges from the water.",
    "A toy lights up in a dark room.",
    "A tool operates on a workbench.",
    "A fruit rolls across a table.",
    "A plant sways in the wind outdoors.",
    "A material reacts to its surroundings.",
    "A group plays tennis on a court.",
]


def prompt_dirname(prompt):
    return prompt.rstrip(".")


# (use_global, use_local, use_regularization) per setup; IID uses none of these.
SETUPS = {
    "iid":          dict(is_iid=True),
    "local":        dict(is_iid=False, use_global=False, use_local=True,  use_regularization=False),
    "local_reg":    dict(is_iid=False, use_global=False, use_local=True,  use_regularization=True),
    "local_global": dict(is_iid=False, use_global=True,  use_local=True,  use_regularization=False),
    "full":         dict(is_iid=False, use_global=True,  use_local=True,  use_regularization=True),
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--setup", required=True, choices=list(SETUPS.keys()))
    p.add_argument("--ckpt_dir", default=str(MODELS / "Wan2.1-T2V-1.3B"),
                   help="Wan 2.1 t2v-1.3B checkpoint dir.")
    p.add_argument("--num_seeds", type=int, default=50, help="50 in the paper.")
    p.add_argument("--batch_size", type=int, default=4, help="Joint batch size; 4 in the paper.")
    return p.parse_args()


def load_latent_models(device, embed_dim=16):
    """Load trained M_v, M_f, M_c and their alignment matrices."""
    v = VideoEmbedModel(embed_dim=embed_dim)
    v.load_state_dict(torch.load(MODELS / "video_embed" / "model.pt"))
    v = remove_sn_from_model(v).eval().to(device)
    A_v = torch.load(MODELS / "video_embed" / "projector.pt").to(device)

    f = FrameEmbedModel(embed_dim=embed_dim)
    f.load_state_dict(torch.load(MODELS / "frame_embed" / "model.pt"))
    f = remove_sn_from_model(f).eval().to(device)
    A_f = torch.load(MODELS / "frame_embed" / "projector.pt").to(device)

    mc = LatentInterpolation()
    mc.load_state_dict(torch.load(MODELS / "latent_interpolation" / "model.pt"))
    mc = remove_sn_from_model(mc).eval().to(device)
    return v, f, mc, A_v, A_f


def main():
    args = parse_args()
    setup = SETUPS[args.setup]

    cfg = WAN_CONFIGS["t2v-1.3B"]
    common = dict(
        size=SIZE_CONFIGS["832*480"],
        frame_num=81,
        shift=8.0,
        sampling_steps=50,
        guide_scale=6.0,
        offload_model=True,
        batch_size=args.batch_size,
    )

    device = "cuda"
    if setup["is_iid"]:
        wan_t2v = wan.WanT2VForProcessedCond(
            config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        )
    else:
        v_model, f_model, mc_model, A_v, A_f = load_latent_models(device)
        wan_t2v = wan.WanT2VForProcessedCondJoint(
            config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
            video_embed_model=v_model,
            frame_embed_model=f_model,
            latent_interpolation_model=mc_model,
        )

    out_root = RESULTS / args.setup
    for prompt in PROMPTS:
        cond = torch.load(DATA / "processed_cond" / f"{prompt}_processed.pt")
        prompt_dir = prompt_dirname(prompt)
        out_dir = out_root / prompt_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        if not setup["is_iid"]:
            f_te = torch.load(DATA / "text_embeds" / prompt_dir / "text_embed.pt").to(device)
            v_te = torch.from_numpy(
                np.load(DATA / "text_embeds" / prompt_dir / "video_model_text_embed.npy")
            ).to(device)
            f_proj = F.normalize((f_te.view(1, -1) @ A_f.t()).view(-1), p=2, dim=-1)
            v_proj = F.normalize((v_te.view(1, -1) @ A_v.t()).view(-1), p=2, dim=-1)

        for seed in range(args.num_seeds):
            last = out_dir / f"{seed}_{args.batch_size - 1:02d}.mp4"
            if last.exists():
                print(f"[skip] {last}")
                continue
            if setup["is_iid"]:
                videos = wan_t2v.generate(cond, seed=seed, **common)
            else:
                videos = wan_t2v.generate(
                    cond, seed=seed,
                    use_global=setup["use_global"],
                    use_local=setup["use_local"],
                    use_regularization=setup["use_regularization"],
                    frame_text_embed_projected=f_proj,
                    video_text_embed_projected=v_proj,
                    **common,
                )
            for idx, vid in enumerate(videos):
                cache_video(
                    tensor=vid[None],
                    save_file=str(out_dir / f"{seed}_{idx:02d}.mp4"),
                    fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1),
                )


if __name__ == "__main__":
    main()
