"""
Evaluate the videos produced by generate.py for one setup.

Computes Vendi-v, Vendi-f, MSE, CNI and writes:
  results/<setup>/diversity.json     (full per-prompt diversity stats)
  results/<setup>/naturalness.json   (full per-prompt CNI stats)
  results/<setup>/consistency.json   (full per-prompt MSE stats)
  results/<setup>/metrics.json       (4-metric headline summary)

Reads videos from   results/<setup>/<prompt>/<seed>_<batch>.mp4
Reads EDEN weights from   models/eden.pt

Setups: iid, local, local_reg, local_global, full

Usage:
  python eval.py --setup full
  python eval.py --setup local_reg --metric diversity
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent      # repo root
DATA = HERE / "data"
MODELS = HERE / "models"
RESULTS = HERE / "results"
EDEN_DIR = HERE / "EDEN"
sys.path.insert(0, str(EDEN_DIR))


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


SETUPS = ["iid", "local", "local_reg", "local_global", "full"]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def summarize_with_ci(values, confidence=0.95, round_to=6):
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    ci_half = float(stats.t.ppf((1 + confidence) / 2, df=max(n - 1, 1)) * se) if n > 1 else 0.0
    return {"mean": round(mean, round_to), "std": round(std, round_to), "ci_half": round(ci_half, round_to)}


def _normalize_K(K, eps=1e-3):
    upper = K[torch.triu(torch.ones_like(K, dtype=torch.bool), diagonal=1)]
    return K / (torch.median(upper).detach() + eps)


def _logdet_psd(K, jitter=1e-3):
    B = K.size(0)
    K = K.to(torch.float32)
    I = torch.eye(B, dtype=K.dtype, device=K.device)
    L, _ = torch.linalg.cholesky_ex(K + jitter * I)
    return 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)


def _vendi_global(data):
    N = data.shape[0]
    K = data @ data.T
    eigs = torch.linalg.eigvalsh(K).clamp(min=0) / N
    pos = eigs[eigs > 0]
    return float(torch.exp(-torch.sum(pos * torch.log(pos))))


def _vendi_local(data):  # [N, T, C]
    N, T, _ = data.shape
    K = torch.einsum('ntc,mtc->nm', data, data) / T
    eigs = torch.linalg.eigvalsh(K).clamp(min=0) / N
    pos = eigs[eigs > 0]
    return float(torch.exp(-torch.sum(pos * torch.log(pos))))


# ---------------------------------------------------------------------------
# Diversity (Vendi-v via VideoPrism, Vendi-f via CLIP)
# ---------------------------------------------------------------------------
def eval_diversity(video_dir, num_seeds, batch_size):
    print(f"[diversity] {video_dir}")
    import imageio.v2 as imageio
    import jax
    import jax.numpy as jnp
    import mediapy as media
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from videoprism import models as vp

    flax_model = vp.get_model("videoprism_lvt_public_v1_base")
    loaded_state = vp.load_pretrained_weights("videoprism_lvt_public_v1_base")

    @jax.jit
    def encode_video(x):
        ve, _, _ = flax_model.apply(loaded_state, x, None, None, train=False)
        return ve / jnp.linalg.norm(ve, axis=-1, keepdims=True)

    NUM_FRAMES, IMAGE_SIZE = 16, 288
    def load_vp_video(path):
        reader = imageio.get_reader(path)
        frames = [fr for fr in reader]
        reader.close()
        video = np.stack(frames, axis=0)
        idx = np.linspace(0, video.shape[0] - 1, NUM_FRAMES).astype(int)
        video = video[idx]
        video = np.stack([media.resize_image(fr, (IMAGE_SIZE, IMAGE_SIZE)) for fr in video], axis=0)
        video = media.to_float01(video).astype("float32")
        return jnp.asarray(np.expand_dims(video, 0))

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    def clip_frames(path):
        feats = []
        with imageio.get_reader(path) as reader:
            for frame in reader:
                inp = clip_proc(images=Image.fromarray(frame), return_tensors="pt")["pixel_values"].to("cuda")
                with torch.no_grad():
                    out = clip.get_image_features(pixel_values=inp)
                    # In most transformers versions this is a Tensor (projected
                    # image_embeds); in some it's a structured output. We always
                    # want the projected vector (CLIP's contrastive embedding
                    # space), NOT the pre-projection 768-dim pooler_output.
                    if isinstance(out, torch.Tensor):
                        feat = out
                    elif hasattr(out, "image_embeds") and out.image_embeds is not None:
                        feat = out.image_embeds
                    elif hasattr(out, "pooler_output"):
                        feat = clip.visual_projection(out.pooler_output)
                    else:
                        raise RuntimeError(f"Unexpected CLIP output type: {type(out)}")
                    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
                feats.append(feat.squeeze(0).cpu())
        return torch.stack(feats)

    gres = {p: {"dpp_list": [], "vendi_score_list": []} for p in PROMPTS}
    lres = {p: {"dpp_list": [], "vendi_score_list": []} for p in PROMPTS}

    for prompt in PROMPTS:
        prompt_dir = video_dir / prompt_dirname(prompt)
        for seed in range(num_seeds):
            bv, bf = [], []
            for b in range(batch_size):
                path = str(prompt_dir / f"{seed}_{b:02d}.mp4")
                bv.append(torch.from_numpy(np.array(encode_video(load_vp_video(path))[0])))
                bf.append(clip_frames(path))

            V = F.normalize(torch.stack(bv), p=2, dim=1)
            nv = (V * V).sum(dim=1, keepdim=True)
            Kg = _normalize_K(nv + nv.T - 2 * V @ V.T)
            Lg = torch.exp(-Kg)
            Ig = torch.eye(Kg.shape[0], dtype=Lg.dtype, device=Lg.device)
            gres[prompt]["dpp_list"].append(float(_logdet_psd(Lg) - _logdet_psd(Lg + Ig)))
            gres[prompt]["vendi_score_list"].append(_vendi_global(V))

            Fs = F.normalize(torch.stack(bf), p=2, dim=-1, eps=1e-8)
            T_BC = Fs.transpose(0, 1)
            nf = (T_BC * T_BC).sum(dim=2, keepdim=True)
            Kl = _normalize_K((nf + nf.transpose(1, 2) - 2 * torch.matmul(T_BC, T_BC.transpose(1, 2))).mean(dim=0))
            Ll = torch.exp(-Kl)
            Il = torch.eye(Kl.shape[0], dtype=Ll.dtype, device=Ll.device)
            lres[prompt]["dpp_list"].append(float(_logdet_psd(Ll) - _logdet_psd(Ll + Il)))
            lres[prompt]["vendi_score_list"].append(_vendi_local(Fs))

    def analyze(res):
        keys = ("dpp_list", "vendi_score_list")
        per_prompt = {
            p: {k.replace("_list", "_summary"): summarize_with_ci(res[p][k]) for k in keys}
            for p in PROMPTS
        }
        all_lists = {k: [] for k in keys}
        for s_ in range(num_seeds):
            for k in keys:
                all_lists[k].append(float(np.mean([res[p][k][s_] for p in PROMPTS])))
        return {
            "summary_per_prompt": per_prompt,
            "dpp_all_prompts_summary":  summarize_with_ci(all_lists["dpp_list"]),
            "vendi_score_all_prompts_summary": summarize_with_ci(all_lists["vendi_score_list"]),
        }

    return {
        "global_diversity_results": gres,
        "local_diversity_results":  lres,
        "global_diversity_summary": analyze(gres),
        "local_diversity_summary":  analyze(lres),
    }


# ---------------------------------------------------------------------------
# Naturalness (CNI)
# ---------------------------------------------------------------------------
def _cni_score(rgb):
    import cv2
    img = rgb.astype(np.float32) / 255.0
    L, u, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2Luv))
    C = np.sqrt(u * u + v * v)
    S = C / np.maximum(L, 1e-12)
    H = (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0
    valid = (L >= 20.0) & (L <= 80.0) & (S > 0.1)
    if not np.any(valid):
        return 0.0
    Hm, Sm = H[valid], S[valid]
    masks = {
        "skin":  ((Hm >= 25.0)  & (Hm <= 70.0),  (0.76, 0.52)),
        "grass": ((Hm >= 95.0)  & (Hm <= 135.0), (0.81, 0.53)),
        "sky":   ((Hm >= 185.0) & (Hm <= 260.0), (0.43, 0.22)),
    }
    counts = {k: int(m.sum()) for k, (m, _) in masks.items()}
    denom = sum(counts.values())
    if denom == 0:
        return 0.0
    total = 0.0
    for k, (m, (mu, sig)) in masks.items():
        if counts[k] == 0:
            continue
        S_avg = float(Sm[m].mean())
        total += counts[k] * float(np.exp(-0.5 * ((S_avg - mu) / sig) ** 2))
    return total / denom


def _video_cni(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = success = 0
    s_sum = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        try:
            s_sum += _cni_score(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            success += 1
        except Exception:
            pass
        n += 1
    cap.release()
    return s_sum / max(success, 1), (n - success) / max(n, 1)


def eval_naturalness(video_dir, num_seeds, batch_size):
    print(f"[naturalness] {video_dir}")
    res = {p: {"naturalness_list": [], "failure_rate_list": []} for p in PROMPTS}
    for prompt in PROMPTS:
        prompt_dir = video_dir / prompt_dirname(prompt)
        for seed in range(num_seeds):
            n_sum = f_sum = 0.0
            for b in range(batch_size):
                n_, f_ = _video_cni(prompt_dir / f"{seed}_{b:02d}.mp4")
                n_sum += n_
                f_sum += f_
            res[prompt]["naturalness_list"].append(n_sum / batch_size)
            res[prompt]["failure_rate_list"].append(f_sum / batch_size)

    per_prompt = {
        p: {"naturalness_summary": summarize_with_ci(r["naturalness_list"]),
            "failure_rate_summary": summarize_with_ci(r["failure_rate_list"])}
        for p, r in res.items()
    }
    nat_all  = [float(np.mean([res[p]["naturalness_list"][s_]  for p in PROMPTS])) for s_ in range(num_seeds)]
    fail_all = [float(np.mean([res[p]["failure_rate_list"][s_] for p in PROMPTS])) for s_ in range(num_seeds)]
    return {
        "video_naturalness_results": res,
        "video_naturalness_summary": {
            "summary_per_prompt": per_prompt,
            "naturalness_all_prompts_summary": summarize_with_ci(nat_all),
            "failure_rate_all_prompts_summary": summarize_with_ci(fail_all),
        },
    }


# ---------------------------------------------------------------------------
# Consistency (MSE via EDEN interpolation)
# ---------------------------------------------------------------------------
def eval_consistency(video_dir, num_seeds, batch_size, eden_config, eden_ckpt):
    print(f"[consistency] {video_dir}")
    import imageio
    import yaml
    from src.models import load_model
    from src.transport import Sampler, create_transport
    from src.utils import InputPadder

    with open(eden_config) as fp:
        cfg = yaml.unsafe_load(fp)
    device = torch.device("cuda:0")
    eden = load_model(cfg["model_name"], **cfg["model_args"])
    ckpt = torch.load(eden_ckpt, map_location="cpu")
    eden.load_state_dict(ckpt["eden"])
    eden.to(device).eval()
    del ckpt
    sample_fn = Sampler(create_transport("Linear", "velocity")).sample_ode(
        sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3,
    )

    def interp(prev, nxt):
        h, w = prev.shape[2:]
        padder = InputPadder([h, w])
        diff = (
            (torch.mean(torch.cosine_similarity(prev, nxt, dim=1), dim=[1, 2]) - cfg["cos_sim_mean"])
            / cfg["cos_sim_std"]
        ).unsqueeze(1).to(device)
        cond_frames = padder.pad(torch.cat((prev, nxt), dim=0))
        nh, nw = cond_frames.shape[2:]
        noise = torch.randn([prev.shape[0], nh // 32 * nw // 32, cfg["model_args"]["latent_dim"]], device=device)
        samples = sample_fn(noise, eden.denoise, cond_frames=cond_frames, difference=diff)[-1]
        latents = samples / cfg["vae_scaler"] + cfg["vae_shift"]
        return padder.unpad(eden.decode(latents).clamp(0.0, 1.0))

    def video_mse(path):
        reader = imageio.get_reader(str(path))
        frames = [fr for fr in reader]
        reader.close()
        if len(frames) < 3:
            return 0.0
        video = torch.from_numpy(np.stack(frames, axis=0)).float().permute(0, 3, 1, 2) / 255.0
        with torch.no_grad():
            prev = video[:-2].to(device)
            nxt = video[2:].to(device)
            cur = video[1:-1].to(device)
            preds = []
            for st in range(0, prev.shape[0], 10):
                preds.append(interp(prev[st:st + 10], nxt[st:st + 10]))
            pred = torch.cat(preds, dim=0)
            mses = [F.mse_loss(cur[i], pred[i]).item() for i in range(cur.shape[0])]
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return float(np.mean(mses))

    all_results = {}
    for prompt in PROMPTS:
        details = []
        for seed in range(num_seeds):
            mse_sum = 0.0
            for b in range(batch_size):
                mse_sum += video_mse(video_dir / prompt_dirname(prompt) / f"{seed}_{b:02d}.mp4")
            details.append({"mse": mse_sum / batch_size})
        all_results[prompt] = {
            "mse": float(np.mean([d["mse"] for d in details])),
            "details": details,
        }

    mse_per_seed = [
        float(np.mean([all_results[p]["details"][s_]["mse"] for p in PROMPTS]))
        for s_ in range(num_seeds)
    ]
    return {
        "all_results": all_results,
        "avg_results": {"mse": float(np.mean([all_results[p]["mse"] for p in PROMPTS]))},
        "all_summary": {"mse_summary": summarize_with_ci(mse_per_seed)},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--setup", required=True, choices=SETUPS)
    p.add_argument("--num_seeds", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--metric", default="all",
                   choices=["diversity", "naturalness", "consistency", "all"])
    p.add_argument("--eden_config", default=str(EDEN_DIR / "configs" / "eval_eden.yaml"))
    p.add_argument("--eden_ckpt",   default=str(MODELS / "eden.pt"))
    return p.parse_args()


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(data, fp, indent=4)


def main():
    args = parse_args()
    out_dir = RESULTS / args.setup
    video_dir = out_dir
    summary = {}

    if args.metric in ("diversity", "all"):
        d = eval_diversity(video_dir, args.num_seeds, args.batch_size)
        _write(out_dir / "diversity.json", d)
        summary["vendi_v"] = d["global_diversity_summary"]["vendi_score_all_prompts_summary"]
        summary["vendi_f"] = d["local_diversity_summary"]["vendi_score_all_prompts_summary"]
    if args.metric in ("naturalness", "all"):
        n = eval_naturalness(video_dir, args.num_seeds, args.batch_size)
        _write(out_dir / "naturalness.json", n)
        summary["cni"] = n["video_naturalness_summary"]["naturalness_all_prompts_summary"]
    if args.metric in ("consistency", "all"):
        c = eval_consistency(video_dir, args.num_seeds, args.batch_size,
                             args.eden_config, args.eden_ckpt)
        _write(out_dir / "consistency.json", c)
        summary["mse"] = c["all_summary"]["mse_summary"]

    metrics_path = out_dir / "metrics.json"
    existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    existing.update(summary)
    _write(metrics_path, existing)
    print(f"\n[summary written to {metrics_path}]")
    print(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()
