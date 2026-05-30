# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Joint sampling with Wan 2.1 t2v-1.3B. ``batch_size`` samples are drawn
jointly so a DPP diversity force and an optional consistency regularization
can be applied along the sampling trajectory.
"""

import gc
import math
import os
import random
import sys
from contextlib import contextmanager

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

from .modules.model import WanModel
from .modules.vae import WanVAE
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def normalize_K(K, eps=1e-3):
    upper = K[torch.triu(torch.ones_like(K, dtype=torch.bool), diagonal=1)]
    return K / (torch.median(upper).detach() + eps)


def logdet_psd_cholesky(K, jitter=1e-3):
    B = K.size(0)
    old_dtype = K.dtype
    K = K.to(dtype=torch.float32)
    I = torch.eye(B, dtype=K.dtype, device=K.device)
    L, _ = torch.linalg.cholesky_ex(K + jitter * I)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
    return logdet.to(dtype=old_dtype)


def normalize_force(force, velocity):
    velocity_scale = (velocity * velocity).sum(dim=(0, 1, 3, 4)).sqrt().mean()
    force_per_T_scale = (force * force).sum(dim=(0, 1, 3, 4), keepdim=True).sqrt()
    return force / force_per_T_scale * velocity_scale


def regularize_force(force, good_gradient):
    """Project the diversity force so it cannot oppose the consistency gradient."""
    good_unit = F.normalize(good_gradient, dim=1, p=2)
    f_par_scalar = (force * good_unit).sum(dim=1, keepdim=True)
    f_vert = force - f_par_scalar * good_unit
    return f_vert + F.relu(f_par_scalar) * good_unit


def remove_component(x, ref):
    ref = F.normalize(ref, p=2, dim=-1).view(1, -1)
    return x - (x * ref).sum(dim=-1, keepdim=True) * ref


class WanT2V:

    def __init__(self, config, checkpoint_dir, device_id=0, rank=0,
                 video_embed_model=None, frame_embed_model=None,
                 latent_interpolation_model=None):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size

        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device,
        )
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False).to(self.device)
        self.sp_size = 1

        self.video_embed_model = video_embed_model
        self.frame_embed_model = frame_embed_model
        self.latent_interpolation_model = latent_interpolation_model

    def compute_diversity(self, x1, use_global, use_local,
                          frame_text_embed_projected,
                          video_text_embed_projected):
        """DPP log-determinant diversity over a joint batch of latents."""
        video_embed = self.video_embed_model(x1)

        B, C, T, H, W = x1.shape
        frames = x1.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        frame_embed = self.frame_embed_model(frames).reshape(B, T, -1)

        # Strip the text-aligned component so diversity isn't dominated by it.
        frame_embed = remove_component(
            frame_embed.view(B * T, -1), frame_text_embed_projected
        ).view(B, T, -1)
        video_embed = remove_component(video_embed, video_text_embed_projected)

        v_norm = (video_embed ** 2).sum(dim=1, keepdim=True)
        K_global = normalize_K(v_norm + v_norm.T - 2 * video_embed @ video_embed.T)

        frame_embed = frame_embed.permute(1, 0, 2)
        f_norm = (frame_embed ** 2).sum(dim=2, keepdim=True)
        K_local = (f_norm + f_norm.transpose(1, 2)
                   - 2 * torch.matmul(frame_embed, frame_embed.transpose(1, 2))).mean(dim=0)
        K_local = normalize_K(K_local)

        if use_global and use_local:
            K = (K_global + K_local) / 2
        elif use_global:
            K = K_global
        elif use_local:
            K = K_local
        else:
            raise ValueError("At least one of use_global / use_local must be True.")

        L = torch.exp(-K)
        I = torch.eye(B, dtype=L.dtype, device=L.device)
        return logdet_psd_cholesky(L) - logdet_psd_cholesky(L + I)

    def compute_consistency(self, latents):
        """-MSE between the interpolation model's prediction and the true centre latent."""
        latents = latents.to(torch.float32)
        B, C, T, H, W = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        before = latents[:, :-2].reshape(-1, C, H, W)
        after = latents[:, 2:].reshape(-1, C, H, W)
        cur = latents[:, 1:-1].reshape(-1, C, H, W)
        pred, _ = self.latent_interpolation_model(before, after)
        return -F.mse_loss(pred, cur)

    def generate(self, cond, size=(832, 480), frame_num=81,
                 shift=5.0, sampling_steps=50, guide_scale=5.0,
                 seed=-1, offload_model=True, batch_size=1,
                 force_weight=0.2,
                 use_global=False, use_local=False, use_regularization=False,
                 frame_text_embed_projected=None,
                 video_text_embed_projected=None):
        if batch_size is None or batch_size < 1:
            batch_size = 1
        target_shape = (
            self.vae.model.z_dim,
            (frame_num - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device).manual_seed(seed)
        noise = [
            torch.randn(*target_shape, dtype=torch.float32,
                        device=self.device, generator=seed_g)
            for _ in range(batch_size)
        ]

        @contextmanager
        def noop_no_sync():
            yield
        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1, use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps

            latents = noise
            arg_c, arg_null = cond["arg_c"], cond["arg_null"]

            for t in tqdm(timesteps):
                timestep = torch.tensor([t], device=self.device)
                self.model.to(self.device)

                noise_pred_cond = self.model(latents, t=timestep, **arg_c)
                noise_pred_uncond = self.model(latents, t=timestep, **arg_null)
                guided = [u + guide_scale * (c - u)
                          for c, u in zip(noise_pred_cond, noise_pred_uncond)]
                noise_pred = torch.stack(guided, dim=0)
                latents_tensor = torch.stack(latents, dim=0)

                sigma = t / 1000.0
                with torch.enable_grad(), autocast(enabled=False):
                    x_req = latents_tensor.clone().detach().to(torch.float32).requires_grad_(True)
                    x1 = x_req - sigma * noise_pred

                    diversity = self.compute_diversity(
                        x1, use_global, use_local,
                        frame_text_embed_projected, video_text_embed_projected,
                    )
                    (force,) = torch.autograd.grad(diversity, x_req, create_graph=False)
                    force = force.contiguous().detach()

                    if use_regularization:
                        consistency = self.compute_consistency(x1)
                        (cons_grad,) = torch.autograd.grad(consistency, x_req, create_graph=False)
                        force = regularize_force(force, cons_grad.detach())

                    force = normalize_force(force, -noise_pred)
                    applied_force = force * force_weight * math.sqrt(max(float(sigma), 0.0) + 1e-12)

                update_vector = noise_pred - applied_force
                temp_x0 = sample_scheduler.step(
                    update_vector, t, latents_tensor,
                    return_dict=False, generator=seed_g,
                )[0]
                latents = [temp_x0[i] for i in range(temp_x0.shape[0])]

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            videos = self.vae.decode(latents)

        del noise, latents, sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        return videos
