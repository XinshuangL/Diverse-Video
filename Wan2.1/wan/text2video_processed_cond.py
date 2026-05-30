# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""IID sampling with Wan 2.1 t2v-1.3B. Pre-computed T5 conditioning is
passed in as ``cond`` and broadcast to ``batch_size`` parallel samples.
"""

import gc
import math
import os
import random
import sys
from contextlib import contextmanager

import torch
import torch.cuda.amp as amp
from tqdm import tqdm

from .modules.model import WanModel
from .modules.vae import WanVAE
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2V:

    def __init__(self, config, checkpoint_dir, device_id=0, rank=0):
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

    def generate(self, cond, size=(832, 480), frame_num=81,
                 shift=5.0, sampling_steps=50, guide_scale=5.0,
                 seed=-1, offload_model=True, batch_size=1):
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
                temp_x0 = sample_scheduler.step(
                    noise_pred, t, latents_tensor,
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
