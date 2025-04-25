import os
import json
import copy
import time
import random
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
import gradio as gr

from diffusers import (
    DiffusionPipeline,
    AutoencoderTiny,
    AutoencoderKL,
    AutoPipelineForImage2Image,
    FluxPipeline,
    FlowMatchEulerDiscreteScheduler)

from huggingface_hub import (
    hf_hub_download,
    HfFileSystem,
    ModelCard,
    snapshot_download)

from diffusers.utils import load_image

import spaces

#---if workspace = local or colab---

# Authenticate with Hugging Face
# from huggingface_hub import login

# Log in to Hugging Face using the provided token
# hf_token = 'hf-token-authentication'
# login(hf_token)

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

# FLUX pipeline
@torch.inference_mode()
def flux_pipe_call_that_returns_an_iterable_of_images(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
    good_vae: Optional[Any] = None,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = self._execution_device

    lora_scale = joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None
    prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    self._num_timesteps = len(timesteps)

    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0]) if self.transformer.config.guidance_embeds else None

    for i, t in enumerate(timesteps):
        if self.interrupt:
            continue

        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]

        latents_for_image = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents_for_image = (latents_for_image / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents_for_image, return_dict=False)[0]
        yield self.image_processor.postprocess(image, output_type=output_type)[0]
        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        torch.cuda.empty_cache()
        
    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    latents = (latents / good_vae.config.scaling_factor) + good_vae.config.shift_factor
    image = good_vae.decode(latents, return_dict=False)[0]
    self.maybe_free_model_hooks()
    torch.cuda.empty_cache()
    yield self.image_processor.postprocess(image, output_type=output_type)[0]

#------------------------------------------------------------------------------------------------------------------------------------------------------------#
loras = [
    #24
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Claymation-XC-LoRA/resolve/main/images/4.png",
        "title": "Claymation XC",
        "repo": "strangerzonehf/Flux-Claymation-XC-LoRA",
        "weights": "Claymation.safetensors",
        "trigger_word": "Claymation"    
    },
    #25
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Icon-Kit-LoRA/resolve/main/images/6.png",
        "title": "Icon Kit",
        "repo": "strangerzonehf/Flux-Icon-Kit-LoRA",
        "weights": "Icon-Kit.safetensors",
        "trigger_word": "Icon Kit"    
    },
    #43
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Cardboard-Art-LoRA/resolve/main/images/6.png",
        "title": "Cardboard Art V1",
        "repo": "strangerzonehf/Flux-Cardboard-Art-LoRA",
        "weights": "cardboard# art.safetensors",
        "trigger_word": "cardboard# art"    
    },
    #53
    {
        "image": "https://huggingface.co/fofr/flux-condensation/resolve/main/images/example_crzf2b8xi.png",
        "title": "Condensation",
        "repo": "fofr/flux-condensation",
        "weights": "lora.safetensors",
        "trigger_word": "CONDENSATION"    
    },
    #63
    {
        "image": "https://huggingface.co/Datou1111/flux-sincity-movie/resolve/main/images/img__00685_.png",
        "title": "Sincity Movie",
        "repo": "Datou1111/flux-sincity-movie",
        "weights": "sincitymov.safetensors",
        "trigger_word": "sincitymov"    
    },
    #88
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Sketch-Sized-LoRA/resolve/main/images/8.png",
        "title": "Sketch Sized",
        "repo": "strangerzonehf/Flux-Sketch-Sized-LoRA",
        "weights": "Sketch_Sized.safetensors",
        "trigger_word": "Sketch Sized"    
    },   
    #60
    {
        "image": "https://huggingface.co/strangerzonehf/Sketch-Paint/resolve/main/images/2.png",
        "title": "Sketch Paint",
        "repo": "strangerzonehf/Sketch-Paint",
        "weights": "Sketch-Paint.safetensors",
        "trigger_word": "Sketch paint"    
    },
    #61
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Smiley-Portrait-LoRA/resolve/main/images/3.png",
        "title": "Smiley Portrait",
        "repo": "strangerzonehf/Flux-Smiley-Portrait-LoRA",
        "weights": "smiley-portrait.safetensors",
        "trigger_word": "smiley portrait"    
    },
    #26
    {
        "image": "https://huggingface.co/strangerzonehf/Gem-Touch-LoRA-Flux/resolve/main/images/333.png",
        "title": "Gem Touch LoRA",
        "repo": "strangerzonehf/Gem-Touch-LoRA-Flux",
        "weights": "GemTouch.safetensors",
        "trigger_word": "Gem Touch"    
    },
    #54
    {
        "image": "https://huggingface.co/AiAF/D-ART-18DART5_LoRA_Flux1/resolve/main/samples_2000-4000/1735935528010__000004000_3.jpg",
        "title": "D-ART Anime",
        "repo": "AiAF/D-ART-18DART5_LoRA_Flux1",
        "weights": "D-ART-Flux1.safetensors",
        "trigger_word": "D-ART \(Artist\), @18dart5, @18dart3, @18dart2, and/or @18dart1"    
    },
    #1
    {
        "image": "https://huggingface.co/strangerzonehf/CMS-3D-Art/resolve/main/images/33.png",
        "title": "CMS 3D Art",
        "repo": "strangerzonehf/CMS-3D-Art",
        "weights": "CMS-3D-Art.safetensors",
        "trigger_word": "CMS 3D Art"    
    },
    #2
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Ultimate-LoRA-Collection/resolve/main/images/image.png",
        "title": "AWPortraitCN2",
        "repo": "Shakker-Labs/AWPortraitCN2",
        "weights": "AWPortraitCN_2.safetensors",
        "trigger_word": ""    
    },
    #3
    {
        "image": "https://huggingface.co/strangerzonehf/3d-Station-Toon/resolve/main/images/5555.png",
        "title": "3d Station Toon",
        "repo": "strangerzonehf/3d-Station-Toon",
        "weights": "3d station toon.safetensors",
        "trigger_word": "3d station toon"    
    },
    #4
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Master-Claymation/resolve/main/images/6.png",
        "title": "Master Claymation",
        "repo": "strangerzonehf/Flux-Master-Claymation",
        "weights": "Master-Claymation.safetensors",
        "trigger_word": "Master Claymation"    
    },
    #5
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Ultimate-LoRA-Collection/resolve/main/images/image2.png",
        "title": "AWPortraitCN",
        "repo": "Shakker-Labs/AWPortraitCN",
        "weights": "AWPortraitCN.safetensors",
        "trigger_word": ""    
    },
    #6
    {
        "image": "https://huggingface.co/strangerzonehf/333-PRO/resolve/main/images/111.png",
        "title": "333 PRO",
        "repo": "strangerzonehf/333-PRO",
        "weights": "333-Pro.safetensors",
        "trigger_word": "333 Pro Sketch"    
    },
    #7
    {
        "image": "https://huggingface.co/strangerzonehf/BnW-Expressions-Flux/resolve/main/images/111.png",
        "title": "BnW Expressions",
        "repo": "strangerzonehf/BnW-Expressions-Flux",
        "weights": "BnW-Expressions.safetensors",
        "trigger_word": "BnW Expressions"    
    },
    #8
    {
        "image": "https://huggingface.co/strangerzonehf/2DAura-Flux/resolve/main/images/666.png",
        "title": "2DAura Flux",
        "repo": "strangerzonehf/2DAura-Flux",
        "weights": "2DAura.safetensors",
        "trigger_word": "2D Aura"    
    },
    #9
    {
        "image": "https://huggingface.co/strangerzonehf/FallenArt-Flux/resolve/main/images/222.png",
        "title": "Fallen Art",
        "repo": "strangerzonehf/FallenArt-Flux",
        "weights": "FallenArt.safetensors",
        "trigger_word": "Fallen Art"    
    },
    #10
    {
        "image": "https://huggingface.co/strangerzonehf/Cardboard-v2-Flux/resolve/main/images/111.png",
        "title": "Cardboard-v2-Flux",
        "repo": "strangerzonehf/Cardboard-v2-Flux",
        "weights": "Cardboard-v2.safetensors",
        "trigger_word": "Cardboard v2"    
    },
    #11
    {
        "image": "https://huggingface.co/strangerzonehf/Qx-Art/resolve/main/images/2.png",
        "title": "Qx Art",
        "repo": "strangerzonehf/Qx-Art",
        "weights": "Qx-Art.safetensors",
        "trigger_word": "Qx-Art"    
    },
    #12
    {
        "image": "https://huggingface.co/strangerzonehf/Realism-H6-Flux/resolve/main/images/3333.png",
        "title": "Realism H6 Flux",
        "repo": "strangerzonehf/Realism-H6-Flux",
        "weights": "Realism H6.safetensors",
        "trigger_word": "Realism H6"    
    },
    #13
    {
        "image": "https://huggingface.co/strangerzonehf/Qs-Sketch/resolve/main/images/5.png",
        "title": "Qs Sketch",
        "repo": "strangerzonehf/Qs-Sketch",
        "weights": "Qs Sketch.safetensors",
        "trigger_word": "Qs Sketch"    
    },
    #14
    {
        "image": "https://huggingface.co/strangerzonehf/Qc-Sketch/resolve/main/images/1.png",
        "title": "Qc Sketch",
        "repo": "strangerzonehf/Qc-Sketch",
        "weights": "Qc-Sketch.safetensors",
        "trigger_word": "Qc-Sketch"    
    },
    #15
    {
        "image": "https://huggingface.co/strangerzonehf/Qw-Sketch/resolve/main/images/4.png",
        "title": "Qw Sketch",
        "repo": "strangerzonehf/Qw-Sketch",
        "weights": "Qw-Sketch.safetensors",
        "trigger_word": "Qw Sketch"    
    },
    #16
    {
        "image": "https://huggingface.co/strangerzonehf/Thread-of-Art-Flux/resolve/main/images/1111.png",
        "title": "Thread of Art",
        "repo": "strangerzonehf/Thread-of-Art-Flux",
        "weights": "Thread-of-Art.safetensors",
        "trigger_word": "Thread of Art"    
    },
    #17
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Automotive-X2-LoRA/resolve/main/images/1.png",
        "title": "Automotive X2",
        "repo": "strangerzonehf/Flux-Automotive-X2-LoRA",
        "weights": "Automotive-X2.safetensors",
        "trigger_word": "Automotive X2"    
    },
    #18
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Automotive-X1-LoRA/resolve/main/images/3.png",
        "title": "Automotive X1",
        "repo": "strangerzonehf/Flux-Automotive-X1-LoRA",
        "weights": "Automotive-X1.safetensors",
        "trigger_word": "Automotive X1"    
    },
    #19
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-3DXL-Partfile-0001/resolve/main/images/4.png",
        "title": "3DXLP1",
        "repo": "strangerzonehf/Flux-3DXL-Partfile-0001",
        "weights": "3DXLP1.safetensors",
        "trigger_word": "3DXLP1"    
    },
    #20
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-3DXL-Partfile-0002/resolve/main/images/44.png",
        "title": "3DXLP2",
        "repo": "strangerzonehf/Flux-3DXL-Partfile-0002",
        "weights": "3DXLP2.safetensors",
        "trigger_word": "3DXLP2"    
    },
    #21
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-3DXL-Partfile-0003/resolve/main/images/222.png",
        "title": "3DXLP3",
        "repo": "strangerzonehf/Flux-3DXL-Partfile-0003",
        "weights": "3DXLP3.safetensors",
        "trigger_word": "3DXLP3"    
    },
    #22
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-3DXL-Partfile-0004/resolve/main/images/4444.png",
        "title": "3DXLP4",
        "repo": "strangerzonehf/Flux-3DXL-Partfile-0004",
        "weights": "3DXLP4.safetensors",
        "trigger_word": "3DXLP4"    
    },
    #23
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Isometric-3D-Cinematography/resolve/main/images/IS1.png",
        "title": "Isometric 3D",
        "repo": "strangerzonehf/Flux-Isometric-3D-Cinematography",
        "weights": "Isometric-3D-Cinematography.safetensors",
        "trigger_word": "Isometric 3D Cinematography"    
    },
    #27
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Microworld-NFT-LoRA/resolve/main/images/4.png",
        "title": "Microworld NFT",
        "repo": "strangerzonehf/Flux-Microworld-NFT-LoRA",
        "weights": "Microworld-NFT.safetensors",
        "trigger_word": "Microworld NFT"    
    },
    #28
    {
        "image": "https://huggingface.co/strangerzonehf/NFT-Plus-InsideOut-Perspective/resolve/main/images/2.png",
        "title": "NFT ++",
        "repo": "strangerzonehf/NFT-Plus-InsideOut-Perspective",
        "weights": "NFT-Plus-InsideOut-Perspective.safetensors",
        "trigger_word": "NFT ++"    
    },
    #29
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Ghibli-Art-LoRA/resolve/main/images/8.png",
        "title": "Half Ghibli",
        "repo": "strangerzonehf/Flux-Ghibli-Art-LoRA",
        "weights": "Ghibli-Art.safetensors",
        "trigger_word": "Ghibli Art"    
    },
    #80
    {
        "image": "https://huggingface.co/fffiloni/dark-pointillisme/resolve/main/images/example_xhq6z88qq.png",
        "title": "Dark Pointillisme",
        "repo": "fffiloni/dark-pointillisme",
        "weights": "dark-pointillisme.safetensors",
        "trigger_word": "in the style of TOK"    
    }, 
        
    #30
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Midjourney-Mix-LoRA/resolve/main/images/mj10.png",
        "title": "Midjourney Mix",
        "repo": "strangerzonehf/Flux-Midjourney-Mix-LoRA",
        "weights": "midjourney-mix.safetensors",
        "trigger_word": "midjourney mix"    
    },
    #31
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Creative-Stocks-LoRA/resolve/main/images/6.png",
        "title": "Creative Stocks",
        "repo": "strangerzonehf/Flux-Creative-Stocks-LoRA",
        "weights": "Creative-Stocks.safetensors",
        "trigger_word": "Creative Stocks"    
    },
    #82
    {
        "image": "https://huggingface.co/fffiloni/carbo-800/resolve/main/images/example_sxso69ocl.png",
        "title": "Carbo 800",
        "repo": "fffiloni/carbo-800",
        "weights": "carbo-800.safetensors",
        "trigger_word": "in the style of TOK"    
    }, 
    #32
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Pixel-Background-LoRA/resolve/main/images/2.png",
        "title": "Flux Pixel",
        "repo": "strangerzonehf/Flux-Pixel-Background-LoRA",
        "weights": "Pixel-Background.safetensors",
        "trigger_word": "Pixel Background"    
    },
    #33
    {
        "image": "https://huggingface.co/strangerzonehf/Multi-perspective-Art-Flux/resolve/main/images/1.png",
        "title": "Multi Perspective Art",
        "repo": "strangerzonehf/Multi-perspective-Art-Flux",
        "weights": "Multi-perspective Art .safetensors",
        "trigger_word": "Multi-perspective Art"    
    },
    #34
    {
        "image": "https://huggingface.co/strangerzonehf/Neon-Impressionism-Flux/resolve/main/images/4.png",
        "title": "Neon Impressionism Flux",
        "repo": "strangerzonehf/Neon-Impressionism-Flux",
        "weights": "Neon Impressionism.safetensors",
        "trigger_word": "Neon Impressionism"    
    },
    #35
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-3DXL-Partfile-0006/resolve/main/images/555.png",
        "title": "3DXLP6",
        "repo": "strangerzonehf/Flux-3DXL-Partfile-0006",
        "weights": "3DXLP6.safetensors",
        "trigger_word": "3DXLP6"    
    },
    #36
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-3DXL-Garment-Mannequin/resolve/main/images/2.png",
        "title": "Garment Mannequin",
        "repo": "strangerzonehf/Flux-3DXL-Garment-Mannequin",
        "weights": "3DXL-Mannequin.safetensors",
        "trigger_word": "3DXL Mannequin"    
    },
    #37
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Xmas-3D-LoRA/resolve/main/images/3.png",
        "title": "Xmas 3D",
        "repo": "strangerzonehf/Flux-Xmas-3D-LoRA",
        "weights": "Flux-Xmas-3D-LoRA.safetensors",
        "trigger_word": "Xmas 3D"    
    },
    #38
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Xmas-Chocolate-LoRA/resolve/main/images/2.png",
        "title": "Xmas Chocolate",
        "repo": "strangerzonehf/Flux-Xmas-Chocolate-LoRA",
        "weights": "Flux-Xmas-Chocolate.safetensors",
        "trigger_word": "Xmas Chocolate"    
    },
    #39
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Xmas-Isometric-Kit-LoRA/resolve/main/images/4.png",
        "title": "Xmas Isometric Kit",
        "repo": "strangerzonehf/Flux-Xmas-Isometric-Kit-LoRA",
        "weights": "Xmas-Isometric-Kit.safetensors",
        "trigger_word": "Xmas Isometric Kit"    
    },
    #40
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Isometric-Site-LoRA/resolve/main/images/1.png",
        "title": "Flux Isometric Site",
        "repo": "strangerzonehf/Flux-Isometric-Site-LoRA",
        "weights": "Isometric-Building.safetensors",
        "trigger_word": "Isometric Building"    
    },
    #41
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-NFT-Art99-LoRA/resolve/main/images/2.png",
        "title": "NFT Art 99",
        "repo": "strangerzonehf/Flux-NFT-Art99-LoRA",
        "weights": "NFT-Art-99.safetensors",
        "trigger_word": "NFT Art 99"    
    },
    #42
    {
        "image": "https://huggingface.co/strangerzonehf/2021-Art-Flux/resolve/main/images/2222.png",
        "title": "2021 Art",
        "repo": "strangerzonehf/2021-Art-Flux",
        "weights": "2021-Art.safetensors",
        "trigger_word": "2021 Art"    
    },
    #44
    {
        "image": "https://huggingface.co/strangerzonehf/New-Journey-Art-Flux/resolve/main/images/3333.png",
        "title": "New Journey Art",
        "repo": "strangerzonehf/New-Journey-Art-Flux",
        "weights": "New-Journey-Art.safetensors",
        "trigger_word": "New Journey Art"    
    },
    #45
    {
        "image": "https://huggingface.co/strangerzonehf/Casual-Pencil-Pro/resolve/main/images/333.png",
        "title": "Casual Pencil",
        "repo": "strangerzonehf/Casual-Pencil-Pro",
        "weights": "CasualPencil.safetensors",
        "trigger_word": "Casual Pencil"    
    },
    #46
    {
        "image": "https://huggingface.co/strangerzonehf/Real-Claymation/resolve/main/images/1.png",
        "title": "Real Claymation",
        "repo": "strangerzonehf/Real-Claymation",
        "weights": "Real-Claymation.safetensors",
        "trigger_word": "Real Claymation"    
    },
    #47
    {
        "image": "https://huggingface.co/strangerzonehf/Embroidery-Art-Flux/resolve/main/images/6.png",
        "title": "Embroidery Art",
        "repo": "strangerzonehf/Embroidery-Art-Flux",
        "weights": "embroidery art.safetensors",
        "trigger_word": "embroidery art"    
    },
    #48
    {
        "image": "https://huggingface.co/strangerzonehf/Whaaaattttt-Flux/resolve/main/images/10.png",
        "title": "Whaaattt Art",
        "repo": "strangerzonehf/Whaaaattttt-Flux",
        "weights": "Whaaattt Art.safetensors",
        "trigger_word": "Whaaattt Art"    
    },
    #49
    {
        "image": "https://huggingface.co/strangerzonehf/Oil-Wall-Art-Flux/resolve/main/images/1.png",
        "title": "Oil Wall Art Flux",
        "repo": "strangerzonehf/Oil-Wall-Art-Flux",
        "weights": "oil-art.safetensors",
        "trigger_word": "oil art"    
    },
    #50
    {
        "image": "https://huggingface.co/fffiloni/deep-blue-v2/resolve/main/images/example_0o2puhiae.png",
        "title": "Deep Blue",
        "repo": "fffiloni/deep-blue-v2",
        "weights": "deep-blue-v2.safetensors",
        "trigger_word": "deep blue, white lines illustration"    
    },
    #51
    {
        "image": "https://huggingface.co/fffiloni/cozy-book-800/resolve/main/images/example_zza0rj1uq.png",
        "title": "Cozy Book 800",
        "repo": "fffiloni/cozy-book-800",
        "weights": "cozy-book-800.safetensors",
        "trigger_word": "in the style of TOK"    
    },
    #52
    {
        "image": "https://huggingface.co/kudzueye/Boreal/resolve/main/images/ComfyUI_00822_.png",
        "title": "Boreal",
        "repo": "kudzueye/Boreal",
        "weights": "boreal-flux-dev-lora-v04_1000_steps.safetensors",
        "trigger_word": "photo"    
    },
    #55
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-St-Shot/resolve/main/images/1.png",
        "title": "Portrait",
        "repo": "strangerzonehf/Flux-St-Shot",
        "weights": "st portrait.safetensors",
        "trigger_word": "st portrait"    
    },
    #56
    {
        "image": "https://huggingface.co/strangerzonehf/Pixelo-Flux/resolve/main/images/2.png",
        "title": "Better Pixel",
        "repo": "strangerzonehf/Pixelo-Flux",
        "weights": "pxl.safetensors",
        "trigger_word": "better pixel"    
    },
    #57
    {
        "image": "https://huggingface.co/strangerzonehf/cinematicShot-Pics-Flux/resolve/main/images/4.png",
        "title": "Cinematic Shot",
        "repo": "strangerzonehf/cinematicShot-Pics-Flux",
        "weights": "cinematic-shot.safetensors",
        "trigger_word": "cinematic shot"    
    },
    #58
    {
        "image": "https://huggingface.co/strangerzonehf/cinematicShot-Pics-Flux/resolve/main/images/4.png",
        "title": "Cinematic Shot",
        "repo": "strangerzonehf/cinematicShot-Pics-Flux",
        "weights": "cinematic-shot.safetensors",
        "trigger_word": "cinematic shot"    
    },
    #59
    {
        "image": "https://huggingface.co/strangerzonehf/OleART/resolve/main/images/1.png",
        "title": "OleART",
        "repo": "strangerzonehf/OleART",
        "weights": "ole.safetensors",
        "trigger_word": "ole art"    
    },
    #62
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-3D-Realism-LoRA/resolve/main/images/4.png",
        "title": "3D Realism",
        "repo": "strangerzonehf/Flux-3D-Realism-LoRA",
        "weights": "3D-Realism.safetensors",
        "trigger_word": "3D Realism"    
    },
    #64
    {
        "image": "https://huggingface.co/saurabhswami/HumaneArt/resolve/main/images/1.jpg",
        "title": "Humane Art",
        "repo": "saurabhswami/HumaneArt",
        "weights": "humaneart.safetensors",
        "trigger_word": "HumaneArt"    
    },
    #65
    {
        "image": "https://huggingface.co/gokaygokay/Flux-Engrave-LoRA/resolve/main/images/image5.jpg",
        "title": "Flux Engrave",
        "repo": "gokaygokay/Flux-Engrave-LoRA",
        "weights": "engrave.safetensors",
        "trigger_word": "NGRVNG, engrave, <<your prompt>>"    
    },
    #66
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-SuperPortrait-v2-LoRA/resolve/main/images/1.png",
        "title": "Super Portrait v2",
        "repo": "strangerzonehf/Flux-SuperPortrait-v2-LoRA",
        "weights": "Super-Portrait-v2.safetensors",
        "trigger_word": "Super Portrait v2"    
    },
    #81
    {
        "image": "https://huggingface.co/fffiloni/cute-comic-800/resolve/main/images/example_geha6pn5l.png",
        "title": "Cute Comic",
        "repo": "fffiloni/cute-comic-800",
        "weights": "cute-comic-800.safetensors",
        "trigger_word": "in the style of TOK"    
    }, 
    #67
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Mexican-CPunk-LoRA/resolve/main/images/3.png",
        "title": "Mexican Cyberpunk",
        "repo": "strangerzonehf/Flux-Mexican-CPunk-LoRA",
        "weights": "Mexican-Cyberpunk.safetensors",
        "trigger_word": "Mexican Cyberpunk"    
    },    
    #68
    {
        "image": "https://huggingface.co/glif-loradex-trainer/steam6332_flux_dev_pink_glitter_dust/resolve/main/samples/1731082349941__000001800_0.jpg",
        "title": "Pink Glitter Dust",
        "repo": "glif-loradex-trainer/steam6332_flux_dev_pink_glitter_dust",
        "weights": "flux_dev_pink_glitter_dust.safetensors",
        "trigger_word": "pink-glitter-dust"    
    },    
    #69
    {
        "image": "https://huggingface.co/glif-loradex-trainer/dham_dham_osteology2/resolve/main/samples/1731452917113__000003000_1.jpg",
        "title": "Osteology 2",
        "repo": "glif-loradex-trainer/dham_dham_osteology2",
        "weights": "dham_osteology2.safetensors",
        "trigger_word": "TOK"    
    },
    #70
    {
        "image": "https://huggingface.co/glif-loradex-trainer/chrysoliteop_Chrysolite_Light_Leaks/resolve/main/samples/1732346113846__000003000_1.jpg",
        "title": "Chrysolite Light Leaks",
        "repo": "glif-loradex-trainer/chrysoliteop_Chrysolite_Light_Leaks",
        "weights": "Chrysolite_Light_Leaks.safetensors",
        "trigger_word": "LTLKS_CHRYLT"    
    },  
    #71
    {
        "image": "https://huggingface.co/AIGCDuckBoss/fluxLora_cute3DModel/resolve/main/images/1.png",
        "title": "Cute 3D",
        "repo": "AIGCDuckBoss/fluxLora_cute3DModel",
        "weights": "flux_cute3DModel.safetensors",
        "trigger_word": "3d illustration"    
    },  
    #72
    {
        "image": "https://huggingface.co/alvdansen/flux_film_foto/resolve/main/images/ComfyUI_00247_.png",
        "title": "Flmft Photo Style",
        "repo": "alvdansen/flux_film_foto",
        "weights": "araminta_k_flux_film_foto.safetensors",
        "trigger_word": "flmft photo style"    
    },  
    #73
    {
        "image": "https://huggingface.co/glif/90s-anime-art/resolve/main/images/glif-90s-anime-lora-araminta-k-hgzcnpjlorspm86jhgpl57ph.jpg",
        "title": "90s Anime Art",
        "repo": "glif/90s-anime-art",
        "weights": "flux_dev_anime.safetensors",
        "trigger_word": "90s anime art styles"    
    },  
    #74
    {
        "image": "https://huggingface.co/Datou1111/Slow-Shutter/resolve/main/images/img__00628_.png",
        "title": "Slow Shutter",
        "repo": "Datou1111/Slow-Shutter",
        "weights": "Slow-Shutter.safetensors",
        "trigger_word": "slow shutter photography motion blur"    
    },  
    #75
    {
        "image": "https://huggingface.co/Datou1111/Yoji_Shinkawa/resolve/main/images/img__00198_.png",
        "title": "Mecha Design Yoji Shinkawa",
        "repo": "Datou1111/Yoji_Shinkawa",
        "weights": "Yoji_Shinkawa.safetensors",
        "trigger_word": "Yoji_Shinkawa"    
    },  
    #76
    {
        "image": "https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-Miniature-World/resolve/main/examples/1_0.8.png",
        "title": "Miniature World ",
        "repo": "Shakker-Labs/FLUX.1-dev-LoRA-Miniature-World",
        "weights": "FLUX-dev-lora-Miniature-World.safetensors",
        "trigger_word": "a meticulously crafted miniature scene"    
    }, 
    #77
    {
        "image": "https://huggingface.co/strangerzonehf/Ctoon-Plus-Plus/resolve/main/images/4.png",
        "title": "Ctoon ++",
        "repo": "strangerzonehf/Ctoon-Plus-Plus",
        "weights": "Ctoon++.safetensors",
        "trigger_word": "Ctoon++"    
    }, 
    #78
    {
        "image": "https://huggingface.co/longnthgmedia/flux_lora_meme_v2/resolve/main/samples/1728923568922__000008000_1.jpg",
        "title": "Lora Meme",
        "repo": "longnthgmedia/flux_lora_meme_v2",
        "weights": "flux_lora_meme_v2.safetensors",
        "trigger_word": ""    
    },
    #79
    {
        "image": "https://huggingface.co/fffiloni/greyscale-tiny-town/resolve/main/images/example_ol1f5bbio.png",
        "title": "Greyscale Tiny Town",
        "repo": "fffiloni/greyscale-tiny-town",
        "weights": "greyscale-tiny-town.safetensors",
        "trigger_word": "greyscale drawing"    
    }, 
    #83
    {
        "image": "https://huggingface.co/strangerzonehf/Dls-ART/resolve/main/images/3.png",
        "title": "Dls-ART",
        "repo": "strangerzonehf/Dls-ART",
        "weights": "Dls-Art.safetensors",
        "trigger_word": "Dls-ART"    
    },            
    #84
    {
        "image": "https://huggingface.co/fffiloni/sweet-brush/resolve/main/images/example_om6c5d6bt.png",
        "title": "Sweet Brush Art",
        "repo": "fffiloni/sweet-brush",
        "weights": "sweet-brush.safetensors",
        "trigger_word": "in the style of TOK"    
    },  
    #85
    {
        "image": "https://huggingface.co/glif/l0w-r3z/resolve/main/images/a19d658b-5d4c-45bc-9df6-f2bec54462a5.png",
        "title": "ReZ",
        "repo": "glif/l0w-r3z",
        "weights": "low-rez_000002000.safetensors",
        "trigger_word": "-r3z"    
    },         
     #86
    {
        "image": "https://huggingface.co/glif-loradex-trainer/fabian3000_mspaint1/resolve/main/samples/1731588572064__000002500_0.jpg",
        "title": "MS Paint",
        "repo": "glif-loradex-trainer/fabian3000_mspaint1",
        "weights": "mspaint1.safetensors",
        "trigger_word": "mspaintstyle"    
    },  
     #87
    {
        "image": "https://huggingface.co/glif-loradex-trainer/fab1an_1970sbookcovers/resolve/main/samples/1740488542933__000001500_1.jpg",
        "title": "1970 Book Cover",
        "repo": "glif-loradex-trainer/fab1an_1970sbookcovers",
        "weights": "1970sbookcovers.safetensors",
        "trigger_word": "1970s sci-fi book cover"    
    },         
    #89
    {
        "image": "https://huggingface.co/strangerzonehf/RGART/resolve/main/images/4.png",
        "title": "RG Art",
        "repo": "strangerzonehf/RGART",
        "weights": "RB-Art.safetensors",
        "trigger_word": "RG Art"    
    },  
    #90
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Super-Paint-LoRA/resolve/main/images/4.png",
        "title": "Super Paint",
        "repo": "strangerzonehf/Flux-Super-Paint-LoRA",
        "weights": "Super-Paint.safetensors",
        "trigger_word": "Super Paint"    
    },    
    #91
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Sketch-Flat-LoRA/resolve/main/images/11.png",
        "title": "Sketch Flat",
        "repo": "strangerzonehf/Flux-Sketch-Flat-LoRA",
        "weights": "Sketch-Flat.safetensors",
        "trigger_word": "Sketch Flat"    
    },                  
    #92
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Claude-Art/resolve/main/images/3.png",
        "title": "Flux Claude Art",
        "repo": "strangerzonehf/Flux-Claude-Art",
        "weights": "claude-art.safetensors",
        "trigger_word": "claude art"    
    },  
    #93
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-Sketch-Scribble-LoRA/resolve/main/images/1.png",
        "title": "Sketch Scribble",
        "repo": "strangerzonehf/Flux-Sketch-Scribble-LoRA",
        "weights": "Sketch-Scribble.safetensors",
        "trigger_word": "Sketch Scribble"    
    },  
    #94
    {
        "image": "https://huggingface.co/strangerzonehf/Flux-3D-Emojies-LoRA/resolve/main/images/2.png",
        "title": "3D Emojies",
        "repo": "strangerzonehf/Flux-3D-Emojies-LoRA",
        "weights": "Flux-3D-Emojies-Mation.safetensors",
        "trigger_word": "3D Emojies"    
    },  
    #95
    {
        "image": "https://huggingface.co/igorriti/flux-360/resolve/main/sample.png",
        "title": "360 Panoramas ",
        "repo": "strangerzonehf/Flux-3D-Emojies-LoRA",
        "weights": "lora.safetensors",
        "trigger_word": "TOK"    
    },  
    #96
    {
        "image": "https://huggingface.co/InstantX/FLUX.1-dev-LoRA-Makoto-Shinkai/resolve/main/images/example_1.png",
        "title": "Makoto Shinkai",
        "repo": "strangerzonehf/Flux-3D-Emojies-LoRA",
        "weights": "Makoto_Shinkai_style.safetensors",
        "trigger_word": "Makoto Shinkai Style"    
    },  
    #97
    {
        "image": "https://huggingface.co/fffiloni/wooly-play-doh/resolve/main/images/example_j0s9hnq2s.png",
        "title": "Wooly Play Doh",
        "repo": "fffiloni/wooly-play-doh",
        "weights": "wooly-play-doh.safetensors",
        "trigger_word": "in the style of TOK"    
    },  
    #98
    {
        "image": "https://huggingface.co/davidrd123/lora-Kirchner-flux/resolve/main/assets/image_0_0.png",
        "title": "Lora Kirchner",
        "repo": "davidrd123/lora-Kirchner-flux",
        "weights": "pytorch_lora_weights.safetensors",
        "trigger_word": "elk_style"    
    },  
    #99
    {
        "image": "https://huggingface.co/AiAF/Urcarta-ucrt_LoRA_Flux1/resolve/main/images/1000671824.png",
        "title": "Urcarta",
        "repo": "AiAF/Urcarta-ucrt_LoRA_Flux1",
        "weights": "Urcarta-urct-裏方-Flux1.safetensors",
        "trigger_word": "Urcarta \(Artist\), @urct, 裏方 \(芸術家\)"    
    },  
    #100
    {
        "image": "https://huggingface.co/mujibanget/vector-illustration/resolve/main/images/7da62627-da2a-4505-bb4e-a38dbf3da45b.png",
        "title": "Vector Illustration",
        "repo": "mujibanget/vector-illustration",
        "weights": "lora-000002.TA_trained.safetensors",
        "trigger_word": "mujibvector, vector"    
    }, 
    #101
    {
        "image": "https://huggingface.co/glif-loradex-trainer/goldenark__WaterColorSketchStyle/resolve/main/samples/1727240451672__000003000_0.jpg",
        "title": "Water Color Sketch",
        "repo": "glif-loradex-trainer/goldenark__WaterColorSketchStyle",
        "weights": "WaterColorSketchStyle.safetensors",
        "trigger_word": "WaterColorSketchStyle"    
    }, 
]

#--------------------------------------------------Model Initialization-----------------------------------------------------------------------------------------#

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = "black-forest-labs/FLUX.1-dev"

#TAEF1 is very tiny autoencoder which uses the same "latent API" as FLUX.1's VAE. FLUX.1 is useful for real-time previewing of the FLUX.1 generation process.#
taef1 = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=dtype).to(device)
good_vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype).to(device)
pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype, vae=taef1).to(device)
pipe_i2i = AutoPipelineForImage2Image.from_pretrained(base_model,
                                                      vae=good_vae,
                                                      transformer=pipe.transformer,
                                                      text_encoder=pipe.text_encoder,
                                                      tokenizer=pipe.tokenizer,
                                                      text_encoder_2=pipe.text_encoder_2,
                                                      tokenizer_2=pipe.tokenizer_2,
                                                      torch_dtype=dtype
                                                     )

MAX_SEED = 2**32-1

pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)

class calculateDuration:
    def __init__(self, activity_name=""):
        self.activity_name = activity_name

    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        if self.activity_name:
            print(f"Elapsed time for {self.activity_name}: {self.elapsed_time:.6f} seconds")
        else:
            print(f"Elapsed time: {self.elapsed_time:.6f} seconds")

def update_selection(evt: gr.SelectData, width, height):
    selected_lora = loras[evt.index]
    new_placeholder = f"Type a prompt for {selected_lora['title']}"
    lora_repo = selected_lora["repo"]
    updated_text = f"### Selected: [{lora_repo}](https://huggingface.co/{lora_repo}) ✅"
    if "aspect" in selected_lora:
        if selected_lora["aspect"] == "portrait":
            width = 768
            height = 1024
        elif selected_lora["aspect"] == "landscape":
            width = 1024
            height = 768
        else:
            width = 1024
            height = 1024
    return (
        gr.update(placeholder=new_placeholder),
        updated_text,
        evt.index,
        width,
        height,
    )

@spaces.GPU(duration=100)
def generate_image(prompt_mash, steps, seed, cfg_scale, width, height, lora_scale, progress):
    pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with calculateDuration("Generating image"):
        # Generate image
        for img in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
            prompt=prompt_mash,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            generator=generator,
            joint_attention_kwargs={"scale": lora_scale},
            output_type="pil",
            good_vae=good_vae,
        ):
            yield img

def generate_image_to_image(prompt_mash, image_input_path, image_strength, steps, cfg_scale, width, height, lora_scale, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    pipe_i2i.to("cuda")
    image_input = load_image(image_input_path)
    final_image = pipe_i2i(
        prompt=prompt_mash,
        image=image_input,
        strength=image_strength,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        width=width,
        height=height,
        generator=generator,
        joint_attention_kwargs={"scale": lora_scale},
        output_type="pil",
    ).images[0]
    return final_image 

@spaces.GPU(duration=100)
def run_lora(prompt, image_input, image_strength, cfg_scale, steps, selected_index, randomize_seed, seed, width, height, lora_scale, progress=gr.Progress(track_tqdm=True)):
    if selected_index is None:
        raise gr.Error("You must select a LoRA before proceeding.🧨")
    selected_lora = loras[selected_index]
    lora_path = selected_lora["repo"]
    trigger_word = selected_lora["trigger_word"]
    if(trigger_word):
        if "trigger_position" in selected_lora:
            if selected_lora["trigger_position"] == "prepend":
                prompt_mash = f"{trigger_word} {prompt}"
            else:
                prompt_mash = f"{prompt} {trigger_word}"
        else:
            prompt_mash = f"{trigger_word} {prompt}"
    else:
        prompt_mash = prompt

    with calculateDuration("Unloading LoRA"):
        pipe.unload_lora_weights()
        pipe_i2i.unload_lora_weights()
        
    #LoRA weights flow
    with calculateDuration(f"Loading LoRA weights for {selected_lora['title']}"):
        pipe_to_use = pipe_i2i if image_input is not None else pipe
        weight_name = selected_lora.get("weights", None)
        
        pipe_to_use.load_lora_weights(
            lora_path, 
            weight_name=weight_name, 
            low_cpu_mem_usage=True
        )
            
    with calculateDuration("Randomizing seed"):
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
            
    if(image_input is not None):
        
        final_image = generate_image_to_image(prompt_mash, image_input, image_strength, steps, cfg_scale, width, height, lora_scale, seed)
        yield final_image, seed, gr.update(visible=False)
    else:
        image_generator = generate_image(prompt_mash, steps, seed, cfg_scale, width, height, lora_scale, progress)
    
        final_image = None
        step_counter = 0
        for image in image_generator:
            step_counter+=1
            final_image = image
            progress_bar = f'<div class="progress-container"><div class="progress-bar" style="--current: {step_counter}; --total: {steps};"></div></div>'
            yield image, seed, gr.update(value=progress_bar, visible=True)
            
        yield final_image, seed, gr.update(value=progress_bar, visible=False)
        
def get_huggingface_safetensors(link):
  split_link = link.split("/")
  if(len(split_link) == 2):
            model_card = ModelCard.load(link)
            base_model = model_card.data.get("base_model")
            print(base_model)
      
            #Allows Both
            if((base_model != "black-forest-labs/FLUX.1-dev") and (base_model != "black-forest-labs/FLUX.1-schnell")):
                raise Exception("Flux LoRA Not Found!")
                
            # Only allow "black-forest-labs/FLUX.1-dev"
            #if base_model != "black-forest-labs/FLUX.1-dev":
                #raise Exception("Only FLUX.1-dev is supported, other LoRA models are not allowed!")
                
            image_path = model_card.data.get("widget", [{}])[0].get("output", {}).get("url", None)
            trigger_word = model_card.data.get("instance_prompt", "")
            image_url = f"https://huggingface.co/{link}/resolve/main/{image_path}" if image_path else None
            fs = HfFileSystem()
            try:
                list_of_files = fs.ls(link, detail=False)
                for file in list_of_files:
                    if(file.endswith(".safetensors")):
                        safetensors_name = file.split("/")[-1]
                    if (not image_url and file.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
                      image_elements = file.split("/")
                      image_url = f"https://huggingface.co/{link}/resolve/main/{image_elements[-1]}"
            except Exception as e:
              print(e)
              gr.Warning(f"You didn't include a link neither a valid Hugging Face repository with a *.safetensors LoRA")
              raise Exception(f"You didn't include a link neither a valid Hugging Face repository with a *.safetensors LoRA")
            return split_link[1], link, safetensors_name, trigger_word, image_url

def check_custom_model(link):
    if(link.startswith("https://")):
        if(link.startswith("https://huggingface.co") or link.startswith("https://www.huggingface.co")):
            link_split = link.split("huggingface.co/")
            return get_huggingface_safetensors(link_split[1])
    else: 
        return get_huggingface_safetensors(link)

def add_custom_lora(custom_lora):
    global loras
    if(custom_lora):
        try:
            title, repo, path, trigger_word, image = check_custom_model(custom_lora)
            print(f"Loaded custom LoRA: {repo}")
            card = f'''
            <div class="custom_lora_card">
              <span>Loaded custom LoRA:</span>
              <div class="card_internal">
                <img src="{image}" />
                <div>
                    <h3>{title}</h3>
                    <small>{"Using: <code><b>"+trigger_word+"</code></b> as the trigger word" if trigger_word else "No trigger word found. If there's a trigger word, include it in your prompt"}<br></small>
                </div>
              </div>
            </div>
            '''
            existing_item_index = next((index for (index, item) in enumerate(loras) if item['repo'] == repo), None)
            if(not existing_item_index):
                new_item = {
                    "image": image,
                    "title": title,
                    "repo": repo,
                    "weights": path,
                    "trigger_word": trigger_word
                }
                print(new_item)
                existing_item_index = len(loras)
                loras.append(new_item)
        
            return gr.update(visible=True, value=card), gr.update(visible=True), gr.Gallery(selected_index=None), f"Custom: {path}", existing_item_index, trigger_word
        except Exception as e:
            gr.Warning(f"Invalid LoRA: either you entered an invalid link, or a non-FLUX LoRA")
            return gr.update(visible=True, value=f"Invalid LoRA: either you entered an invalid link, a non-FLUX LoRA"), gr.update(visible=False), gr.update(), "", None, ""
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(), "", None, ""

def remove_custom_lora():
    return gr.update(visible=False), gr.update(visible=False), gr.update(), "", None, ""

run_lora.zerogpu = True

css = '''
#gen_btn{height: 100%}
#gen_column{align-self: stretch}
#title{text-align: center}
#title h1{font-size: 3em; display:inline-flex; align-items:center}
#title img{width: 100px; margin-right: 0.5em}
#gallery .grid-wrap{height: 10vh}
#lora_list{background: var(--block-background-fill);padding: 0 1em .3em; font-size: 90%}
.card_internal{display: flex;height: 100px;margin-top: .5em}
.card_internal img{margin-right: 1em}
.styler{--form-gap-width: 0px !important}
#progress{height:30px}
#progress .generating{display:none}
.progress-container {width: 100%;height: 30px;background-color: #f0f0f0;border-radius: 15px;overflow: hidden;margin-bottom: 20px}
.progress-bar {height: 100%;background-color: #4f46e5;width: calc(var(--current) / var(--total) * 100%);transition: width 0.5s ease-in-out}
'''

with gr.Blocks(theme=gr.themes.Soft(), css=css, delete_cache=(60, 60)) as app:
    title = gr.HTML(
        """<h1>FLUX LoRA DLC2🔥</h1>""",
        elem_id="title",
    )
    selected_index = gr.State(None)
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Prompt", lines=1, placeholder=":/ choose the LoRA and type the prompt ")
        with gr.Column(scale=1, elem_id="gen_column"):
            generate_button = gr.Button("Generate", variant="primary", elem_id="gen_btn")
    with gr.Row():
        with gr.Column():
            selected_info = gr.Markdown("")
            gallery = gr.Gallery(
                [(item["image"], item["title"]) for item in loras],
                label="100+ LoRA DLC's",
                allow_preview=False,
                columns=3,
                elem_id="gallery",
                show_share_button=False
            )
            with gr.Group():
                custom_lora = gr.Textbox(label="Enter Custom LoRA", placeholder="prithivMLmods/Canopus-LoRA-Flux-Anime")
                gr.Markdown("[Check the list of FLUX LoRA's](https://huggingface.co/models?other=base_model:adapter:black-forest-labs/FLUX.1-dev)", elem_id="lora_list")
            custom_lora_info = gr.HTML(visible=False)
            custom_lora_button = gr.Button("Remove custom LoRA", visible=False)
        with gr.Column():
            progress_bar = gr.Markdown(elem_id="progress",visible=False)
            result = gr.Image(label="Generated Image", format="png")

    with gr.Row():
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                input_image = gr.Image(label="Input image", type="filepath")
                image_strength = gr.Slider(label="Denoise Strength", info="Lower means more image influence", minimum=0.1, maximum=1.0, step=0.01, value=0.75)
            with gr.Column():
                with gr.Row():
                    cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=20, step=0.5, value=3.5)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=28)
                
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=256, maximum=1536, step=64, value=1024)
                    height = gr.Slider(label="Height", minimum=256, maximum=1536, step=64, value=1024)
                
                with gr.Row():
                    randomize_seed = gr.Checkbox(True, label="Randomize seed")
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, randomize=True)
                    lora_scale = gr.Slider(label="LoRA Scale", minimum=0, maximum=3, step=0.01, value=0.95)

    gallery.select(
        update_selection,
        inputs=[width, height],
        outputs=[prompt, selected_info, selected_index, width, height]
    )
    custom_lora.input(
        add_custom_lora,
        inputs=[custom_lora],
        outputs=[custom_lora_info, custom_lora_button, gallery, selected_info, selected_index, prompt]
    )
    custom_lora_button.click(
        remove_custom_lora,
        outputs=[custom_lora_info, custom_lora_button, gallery, selected_info, selected_index, custom_lora]
    )
    gr.on(
        triggers=[generate_button.click, prompt.submit],
        fn=run_lora,
        inputs=[prompt, input_image, image_strength, cfg_scale, steps, selected_index, randomize_seed, seed, width, height, lora_scale],
        outputs=[result, seed, progress_bar]
    )

app.queue()
app.launch(ssr_mode=False)