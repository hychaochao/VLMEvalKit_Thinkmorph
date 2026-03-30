# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Batch inference support for ThinkMorph in VLMEvalKit.

Extends InterleaveInferencer with batch processing capabilities by leveraging
the model's native batch support through proper initialization of kv_lens and
ropes as lists with length = batch_size.

Reference: ThinkMorph-Infer/infer/batch_inferencer.py and
           ThinkMorph-Infer/infer/batch_interleaved_inferencer.py
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
import torch
from tqdm import tqdm

from .data.data_utils import pil_img2rgb
from .modeling.bagel.qwen2_navit import NaiveCache
from .inferencer import (
    InterleaveInferencer,
    VLM_THINK_SYSTEM_PROMPT,
    GEN_THINK_SYSTEM_PROMPT,
)

IMAGE_REQUEST_MARKERS = ["<image_start>"]


class BatchInterleaveInferencer(InterleaveInferencer):
    """Batch inference extending InterleaveInferencer.

    Processes multiple samples in parallel by initializing kv_lens and ropes
    as lists of length batch_size.  Supports optional multi-GPU CFG parallel
    when ``cfg_parallel_workers`` is provided.
    """

    def __init__(
        self,
        model,
        vae_model,
        tokenizer,
        vae_transform,
        vit_transform,
        new_token_ids,
        cfg_parallel_workers: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        super().__init__(
            model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids,
        )
        self.cfg_parallel_workers = cfg_parallel_workers or {}
        self.cfg_parallel_executor = None
        if len(self.cfg_parallel_workers) > 1:
            self.cfg_parallel_executor = ThreadPoolExecutor(
                max_workers=len(self.cfg_parallel_workers),
                thread_name_prefix="cfg_parallel",
            )

    # ------------------------------------------------------------------ #
    #  Context initialisation
    # ------------------------------------------------------------------ #

    def init_gen_context(self, batch_size: int = 1) -> Dict:
        gen_context = {
            "kv_lens": [0] * batch_size,
            "ropes": [0] * batch_size,
            "past_key_values": NaiveCache(
                self.model.config.llm_config.num_hidden_layers
            ),
        }
        return gen_context

    # ------------------------------------------------------------------ #
    #  Batch text generation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def gen_text_batch(
        self,
        gen_context: Dict,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
    ) -> List[str]:
        """Generate text for all samples in the batch."""
        past_key_values = deepcopy(gen_context["past_key_values"])
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input = self.model.prepare_start_tokens(
            kv_lens, ropes, self.new_token_ids
        )

        output_sequences = self.model.batch_generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids["eos_token_id"],
            **generation_input,
        )

        texts = []
        for seq in output_sequences:
            if len(seq) > 0:
                decoded = self.tokenizer.decode(seq)
                decoded = decoded.split("<|im_end|>")[0]
                if "<|im_start|>" in decoded:
                    decoded = decoded.split("<|im_start|>")[1]
                texts.append(decoded)
            else:
                texts.append("")

        return texts

    # ------------------------------------------------------------------ #
    #  Batch context updates
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _encode_vae_images_shared_noise(
        self,
        padded_images: torch.Tensor,
        noise_seed: Optional[int],
    ) -> torch.Tensor:
        device = padded_images.device
        padded_images = padded_images.to(device=device, dtype=torch.bfloat16)
        if noise_seed is None or padded_images.shape[0] <= 1:
            return self.vae_model.encode(padded_images)

        latent_stats = self.vae_model.encoder(padded_images)
        mean, logvar = torch.chunk(latent_stats, 2, dim=1)
        std = torch.exp(0.5 * logvar)

        generator = torch.Generator(device=str(device))
        generator.manual_seed(noise_seed)
        sample_noise = torch.randn(
            mean[0].shape,
            generator=generator,
            device=device,
            dtype=mean.dtype,
        )
        noise = sample_noise.unsqueeze(0).expand_as(mean)
        latent = mean + std * noise
        return self.vae_model.scale_factor * (latent - self.vae_model.shift_factor)

    @torch.no_grad()
    def _forward_cache_update_vae_shared_noise(
        self,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List[Tuple[int, int]],
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
        noise_seed: Optional[int] = None,
    ) -> NaiveCache:
        if noise_seed is None or padded_images.shape[0] <= 1:
            return self.model.forward_cache_update_vae(
                self.vae_model,
                past_key_values,
                padded_images=padded_images,
                patchified_vae_latent_shapes=patchified_vae_latent_shapes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_timesteps=packed_timesteps,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_seqlens=packed_seqlens,
                packed_indexes=packed_indexes,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
            )

        original_encode = self.vae_model.encode

        def encode_with_shared_noise(vae_model, x):
            return self._encode_vae_images_shared_noise(x, noise_seed=noise_seed)

        self.vae_model.encode = MethodType(encode_with_shared_noise, self.vae_model)
        try:
            return self.model.forward_cache_update_vae(
                self.vae_model,
                past_key_values,
                padded_images=padded_images,
                patchified_vae_latent_shapes=patchified_vae_latent_shapes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_timesteps=packed_timesteps,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_seqlens=packed_seqlens,
                packed_indexes=packed_indexes,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
            )
        finally:
            self.vae_model.encode = original_encode

    @torch.no_grad()
    def batch_update_context_text(
        self, texts: List[str], gen_context: Dict
    ) -> Dict:
        """Update context with multiple text prompts in batch."""
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input, new_kv_lens, new_ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=texts,
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(
            past_key_values, **generation_input
        )

        gen_context["kv_lens"] = new_kv_lens
        gen_context["ropes"] = new_ropes
        gen_context["past_key_values"] = past_key_values
        return gen_context

    @torch.no_grad()
    def batch_update_context_image(
        self,
        images: List[Image.Image],
        gen_context: Dict,
        vae: bool = True,
        vit: bool = True,
        noise_seed: Optional[int] = None,
    ) -> Dict:
        """Update context with multiple images in batch."""
        assert vae or vit
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        if vae:
            generation_input, new_kv_lens, new_ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=images,
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self._forward_cache_update_vae_shared_noise(
                past_key_values=past_key_values,
                noise_seed=noise_seed,
                **generation_input,
            )
            kv_lens = new_kv_lens
            ropes = new_ropes

        if vit:
            generation_input, new_kv_lens, new_ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=images,
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(
                past_key_values, **generation_input
            )
            kv_lens = new_kv_lens
            ropes = new_ropes

        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values
        return gen_context

    # ------------------------------------------------------------------ #
    #  Multi-GPU CFG parallel helpers
    # ------------------------------------------------------------------ #

    def _copy_naive_cache_to_device(self, past_key_values, device):
        copied = NaiveCache(past_key_values.num_layers)
        for layer_idx in range(past_key_values.num_layers):
            key_cache = past_key_values.key_cache[layer_idx]
            if key_cache is None:
                continue
            copied.key_cache[layer_idx] = key_cache.to(device, non_blocking=True)
            copied.value_cache[layer_idx] = past_key_values.value_cache[
                layer_idx
            ].to(device, non_blocking=True)
        return copied

    def _move_generation_input_to_device(self, generation_input, device, skip_keys=None):
        moved = {}
        skip_keys = skip_keys or set()
        for key, value in generation_input.items():
            if key in skip_keys or not torch.is_tensor(value):
                moved[key] = value
            else:
                moved[key] = value.to(device, non_blocking=True)
        return moved

    def _can_use_multi_gpu_cfg_parallel(
        self, use_cfg_parallel, cfg_text_scale, cfg_img_scale
    ):
        if not use_cfg_parallel or not self.cfg_parallel_workers:
            return False
        required_workers = ["base"]
        if cfg_text_scale > 1.0:
            required_workers.append("cfg_text")
        if cfg_img_scale > 1.0:
            required_workers.append("cfg_img")
        return all(w in self.cfg_parallel_workers for w in required_workers)

    def _build_cfg_branch_state(
        self,
        worker_name,
        image_generation_input,
        branch_context,
        packed_query_position_ids,
        packed_query_indexes,
        key_values_lens,
        packed_key_value_indexes,
    ):
        worker = self.cfg_parallel_workers[worker_name]
        device = worker["device"]
        branch_state = self._move_generation_input_to_device(
            image_generation_input, device, skip_keys={"packed_init_noises"},
        )
        branch_state["device"] = device
        branch_state["model"] = worker["model"]
        branch_state["past_key_values"] = self._copy_naive_cache_to_device(
            branch_context["past_key_values"], device
        )
        branch_state["packed_position_ids"] = packed_query_position_ids.to(
            device, non_blocking=True
        )
        branch_state["packed_indexes"] = packed_query_indexes.to(
            device, non_blocking=True
        )
        branch_state["key_values_lens"] = key_values_lens.to(device, non_blocking=True)
        branch_state["packed_key_value_indexes"] = packed_key_value_indexes.to(
            device, non_blocking=True
        )
        return branch_state

    def _run_cfg_branch_step(self, branch_state, x_t, timestep):
        branch_state["model"].language_model.model.enable_taylorseer = False
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            return branch_state["model"]._forward_flow(
                x_t=x_t.to(branch_state["device"], non_blocking=True),
                timestep=timestep.to(branch_state["device"], non_blocking=True),
                packed_vae_token_indexes=branch_state["packed_vae_token_indexes"],
                packed_vae_position_ids=branch_state["packed_vae_position_ids"],
                packed_text_ids=branch_state["packed_text_ids"],
                packed_text_indexes=branch_state["packed_text_indexes"],
                packed_indexes=branch_state["packed_indexes"],
                packed_position_ids=branch_state["packed_position_ids"],
                packed_seqlens=branch_state["packed_seqlens"],
                key_values_lens=branch_state["key_values_lens"],
                past_key_values=branch_state["past_key_values"],
                packed_key_value_indexes=branch_state["packed_key_value_indexes"],
                cfg_text_scale=1.0,
                cfg_img_scale=1.0,
                cfg_type="sequential",
            )

    def _apply_cfg_guidance(
        self,
        v_t,
        cfg_text_v_t,
        cfg_img_v_t,
        sample_token_lens,
        cfg_text_scale,
        cfg_img_scale,
        cfg_renorm_min,
        cfg_renorm_type,
    ):
        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(
                    min=cfg_renorm_min, max=1.0
                )
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    return cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                return v_t_text

            v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
            if cfg_img_scale > 1.0:
                v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
            else:
                v_t_ = v_t_text_

            if cfg_renorm_type == "global":
                norm_v_t = self.model._compute_samplewise_global_norms(
                    v_t, sample_token_lens
                )
                norm_v_t_ = self.model._compute_samplewise_global_norms(
                    v_t_, sample_token_lens
                )
            elif cfg_renorm_type == "channel":
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
            else:
                raise NotImplementedError(f"{cfg_renorm_type} is not supported")

            scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(
                min=cfg_renorm_min, max=1.0
            )
            return v_t_ * scale

        return v_t

    # ------------------------------------------------------------------ #
    #  Multi-GPU image generation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def gen_image_multi_gpu(
        self,
        image_shapes: List[Tuple[int, int]],
        gen_context: Dict,
        cfg_text_precontext: Dict,
        cfg_img_precontext: Dict,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: Tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        num_timesteps: int = 50,
        timestep_shift: float = 3.0,
        enable_taylorseer: bool = False,
        noise_seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Batch image generation using multi-GPU CFG parallel."""
        if enable_taylorseer:
            raise NotImplementedError(
                "TaylorSeer is not supported in multi-GPU CFG parallel mode."
            )

        base_worker = self.cfg_parallel_workers["base"]
        base_model = base_worker["model"]
        base_device = base_worker["device"]

        image_generation_input = base_model.prepare_vae_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            image_sizes=image_shapes,
            new_token_ids=self.new_token_ids,
            noise_seed=noise_seed,
        )
        generation_input_cfg_text = base_model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_precontext["kv_lens"],
            curr_rope=cfg_text_precontext["ropes"],
            image_sizes=image_shapes,
        )
        generation_input_cfg_img = base_model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_precontext["kv_lens"],
            curr_rope=cfg_img_precontext["ropes"],
            image_sizes=image_shapes,
        )

        branch_states = {
            "base": self._build_cfg_branch_state(
                "base",
                image_generation_input,
                gen_context,
                image_generation_input["packed_position_ids"],
                image_generation_input["packed_indexes"],
                image_generation_input["key_values_lens"],
                image_generation_input["packed_key_value_indexes"],
            )
        }
        if cfg_text_scale > 1.0:
            branch_states["cfg_text"] = self._build_cfg_branch_state(
                "cfg_text",
                image_generation_input,
                cfg_text_precontext,
                generation_input_cfg_text["cfg_packed_position_ids"],
                generation_input_cfg_text["cfg_packed_query_indexes"],
                generation_input_cfg_text["cfg_key_values_lens"],
                generation_input_cfg_text["cfg_packed_key_value_indexes"],
            )
        if cfg_img_scale > 1.0:
            branch_states["cfg_img"] = self._build_cfg_branch_state(
                "cfg_img",
                image_generation_input,
                cfg_img_precontext,
                generation_input_cfg_img["cfg_packed_position_ids"],
                generation_input_cfg_img["cfg_packed_query_indexes"],
                generation_input_cfg_img["cfg_key_values_lens"],
                generation_input_cfg_img["cfg_packed_key_value_indexes"],
            )

        x_t = image_generation_input["packed_init_noises"].to(
            base_device, non_blocking=True
        )
        sample_token_lens = (branch_states["base"]["packed_seqlens"] - 2).tolist()
        timesteps = torch.linspace(1, 0, num_timesteps, device=base_device)
        timesteps = timestep_shift * timesteps / (
            1 + (timestep_shift - 1) * timesteps
        )
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(
            enumerate(timesteps), total=len(timesteps), desc="Denoising",
        ):
            timestep = torch.full((x_t.shape[0],), t, device=base_device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0

            active_branch_names = ["base"]
            if cfg_text_scale_ > 1.0:
                active_branch_names.append("cfg_text")
            if cfg_img_scale_ > 1.0:
                active_branch_names.append("cfg_img")

            if self.cfg_parallel_executor is None or len(active_branch_names) == 1:
                branch_outputs = {
                    name: self._run_cfg_branch_step(
                        branch_states[name], x_t, timestep
                    )
                    for name in active_branch_names
                }
            else:
                futures = {
                    name: self.cfg_parallel_executor.submit(
                        self._run_cfg_branch_step,
                        branch_states[name],
                        x_t,
                        timestep,
                    )
                    for name in active_branch_names
                }
                branch_outputs = {
                    name: future.result() for name, future in futures.items()
                }

            v_t = branch_outputs["base"].to(base_device, non_blocking=True)
            cfg_text_v_t = None
            cfg_img_v_t = None
            if cfg_text_scale_ > 1.0:
                cfg_text_v_t = branch_outputs["cfg_text"].to(
                    base_device, non_blocking=True
                )
            if cfg_img_scale_ > 1.0:
                cfg_img_v_t = branch_outputs["cfg_img"].to(
                    base_device, non_blocking=True
                )

            v_t = self._apply_cfg_guidance(
                v_t,
                cfg_text_v_t,
                cfg_img_v_t,
                sample_token_lens,
                cfg_text_scale_,
                cfg_img_scale_,
                cfg_renorm_min,
                cfg_renorm_type,
            )
            x_t = x_t - v_t.to(x_t.device) * dts[i]

        unpacked_latent = x_t.split(sample_token_lens)
        return [
            self.decode_image(latent, shape)
            for latent, shape in zip(unpacked_latent, image_shapes)
        ]

    # ------------------------------------------------------------------ #
    #  Batch image generation (single-model or multi-GPU)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def batch_gen_image(
        self,
        image_shapes: List[Tuple[int, int]],
        gen_context: Dict,
        cfg_text_context: Optional[Dict] = None,
        cfg_img_context: Optional[Dict] = None,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: Tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        num_timesteps: int = 50,
        timestep_shift: float = 3.0,
        enable_taylorseer: bool = False,
        noise_seed: Optional[int] = None,
        use_cfg_parallel: bool = False,
    ) -> List[Image.Image]:
        """Generate multiple images in batch.

        When ``use_cfg_parallel=True`` and enough GPUs are available, CFG
        branches (base / cfg_text / cfg_img) run on separate GPUs via
        ThreadPoolExecutor.
        """
        if self._can_use_multi_gpu_cfg_parallel(
            use_cfg_parallel, cfg_text_scale, cfg_img_scale
        ):
            return self.gen_image_multi_gpu(
                image_shapes=image_shapes,
                gen_context=gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
                enable_taylorseer=enable_taylorseer,
                noise_seed=noise_seed,
            )

        batch_size = len(image_shapes)
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=image_shapes,
            new_token_ids=self.new_token_ids,
            noise_seed=noise_seed,
        )

        # Text CFG context
        if cfg_text_context is not None:
            cfg_text_past_key_values = cfg_text_context["past_key_values"]
            cfg_text_kv_lens = cfg_text_context["kv_lens"]
            cfg_text_ropes = cfg_text_context["ropes"]
        else:
            cfg_text_past_key_values = NaiveCache(
                self.model.config.llm_config.num_hidden_layers
            )
            cfg_text_kv_lens = [0] * batch_size
            cfg_text_ropes = [0] * batch_size

        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_kv_lens,
            curr_rope=cfg_text_ropes,
            image_sizes=image_shapes,
        )

        # Image CFG context
        if cfg_img_context is not None:
            cfg_img_past_key_values = cfg_img_context["past_key_values"]
            cfg_img_kv_lens = cfg_img_context["kv_lens"]
            cfg_img_ropes = cfg_img_context["ropes"]
        else:
            cfg_img_past_key_values = NaiveCache(
                self.model.config.llm_config.num_hidden_layers
            )
            cfg_img_kv_lens = [0] * batch_size
            cfg_img_ropes = [0] * batch_size

        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_kv_lens,
            curr_rope=cfg_img_ropes,
            image_sizes=image_shapes,
        )

        unpacked_latents = self.model.generate_image(
            past_key_values=past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_text_packed_position_ids=generation_input_cfg_text["cfg_packed_position_ids"],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_query_indexes=generation_input_cfg_text["cfg_packed_query_indexes"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text["cfg_packed_key_value_indexes"],
            cfg_img_past_key_values=cfg_img_past_key_values,
            cfg_img_packed_position_ids=generation_input_cfg_img["cfg_packed_position_ids"],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_query_indexes=generation_input_cfg_img["cfg_packed_query_indexes"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img["cfg_packed_key_value_indexes"],
            enable_taylorseer=enable_taylorseer,
            cfg_type="parallel" if use_cfg_parallel else "sequential",
            **generation_input,
        )

        images = []
        for latent, shape in zip(unpacked_latents, image_shapes):
            images.append(self.decode_image(latent, shape))
        return images

    # ------------------------------------------------------------------ #
    #  Cache selection helpers (for multi-round interleaved inference)
    # ------------------------------------------------------------------ #

    def _extract_single_sample_cache(
        self,
        past_key_values: NaiveCache,
        kv_lens: List[int],
        sample_idx: int,
    ) -> NaiveCache:
        """Extract KV cache for a single sample from the packed batch cache."""
        start_idx = sum(kv_lens[:sample_idx])
        end_idx = start_idx + kv_lens[sample_idx]

        selected_cache = NaiveCache(past_key_values.num_layers)
        for layer_idx in range(past_key_values.num_layers):
            if past_key_values.key_cache[layer_idx] is None:
                continue
            selected_cache.key_cache[layer_idx] = past_key_values.key_cache[
                layer_idx
            ][start_idx:end_idx].clone()
            selected_cache.value_cache[layer_idx] = past_key_values.value_cache[
                layer_idx
            ][start_idx:end_idx].clone()
        return selected_cache

    def _select_batch_context(
        self, gen_context: Dict, sample_indices: List[int]
    ) -> Dict:
        """Select a subset of samples from a batch context."""
        selected_context = {
            "kv_lens": [gen_context["kv_lens"][idx] for idx in sample_indices],
            "ropes": [gen_context["ropes"][idx] for idx in sample_indices],
            "past_key_values": NaiveCache(
                self.model.config.llm_config.num_hidden_layers
            ),
        }
        if not sample_indices:
            return selected_context

        extracted_caches = [
            self._extract_single_sample_cache(
                gen_context["past_key_values"],
                gen_context["kv_lens"],
                sample_idx,
            )
            for sample_idx in sample_indices
        ]

        for layer_idx in range(selected_context["past_key_values"].num_layers):
            key_chunks = [
                cache.key_cache[layer_idx]
                for cache in extracted_caches
                if cache.key_cache[layer_idx] is not None
            ]
            if not key_chunks:
                continue
            value_chunks = [
                cache.value_cache[layer_idx]
                for cache in extracted_caches
                if cache.value_cache[layer_idx] is not None
            ]
            selected_context["past_key_values"].key_cache[layer_idx] = torch.cat(
                key_chunks, dim=0
            )
            selected_context["past_key_values"].value_cache[layer_idx] = torch.cat(
                value_chunks, dim=0
            )

        return selected_context

    # ------------------------------------------------------------------ #
    #  Input validation
    # ------------------------------------------------------------------ #

    def _validate_batch_inputs(
        self, batch_input_lists: List[List[Union[str, Image.Image]]]
    ) -> List[type]:
        """Validate all samples have the same input structure."""
        if not batch_input_lists:
            raise ValueError("batch_input_lists must not be empty.")

        first_input_len = len(batch_input_lists[0])
        for sample_idx, inputs in enumerate(batch_input_lists):
            if len(inputs) != first_input_len:
                raise ValueError(
                    f"All samples must have same number of inputs. "
                    f"Sample 0 has {first_input_len}, sample {sample_idx} has {len(inputs)}"
                )

        input_type_sequence: List[type] = []
        for input_idx in range(first_input_len):
            ref = batch_input_lists[0][input_idx]
            if isinstance(ref, str):
                input_type_sequence.append(str)
            elif isinstance(ref, Image.Image):
                input_type_sequence.append(Image.Image)
            else:
                raise ValueError(f"Unsupported input type: {type(ref)}")

            for sample_idx in range(1, len(batch_input_lists)):
                item = batch_input_lists[sample_idx][input_idx]
                expected = input_type_sequence[input_idx]
                if expected is str and not isinstance(item, str):
                    raise ValueError(
                        f"Type mismatch at position {input_idx}: "
                        f"sample 0 is str, sample {sample_idx} is {type(item)}"
                    )
                if expected is Image.Image and not isinstance(item, Image.Image):
                    raise ValueError(
                        f"Type mismatch at position {input_idx}: "
                        f"sample 0 is Image, sample {sample_idx} is {type(item)}"
                    )

        return input_type_sequence

    # ------------------------------------------------------------------ #
    #  Main batch interleaved inference
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def batch_interleave_inference(
        self,
        batch_input_lists: List[List[Union[str, Image.Image]]],
        think: bool = False,
        understanding_output: bool = False,
        max_think_token_n: int = 1000,
        do_sample: bool = False,
        text_temperature: float = 1.0,
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: List[float] = [0.4, 1.0],
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        image_shapes: Union[Tuple[int, int], List[Tuple[int, int]]] = (1024, 1024),
        enable_taylorseer: bool = False,
        noise_seed: Optional[int] = None,
        use_cfg_parallel: bool = False,
        max_rounds: int = 3,
    ) -> List[List[Union[str, Image.Image]]]:
        """Run batch interleaved inference on multiple samples.

        All samples must have the same input structure (same number and types
        of inputs at each position).

        Args:
            batch_input_lists: List of input lists, one per sample.
            think: Enable thinking mode.
            understanding_output: If True, generate text only.
            max_think_token_n: Max tokens for text generation.
            do_sample: Use sampling for text generation.
            text_temperature: Temperature for text sampling.
            cfg_text_scale: CFG scale for text.
            cfg_img_scale: CFG scale for image.
            cfg_interval: CFG timestep interval.
            timestep_shift: Timestep shift for diffusion.
            num_timesteps: Number of diffusion steps.
            cfg_renorm_min: CFG renormalization minimum.
            cfg_renorm_type: CFG renormalization type.
            image_shapes: Image shape(s) for generation.
            enable_taylorseer: Enable TaylorSeer optimization.
            noise_seed: Random seed for noise generation.
            use_cfg_parallel: Use multi-GPU CFG parallel.
            max_rounds: Max interleaved generation rounds.

        Returns:
            List of output lists, one per sample.
        """
        batch_size = len(batch_input_lists)
        input_type_sequence = self._validate_batch_inputs(batch_input_lists)

        if isinstance(image_shapes, tuple):
            image_shapes = [image_shapes] * batch_size
        else:
            image_shapes = list(image_shapes)

        gen_context = self.init_gen_context(batch_size=batch_size)
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        outputs: List[List[Union[str, Image.Image]]] = [[] for _ in range(batch_size)]

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # System prompt
            if think:
                system_prompt = (
                    VLM_THINK_SYSTEM_PROMPT
                    if understanding_output
                    else GEN_THINK_SYSTEM_PROMPT
                )
                system_prompts = [system_prompt] * batch_size
                gen_context = self.batch_update_context_text(
                    system_prompts, gen_context
                )
                cfg_img_context = self.batch_update_context_text(
                    system_prompts, cfg_img_context
                )

            # Process each input position across all samples
            for input_idx, input_type in enumerate(input_type_sequence):
                if input_type is str:
                    texts = [
                        batch_input_lists[sample_idx][input_idx]
                        for sample_idx in range(batch_size)
                    ]
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.batch_update_context_text(texts, gen_context)
                    cfg_img_context = self.batch_update_context_text(
                        texts, cfg_img_context
                    )
                    continue

                # Image input
                images = [
                    batch_input_lists[sample_idx][input_idx]
                    for sample_idx in range(batch_size)
                ]
                processed_images = [
                    self.vae_transform.resize_transform(pil_img2rgb(image))
                    for image in images
                ]
                for sample_idx, processed_image in enumerate(processed_images):
                    image_shapes[sample_idx] = processed_image.size[::-1]
                gen_context = self.batch_update_context_image(
                    processed_images,
                    gen_context,
                    vae=not understanding_output,
                    noise_seed=noise_seed,
                )
                cfg_text_context = deepcopy(gen_context)

            # ---- Generate outputs ----

            if understanding_output:
                generated_texts = self.gen_text_batch(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                )
                for sample_idx, text in enumerate(generated_texts):
                    outputs[sample_idx].append(text)
                return outputs

            # Generation output (no think)
            if not think:
                generated_images = self.batch_gen_image(
                    image_shapes=image_shapes,
                    gen_context=gen_context,
                    cfg_text_context=cfg_text_context,
                    cfg_img_context=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=tuple(cfg_interval),
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    num_timesteps=num_timesteps,
                    timestep_shift=timestep_shift,
                    enable_taylorseer=enable_taylorseer,
                    noise_seed=noise_seed,
                    use_cfg_parallel=use_cfg_parallel,
                )
                for sample_idx, img in enumerate(generated_images):
                    outputs[sample_idx].append(img)
                return outputs

            # Think mode: multi-round text + image generation
            active_indices = list(range(batch_size))
            active_image_shapes = list(image_shapes)
            active_gen_context = gen_context
            active_cfg_text_context = cfg_text_context
            active_cfg_img_context = cfg_img_context

            rounds = 0
            while active_indices and rounds < max_rounds:
                generated_texts = self.gen_text_batch(
                    active_gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                )
                for local_idx, text in enumerate(generated_texts):
                    outputs[active_indices[local_idx]].append(text)

                active_gen_context = self.batch_update_context_text(
                    generated_texts, active_gen_context
                )

                image_request_local_indices = [
                    local_idx
                    for local_idx, text in enumerate(generated_texts)
                    if any(marker in text for marker in IMAGE_REQUEST_MARKERS)
                ]
                if not image_request_local_indices:
                    break

                round_indices = [
                    active_indices[local_idx]
                    for local_idx in image_request_local_indices
                ]
                round_image_shapes = [
                    active_image_shapes[local_idx]
                    for local_idx in image_request_local_indices
                ]
                round_gen_context = self._select_batch_context(
                    active_gen_context, image_request_local_indices
                )
                round_cfg_text_context = self._select_batch_context(
                    active_cfg_text_context, image_request_local_indices
                )
                round_cfg_img_context = self._select_batch_context(
                    active_cfg_img_context, image_request_local_indices
                )

                generated_images = self.batch_gen_image(
                    image_shapes=round_image_shapes,
                    gen_context=round_gen_context,
                    cfg_text_context=round_cfg_text_context,
                    cfg_img_context=round_cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=tuple(cfg_interval),
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    num_timesteps=num_timesteps,
                    timestep_shift=timestep_shift,
                    enable_taylorseer=enable_taylorseer,
                    noise_seed=noise_seed,
                    use_cfg_parallel=use_cfg_parallel,
                )

                processed_generated_images = []
                for global_idx, img in zip(round_indices, generated_images):
                    outputs[global_idx].append(img)
                    processed_generated_images.append(
                        self.vae_transform.resize_transform(pil_img2rgb(img))
                    )

                active_gen_context = self.batch_update_context_image(
                    processed_generated_images,
                    round_gen_context,
                    vae=not understanding_output,
                )
                active_indices = round_indices
                active_image_shapes = round_image_shapes
                active_cfg_text_context = round_cfg_text_context
                active_cfg_img_context = round_cfg_img_context
                rounds += 1

        return outputs

    def batch_inference(self, *args, **kwargs):
        return self.batch_interleave_inference(*args, **kwargs)

    def batch_interleaved_inference(self, *args, **kwargs):
        return self.batch_interleave_inference(*args, **kwargs)

    def __call__(self, batch_input_lists, **kwargs):
        return self.batch_interleave_inference(batch_input_lists, **kwargs)
