from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
import random
import uuid

import numpy as np
import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from PIL import Image
from .thinkmorph import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
    Qwen2Tokenizer,
    load_ae,
    add_special_tokens,
    ImageTransform,
    InterleaveInferencer,
    BatchInterleaveInferencer,
)

from .base import BaseModel


SAME_DEVICE_MODULES = [
    "language_model.model.embed_tokens",
    "time_embedder",
    "latent_pos_embed",
    "vae2llm",
    "llm2vae",
    "connector",
    "vit_pos_embed",
]


def _build_empty_bagel_model(llm_config, vit_config, vae_config):
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=True
        )
    return model


def _load_bagel_model(
    llm_config,
    vit_config,
    vae_config,
    model_checkpoint,
    target_device=None,
):
    model = _build_empty_bagel_model(llm_config, vit_config, vae_config)

    if target_device is None:
        max_mem_per_gpu = "80GiB"
        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(SAME_DEVICE_MODULES[0], "cuda:0")
            for module_name in SAME_DEVICE_MODULES:
                if module_name in device_map:
                    device_map[module_name] = first_device
                else:
                    device_map[module_name] = "cuda:0"
        else:
            first_device = device_map.get(SAME_DEVICE_MODULES[0])
            for module_name in SAME_DEVICE_MODULES:
                if module_name in device_map:
                    device_map[module_name] = first_device
    else:
        device_map = {"": target_device}
        for module_name in SAME_DEVICE_MODULES:
            device_map[module_name] = target_device

    offload_folder = "/tmp/offload"
    if target_device is not None:
        offload_folder = f"/tmp/offload_{str(target_device).replace(':', '_')}"

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=model_checkpoint,
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder=offload_folder,
    )
    return model.eval()


def _load_bagel_model_for_device(
    llm_config,
    vit_config,
    vae_config,
    model_checkpoint,
    target_device,
):
    return _load_bagel_model(
        deepcopy(llm_config),
        deepcopy(vit_config),
        deepcopy(vae_config),
        model_checkpoint,
        target_device=target_device,
    )


class ThinkMorph(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='ThinkMorph/ThinkMorph-7B', think=True, understanding_output=True, save_dir=None, temperature=0.3, max_think_token_n=4096, **kwargs):
        # self.check_install()
        assert model_path is not None
        if not understanding_output:
            assert save_dir is not None
        self.model_path = model_path
        self.understanding_output = understanding_output
        self.save_dir = save_dir
        self.think = think
        self.temperature = temperature
        self.max_think_token_n = max_think_token_n
        self.batch_size = kwargs.get("batch_size")
        if self.batch_size is not None:
            self.batch_size = int(self.batch_size)
            if self.batch_size <= 0:
                raise ValueError("batch_size must be a positive integer.")
        self.use_batch_inferencer = bool(self.batch_size and self.batch_size > 1)
        self.use_cfg_parallel = bool(kwargs.get("use_cfg_parallel", False))

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

        # tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # transforms
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        do_sample = kwargs.get("do_sample", True)
        cfg_text_scale = kwargs.get("cfg_text_scale", 4.0)
        cfg_img_scale = kwargs.get("cfg_img_scale", 2.0)
        cfg_interval = list(kwargs.get("cfg_interval", [0.0, 1.0]))
        timestep_shift = kwargs.get("timestep_shift", 3.0)
        num_timesteps = kwargs.get("num_timesteps", 50)
        cfg_renorm_min = kwargs.get("cfg_renorm_min", 0.0)
        cfg_renorm_type = kwargs.get("cfg_renorm_type", "text_channel")
        enable_taylorseer = kwargs.get("enable_taylorseer", False)
        noise_seed = kwargs.get("noise_seed")
        max_rounds = kwargs.get("max_rounds", 3)

        model_checkpoint = os.path.join(model_path, "model.safetensors")
        cfg_parallel_worker_names = []
        if self.use_cfg_parallel and not understanding_output:
            if cfg_text_scale > 1.0:
                cfg_parallel_worker_names.append("cfg_text")
            if cfg_img_scale > 1.0:
                cfg_parallel_worker_names.append("cfg_img")

            required_gpu_num = 1 + len(cfg_parallel_worker_names)
            if torch.cuda.device_count() < required_gpu_num:
                print(
                    f"Visible GPU count {torch.cuda.device_count()} is not enough for "
                    f"multi-GPU CFG parallel (need {required_gpu_num}). "
                    "Falling back to single-model CFG parallel."
                )
                cfg_parallel_worker_names = []

        cfg_parallel_workers = {}
        if cfg_parallel_worker_names:
            worker_specs = [("base", "cuda:0")] + [
                (worker_name, f"cuda:{worker_idx}")
                for worker_idx, worker_name in enumerate(
                    cfg_parallel_worker_names, start=1
                )
            ]
            with ThreadPoolExecutor(
                max_workers=len(worker_specs),
                thread_name_prefix="cfg_model_loader",
            ) as executor:
                future_map = {
                    worker_name: (
                        worker_device,
                        executor.submit(
                            _load_bagel_model_for_device,
                            llm_config,
                            vit_config,
                            vae_config,
                            model_checkpoint,
                            worker_device,
                        ),
                    )
                    for worker_name, worker_device in worker_specs
                }

                for worker_name, (worker_device, future) in future_map.items():
                    cfg_parallel_workers[worker_name] = {
                        "model": future.result(),
                        "device": worker_device,
                    }

            model = cfg_parallel_workers["base"]["model"]
            vae_device = cfg_parallel_workers["base"]["device"]
        else:
            model = _load_bagel_model(
                llm_config,
                vit_config,
                vae_config,
                model_checkpoint,
            )
            vae_device = "cuda" if torch.cuda.is_available() else "cpu"

        vae_model = vae_model.to(vae_device).to(torch.bfloat16).eval()
        print('Model loaded successfully')

        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.cfg_parallel_workers = cfg_parallel_workers

        self.inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids,
        )

        self.batch_inferencer = None
        if self.use_batch_inferencer:
            self.batch_inferencer = BatchInterleaveInferencer(
                model=self.model,
                vae_model=self.vae_model,
                tokenizer=self.tokenizer,
                vae_transform=self.vae_transform,
                vit_transform=self.vit_transform,
                new_token_ids=self.new_token_ids,
                cfg_parallel_workers=self.cfg_parallel_workers,
            )

        seed = 42
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if understanding_output:
            inference_hyper = dict(
                max_think_token_n=self.max_think_token_n,
                do_sample=do_sample,
                text_temperature=self.temperature,
                max_rounds=max_rounds,
            )
        else:
            inference_hyper = dict(
                max_think_token_n=self.max_think_token_n,
                do_sample=do_sample,
                text_temperature=self.temperature,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                enable_taylorseer=enable_taylorseer,
                noise_seed=noise_seed,
                max_rounds=max_rounds,
            )
            if self.use_cfg_parallel:
                inference_hyper["use_cfg_parallel"] = True

        self.inference_hyper = inference_hyper


    def build_thinkmorph_input(self, message):
        # according to https://github.com/ByteDance-Seed/Bagel/issues/83

        images = []
        text_parts = []
        image_counter = 1  

        for m in message:
            if m['type'] == 'image':
                val = m['value']
                if isinstance(val, str):
                    img = Image.open(val).convert("RGB")
                elif isinstance(val, Image.Image):
                    img = val
                else:
                    raise TypeError(f"Unsupported image input type {type(val)}")
                
                images.append(img)
                text_parts.append(f"<img><|image_{image_counter}|></img>")
                image_counter += 1

            elif m['type'] == 'text':
                text_parts.append(m['value'])
            else:
                raise ValueError(f"Unsupported message type {m['type']}")

        if not images:
            raise ValueError("Bagel requires at least one image input")

        final_text = " ".join(text_parts)
        input_list = images + [final_text]
        return input_list

    def _run_single_inference(self, input_list, understanding_output):
        return self.inferencer.interleave_inference(
            input_list,
            think=self.think,
            understanding_output=understanding_output,
            **self.inference_hyper,
        )

    def _run_batch_inference(self, batch_input_lists, understanding_output):
        if not self.use_batch_inferencer:
            return [
                self._run_single_inference(
                    input_list, understanding_output=understanding_output
                )
                for input_list in batch_input_lists
            ]

        return self.batch_inferencer.batch_interleave_inference(
            batch_input_lists=batch_input_lists,
            think=self.think,
            understanding_output=understanding_output,
            **self.inference_hyper,
        )

    def _format_generation_outputs(self, output_list):
        results = []
        text_round = 0

        for idx, out_item in enumerate(output_list):
            if isinstance(out_item, str):
                out_item = f"Round_{text_round}:\n" + out_item
                results.append(out_item)
                text_round += 1
            elif isinstance(out_item, Image.Image):
                out_img_path = os.path.join(
                    self.save_dir,
                    f"thinkmorph_out_{uuid.uuid4().hex[:8]}_{idx}.jpg",
                )
                out_item.save(out_img_path)
                results.append(f"[Image: {out_img_path}]")

        return "\n".join(results)

    def _chunk_indexed_samples(self, indexed_samples):
        chunk_size = self.batch_size or len(indexed_samples)
        for start_idx in range(0, len(indexed_samples), chunk_size):
            yield indexed_samples[start_idx : start_idx + chunk_size]
    
    def generate_inner(self, message, dataset=None):
        input_list = self.build_thinkmorph_input(message)
        if self.use_batch_inferencer:
            output_list = self._run_batch_inference(
                [input_list], understanding_output=self.understanding_output
            )[0]
        else:
            output_list = self._run_single_inference(
                input_list, understanding_output=self.understanding_output
            )

        if self.understanding_output:
            final_output = output_list[0] if output_list else ""
        else:
            final_output = self._format_generation_outputs(output_list)
            print(final_output)

        return final_output

    def _sample_signature(self, input_list):
        """Return a tuple of types ('str'/'image') describing the input structure."""
        sig = []
        for item in input_list:
            if isinstance(item, str):
                sig.append("str")
            elif isinstance(item, Image.Image):
                sig.append("image")
            else:
                sig.append(type(item).__name__)
        return tuple(sig)

    def generate_inner_batch(self, messages, dataset=None):
        """Batch inference for multiple messages.

        Groups messages by input structure, processes each group in batch,
        and returns results in the original order.

        Args:
            messages: List of message lists (same format as generate_inner input).
            dataset: Optional dataset name.

        Returns:
            List of output strings, one per message.
        """
        from collections import OrderedDict

        # Build input lists for all messages
        all_input_lists = []
        for message in messages:
            all_input_lists.append(self.build_thinkmorph_input(message))

        # Group by input structure (same number and types of inputs)
        grouped = OrderedDict()
        for idx, input_list in enumerate(all_input_lists):
            sig = self._sample_signature(input_list)
            grouped.setdefault(sig, []).append((idx, input_list))

        all_outputs = [None] * len(messages)

        if not self.use_batch_inferencer:
            for sample_idx, input_list in enumerate(all_input_lists):
                output_list = self._run_single_inference(
                    input_list, understanding_output=self.understanding_output
                )
                if self.understanding_output:
                    all_outputs[sample_idx] = output_list[0] if output_list else ""
                else:
                    all_outputs[sample_idx] = self._format_generation_outputs(
                        output_list
                    )
            return all_outputs

        for _, indexed_samples in grouped.items():
            for chunk in self._chunk_indexed_samples(indexed_samples):
                chunk_indices = [idx for idx, _ in chunk]
                chunk_inputs = [inp for _, inp in chunk]
                batch_outputs = self._run_batch_inference(
                    chunk_inputs, understanding_output=self.understanding_output
                )

                for sample_idx, output_list in zip(chunk_indices, batch_outputs):
                    if self.understanding_output:
                        all_outputs[sample_idx] = output_list[0] if output_list else ""
                    else:
                        all_outputs[sample_idx] = self._format_generation_outputs(
                            output_list
                        )

        return all_outputs
