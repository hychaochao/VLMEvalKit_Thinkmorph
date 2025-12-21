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
)

from .base import BaseModel


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

        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # transforms
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        # device map
        max_mem_per_gpu = "40GiB"
        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )

        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "model.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload"
        )
        model = model.eval()
        print(f'Model loaded successfully')

        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

        self.inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids,
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
                do_sample=True,
                text_temperature=self.temperature,
            )
        else:
            inference_hyper = dict(
                max_think_token_n=self.max_think_token_n,
                do_sample=True,
                text_temperature=self.temperature,
                cfg_text_scale=4.0,
                cfg_img_scale=2.0,
                cfg_interval=[0.0, 1.0],
                timestep_shift=3.0,
                num_timesteps=50,
                cfg_renorm_min=0.0,
                cfg_renorm_type="text_channel",
            )

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
    
    def generate_inner(self, message, dataset=None):
        input_list = self.build_thinkmorph_input(message)

        if self.understanding_output:
            output_dict = self.inferencer(input_list=input_list, think=self.think, 
                                        understanding_output=True, **self.inference_hyper)
            final_output = output_dict[0]
            
        else:
            output_list = self.inferencer(input_list=input_list, think=self.think, **self.inference_hyper)
            results = []
            text_round = 0

            for idx, out_item in enumerate(output_list):
                if isinstance(out_item, str):
                    out_item = f"Round_{text_round}:\n" + out_item
                    results.append(out_item)
                    text_round += 1
                elif isinstance(out_item, Image.Image):
                    out_img_path = os.path.join(self.save_dir, f"thinkmorph_out_{uuid.uuid4().hex[:8]}_{idx}.jpg")
                    out_item.save(out_img_path)
                    results.append(f"[Image: {out_img_path}]")

            final_output = "\n".join(results)
            print(final_output)

        return final_output