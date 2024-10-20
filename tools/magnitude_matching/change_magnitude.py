import copy
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
)


def compute_base_magnitute_ratio():
    model_kwargs = dict(
        trust_remote_code=True,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16,
        use_cache=True,
        device_map=None,
        cache_dir='./model_cache'
    )
    model_name_or_path = "model_cache/llama3-8b"
    llama3_model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )

    model_name_or_path = "model_cache/llama3_1-8b"
    llama31_model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )
    # print(len(llama3_model.model.layers))
    magnitude_ratio_dict = {}
    for layer_idx in range(32):
        source_q_norm = torch.norm(llama3_model.model.layers[layer_idx].self_attn.q_proj.weight.data.float())
        source_k_norm = torch.norm(llama3_model.model.layers[layer_idx].self_attn.k_proj.weight.data.float())
        source_v_norm = torch.norm(llama3_model.model.layers[layer_idx].self_attn.v_proj.weight.data.float())
        source_o_norm = torch.norm(llama3_model.model.layers[layer_idx].self_attn.o_proj.weight.data.float())

        target_q_norm = torch.norm(llama31_model.model.layers[layer_idx].self_attn.q_proj.weight.data.float())
        target_k_norm = torch.norm(llama31_model.model.layers[layer_idx].self_attn.k_proj.weight.data.float())
        target_v_norm = torch.norm(llama31_model.model.layers[layer_idx].self_attn.v_proj.weight.data.float())
        target_o_norm = torch.norm(llama31_model.model.layers[layer_idx].self_attn.o_proj.weight.data.float())

        # print(f"================Layer {layer_idx}================")
        # print(source_q_norm, target_q_norm, source_q_norm/target_q_norm)
        # print(source_k_norm, target_k_norm, source_k_norm/target_k_norm)
        # print(source_v_norm, target_v_norm, source_v_norm/target_v_norm)
        # print(source_o_norm, target_o_norm, source_o_norm/target_o_norm)
        # print()

        magnitude_ratio_dict[f"layer{layer_idx}_q"] = target_q_norm/source_q_norm
        magnitude_ratio_dict[f"layer{layer_idx}_k"] = target_k_norm/source_k_norm
        magnitude_ratio_dict[f"layer{layer_idx}_v"] = target_v_norm/source_v_norm
        magnitude_ratio_dict[f"layer{layer_idx}_o"] = target_o_norm/source_o_norm

    return magnitude_ratio_dict


def load_lora_weights(model = "source"):
    if model == "source":
        adapter_path = "ckpt/instruct_lm/llama3_for_llama31/adapter_model.safetensors"
    elif model == "target":
        adapter_path = "ckpt/instruct_lm/llama31_alpha128_r64/checkpoint-16797/adapter_model.safetensors"
    tensors = {}
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def change_lora_magnitude():
    magnitude_ratio_dict = compute_base_magnitute_ratio()
    llama3_lora_tensors = load_lora_weights(model="source")
    llama31_lora_tensors = load_lora_weights(model="target")
    # for k,v in llama31_lora_tensors.items():
    #     print(k, v.size())

    device = torch.device("cuda")

    new_llama3_lora_tensors = copy.deepcopy(llama3_lora_tensors)

    for layer_idx in range(32):
        print(f"================Layer {layer_idx}================")
        for module in ["q", "k", "v", "o"]:
            lora_B = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}_proj.lora_B.weight"
            lora_A = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}_proj.lora_A.weight"

            lora_B = llama3_lora_tensors[lora_B].to(device).float()
            lora_A = llama3_lora_tensors[lora_A].to(device).float()

            module_mat = torch.matmul(lora_B, lora_A)
            source_norm = torch.norm(module_mat)

            lora_B = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}_proj.lora_B.weight"
            lora_A = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}_proj.lora_A.weight"
            lora_B = llama31_lora_tensors[lora_B].to(device).float()
            lora_A = llama31_lora_tensors[lora_A].to(device).float()

            module_mat = torch.matmul(lora_B, lora_A)
            target_norm = torch.norm(module_mat)

            key = f"layer{layer_idx}_{module}"
            # target / source
            base_ratio = magnitude_ratio_dict[key]
            curr_ratio = target_norm / source_norm
            upscale_ratio = curr_ratio / base_ratio

            lora_B_name = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}_proj.lora_B.weight"
            lora_A_name = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}_proj.lora_A.weight"
            lora_B = llama3_lora_tensors[lora_B_name].to(device)
            lora_A = llama3_lora_tensors[lora_A_name].to(device)
            # new_lora_B = lora_B * torch.sqrt(torch.tensor(upscale_ratio).float().to(device))
            new_lora_B = lora_B * torch.tensor(upscale_ratio).float().to(device)
            new_module_mat = torch.matmul(new_lora_B, lora_A)
            new_norm = torch.norm(new_module_mat)
            new_llama3_lora_tensors[lora_B_name] = new_lora_B.cpu().bfloat16()

            print(source_norm, target_norm, target_norm/source_norm)
            print(target_norm/new_norm, base_ratio)
        print()

    filename = "ckpt/instruct_lm/llama3_for_llama31_rescale/adapter_model.safetensors"
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    save_file(new_llama3_lora_tensors, filename)

if __name__ == "__main__":
    # main()
    change_lora_magnitude()

