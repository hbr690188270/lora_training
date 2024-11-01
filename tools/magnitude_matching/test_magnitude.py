import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
)


def load_lora_weights(model = "source"):
    if model == "source":
        # adapter_path = "ckpt/instruct_lm/llama3_alpha128_r64/checkpoint-16797/adapter_model.safetensors"
        adapter_path = "ckpt/ptr/mistral-7b-v3-gsm8k/adapter_model.safetensors"
    elif model == "target":
        adapter_path = "ckpt/instruct_lm/llama31_alpha128_r64/checkpoint-16797/adapter_model.safetensors"
    tensors = {}
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def compute_lora_magnitude():
    llama3_lora_tensors = load_lora_weights(model="source")
    llama31_lora_tensors = load_lora_weights(model="target")
    # for k,v in llama31_lora_tensors.items():
    #     print(k, v.size())

    device = torch.device("cuda")

    # for layer_idx in range(32):
    for layer_idx in [1, 5, 31]:
        print(f"================Layer {layer_idx}================")
        for module in ["q", "k", "v", "o"]:
            lora_B = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}_proj.lora_B.weight"
            lora_A = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}_proj.lora_A.weight"

            lora_B = llama3_lora_tensors[lora_B].to(device)
            lora_A = llama3_lora_tensors[lora_A].to(device)

            module_mat = torch.matmul(lora_B, lora_A)
            source_norm = torch.norm(module_mat)

            lora_B = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}_proj.lora_B.weight"
            lora_A = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}_proj.lora_A.weight"
            lora_B = llama31_lora_tensors[lora_B].to(device)
            lora_A = llama31_lora_tensors[lora_A].to(device)

            module_mat = torch.matmul(lora_B, lora_A)
            target_norm = torch.norm(module_mat)

            print(source_norm, target_norm, source_norm/target_norm)
        print()


def main():
    model_kwargs = dict(
        trust_remote_code=True,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16,
        use_cache=True,
        device_map=None,
        cache_dir='./model_cache'
    )
    # model_name_or_path = "model_cache/llama3-8b"
    model_name_or_path = "model_cache/mistral-7b-v3"
    llama3_model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )

    model_name_or_path = "model_cache/llama3_1-8b"
    llama31_model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )
    print(len(llama3_model.model.layers))
    # for layer_idx in range(32):
    for layer_idx in [1, 5, 31]:
        source_q_norm = torch.norm(llama3_model.model.layers[layer_idx].self_attn.q_proj.weight.data)
        source_k_norm = torch.norm(llama3_model.model.layers[layer_idx].self_attn.k_proj.weight.data)
        source_v_norm = torch.norm(llama3_model.model.layers[layer_idx].self_attn.v_proj.weight.data)
        source_o_norm = torch.norm(llama3_model.model.layers[layer_idx].self_attn.o_proj.weight.data)

        target_q_norm = torch.norm(llama31_model.model.layers[layer_idx].self_attn.q_proj.weight.data)
        target_k_norm = torch.norm(llama31_model.model.layers[layer_idx].self_attn.k_proj.weight.data)
        target_v_norm = torch.norm(llama31_model.model.layers[layer_idx].self_attn.v_proj.weight.data)
        target_o_norm = torch.norm(llama31_model.model.layers[layer_idx].self_attn.o_proj.weight.data)

        print(f"================Layer {layer_idx}================")
        print(source_q_norm, target_q_norm, source_q_norm/target_q_norm)
        print(source_k_norm, target_k_norm, source_k_norm/target_k_norm)
        print(source_v_norm, target_v_norm, source_v_norm/target_v_norm)
        print(source_o_norm, target_o_norm, source_o_norm/target_o_norm)
        print()

if __name__ == "__main__":
    # main()
    compute_lora_magnitude()

