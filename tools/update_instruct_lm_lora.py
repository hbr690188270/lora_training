import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
)


def load_lora_weights():
    adapter_path = "ckpt/instruct_lm/llama3_alpha128_r64/checkpoint-6000/adapter_model.safetensors"
    tensors = {}
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def main():
    lora_tensors = load_lora_weights()

    model_name_or_path = "model_cache/llama3-8b"
    model_kwargs = dict(
        trust_remote_code=True,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16,
        use_cache=True,
        device_map=None,
        cache_dir='./model_cache'
    )
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )
    pretrained_embeddings = model.model.embed_tokens.weight.data
    pretrained_lm_head = model.lm_head.weight.data

    llama31_model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        "model_cache/llama3_1-8b",
        **model_kwargs
    )
    llama31_embeddings = llama31_model.model.embed_tokens.weight.data
    llama31_lm_head = llama31_model.lm_head.weight.data

    lora_embeddings = lora_tensors["base_model.model.model.embed_tokens.weight"]
    lora_lm_head = lora_tensors["base_model.model.lm_head.weight"]

    assert lora_embeddings.size() == llama31_embeddings.size()
    assert llama31_embeddings.size() == pretrained_embeddings.size()

    updated_lora_embeddings = lora_embeddings - pretrained_embeddings + llama31_embeddings
    updated_lora_lm_head = lora_lm_head - pretrained_lm_head + llama31_lm_head

    lora_tensors["base_model.model.model.embed_tokens.weight"] = updated_lora_embeddings
    lora_tensors["base_model.model.lm_head.weight"] = updated_lora_lm_head

    save_name = "ckpt/instruct_lm/llama3_for_llama31/adapter_model.safetensors"
    save_file(lora_tensors, save_name)

if __name__ == "__main__":
    main()

