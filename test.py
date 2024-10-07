from safetensors import safe_open
from safetensors.torch import save_file


def load_lora_weights():
    adapter_path = "ckpt/instruct_lm/llama3_alpha128_r64/checkpoint-4000/adapter_model.safetensors"
    tensors = {}
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def main():
    tensors = load_lora_weights()
    for k,v in tensors.items():
        print(f"{k}: {v.shape}")

if __name__ == "__main__":
    main()


