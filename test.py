from dataclasses import dataclass, field
from typing import List

import datasets
import torch
import torch.nn.functional as F
from peft import LoraConfig
from safetensors import safe_open
from transformers import (
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from yaml import safe_load

from src.data_utils import load_flan_subset
from src.experiments.lora_transform.train_utils import FLAN_PATH


COMMON_MODEL_ARGS = dict(
    torch_dtype=torch.bfloat16,
    use_peft=True,
    trust_remote_code=True,
    use_flash_attention_2=True,
    lora_r=64,
    lora_alpha=128,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_modules_to_save=None,
    lora_transform_type="PQBAST",
    lora_dropout=0.05,
)

@dataclass
class LoraConfigV2(LoraConfig):
    transform_r_multiple: int = field(default=1, metadata={"help": "Lora attention dimension"})


def load_lora_weights():
    adapter_path = "ckpt/instruct_lm/llama3_alpha128_r64/checkpoint-16797/adapter_model.safetensors"
    tensors = {}
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def load_llama():
    model = AutoModelForCausalLM.from_pretrained(
        "model_cache/llama3_1-8b-instruct",
    )
    print([key for key, _ in model.named_modules()])

def load_yaml():
    with open("configs/instruct_lm_llama3/config_v2.yaml", "r") as f:
        content = safe_load(f)
    for k,v in content.items():
        print(k, v)

def compare_lora_transform():
    source_path = "ckpt/instruct_lm/llama3_for_llama31/adapter_model.safetensors"
    # target_path = "ckpt/instruct_lm/llama3_transform_for_31_alpha128_r64-step2000/adapter_model.safetensors"
    target_path = "ckpt/instruct_lm/llama3_transform_for_31_alpha128_r64/checkpoint-11500/adapter_model.safetensors"

    source_tensors = {}
    with safe_open(source_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            source_tensors[key] = f.get_tensor(key)

    finetuned_tensors = {}
    with safe_open(target_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            finetuned_tensors[key] = f.get_tensor(key)
    # for k,v in finetuned_tensors.items():
    #     print(f"{k}, {v.size()}")
    # key = "base_model.model.model.layers.9.self_attn.o_proj.lora_A.weight"
    key = "base_model.model.model.layers.31.self_attn.o_proj.lora_transform_matrix.weight"
    # source_value = source_tensors[key]
    target_value = finetuned_tensors[key]
    # diff = source_value - target_value
    # print(diff.mean())

    print(target_value)

def parameter_matching(
    source_train_loraA: List[torch.Tensor],
    source_train_loraB: List[torch.Tensor],
    target_train_loraA: List[torch.Tensor],
    target_train_loraB: List[torch.Tensor],
):
    """
    Args:
        source_train_loraA: a list of torch.FloatTensor. Each is 32-by-64-by-4096
        source_train_loraB: a list of torch.FloatTensor. Each is 32-by-4096-by-64
        source_train_loraA: a list of torch.FloatTensor. Each is 32-by-64-by-4096
        source_train_loraB: a list of torch.FloatTensor. Each is 32-by-4096-by-64
    """
    device = torch.device("cuda")

    transform_matrices = torch.ones(
        source_train_loraB[0].size(0),
        source_train_loraB[0].size(2),
        source_train_loraB[0].size(2),
    ).float().to(device)

    transform_matrices.requires_grad_(True)

    all_train_losses = []
    num_epochs = 100
    lr = 1e-3
    for _ in range(num_epochs):
        all_losses = []
        for idx in range(len(source_train_loraA)):

            curr_tgt_loraB = target_train_loraB[idx].to(device)
            curr_tgt_loraA = target_train_loraA[idx].to(device)
            with torch.no_grad():
                target_weights = torch.einsum(
                    "bij,bjk->bik",
                    curr_tgt_loraB,
                    curr_tgt_loraA,
                )

            curr_src_loraB = source_train_loraB[idx].to(device)
            curr_src_loraA = source_train_loraA[idx].to(device)
            source_transformed_weights = torch.einsum(
                "bij,bjk->bik",
                curr_src_loraB,
                torch.einsum(
                    "brr,brd->brd",
                    transform_matrices,
                    curr_src_loraA,
                )
            )
            mse_loss = F.mse_loss(source_transformed_weights, target_weights)
            mse_loss.backward()
            all_losses.append(mse_loss)
            torch.cuda.empty_cache()

            transform_matrices.data = transform_matrices.data - lr * transform_matrices.grad
            transform_matrices.grad.zero_()


        total_loss = torch.mean(all_losses)
        all_train_losses.append(total_loss.item())

def torch_indexing():
    matrix = torch.tensor(
        [
            [1,2,3],
            [4,5,6],
            [7,8,9],
        ]
    )
    indices = [0,2]
    res = matrix[indices]
    print(res)

def test_flan():
    flan_v2_dataset = datasets.load_from_disk(FLAN_PATH)
    task_names = flan_v2_dataset.unique("task_name")
    print("Num Tasks: ", len(task_names))
    for task_name in task_names:
        print(f"  {task_name}")
    subset = load_flan_subset(
        flan_v2_dataset,
        taskname="wiki_bio_key_content",
    )
    print(subset)
    print(subset["source"][:4])
    print(subset["target"][:4])

def test_load_tokenizer():
    from src.data_utils import get_tokenizer
    tokenizer = get_tokenizer("model_cache/llama3-8b")
    print(tokenizer)

def test_train_recipe():
    from src.experiments.lora_transform.train_utils import TASKSET_ID_TO_TASKS
    tasks = TASKSET_ID_TO_TASKS["v3"]
    print(len(tasks))

def test_lora_cfg():
    cfg = LoraConfigV2()
    print(cfg)

def test_dict():
    cfgs = dict(
        a="a",
        **COMMON_MODEL_ARGS
    )
    print(cfgs)

def test_fineweb():
    from datasets import load_dataset
    # use name="sample-10BT" to use the 10BT sample
    fw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)



def main():
    # lora_tensors = load_lora_weights()
    # for k,v in lora_tensors.items():
        # print(f"{k}, {v.size()}")
    # load_llama()
    # load_yaml()
    # compare_lora_transform()
    # torch_indexing()
    test_flan()
    # test_load_tokenizer()
    # test_lora_cfg()
    # test_dict()

if __name__ == "__main__":
    main()


