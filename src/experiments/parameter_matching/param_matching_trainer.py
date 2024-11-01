"""
Perform parameter matching for LoRA transfer

python -m src.experiments.parameter_matching.param_matching_trainer \
    --source_model=llama3 \
    --target_model=mistral \
    --train_tasks=arc,winogrande,piqa \
    --eval_tasks=gsm8k,hellaswag
"""
import os
import shutil
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from absl import app, flags
from safetensors import safe_open
from safetensors.torch import save_file

from src.common import move_to_target_device

FLAGS = flags.FLAGS
num_layers = 32
num_heads = 32
num_kv_heads = 8


def set_flags():
    flags.DEFINE_enum(
        "source_model",
        None,
        ["llama3", "llama31"],
        help="The source model on which the LoRA adapters is trained.",
    )
    flags.DEFINE_enum(
        "target_model",
        None,
        ["llama31", "mistral"],
        help="The target model that the LoRA adapters will be applied to.",
    )
    flags.DEFINE_list(
        "train_tasks",
        None,
        help="The tasks where LoRA adapters are trained on should be used for training",
    )
    flags.DEFINE_list(
        "eval_tasks",
        None,
        required=False,
        help="The tasks where the transformed LoRA adapters are evaluated",
    )
    flags.DEFINE_string(
        "output_path",
        None,
        help="Path to save the converted LoRA weights",
    )

def load_safetensors(train=True, source=True, repeat_kv=True):
    if train:
        all_tasks = FLAGS.train_tasks
    else:
        all_tasks = FLAGS.eval_tasks

    all_task_lora_A = []
    all_task_lora_B = []
    for task in all_tasks:
        if source:
            if FLAGS.source_model == "llama3":
                weight_path = f"ckpt/ptr/llama3-8b-{task}/adapter_model.safetensors"
            elif FLAGS.source_model == "llama31":
                weight_path = f"ckpt/ptr/llama3_1-8b-{task}/adapter_model.safetensors"
            else:
                raise NotImplementedError()
        else:
            if FLAGS.target_model == "llama31":
                weight_path = f"ckpt/ptr/llama3_1-8b-{task}/adapter_model.safetensors"
            elif FLAGS.target_model == "mistral":
                weight_path = f"ckpt/ptr/mistral-7b-v3-{task}/adapter_model.safetensors"
            else:
                raise NotImplementedError()

        tensor_dict = {}
        with safe_open(weight_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor_dict[key] = f.get_tensor(key)

        task_lora_A = []
        task_lora_B = []

        for idx in range(num_layers):
            for module in ["q", "k", "v", "o"]:
                lora_A_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_A.weight"
                lora_B_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_B.weight"

                lora_A = tensor_dict[lora_A_key].float()
                lora_B = tensor_dict[lora_B_key].float()
                lora_rank = lora_B.size(1)
                if module in ["k", "v"] and repeat_kv:
                    chunked_B = torch.chunk(lora_B, chunks=num_kv_heads, dim=0)
                    chunked_B = torch.stack(chunked_B, dim=0)
                    repeat_B = chunked_B.repeat(1, num_heads//num_kv_heads, 1)
                    lora_B = repeat_B.reshape(-1, lora_rank)

                task_lora_A.append(lora_A)
                task_lora_B.append(lora_B)

        task_lora_A = torch.stack(task_lora_A)
        task_lora_B = torch.stack(task_lora_B)

        all_task_lora_A.append(task_lora_A)
        all_task_lora_B.append(task_lora_B)

    return all_task_lora_A, all_task_lora_B

def do_eval(
    source_eval_loraA: List[torch.Tensor],
    source_eval_loraB: List[torch.Tensor],
    target_eval_loraA: List[torch.Tensor],
    target_eval_loraB: List[torch.Tensor],
    transform_matrices: torch.Tensor,
):
    all_losses = []
    for idx in range(len(source_eval_loraA)):
        target_weights = torch.einsum(
            "bij,bjk->bik",
            target_eval_loraB[idx],
            target_eval_loraA[idx],
        )
        source_transformed_weights = torch.einsum(
            "bij,bjk->bik",
            source_eval_loraB[idx],
            torch.einsum(
                "brr,brd->brd",
                transform_matrices,
                source_eval_loraA[idx],
            )
        )
        source_transformed_weights = source_transformed_weights * 100
        target_weights = target_weights * 100
        mse_loss = F.mse_loss(source_transformed_weights, target_weights)
        all_losses.append(mse_loss.item())
    total_loss = np.mean(all_losses)
    return total_loss

def convert_transform_matrices_to_safetensor_dict(
    transform_matrices: torch.nn.Parameter,
):
    _transform_matrices = transform_matrices.data

    tensor_dict = {}
    # transform_matrices is sequentially concatenated using the lora transform matrix for each
    # Lora adapter with the following orders
    count = 0
    for idx in range(num_layers):
        for module in ["q", "k", "v", "o"]:
            curr_transform_matrix = _transform_matrices[count]
            name = f"layer{idx}_{module}_transform"
            tensor_dict[name] = curr_transform_matrix
            count += 1

    return tensor_dict

def apply_transform_to_target_tasks(
    transform_tensor_dict: Dict[str, torch.Tensor]
):
    device = torch.device("cuda")
    all_tasks = FLAGS.eval_tasks
    for task in all_tasks:
        if FLAGS.source_model == "llama31":
            weight_path = f"ckpt/ptr/llama3_1-8b-{task}/adapter_model.safetensors"
            config_path = f"ckpt/ptr/llama3_1-8b-{task}/adapter_config.json"
        elif FLAGS.source_model == "llama3":
            weight_path = f"ckpt/ptr/llama3-8b-{task}/adapter_model.safetensors"
            config_path = f"ckpt/ptr/llama3-8b-{task}/adapter_config.json"
        else:
            raise NotImplementedError()

        tensor_dict = {}
        with safe_open(weight_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor_dict[key] = f.get_tensor(key)

        for idx in range(num_layers):
            for module in ["q", "k", "v", "o"]:
                transform_name = f"layer{idx}_{module}_transform"
                transform_tensor = transform_tensor_dict[transform_name].to(device)

                lora_A_name = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_A.weight"
                lora_A = tensor_dict[lora_A_name].to(device).float()
                new_lora_A = torch.matmul(transform_tensor, lora_A).bfloat16().cpu()
                tensor_dict[lora_A_name] = new_lora_A

        output_path = f"ckpt/ptr_transform/{FLAGS.source_model}-{FLAGS.target_model}-{task}/adapter_model.safetensors"
        output_config_path = f"ckpt/ptr_transform/{FLAGS.source_model}-{FLAGS.target_model}-{task}/adapter_config.json"
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        save_file(tensor_dict, output_path)
        shutil.copy2(config_path, output_config_path)





def main(argv):
    device = torch.device("cuda")

    source_train_loraA, source_train_loraB = load_safetensors(train=True, source=True)
    target_train_loraA, target_train_loraB = load_safetensors(train=True, source=False)

    source_train_loraA = move_to_target_device(source_train_loraA, device)
    source_train_loraB = move_to_target_device(source_train_loraB, device)
    target_train_loraA = move_to_target_device(target_train_loraA, device)
    target_train_loraB = move_to_target_device(target_train_loraB, device)

    source_eval_loraA, source_eval_loraB = load_safetensors(train=False, source=True)
    target_eval_loraA, target_eval_loraB = load_safetensors(train=False, source=False)

    source_eval_loraA = move_to_target_device(source_eval_loraA, device)
    source_eval_loraB = move_to_target_device(source_eval_loraB, device)
    target_eval_loraA = move_to_target_device(target_eval_loraA, device)
    target_eval_loraB = move_to_target_device(target_eval_loraB, device)


    transform_matrices = torch.nn.Parameter(
        data = torch.ones(
            source_train_loraB[0].size(0),
            source_train_loraB[0].size(2),
            source_train_loraB[0].size(2),
            device=device,
        ).float(),
        requires_grad=True,
    )

    optimizer = torch.optim.SGD([transform_matrices], lr=20000.0)
    transform_matrices = transform_matrices.to(device)

    num_qkvo = source_train_loraB[0].size(0)
    batch_size = 16
    num_batches = num_qkvo // batch_size
    all_train_losses = []
    all_eval_losses = []
    num_epochs = 500
    # num_epochs = 5
    for _ in tqdm.tqdm(range(num_epochs)):
        all_losses = []
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        for idx in range(len(source_train_loraA)):
            for batch_id in range(num_batches):
                curr_tgt_loraB = (
                    target_train_loraB[idx][batch_id * batch_size: (batch_id + 1) * batch_size]
                )
                curr_tgt_loraA = (
                    target_train_loraA[idx][batch_id * batch_size: (batch_id + 1) * batch_size]
                )
                # curr_tgt_loraB = move_to_target_device(
                #     target_train_loraB[idx][batch_id * batch_size: (batch_id + 1) * batch_size],
                #     device,
                # )
                # curr_tgt_loraA = move_to_target_device(
                #     target_train_loraA[idx][batch_id * batch_size: (batch_id + 1) * batch_size],
                #     device,
                # )
                with torch.no_grad():
                    target_weights = torch.einsum(
                        "bij,bjk->bik",
                        curr_tgt_loraB,
                        curr_tgt_loraA,
                    )

                curr_src_loraB = (
                    source_train_loraB[idx][batch_id * batch_size: (batch_id + 1) * batch_size]
                )
                curr_src_loraA = (
                    source_train_loraA[idx][batch_id * batch_size: (batch_id + 1) * batch_size]
                )
                # curr_src_loraB = move_to_target_device(
                #     source_train_loraB[idx][batch_id * batch_size: (batch_id + 1) * batch_size],
                #     device,
                # )
                # curr_src_loraA = move_to_target_device(
                #     source_train_loraA[idx][batch_id * batch_size: (batch_id + 1) * batch_size],
                #     device,
                # )
                source_transformed_weights = torch.einsum(
                    "bij,bjk->bik",
                    curr_src_loraB,
                    torch.einsum(
                        "brr,brd->brd",
                        transform_matrices[batch_id * batch_size: (batch_id + 1) * batch_size],
                        curr_src_loraA,
                    )
                )
                source_transformed_weights = source_transformed_weights * 100
                target_weights = target_weights * 100
                mse_loss = F.mse_loss(source_transformed_weights, target_weights)

                # optimizer.zero_grad()
                mse_loss.backward()
                # optimizer.step()

                all_losses.append(mse_loss.item())
                torch.cuda.empty_cache()

        optimizer.step()

        total_loss = np.mean(all_losses)
        # total_loss.backward()

        # with torch.no_grad():
        #     transform_matrices.data = transform_matrices.data - lr * transform_matrices.grad
        #     transform_matrices.grad.zero_()

        with torch.no_grad():
            eval_loss = do_eval(
                source_eval_loraA,
                source_eval_loraB,
                target_eval_loraA,
                target_eval_loraB,
                transform_matrices,
            )
        all_train_losses.append(total_loss.item())
        all_eval_losses.append(eval_loss.item())
        print(all_train_losses[-1], all_eval_losses[-1])
        torch.cuda.empty_cache()

    transform_dict = convert_transform_matrices_to_safetensor_dict(transform_matrices)
    apply_transform_to_target_tasks(transform_dict)
    plt.figure(figsize=(10, 6))
    plt.plot(all_train_losses, label='Training Loss', marker='o')
    plt.title("Training Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    save_path = "logs/figures/train_loss.pdf"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(all_eval_losses, label='Evaluation Loss', marker='s')
    plt.title("Evaluation Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    save_path = "logs/figures/eval_loss.pdf"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.clf()


if __name__ == "__main__":
    set_flags()
    app.run(main)







