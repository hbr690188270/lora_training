"""
python tools/permute_lora_weights.py --task=dream_read_the_following_conversation_and_answer_the_question --model=llama3
"""

import os
from typing import List, Union

import numpy as np
import torch
from absl import app, flags
from safetensors import safe_open
from safetensors.torch import save_file

PHI3_NUM_ATTN_HEADS = 32
PHI3_HIDDEN_SIZE = 3072
PHI3_HEAD_SIZE = PHI3_HIDDEN_SIZE // PHI3_NUM_ATTN_HEADS

LLAMA3_NUM_ATTN_HEADS = 32
LLAMA3_NUM_GROUPS = 8
LLAMA3_HIDDEN_SIZE = 4096

MODEL_NAME_CONVERTER = {
    "phi3": "model_cache/phi-3-mini-4k-instruct",
    "phi35": "model_cache/phi-3.5-mini-instruct",
    "llama3": "model_cache/llama3-8b-instruct",
    "llama31": "model_cache/llama3_1-8b-instruct",
}
FLAGS = flags.FLAGS

def set_eval_args():
    flags.DEFINE_enum(
        "task",
        None,
        [
            "adversarial_qa_dbert_answer_the_following_q",
            "cos_e_v1_11_generate_explanation_given_text",
            "dream_read_the_following_conversation_and_answer_the_question",
            "glue_qnli_2_0_0"
        ],
        help="Task to be performed. Choose from the available tasks.",
    )
    flags.DEFINE_enum(
        "model",
        None,
        [
            "phi3",
            "phi35",
            "llama3",
            "llama31"
        ],
        help="model to be evauated.",
    )

def load_lora_weights():
    adapter_path = f"ckpt/{FLAGS.model}/{FLAGS.task}_alpha128_r64_chat/adapter_model.safetensors"
    tensors = {}
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def permute_linear_layer_param(
    *,
    weight_mat: torch.Tensor,
    chunk_num: int,
    permutation_idxs: np.ndarray,
    permute_rows = True,
):
    """Permute the parameters of a certain linear layer in transformer
    The permutation is grouped by the attention heads

    Args:
        weight_mat: a tensor with shape [out_features, in_features],
            or with shape [out_features, lora_dim]
        chunk_num: the size of attention head
    """
    out_features, in_features = weight_mat.size()
    if permute_rows:
        num_heads = chunk_num
        _weight_mat = weight_mat.reshape(shape = [num_heads, -1, in_features])
        permutation_idxs = torch.from_numpy(permutation_idxs).long()
        permuted_weight_mat = _weight_mat.index_select(dim=0, index=permutation_idxs)
        permuted_weight_mat = permuted_weight_mat.reshape(out_features, in_features)
    else:
        num_heads = chunk_num
        _weight_mat = weight_mat.reshape(shape = [out_features, num_heads, -1])
        permutation_idxs = torch.from_numpy(permutation_idxs).long()
        permuted_weight_mat = _weight_mat.index_select(dim=1, index=permutation_idxs)
        permuted_weight_mat = permuted_weight_mat.reshape(out_features, in_features)

    return permuted_weight_mat

def random_permute(
    weight_mat: torch.Tensor,
    permutation_idxs: np.ndarray,
):
    out_features, _in_features = weight_mat.size()
    permutation_idxs = torch.from_numpy(permutation_idxs).long()
    assert out_features == permutation_idxs.size(0)
    permuted_weight_mat = weight_mat.index_select(dim=0, index=permutation_idxs)
    return permuted_weight_mat

def load_permutation_configs():
    if FLAGS.model in ["phi3", "phi35"]:
        target_number_of_heads = PHI3_NUM_ATTN_HEADS
    elif FLAGS.model in ["llama3", "llama31"]:
        target_number_of_heads = LLAMA3_NUM_GROUPS
    else:
        raise NotImplementedError
    permute_first_2_heads = np.arange(target_number_of_heads)
    permute_first_2_heads[0] = 1
    permute_first_2_heads[1] = 0

    permute_last_2_heads = np.arange(target_number_of_heads)
    permute_last_2_heads[target_number_of_heads-1] = target_number_of_heads - 2
    permute_last_2_heads[target_number_of_heads-2] = target_number_of_heads - 1

    np.random.seed(111)
    permute_all_heads = np.random.permutation(target_number_of_heads)
    print(permute_all_heads)

    configs = {
        "first_2": permute_first_2_heads,
        "last_2": permute_last_2_heads,
        "all": permute_all_heads
    }

    return configs

def generate_permutated_lora():
    # TODO: Avoid hard-coded checkpoint
    ckpt_path = "ckpt/ckpt_phi3/dream_read_the_following_conversation_and_answer_the_question/checkpoint-2500/adapter_model.safetensors"

    tensors = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)


    configs = load_permutation_configs()
    for config_name, permutation_idxs in configs.items():
        permuted_tensors = {}
        _tensors = tensors.copy()
        for key, weight_mat in _tensors.items():
            lora_part = key.split(".")[-2]
            param_name = key.split(".")[-3]
            assert lora_part in ["lora_B", "lora_A"]
            if lora_part == "lora_A":
                permuted_tensors[key] = weight_mat
                continue
            if param_name == "o_proj":
                permuted_mat = permute_linear_layer_param(
                    weight_mat=weight_mat,
                    chunk_num=PHI3_HEAD_SIZE,
                    permutation_idxs=permutation_idxs,
                )
                permuted_tensors[key] = permuted_mat
            elif param_name == "qkv_proj":
                q,k,v = torch.chunk(weight_mat, chunks=3, dim=0)
                permuted_qkv = []
                for mat in [q,k,v]:
                    permuted_mat = permute_linear_layer_param(
                        weight_mat=mat,
                        chunk_num=PHI3_NUM_ATTN_HEADS,
                        permutation_idxs=permutation_idxs,
                    )
                    permuted_qkv.append(permuted_mat)
                permuted_qkv = torch.cat(permuted_qkv, dim = 0)
                assert permuted_qkv.size(0) == weight_mat.size(0)
                permuted_tensors[key] = permuted_qkv

            else:
                raise NotImplementedError
        save_name = f"ckpt/permuted_dream/{config_name}/adapter_model.safetensors"
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))
        save_file(permuted_tensors, save_name)

# pylint: disable-next=too-many-branches
def generate_structured_permutation(layer_idx: Union[List[int], int]):
    """
    Permute the dimensions of LoRA adapters while keeping their structure unchanged.
    Given LoRA parameters from a transformer layer (Q, K, V, O), we permute:
        B_Q, B_K, B_V: permutation on rows
        B_O: permutation on columns
    This guarantees the output vector do not change for the model.

    Args:
        layer_idx: a integer or a list of integers
    """
    tensors = load_lora_weights()
    if type(layer_idx) is list:
        for idx in range(len(layer_idx) - 1):
            assert layer_idx[idx] == layer_idx[idx+1] - 1
    else:
        layer_idx = [layer_idx]

    permutation_idxs = load_permutation_configs()["all"]
    permuted_tensors = {}
    for key, weight_mat in tensors.items():
        lora_part = key.split(".")[-2]
        param_name = key.split(".")[-3]
        assert lora_part in ["lora_B", "lora_A"]

        param_layer = int(key.split("layers.")[-1].split(".")[0])
        if param_layer not in layer_idx:
            permuted_tensors[key] = weight_mat
            continue

        if FLAGS.model == "llama3":
            if param_name == "o_proj":
                if lora_part == "lora_B":
                    permuted_tensors[key] = weight_mat
                else:
                    permuted_mat = permute_linear_layer_param(
                        weight_mat=weight_mat,
                        chunk_num=LLAMA3_NUM_GROUPS,
                        permutation_idxs=permutation_idxs,
                        permute_rows=False,
                        # permute_rows=True,
                    )
                    permuted_tensors[key] = permuted_mat
            elif param_name in ["q_proj", "k_proj", "v_proj"]:
                if lora_part == "lora_A":
                    permuted_tensors[key] = weight_mat
                else:
                    permuted_mat = permute_linear_layer_param(
                        weight_mat=weight_mat,
                        chunk_num=LLAMA3_NUM_GROUPS,
                        permutation_idxs=permutation_idxs,
                        # permute_rows=False,
                    )
                    permuted_tensors[key] = permuted_mat
            else:
                raise NotImplementedError
        elif FLAGS.model == "phi3":
            if param_name == "o_proj":
                if lora_part == "lora_B":
                    permuted_tensors[key] = weight_mat
                else:
                    permuted_mat = permute_linear_layer_param(
                        weight_mat=weight_mat,
                        chunk_num=PHI3_NUM_ATTN_HEADS,
                        permutation_idxs=permutation_idxs,
                        permute_rows=False,
                    )
                    permuted_tensors[key] = permuted_mat
            elif param_name == "qkv_proj":
                if lora_part == "lora_B":
                    permuted_tensors[key] = weight_mat
                else:
                    q,k,v = torch.chunk(weight_mat, chunks=3, dim=0)
                    permuted_qkv = []
                    for mat in [q,k,v]:
                        permuted_mat = permute_linear_layer_param(
                            weight_mat=mat,
                            chunk_num=PHI3_NUM_ATTN_HEADS,
                            permutation_idxs=permutation_idxs,
                        )
                        permuted_qkv.append(permuted_mat)
                    permuted_qkv = torch.cat(permuted_qkv, dim = 0)
                    assert permuted_qkv.size(0) == weight_mat.size(0)
                    permuted_tensors[key] = permuted_qkv
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    layer_info = f"layer{layer_idx[0]}" if len(layer_idx) == 1 else "layer" + "-".join([f"{layer_idx[0]}", f"{layer_idx[-1]}"])
    save_name = f"ckpt/permuted/{FLAGS.task}/{FLAGS.model}/{layer_info}/adapter_model.safetensors"
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    save_file(permuted_tensors, save_name)

def main(argv):
    # generate_permutated_lora()
    layer_idx = [x for x in range(5)]
    generate_structured_permutation(layer_idx=layer_idx)

if __name__ == "__main__":
    set_eval_args()
    app.run(main)

