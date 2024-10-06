import os
import sys

import numpy as np
from transformers import LlamaForCausalLM

# Add the parent directory to the sys.path so that Python can find `tools/`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.permute_lora_weights import (
    LLAMA3_NUM_GROUPS,
    load_permutation_configs,
    permute_linear_layer_param,
)


def permute_llama_layer(
    llm: LlamaForCausalLM,
    layer_idx: int,
):
    """
    permute the QKVO attention matrices of an LLM
    """
    permutation_idxs = load_permutation_configs()["all"]
    q_mat = llm.model.layers[layer_idx].self_attn.q_proj.weight.data
    k_mat = llm.model.layers[layer_idx].self_attn.k_proj.weight.data
    v_mat = llm.model.layers[layer_idx].self_attn.v_proj.weight.data
    o_mat = llm.model.layers[layer_idx].self_attn.o_proj.weight.data

    q_mat_permuted = permute_linear_layer_param(
        weight_mat=q_mat,
        chunk_num=LLAMA3_NUM_GROUPS,
        permutation_idxs=permutation_idxs,
    )
    k_mat_permuted = permute_linear_layer_param(
        weight_mat=k_mat,
        chunk_num=LLAMA3_NUM_GROUPS,
        permutation_idxs=permutation_idxs,
    )
    v_mat_permuted = permute_linear_layer_param(
        weight_mat=v_mat,
        chunk_num=LLAMA3_NUM_GROUPS,
        permutation_idxs=permutation_idxs,
    )
    o_mat_permuted = permute_linear_layer_param(
        weight_mat=o_mat,
        chunk_num=LLAMA3_NUM_GROUPS,
        permutation_idxs=permutation_idxs,
        permute_rows=False,
    )

    llm.model.layers[layer_idx].self_attn.q_proj.weight.data = q_mat_permuted
    llm.model.layers[layer_idx].self_attn.k_proj.weight.data = k_mat_permuted
    llm.model.layers[layer_idx].self_attn.v_proj.weight.data = v_mat_permuted
    llm.model.layers[layer_idx].self_attn.o_proj.weight.data = o_mat_permuted

def permute_attn_layer(
    attn,
    chunk_num: int,
):
    """
    permute the QKVO attention matrices of an LLM
    """
    # permutation_idxs = np.array([x for x in range(chunk_num)])
    permutation_idxs = np.random.permutation(chunk_num)
    q_mat = attn.q_proj.weight.data
    k_mat = attn.k_proj.weight.data
    v_mat = attn.v_proj.weight.data
    o_mat = attn.o_proj.weight.data

    q_mat_permuted = permute_linear_layer_param(
        weight_mat=q_mat,
        chunk_num=chunk_num,
        permutation_idxs=permutation_idxs,
    )
    k_mat_permuted = permute_linear_layer_param(
        weight_mat=k_mat,
        chunk_num=chunk_num,
        permutation_idxs=permutation_idxs,
    )
    v_mat_permuted = permute_linear_layer_param(
        weight_mat=v_mat,
        chunk_num=chunk_num,
        permutation_idxs=permutation_idxs,
    )
    o_mat_permuted = permute_linear_layer_param(
        weight_mat=o_mat,
        chunk_num=chunk_num,
        permutation_idxs=permutation_idxs,
        permute_rows=False,
    )

    attn.q_proj.weight.data = q_mat_permuted
    attn.k_proj.weight.data = k_mat_permuted
    attn.v_proj.weight.data = v_mat_permuted
    attn.o_proj.weight.data = o_mat_permuted

