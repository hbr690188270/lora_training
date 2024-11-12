import copy

import torch
from absl.testing import absltest
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm

from .permute_utils import permute_attn_layer

cfg = LlamaConfig(
    hidden_size = 4096,
    intermediate_size = 14336,
    num_attention_heads = 32,
    num_key_value_heads = 8,
    vocab_size = 128256,
)

device = torch.device("cuda:0")

class PermutationTest(absltest.TestCase):
    def test_permute(self, ):
        input_layernorm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        post_attention_layernorm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        llama_attn = LlamaAttention(cfg, layer_idx=0)
        orig_attn = copy.deepcopy(llama_attn)
        permute_attn_layer(
            llama_attn,
            chunk_num=8,
        )

        llama_attn = llama_attn.to(device)
        orig_attn = orig_attn.to(device)
        input_layernorm = input_layernorm.to(device)
        post_attention_layernorm = post_attention_layernorm.to(device)

        hidden_states = torch.rand(size=[1, 4, 4096]).to(device)
        residual1 = hidden_states

        hidden_states1 = input_layernorm(hidden_states)
        position_ids = torch.arange(0, 4).view(1,-1).to(device)
        res1, weight1 = llama_attn(
            hidden_states=hidden_states1,
            position_ids=position_ids,
            output_attentions=True,
        )[:2]
        res1 = residual1 + res1
        res1 = post_attention_layernorm(res1)



        residual2 = hidden_states
        hidden_states2 = input_layernorm(hidden_states)
        res2, weight2 = orig_attn(
            hidden_states=hidden_states2,
            position_ids=position_ids,
            output_attentions=True,
        )[:2]
        res2 = residual2 + res2
        res2 = post_attention_layernorm(res2)


        print(res1)
        print(res2)

        # print(weight1[0,])
        # print(weight2[0,])

        print(llama_attn.q_proj.weight.data)
        print(orig_attn.q_proj.weight.data)
