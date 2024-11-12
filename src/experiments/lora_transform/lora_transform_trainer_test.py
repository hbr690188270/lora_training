"""
The trainer to train a small transformation matrix for LoRA adapters
CUDA_VISIBLE_DEVICES=5 python -m src.experiments.lora_transform.lora_transform_trainer_test
"""

import logging
import os

import datasets
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, set_seed

from src.data_utils import (
    get_instruct_lm_tokenizer,
)
from src.experiments.instruct_lm.input_preprocess import (
    instruct_lm_preprocessor,
)
from src.experiments.lora_transform.lora_transform_model import (
    TransformLoraModel,
)

logger = logging.getLogger(__name__)

SOURCE_LORA_PATH="ckpt/instruct_lm/llama3_for_llama31"


def main():
    set_seed(22)
    tokenizer = get_instruct_lm_tokenizer("model_cache/llama3_1-8b")
    preprocessor = instruct_lm_preprocessor(
        tokenizer=tokenizer,
        max_len=2048,
        eot_id=128002,
        prepend_eos=False,
    )

    dataset = datasets.load_dataset("nvidia/Daring-Anteater", split="train")
    dataset = dataset.map(
        preprocessor.process_daring_anteater,
        num_proc=32,
        remove_columns=['system', 'mask', 'dataset', 'conversations'],
        batched=False,
    )

    logger.info("*** Load pretrained model ***")

    torch_dtype = torch.bfloat16
    model_kwargs = dict(
        trust_remote_code=True,
        use_flash_attention_2=True,
        torch_dtype=torch_dtype,
        use_cache=True,
        device_map=None,
        cache_dir='./model_cache'
    )

    model = AutoModelForCausalLM.from_pretrained("model_cache/llama3_1-8b", **model_kwargs)
    # peft_config = LoraConfig.from_json_file(
    peft_config = LoraConfig.from_pretrained(
        SOURCE_LORA_PATH,
    )
    print(peft_config)

    # Load the fine-tuned LoRA from LLaMA3
    model = TransformLoraModel(model, peft_config)
    print(model.base_model)
    import ipdb
    ipdb.set_trace()
    model.load_adapter(SOURCE_LORA_PATH, adapter_name="default")
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "transform_matrix" in name:
            param.requires_grad_(True)
        print(name, param.requires_grad)
    model.print_trainable_parameters()


if __name__ == "__main__":
    main()


