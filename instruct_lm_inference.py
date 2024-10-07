"""
CUDA_VISIBLE_DEVICES=4 python gen.py \
    --task=dream_read_the_following_conversation_and_answer_the_question \
    --model=llama31 \
    --adapter_source=llama3 \
    --apply_chat_template

CUDA_VISIBLE_DEVICES=5 python gen.py \
    --task=dream_read_the_following_conversation_and_answer_the_question \
    --model=llama31 \
    --adapter_source=none \
    --apply_chat_template

CUDA_VISIBLE_DEVICES=6 python gen.py \
    --task=dream_read_the_following_conversation_and_answer_the_question \
    --model=llama31 \
    --adapter_source=llama31 \
    --apply_chat_template
"""

import logging

import datasets
import json
import numpy as np
import torch
import tqdm
import transformers
from absl import app, flags
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from src.common import move_to_target_device
from src.data_utils import (
    DataCollatorForInstructLM,
    get_instruct_lm_tokenizer,
)
from src.experiments.instruct_lm.input_preprocess import (
    EOT_TOKEN,
)

logger = logging.getLogger(__name__)


MODEL_NAME_CONVERTER = {
    "llama3": "model_cache/llama3-8b",
    "llama31": "model_cache/llama3_1-8b",
}
FLAGS = flags.FLAGS

def set_eval_args():
    flags.DEFINE_enum(
        "task",
        None,
        [
            "ifeval",
        ],
        help="Task to be performed. Choose from the available tasks.",
    )
    flags.DEFINE_enum(
        "model",
        None,
        [
            "llama3",
            "llama31"
        ],
        help="model to be evauated.",
    )
    flags.DEFINE_enum(
        "adapter_source",
        None,
        [
            "llama3",
            "llama31",
            "none", # none means we do not load adpaters
        ],
        help="which model's adapter to load. None means do not load any adapters",
    )

SYSTEM_PROMPT = "system\nA conversation between a user and a helpful assistant.<turn_end>"
def apply_chat_template(prompt: str):
    formated_prompt = SYSTEM_PROMPT + " user\n" + prompt + EOT_TOKEN
    formated_prompt += " assistant\n"
    return formated_prompt

def main(argv):
    model_name_or_path = MODEL_NAME_CONVERTER[FLAGS.model]
    tokenizer = get_instruct_lm_tokenizer(
        model_name_or_path,
    )
    if dataset == "ifeval":
        dataset = datasets.load_dataset("google/IFEval", split="train")
        prompts = dataset["prompt"]
        prompts = [apply_chat_template(x) for x in prompts]
    else:
        raise NotImplementedError

    prompt_tokens_ids = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_tensors=None,
    )["input_ids"]

    logger.info("*** Load pretrained model ***")
    device = torch.device("cuda")

    model_kwargs = dict(
        trust_remote_code=True,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16,
        use_cache=True,
        device_map=None,
        cache_dir='./model_cache'
    )

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model = model.to(device)

    if FLAGS.adapter_source != "none":
        adapter_dir = f"ckpt/insruct_lm/{FLAGS.adapter_source}_alpha128_r64/"
        model.load_adapter(adapter_dir, adapter_name = "sft")
        model.set_adapter(["sft"])

    model.eval()


    logger.info("*** Generation ***")

    generation_config = transformers.GenerationConfig(
        max_new_tokens = 128,
        do_sample=False,
        num_beams=1,
        stop_strings=["<turn_end>"]
    )

    batch_size = 4
    all_batches = []
    curr_batch = []
    for idx in range(len(prompt_tokens_ids)):
        curr_token_ids = prompt_tokens_ids[idx]
        curr_batch.append(curr_token_ids)
        if len(curr_batch) == batch_size or idx == len(prompt_tokens_ids) - 1:
            all_batches.append(curr_batch)
            curr_batch = []

    generation_logs = []

    with torch.no_grad():
        for batch in tqdm.tqdm(all_batches):
            max_length = max([len(x) for x in batch])
            all_input_ids = []
            all_attention_masks = []
            for prompt_idxs in batch:
                curr_length = len(prompt_idxs)
                difference = max_length - curr_length
                attention_mask = [0] * difference + [1] * curr_length
                pad_input_ids = [tokenizer.pad_token_id] * difference + prompt_idxs
            all_input_ids.append(pad_input_ids)
            all_attention_masks.append(attention_mask)

            prefix = torch.LongTensor(all_input_ids)
            attention_mask = torch.LongTensor(all_attention_masks)
            prefix = move_to_target_device(prefix, device)
            attention_mask = move_to_target_device(attention_mask, device)

            outputs = model.generate(
                inputs=prefix,
                generation_config=generation_config,
                return_dict_in_generate=True
            )
            input_length = prefix.shape[1]
            print(type(outputs))
            generated_token_ids = outputs.sequences[:, input_length:]
            generated_texts = [tokenizer.decode(x, skip_special_tokens=True) for x in generated_token_ids]

            prompts = [tokenizer.decode(x, skip_special_tokens=True) for x in prefix]

            for idx in range(len(generated_texts)):
                print(f"prompt: {prompts[idx]}")
                print(f"generation: {generated_texts[idx]}")
                print("\n\n")

                log = {
                    "prompt": prompts[idx],
                    "response": generated_texts[idx],
                }
                generation_logs.append(log)
    
    with open("ifeval_logs.jsonl", "w") as f:
        for log in generation_logs:
            f.write(json.dumps(log) + "\n")

if __name__ == "__main__":
    set_eval_args()
    app.run(main=main)
