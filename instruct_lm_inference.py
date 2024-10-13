"""
CUDA_VISIBLE_DEVICES=3 python instruct_lm_inference.py \
    --task=ifeval \
    --model=llama3 \
    --adapter_source=llama3

CUDA_VISIBLE_DEVICES=2 python instruct_lm_inference.py \
    --task=ifeval \
    --model=llama31 \
    --adapter_source=llama31

CUDA_VISIBLE_DEVICES=1 python instruct_lm_inference.py \
    --task=ifeval \
    --model=llama31 \
    --adapter_source=none

CUDA_VISIBLE_DEVICES=0 python instruct_lm_inference.py \
    --task=ifeval \
    --model=llama31 \
    --adapter_source=llama3_converted

"""

import json
import logging

import torch
import tqdm
import transformers
from absl import app, flags
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaForCausalLM

import datasets
from src.common import move_to_target_device
from src.data_utils import (
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
            "llama3_converted",
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
    if FLAGS.task == "ifeval":
        dataset = datasets.load_dataset("google/IFEval", split="train")
        orig_prompts = dataset["prompt"]
        prompts = [apply_chat_template(x) for x in orig_prompts]
    else:
        raise NotImplementedError
    print(prompts[:3])

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

    print(model_name_or_path)
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )
    model = model.to(device)

    if FLAGS.adapter_source != "none":
        if FLAGS.adapter_source == "llama3_converted":
            adapter_dir = "ckpt/instruct_lm/llama3_for_llama31/"
            print("use the updated lora version")
        else:
            # adapter_dir = f"ckpt/instruct_lm/{FLAGS.adapter_source}_alpha128_r64/"
            # adapter_dir = f"ckpt/instruct_lm/{FLAGS.adapter_source}_alpha128_r64/checkpoint-6000"
            adapter_dir = f"ckpt/instruct_lm/{FLAGS.adapter_source}_alpha128_r64/checkpoint-16797"
        # model.load_adapter(adapter_dir, adapter_name = "sft")
        # model.set_adapter(["sft"])
        model: LlamaForCausalLM = PeftModel.from_pretrained(model, adapter_dir)
        # model.merge_and_unload(progressbar=True)
        # model = model.model

    model.eval()


    logger.info("*** Generation ***")

    generation_config = transformers.GenerationConfig(
        max_new_tokens=1000,
        do_sample=False,
        num_beams=1,
        stop_strings=["<turn_end>"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.pad_token_id,
    )

    batch_size = 2
    all_batches = []
    curr_batch = []
    for idx in range(len(prompt_tokens_ids)):
        curr_token_ids = prompt_tokens_ids[idx]
        curr_batch.append(curr_token_ids)
        if len(curr_batch) == batch_size or idx == len(prompt_tokens_ids) - 1:
            all_batches.append(curr_batch)
            curr_batch = []

    generation_logs = []
    global_index = 0

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
                return_dict_in_generate=True,
                tokenizer=tokenizer,
                attention_mask=attention_mask,
            )
            input_length = prefix.shape[1]
            generated_token_ids = outputs.sequences[:, input_length:]
            generated_texts = [tokenizer.decode(x, skip_special_tokens=True) for x in generated_token_ids]

            # prompts = [tokenizer.decode(x, skip_special_tokens=True) for x in prefix]

            for idx in range(len(generated_texts)):
                print(f"prompt: {orig_prompts[global_index]}")
                print(f"generation: {generated_texts[idx]}")
                print("\n\n")

                log = {
                    "prompt": orig_prompts[global_index],
                    "response": generated_texts[idx],
                }
                generation_logs.append(log)
                global_index += 1

    with open(f"generations/ifeval_logs_{FLAGS.adapter_source}_to_{FLAGS.model}.jsonl", "w") as f:
        for log in generation_logs:
            f.write(json.dumps(log) + "\n")

if __name__ == "__main__":
    set_eval_args()
    app.run(main=main)
