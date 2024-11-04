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

import numpy as np
import torch
import tqdm
import transformers
from absl import app, flags
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from src.common import move_to_target_device
from src.data_utils import (
    DataCollatorCompletionOnly,
    DataCollatorWithPaddingSFT,
    get_tokenizer,
    load_taskdataset,
)

logger = logging.getLogger(__name__)


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
    flags.DEFINE_enum(
        "adapter_source",
        None,
        [
            "phi3",
            "phi35",
            "llama3",
            "llama31",
            "none", # none means we do not load adpaters
        ],
        help="which model's adapter to load. None means do not load any adapters",
    )
    flags.DEFINE_boolean(
        "apply_chat_template",
        default=False,
        help="whether apply chat template for each example."
    )

def main(argv):
    model_name_or_path = MODEL_NAME_CONVERTER[FLAGS.model]
    tokenizer = get_tokenizer(
        model_name_or_path,
    )
    if FLAGS.apply_chat_template:
        assert FLAGS.model in ["llama3", "llama31"]
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        print(f"response_template: {response_template}")
        data_collator = DataCollatorCompletionOnly(
            response_token_ids=response_token_ids,
            tokenizer=tokenizer,
            padding=transformers.utils.PaddingStrategy.LONGEST,
            max_length=768,
        )
    else:
        data_collator = DataCollatorWithPaddingSFT(
            tokenizer=tokenizer,
            padding=transformers.utils.PaddingStrategy.LONGEST,
            max_length=768,
        )

    dataset_dict = load_taskdataset(
        taskname=FLAGS.task,
        tokenizer=tokenizer,
        apply_chat_template=FLAGS.apply_chat_template,
        for_generation=True
    )

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
        adapter_dir = f"ckpt/{FLAGS.adapter_source}/{FLAGS.task}_alpha128_r64_chat/"
        model.load_adapter(adapter_dir, adapter_name = "sft")
        model.set_adapter(["sft"])
        # adapter_dir = f"ckpt/{FLAGS.adapter_source}/{FLAGS.task}"
        # model.load_adapter(adapter_dir, adapter_name = "sft")
        # model.set_adapter(["sft"])

    model.eval()

    # eval_dataset = dataset_dict["val"]
    eval_dataset = dataset_dict["test"].select(np.arange(128))
    eval_dataloader = DataLoader(eval_dataset, batch_size = 16, collate_fn=data_collator)

    logger.info("*** Generation ***")

    generation_config = transformers.GenerationConfig(
        max_new_tokens = 128,
        do_sample=False,
        num_beams=1,
    )

    with torch.no_grad():
        for batch in tqdm.tqdm(eval_dataloader):
            batch = move_to_target_device(batch, device)
            prefix = batch["input_ids"]
            outputs = model.generate(
                inputs=prefix,
                generation_config=generation_config,
                return_dict_in_generate=True
            )
            input_length = prefix.shape[1]
            print(type(outputs))
            generated_token_ids = outputs.sequences[:, input_length:]
            generated_texts = [tokenizer.decode(x, skip_special_tokens=True) for x in generated_token_ids]

            ground_truth_outputs_ids = batch["completion_ids"]
            groupd_truth_outputs = [tokenizer.decode(x) for x in ground_truth_outputs_ids]

            prompts = [tokenizer.decode(x, skip_special_tokens=True) for x in prefix]

            for idx in range(len(generated_texts)):
                print(f"prompt: {prompts[idx]}")
                print(f"ground-truth: {groupd_truth_outputs[idx]}")
                print(f"generation: {generated_texts[idx]}")
                print("\n\n")

if __name__ == "__main__":
    set_eval_args()
    app.run(main=main)
