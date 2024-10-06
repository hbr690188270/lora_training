"""
CUDA_VISIBLE_DEVICES=4 python eval.py \
    --task=dream_read_the_following_conversation_and_answer_the_question \
    --model=llama3 \
    --adapter_source=llama3 \
    --apply_chat_template

CUDA_VISIBLE_DEVICES=4 python eval.py \
    --task=dream_read_the_following_conversation_and_answer_the_question \
    --model=llama31 \
    --adapter_source=llama3 \
    --adapter_path=ckpt/permuted/dream_read_the_following_conversation_and_answer_the_question/llama3/layer0-9/ \
    --apply_chat_template

CUDA_VISIBLE_DEVICES=0 python eval.py \
    --task=dream_read_the_following_conversation_and_answer_the_question \
    --model=llama3 \
    --apply_chat_template

CUDA_VISIBLE_DEVICES=4 python eval.py \
    --task=dream_read_the_following_conversation_and_answer_the_question \
    --model=llama31 \
    --adapter_source=llama3 \
    --apply_chat_template
"""

import logging

import numpy as np
import torch
import tqdm
import transformers
from absl import app, flags
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, LlamaForCausalLM

from src.common import move_to_target_device
from src.data_utils import (
    DataCollatorCompletionOnly,
    DataCollatorWithPaddingSFT,
    get_tokenizer,
    load_taskdataset,
)
from src.permute_utils import permute_llama_layer

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
    flags.DEFINE_string(
        "adapter_path",
        None,
        help="The path of the saved adapter.",
        required=False,
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
    )

    logger.info("*** Load pretrained model ***")
    device = torch.device("cuda")

    model_kwargs = dict(
        trust_remote_code=True,
        # use_flash_attention_2=True,
        use_flash_attention_2=False,
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.float32,
        use_cache=True,
        device_map=None,
        cache_dir='./model_cache',
    )


    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    if FLAGS.adapter_path is None and FLAGS.adapter_source is not None:
        # adapter_dir = f"ckpt/{FLAGS.adapter_source}/{FLAGS.task}"
        adapter_dir = f"ckpt/{FLAGS.adapter_source}/{FLAGS.task}_alpha128_r64_chat/checkpoint-1500"
        model.load_adapter(adapter_dir, adapter_name = "sft")
        model.set_adapter(["sft"])
    elif FLAGS.adapter_path is not None:
        adapter_dir = FLAGS.adapter_path
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.merge_and_unload(progressbar=True)
        model = model.model
        # model.load_adapter(adapter_dir, adapter_name = "sft")
        # model.set_adapter(["sft"])

        # permuted_layers_info = adapter_dir.split("layer")[1]
        # if permuted_layers_info.endswith("/"):
        #     permuted_layers_info = permuted_layers_info[:-1]
        # if "-" in permuted_layers_info:
        #     start_layer, end_layer = permuted_layers_info.split("-", 1)
        #     start_layer = int(start_layer)
        #     end_layer = int(end_layer)
        #     layer_list = [x for x in range(start_layer, end_layer + 1)]
        # else:
        #     tgt_layer = int(permuted_layers_info)
        #     layer_list = [tgt_layer]
        # print(layer_list)
        # for layer_idx in layer_list:
        #     permute_llama_layer(model, layer_idx=layer_idx)


    # layer_list = [x for x in range(10)]
    # for layer_idx in layer_list:
    #     permute_llama_layer(model, layer_idx=layer_idx)
    model = model.to(device)
    model.eval()

    eval_dataset = dataset_dict["val"].select(np.arange(32))
    # test_dataset = dataset_dict["test"]
    # eval_dataloader = DataLoader(eval_dataset, batch_size = 16, collate_fn=data_collator, shuffle = False)
    eval_dataloader = DataLoader(eval_dataset, batch_size = 4, collate_fn=data_collator, shuffle = False)

    logger.info("*** Evaluate ***")
    all_loss = []
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_dataloader):
            batch = move_to_target_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            all_loss.append(loss)

    all_loss = torch.stack(all_loss)
    avg_loss = torch.mean(all_loss).item()
    print(f"loss: {avg_loss}")

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    set_eval_args()
    app.run(main=main)
