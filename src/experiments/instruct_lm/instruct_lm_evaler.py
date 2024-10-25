"""
CUDA_VISIBLE_DEVICES=1 python instruct_lm_evaler.py \
    --task=daring_anteater \
    --model=llama31 \
    --adapter_source=llama3_converted_and_absorbed

CUDA_VISIBLE_DEVICES=2 python instruct_lm_evaler.py \
    --task=daring_anteater \
    --model=llama31 \
    --adapter_source=llama3_converted
"""

import logging

import numpy as np
import torch
import tqdm
from absl import app, flags
from peft import LoraConfig, PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, LlamaForCausalLM

import datasets
from src.common import move_to_target_device
from src.data_utils import (
    DataCollatorForInstructLM,
    get_instruct_lm_tokenizer,
)
from src.experiments.instruct_lm.input_preprocess import (
    instruct_lm_preprocessor,
)
from src.experiments.lora_transform.lora_transform_model import (
    TransformLoraModel,
)

logger = logging.getLogger(__name__)


MODEL_NAME_CONVERTER = {
    "llama3": "model_cache/llama3-8b",
    "llama31": "model_cache/llama3_1-8b",
}
ADAPTER_PATH_CONVERTER = {
    "llama3": "ckpt/instruct_lm/llama3_alpha128_r64",
    "llama31": "ckpt/instruct_lm/llama31_alpha128_r64/checkpoint-16797",
    "llama3_converted": "ckpt/instruct_lm/llama3_for_llama31",
    "llama3_identity_transform": "ckpt/instruct_lm/llama3_alpha128_r64",
    "llama3_converted_and_identity_transform": "ckpt/instruct_lm/llama3_for_llama31",
    "llama31_identity_transform": "ckpt/instruct_lm/llama31_alpha128_r64/checkpoint-16797",
    "llama3_converted_and_trained_transform": "ckpt/instruct_lm/llama3_transform_for_31_alpha128_r64",
    "llama3_converted_and_absorbed": "ckpt/instruct_lm/llama3_transform_for_31_absorbed",
}

FLAGS = flags.FLAGS

def set_eval_args():
    flags.DEFINE_enum(
        "task",
        None,
        [
            "daring_anteater",
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
            "llama3_converted_rescale",
            "llama31",
            "llama3_identity_transform",
            "llama3_converted_and_identity_transform",
            "llama31_identity_transform",
            "llama3_converted_and_trained_transform",
            "llama3_converted_and_absorbed",
            "none", # none means we do not load adpaters
        ],
        help="which model's adapter to load. None means do not load any adapters",
    )

def main(argv):
    model_name_or_path = MODEL_NAME_CONVERTER[FLAGS.model]
    tokenizer = get_instruct_lm_tokenizer(
        model_name_or_path,
    )

    preprocessor = instruct_lm_preprocessor(
        tokenizer=tokenizer,
        max_len=2048,
        eot_id=128002,
        prepend_eos=False,
    )

    data_collator = DataCollatorForInstructLM(
        tokenizer=tokenizer,
    )

    if FLAGS.task == "daring_anteater":
        dataset = datasets.load_dataset("nvidia/Daring-Anteater", split="train")
        dataset = dataset.map(
            preprocessor.process_daring_anteater,
            num_proc=32,
            remove_columns=['system', 'mask', 'dataset', 'conversations'],
            batched=False,
        )
        np.random.seed(222)
        num_examples = len(dataset)
        rand_indices = np.random.permutation(num_examples)
        dataset = dataset.select(rand_indices)
        # dataset = dataset.shuffle()
        train_dataset = dataset.select(np.arange(int(num_examples * 0.9)))
        eval_dataset = dataset.select(
            np.arange(int(num_examples * 0.9), int(num_examples * 0.95)),
        )
        test_dataset = dataset.select(
            np.arange(int(num_examples * 0.95), num_examples,)
        )

        eval_dataset = eval_dataset.select(np.arange(200))
        print(train_dataset)
        print(eval_dataset)
        print(test_dataset)
    else:
        raise NotImplementedError()


    logger.info("*** Load pretrained model ***")
    device = torch.device("cuda")

    model_kwargs = dict(
        trust_remote_code=True,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16,
        use_cache=True,
        device_map=None,
        cache_dir='./model_cache',
    )


    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    if FLAGS.adapter_source is not None:
        adapter_dir = ADAPTER_PATH_CONVERTER[FLAGS.adapter_source]
        assert adapter_dir is not None

        if "transform" in FLAGS.adapter_source:
            peft_config = LoraConfig.from_pretrained(
                adapter_dir,
            )
            model = TransformLoraModel(model, peft_config)
            model.load_adapter(adapter_dir, "default")
        else:
            model = PeftModel.from_pretrained(model, adapter_dir)
        # model = model.model

    model = model.to(device)
    model.eval()

    eval_dataloader = DataLoader(eval_dataset, batch_size = 4, collate_fn=data_collator, shuffle = False)

    logger.info("*** Evaluate ***")
    all_loss = []
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_dataloader):
            # print(batch.keys())
            # pause = input("???")
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
