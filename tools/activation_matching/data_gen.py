"""
python tools/activation_matching/data_gen.py \
    --dataset=Daring-Anteater \
    --model=llama31 \
    --adapter_source=llama31 \
    --num_examples=60

python tools/activation_matching/data_gen.py \
    --dataset=Daring-Anteater \
    --model=llama31 \
    --adapter_source=llama3_converted \
    --num_examples=60
"""

import torch
from absl import app, flags
from peft import PeftModel
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm.auto import tqdm as auto_tqdm

import datasets
from src.common import move_to_target_device
from src.data_utils import (
    DataCollatorForInstructLM,
    get_instruct_lm_tokenizer,
)
from src.experiments.instruct_lm.input_preprocess import (
    EOT_TOKEN,
    instruct_lm_preprocessor,
)
from tools.custome_llama import (
    LlamaForCausalLM,
)

SYSTEM_PROMPT = "system\nA conversation between a user and a helpful assistant.<turn_end>"
def apply_chat_template(prompt: str):
    formated_prompt = SYSTEM_PROMPT + " user\n" + prompt + EOT_TOKEN
    formated_prompt += " assistant\n"
    return formated_prompt

MODEL_NAME_CONVERTER = {
    "llama3": "model_cache/llama3-8b",
    "llama31": "model_cache/llama3_1-8b",
}
FLAGS = flags.FLAGS

def set_eval_args():
    flags.DEFINE_enum(
        "dataset",
        None,
        [
            "Daring-Anteater",
        ],
        help="Dataset that used to collect intermediate outputs after equipped with LoRA.",
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
    flags.DEFINE_integer(
        "num_examples",
        None,
        help="number of examples to be evaluated",
    )


def main(argv):
    model_name_or_path = MODEL_NAME_CONVERTER[FLAGS.model]
    tokenizer = get_instruct_lm_tokenizer(
        model_name_or_path,
    )
    print(SYSTEM_PROMPT + " user\n")
    prefix_tokens = tokenizer(
        SYSTEM_PROMPT + " user\n",
        add_special_tokens=False,
        truncation=True,
    )["input_ids"]
    prefix_length = len(prefix_tokens)
    print(f"Prefix length: {prefix_length}")

    preprocessor = instruct_lm_preprocessor(
        tokenizer=tokenizer,
        max_len=2048,
        eot_id=128002,
        prepend_eos=False,
    )

    data_collator = DataCollatorForInstructLM(
        tokenizer=tokenizer,
    )

    dataset = datasets.load_dataset("nvidia/Daring-Anteater", split="train")
    dataset = dataset.map(
        preprocessor.process_daring_anteater,
        num_proc=32,
        remove_columns=['system', 'mask', 'dataset', 'conversations'],
        batched=False,
    )

    device = torch.device("cuda")
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        trust_remote_code=True,
    )
    model = model.to(device)

    if FLAGS.adapter_source != "none":
        if FLAGS.adapter_source == "llama3_converted":
            adapter_dir = "ckpt/instruct_lm/llama3_for_llama31/"
            print("use the updated lora version")
        else:
            adapter_dir = f"ckpt/instruct_lm/{FLAGS.adapter_source}_alpha128_r64/checkpoint-16797"
    model: LlamaForCausalLM = PeftModel.from_pretrained(model, adapter_dir)
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=1,shuffle=False,collate_fn=data_collator)

    prog_bar = auto_tqdm(range(FLAGS.num_examples))
    LLAMA_8B_LAYERS = 32
    all_per_layer_attention_outputs = [[] for x in range(LLAMA_8B_LAYERS)]

    count = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = move_to_target_device(batch, device)
            outputs = model(
                **batch,
            )
            per_layer_attention_outputs = outputs.per_layer_attention_outputs
            # import ipdb
            # ipdb.set_trace()
            for idx in range(LLAMA_8B_LAYERS):
                # remove the batch dimension
                attn_output = per_layer_attention_outputs[idx].squeeze(0)
                attn_output = attn_output[prefix_length: 500]
                all_per_layer_attention_outputs[idx].append(attn_output)
            prog_bar.update(1)
            count += 1
            if count > FLAGS.num_examples:
                break

    all_per_layer_attention_outputs = [
        torch.cat(x,dim=0) for x in all_per_layer_attention_outputs
    ]
    # import ipdb
    # ipdb.set_trace()

    save_tensors = {
        f"layer_{idx}": all_per_layer_attention_outputs[idx]
        for idx in range(LLAMA_8B_LAYERS)
    }
    filename = f"logs/activations/{FLAGS.model}_{FLAGS.adapter_source}_{FLAGS.num_examples}.safetensor"
    save_file(tensors=save_tensors, filename=filename)


if __name__ == "__main__":
    set_eval_args()
    app.run(main)

