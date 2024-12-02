"""
Trainer for LoRA fine-tuning on pre-training evaluation tasks, such as GSM8K, ARC, and Sciq.
To view all available configs, run:

```
import src.experiments.pretrain_tasks.pretrain_task_trainer as sp
configs = sp.named_trainer_configs()
for key in configs.keys():
    print(key)
```

CUDA_VISIBLE_DEVICES=3,4,5,6 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --main_process_port 29504 --config_file configs/a6000_config.yaml \
    src/experiments/pretrain_tasks/pretrain_task_trainer.py \
    --config_name="ptr-llama3-8b-arc_challenge"

"""

import logging
import os

import datasets
import numpy as np
import torch
from absl import app, flags
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, set_seed

from src.data_utils import (
    DataCollatorForInstructLM,
    get_tokenizer,
)
from src.experiments.lora_transform.lora_transform_model import (
    PQBASTLoraModel,
)
from src.experiments.lora_transform.multi_task_trainer import LoraConfigV2
from src.experiments.lora_transform.train_utils import (
    FLAN_PATH,
    TRAINING_RECIPE,
)
from src.experiments.pretrain_tasks.input_preprocess import (
    PretrainingTaskPreprocessor,
    get_dataset_and_preprocess_fn,
)

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS
os.environ["WANDB_PROJECT"]="LoRA-Transfer"

def set_flags():
    flags.DEFINE_string(
        "model_path",
        None,
        help="path to the huggingface model",
    )
    flags.DEFINE_string(
        "adapter_path",
        None,
        help="Path to the directory that stores the adapter and its config",
    )
    flags.DEFINE_string(
        "task",
        None,
        help="The FLAN task to be evaluated."
    )


def main(argv):
    set_seed(42)

    tokenizer = get_tokenizer(
        FLAGS.model_path,
    )
    preprocessor = PretrainingTaskPreprocessor(
        tokenizer=tokenizer,
        max_len=768,
    )

    data_collator = DataCollatorForInstructLM(
        tokenizer=tokenizer,
    )
    flan_v2_dataset = datasets.load_from_disk(FLAN_PATH)
    dataset, preprocess_fn, remove_columns = get_dataset_and_preprocess_fn(
        task=FLAGS.task,
        preprocessor=preprocessor,
        FLAN_dataset=flan_v2_dataset,
    )
    dataset = dataset.map(
        preprocess_fn,
        num_proc=32,
        remove_columns=remove_columns,
        batched=False,
    )
    np.random.seed(222)
    num_examples = len(dataset)
    rand_indices = np.random.permutation(num_examples)
    dataset = dataset.select(rand_indices)
    train_dataset = dataset.select(np.arange(int(num_examples * 0.9)))
    eval_dataset = dataset.select(
        np.arange(int(num_examples * 0.9), int(num_examples * 0.95)),
    )
    test_dataset = dataset.select(
        np.arange(int(num_examples * 0.95), num_examples,)
    )

    logger.info("*** Load pretrained model ***")

    torch_dtype = torch.bfloat16
    model_kwargs = dict(
        trust_remote_code=True,
        use_flash_attention_2=True,
        torch_dtype=torch_dtype,
        use_cache=True,
        device_map=None,
        cache_dir="./model_cache"
    )

    model = AutoModelForCausalLM.from_pretrained(FLAGS.model_path, **model_kwargs)
    # lora_cfg_path = os.path.join(FLAGS.adapter_path, "adapter_config.json")
    # loaded_attributes = LoraConfigV2.from_json_file(lora_cfg_path)
    # peft_config = LoraConfigV2(**loaded_attributes)

    lora_cfg_path = os.path.join(FLAGS.adapter_path, "adapter_config.json")
    loaded_attributes = LoraConfig.from_json_file(lora_cfg_path)
    peft_config = LoraConfig(**loaded_attributes)

    model = get_peft_model(model, peft_config)
    model.load_adapter(FLAGS.adapter_path, adapter_name="sft")

    model.print_trainable_parameters()

    print(train_dataset)
    print(eval_dataset)
    print(test_dataset)

    training_args = TRAINING_RECIPE["a6000"]["default"]
    training_args.report_to = []
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    metrics = trainer.evaluate(eval_dataset=test_dataset)
    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    set_flags()
    app.run(main)



