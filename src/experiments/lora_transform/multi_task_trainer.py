"""
The trainer to train a small transformation matrix based on multiple tasks and the datasets

To view all available configs, run:

```
import src.experiments.lora_transform.multi_task_trainer as mt
configs = mt.named_trainer_configs()
print(configs.keys())
```

CUDA_VISIBLE_DEVICES=3,4,5,6 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --main_process_port 29504 \
    --config_file configs/a6000_config.yaml \
    src/experiments/lora_transform/lora_transform_trainer_v2.py \
    --config_name=PQBA-llama3-8b-mistral-7b-v3-hellaswag-lr5e-4
"""

import logging
import os
import sys
from dataclasses import dataclass, replace
from typing import Dict, Tuple

import datasets
import numpy as np
import torch
import transformers
from absl import app, flags
from peft import LoraConfig
from transformers import AutoModelForCausalLM, Trainer, set_seed

from src.cmd_parser import (
    SFTConfig,
)
from src.data_utils import (
    DataCollatorForInstructLM,
    get_tokenizer,
)
from src.experiments.lora_transform.lora_transform_model import (
    PQBALoraModel,
    PQBASTLoraModel,
)
from src.experiments.lora_transform.train_utils import (
    DATASET_TO_INDEX,
    FLAN_PATH,
    TASKSET_ID_TO_TASKS,
    TRAINING_RECIPE,
)
from src.experiments.pretrain_tasks.input_preprocess import (
    PretrainingTaskPreprocessor,
    get_dataset_and_preprocess_fn,
)

logger = logging.getLogger(__name__)
os.environ["WANDB_PROJECT"]="LoRA-Transfer"

FLAGS = flags.FLAGS

@dataclass
class LoraConfigV2(LoraConfig):
    pass

def set_flags():
    flags.DEFINE_string(
        "config_name",
        None,
        help="Name of the trainer config. Must be defined in `named_trainer_configs()` function",
    )

def load_PQBA_transform_configs():
    trainer_configs: Dict[str, Tuple[Dict, Dict, SFTConfig]] = {}
    config_params = [
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v1", "a6000"],
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v2", "a6000"],
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v1", "h100"],
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v2", "h100"],
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v3", "h100"],
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v4", "h100"],
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v5", "h100"],
    ]
    for source_model, target_model, task_set_id, server_cfg in config_params:
        src_ = os.path.basename(source_model)
        tgt_ = os.path.basename(target_model)
        if source_model == target_model:
            continue
        model_args = dict(
            source_model=source_model,
            target_model=target_model,
            torch_dtype=torch.bfloat16,
            use_peft=True,
            trust_remote_code=True,
            use_flash_attention_2=True,
            lora_r=64,
            lora_alpha=128,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_modules_to_save=None,
            lora_transform_type="PQBA",
            lora_dropout=0.05,
        )
        data_args = dict(
            training_task=task_set_id,
        )
        recipe_names = ["default", "lr5e-5", "lr1e-4", "lr5e-4", "test"]
        for recipe_name in recipe_names:
            sft_training_args = TRAINING_RECIPE[server_cfg][recipe_name]
            if sft_training_args is None:
                continue
            output_dir = (
                f"ckpt/PQBA_transform/{src_}-{tgt_}-mt{task_set_id}-{recipe_name}-{server_cfg}/"
            )
            run_name = (
                f"ckpt/PQBA_transform/{src_}-{tgt_}-mt{task_set_id}-{recipe_name}-{server_cfg}/"
            )
            sft_training_args = replace(
                sft_training_args,
                output_dir=output_dir,
                run_name=run_name,
            )
            cfg_name = (
                f"PQBA-{src_}-{tgt_}-mt{task_set_id}-{recipe_name}-{server_cfg}"
            )
            trainer_configs[cfg_name] = (model_args, data_args, sft_training_args)
    return trainer_configs

def load_PQBAST_transform_configs():
    trainer_configs: Dict[str, Tuple[Dict, Dict, SFTConfig]] = {}
    config_params = [
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v1", "a6000"],
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v2", "a6000"],
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v1", "h100"],
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v2", "h100"],
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v3", "h100"],
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v4", "h100"],
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v5", "h100"],
    ]
    for source_model, target_model, task_set_id, server_cfg in config_params:
        src_ = os.path.basename(source_model)
        tgt_ = os.path.basename(target_model)
        if source_model == target_model:
            continue
        model_args = dict(
            source_model=source_model,
            target_model=target_model,
            torch_dtype=torch.bfloat16,
            use_peft=True,
            trust_remote_code=True,
            use_flash_attention_2=True,
            lora_r=64,
            lora_alpha=128,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_modules_to_save=None,
            lora_transform_type="PQBAST",
            lora_dropout=0.05,
        )
        data_args = dict(
            training_task=task_set_id,
        )
        recipe_names = [
            "default", "lr5e-5", "lr1e-4", "lr5e-4", "lr1e-4-bsz4", "lr1e-4-bsz4-epoch2", "test"
        ]
        for recipe_name in recipe_names:
            sft_training_args = TRAINING_RECIPE[server_cfg][recipe_name]
            if sft_training_args is None:
                continue
            output_dir = (
                f"ckpt/PQBAST_transform/{src_}-{tgt_}-mt{task_set_id}-{recipe_name}-{server_cfg}/"
            )
            run_name = (
                f"ckpt/PQBAST_transform/{src_}-{tgt_}-mt{task_set_id}-{recipe_name}-{server_cfg}/"
            )
            sft_training_args = replace(
                sft_training_args,
                output_dir=output_dir,
                run_name=run_name,
            )
            cfg_name = (
                f"PQBAST-{src_}-{tgt_}-mt{task_set_id}-{recipe_name}-{server_cfg}"
            )
            trainer_configs[cfg_name] = (model_args, data_args, sft_training_args)
    return trainer_configs


def named_trainer_configs():
    trainer_configs: Dict[str, Tuple[Dict, Dict, SFTConfig]] = {}

    # multi_task_configs = load_PQBA_transform_configs()
    # trainer_configs.update(multi_task_configs)

    multi_task_PQBAST_configs = load_PQBAST_transform_configs()
    trainer_configs.update(multi_task_PQBAST_configs)

    return trainer_configs


def main(argv):
    trainer_configs = named_trainer_configs()
    config_name = FLAGS.config_name
    if config_name not in trainer_configs:
        print(f"Found unrecognized config name `{config_name}`")
        print(f"Do you mean {list(trainer_configs.keys())}?")
        raise ValueError()

    model_args, data_args, training_args = trainer_configs[config_name]

    print(training_args.output_dir)
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    tokenizer = get_tokenizer(
        model_args["target_model"],
    )
    preprocessor = PretrainingTaskPreprocessor(
        tokenizer=tokenizer,
        max_len=training_args.max_seq_length,
    )

    data_collator = DataCollatorForInstructLM(
        tokenizer=tokenizer,
    )
    training_tasks = TASKSET_ID_TO_TASKS[data_args["training_task"]]
    extra_eval_tasks = TASKSET_ID_TO_TASKS["v1"] + TASKSET_ID_TO_TASKS["v2"]
    all_tasks = training_tasks + extra_eval_tasks
    flan_v2_dataset = None
    if data_args["training_task"] in ["v3", "v4", "v5"]:
        flan_v2_dataset = datasets.load_from_disk(FLAN_PATH)

    all_train_datasets = []
    all_valid_datasets = []
    all_test_datasets = []
    # for task in training_tasks:
    for task in all_tasks:
        dataset, preprocess_fn, remove_columns = get_dataset_and_preprocess_fn(
            task=task,
            preprocessor=preprocessor,
            FLAN_dataset=flan_v2_dataset,
        )
        dataset = dataset.map(
            preprocess_fn,
            num_proc=32,
            remove_columns=remove_columns,
            batched=False,
        )
        dataset = dataset.add_column("dataset_index", [DATASET_TO_INDEX[task]] * len(dataset))
        # dataset = dataset.add_column("dataset_index", [task] * len(dataset))

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

        if task in training_tasks:
            all_train_datasets.append(train_dataset)
        all_valid_datasets.append(eval_dataset)
        all_test_datasets.append(test_dataset)

    all_train_datasets = all_train_datasets
    data_sizes = np.array([len(x) for x in all_train_datasets])
    data_sizes = np.power(data_sizes, 0.5)
    data_sizes = data_sizes / np.sum(data_sizes)
    data_sizes = data_sizes.tolist()
    print(f"dataset sizes: {data_sizes}")

    interleaved_dataset = datasets.interleave_datasets(
        all_train_datasets,
        probabilities=data_sizes,
        seed=training_args.seed,
        stopping_strategy="all_exhausted",
    )
    eval_dataset = datasets.DatasetDict(
        {all_tasks[idx]: all_valid_datasets[idx] for idx in range(len(all_tasks))}
    )
    test_dataset = datasets.DatasetDict(
        {all_tasks[idx]: all_test_datasets[idx] for idx in range(len(all_tasks))}
    )
    # eval_dataset = datasets.concatenate_datasets(all_valid_datasets)
    # test_dataset = datasets.concatenate_datasets(all_test_datasets)

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

    model = AutoModelForCausalLM.from_pretrained(model_args["target_model"], **model_kwargs)
    # model = CustomMistral.from_pretrained(model_args["target_model"], **model_kwargs)
    peft_config = LoraConfig(
        r=model_args["lora_r"],
        lora_alpha=model_args["lora_alpha"],
        lora_dropout=model_args["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=model_args["lora_target_modules"],
        modules_to_save=model_args["lora_modules_to_save"],
    )

    # Load the fine-tuned LoRA from LLaMA3
    if model_args["lora_transform_type"] == "BTA":
        raise NotImplementedError("Only support PQBA for multi-task learning!")
    elif model_args["lora_transform_type"] == "PQBA":
        model = PQBALoraModel(model, peft_config)
    elif model_args["lora_transform_type"] == "PQBAST":
        model = PQBASTLoraModel(model, peft_config)
    else:
        raise NotImplementedError()

    src_ = os.path.basename(model_args["source_model"])
    # for task in training_tasks:
    for task in all_tasks:
        model.add_adapter(adapter_name=task, peft_config=peft_config)
        source_lora_path = f"ckpt/ptr/{src_}-{task}"
        print(f"loading adapters from {source_lora_path}")
        model.load_adapter(source_lora_path, adapter_name=task)
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "transform_matrix" in name:
            param.requires_grad_(True)
    model.print_trainable_parameters()

    # training_args.eval_on_start = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=interleaved_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


    logger.info("*** Train ***")
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    set_flags()
    app.run(main)


