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

import itertools
import logging
import os
import sys
from dataclasses import replace
from typing import Dict, Tuple

import datasets
import numpy as np
import torch
import transformers
from absl import app, flags
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, set_seed

from src.cmd_parser import (
    DataArguments,
    ModelArguments,
    SFTConfig,
)
from src.data_utils import (
    DataCollatorForInstructLM,
    get_tokenizer,
)
from src.experiments.lora_transform.train_utils import (
    FLAN_PATH,
    INDEX_TO_DATASET,
    TASKSET_ID_TO_TASKS,
    h100_lr25_short_seq_train_config,
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
        "config_name",
        None,
        help="Name of the trainer config. Must be defined in `named_trainer_configs()` function",
    )


def load_pretraining_tasks_configs()->Dict[str, Tuple[ModelArguments, DataArguments, SFTConfig]]:
    trainer_configs = {}
    # models = ["model_cache/llama3-8b", "model_cache/mistral-7b-v3"]
    models = ["model_cache/llama3-8b"]
    pretrain_pcts = [0, 30]
    lora_ranks = [64, 512]
    # all_cfgs = itertools.product(models, pretrain_pcts, INDEX_TO_DATASET)
    all_cfgs = itertools.product(models, pretrain_pcts, INDEX_TO_DATASET[:30], lora_ranks)
    for model_path, ptr_pct, taskname, lora_r in all_cfgs:
        model_name_for_shot = os.path.basename(model_path)
        model_args = ModelArguments(
            model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            use_peft=True,
            trust_remote_code=True,
            use_flash_attention_2=True,
            lora_r=lora_r,
            lora_alpha=128,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_modules_to_save=None,
        )
        data_args = DataArguments(
            dataset_name=taskname,
            ptr_pct=ptr_pct,
        )
        sft_training_args = h100_lr25_short_seq_train_config
        output_dir = (
            f"ckpt/ptr/{model_name_for_shot}-{taskname}"
        )
        run_name = (
            f"ptr-{model_name_for_shot}-{taskname}"
        )
        if ptr_pct != 0:
            output_dir += f"-ptr{ptr_pct}"
            run_name += f"-ptr{ptr_pct}"
        if lora_r != 64:
            output_dir += f"-lora{ptr_pct}"
            run_name += f"-lora{ptr_pct}"

        sft_training_args = replace(
            sft_training_args,
            output_dir=output_dir,
            run_name=run_name,
        )

        cfg_name = f"ptr-{model_name_for_shot}-{taskname}-ptr{ptr_pct}-lora{lora_r}"
        trainer_configs[cfg_name] = (model_args, data_args, sft_training_args)

    return trainer_configs


def named_trainer_configs()->Dict[str, Tuple[ModelArguments, DataArguments, SFTConfig]]:
    trainer_configs = {}
    pretraining_configs = load_pretraining_tasks_configs()
    trainer_configs.update(pretraining_configs)
    return trainer_configs

def main(argv):
    trainer_configs = named_trainer_configs()
    config_name = FLAGS.config_name
    if config_name not in trainer_configs:
        print(f"Found unrecognized config name `{config_name}`")
        print(f"Do you mean {list(trainer_configs.keys())}?")
        raise ValueError()

    model_args, data_args, training_args = trainer_configs[config_name]

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

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Config name: {config_name}")

    tokenizer = get_tokenizer(
        model_args.model_name_or_path,
    )
    preprocessor = PretrainingTaskPreprocessor(
        tokenizer=tokenizer,
        max_len=training_args.max_seq_length,
    )

    data_collator = DataCollatorForInstructLM(
        tokenizer=tokenizer,
    )
    flan_v2_dataset = None
    flan_tasks = TASKSET_ID_TO_TASKS["v6"] + TASKSET_ID_TO_TASKS["v5"]
    if data_args.dataset_name in flan_tasks:
        flan_v2_dataset = datasets.load_from_disk(FLAN_PATH)
    dataset, preprocess_fn, remove_columns = get_dataset_and_preprocess_fn(
        task=data_args.dataset_name,
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
    if data_args.ptr_pct > 0:
        ptr_dataset, ptr_preprocess_fn, ptr_remove_columns = get_dataset_and_preprocess_fn(
            task="pretrain",
            preprocessor=preprocessor,
        )
        ptr_dataset = ptr_dataset.map(
            ptr_preprocess_fn,
            num_proc=32,
            remove_columns=ptr_remove_columns,
            batched=False,
        )
        np.random.seed(222)
        num_examples = len(ptr_dataset)
        rand_indices = np.random.permutation(num_examples)
        ptr_dataset = ptr_dataset.select(rand_indices)
        train_ptr_dataset = ptr_dataset.select(np.arange(int(num_examples * 0.9)))
        eval_ptr_dataset = ptr_dataset.select(
            np.arange(int(num_examples * 0.9), int(num_examples * 0.905)),
        )
        test_ptr_dataset = ptr_dataset.select(
            np.arange(int(num_examples * 0.995), num_examples,)
        )

        probabilities = np.array([100 - data_args.ptr_pct, data_args.ptr_pct])
        probabilities = probabilities / np.sum(probabilities)
        train_dataset = datasets.interleave_datasets(
            [train_dataset, train_ptr_dataset],
            probabilities=probabilities,
            seed=training_args.seed,
            stopping_strategy="first_exhausted",
        )

        eval_dataset = datasets.DatasetDict(
            {
                data_args.dataset_name: eval_dataset,
                "pretrain": eval_ptr_dataset,
            }
        )
        test_dataset = datasets.DatasetDict(
            {
                data_args.dataset_name: test_dataset,
                "pretrain": test_ptr_dataset,
            }
        )


    logger.info("*** Load pretrained model ***")

    torch_dtype = torch.bfloat16
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=None,
        cache_dir="./model_cache"
    )

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(train_dataset)
    print(eval_dataset)
    print(test_dataset)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


    logger.info("*** Train ***")
    train_result = trainer.train(resume_from_checkpoint=None)
    # train_result = trainer.train(resume_from_checkpoint=True)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
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



