"""
The trainer to train a small transformation matrix based on multiple tasks and the datasets

To view all available configs, run:

```
import src.experiments.lora_transform.mutli_task_trainer as mt
configs = mt.named_trainer_configs()
print(configs.keys())
```
"""

import logging
import os
import sys
from dataclasses import replace
from typing import Dict, Tuple

import numpy as np
import torch
import transformers
from absl import app, flags
from peft import LoraConfig
from transformers import AutoModelForCausalLM, Trainer, set_seed

import datasets
from src.cmd_parser import (
    SFTConfig,
)
from src.data_utils import (
    DataCollatorForInstructLM,
    get_instruct_lm_tokenizer,
    get_tokenizer,
)
from src.experiments.instruct_lm.input_preprocess import (
    instruct_lm_preprocessor,
)
from src.experiments.lora_transform.lora_transform_model import (
    PQBALoraModel,
    TransformLoraModel,
)
from src.experiments.lora_transform.train_utils import (
    TRAINING_RECIPE,
)
from src.experiments.pretrain_tasks.input_preprocess import (
    pretraining_task_preprocessor,
)

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS

TASKSET_ID_TO_TASKS = {
    "v1"
}

def set_flags():
    flags.DEFINE_string(
        "config_name",
        None,
        help="Name of the trainer config. Must be defined in `named_trainer_configs()` function",
    )

def load_PQBA_transform_configs():
    trainer_configs: Dict[str, Tuple[Dict, Dict, SFTConfig]] = {}
    config_params = [
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v1",],
        ["model_cache/llama3-8b", "model_cache/mistral-7b-v3", "v2",]
    ]
    for source_model, target_model, task_set_id in config_params:
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
            training_task=training_task,
        )
        if training_task == "sft":
            recipe_names = ["sft"]
        else:
            recipe_names = ["ptr_default", "ptr_lr5e-5", "ptr_lr1e-4", "ptr_lr5e-4"]


def get_configs(
    lora_source_model: str = "model_cache/llama3-8b",
    lora_target_model: str = "model_cache/mistral-7b-v3",

):
    trainer_configs: Dict[str, Tuple[Dict, Dict, SFTConfig]] = {}
    src_ = os.path.basename(lora_source_model)
    tgt_ = os.path.basename(lora_target_model)
    assert lora_source_model == lora_target_model

    model_args = dict(
        source_model=lora_source_model,
        target_model=lora_target_model,
        torch_dtype=torch.bfloat16,
        use_peft=True,
        trust_remote_code=True,
        use_flash_attention_2=True,
        lora_r=64,
        lora_alpha=128,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_modules_to_save=None,
        lora_transform_type="BTA",
        lora_dropout=0.05,
    )
    for training_task in [
        "sft", "arc", "arc_challenge", "arc_easy",
        "gsm8k", "hellaswag", "winogrande", "piqa"
    ]:
        data_args = dict(
            training_task=training_task,
        )
        if training_task == "sft":
            recipe_names = ["sft"]
        else:
            recipe_names = ["ptr_default", "ptr_lr5e-5", "ptr_lr1e-4", "ptr_lr5e-4"]
        for recipe_name in recipe_names:
            sft_training_args = TRAINING_RECIPE[recipe_name]
            output_dir = (
                f"ckpt/lora_transform/{src_}-{tgt_}-{training_task}-{recipe_name}/"
            )
            run_name = (
                f"ckpt/lora_transform/{src_}-{tgt_}-{training_task}-{recipe_name}/"
            )
            sft_training_args = replace(
                sft_training_args,
                output_dir=output_dir,
                run_name=run_name,
            )
            cfg_name = (
                f"lora_transform-{src_}-{tgt_}-{training_task}-{recipe_name}"
            )
            trainer_configs[cfg_name] = (model_args, data_args, sft_training_args)
    return trainer_configs

def named_trainer_configs():
    trainer_configs: Dict[str, Tuple[Dict, Dict, SFTConfig]] = {}
    pretraining_configs = load_lora_transform_configs()
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

    if data_args["training_task"] == "sft":
        tokenizer = get_instruct_lm_tokenizer(
            model_args["target_model"],
        )
        preprocessor = instruct_lm_preprocessor(
            tokenizer=tokenizer,
            max_len=2048,
            eot_id=128002,
            prepend_eos=False,
        )
    else:
        tokenizer = get_tokenizer(
            model_args["target_model"],
        )
        preprocessor = pretraining_task_preprocessor(
            tokenizer=tokenizer,
            max_len=training_args.max_seq_length,
        )

    data_collator = DataCollatorForInstructLM(
        tokenizer=tokenizer,
    )
    if data_args["training_task"] == "gsm8k":
        dataset = datasets.load_dataset("openai/gsm8k", "main", split="train")
        preprocess_fn = preprocessor.process_gsm8k
        remove_columns=["question", "answer"]
    elif data_args["training_task"] == "arc_challenge":
        dataset = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        preprocess_fn = preprocessor.process_arc
        remove_columns=["question", "id", "choices", "answerKey"]
    elif data_args["training_task"] == "arc_easy":
        dataset = datasets.load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
        preprocess_fn = preprocessor.process_arc
        remove_columns=["question", "id", "choices", "answerKey"]
    elif data_args["training_task"] == "arc":
        arc_challenge = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        arc_easy = datasets.load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
        dataset = datasets.concatenate_datasets([arc_challenge, arc_easy],)
        preprocess_fn = preprocessor.process_arc
        remove_columns=["question", "id", "choices", "answerKey"]
    elif data_args["training_task"] == "hellaswag":
        dataset = datasets.load_dataset("Rowan/hellaswag",   split="train")
        preprocess_fn = preprocessor.process_hellaswag
        remove_columns=[
            "ind", "activity_label", "ctx_a", "ctx_b",
            "ctx", "endings", "split", "split_type", "label"
        ]
    elif data_args["training_task"] == "piqa":
        dataset = datasets.load_dataset("ybisk/piqa",   split="train", trust_remote_code=True)
        preprocess_fn = preprocessor.process_piqa
        remove_columns=["label", "goal", "sol1", "sol2"]
    elif data_args["training_task"] == "winogrande":
        dataset = datasets.load_dataset(
            "allenai/winogrande",
            "winogrande_xl",
            split="train",
            trust_remote_code=True
        )
        preprocess_fn = preprocessor.process_winogrand
        remove_columns=["sentence", "option1", "option2", "answer"]
    elif data_args["training_task"] == "sft":
        dataset = datasets.load_dataset("nvidia/Daring-Anteater", split="train")
        preprocess_fn = preprocessor.process_daring_anteater
        remove_columns=['system', 'mask', 'dataset', 'conversations']
    else:
        raise NotImplementedError()

    dataset = dataset.map(
        preprocess_fn,
        num_proc=32,
        remove_columns=remove_columns,
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

    model = AutoModelForCausalLM.from_pretrained(model_args["target_model"], **model_kwargs)
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
        model = TransformLoraModel(model, peft_config)
    elif model_args["lora_transform_type"] == "PQBA":
        model = PQBALoraModel(model, peft_config)
    else:
        raise NotImplementedError()

    src_ = os.path.basename(model_args["source_model"])
    if data_args["training_task"] == "sft":
        if model_args["source_model"] == "llama3":
            source_lora_path = "ckpt/instruct_lm/llama3_alpha128_r64"
        elif model_args["source_model"] == "llama31":
            source_lora_path = "ckpt/instruct_lm/llama31_alpha128_r64"
        else:
            raise NotImplementedError()
    else:
        source_lora_path = f"ckpt/ptr/{src_}-{data_args['training_task']}"
    print(f"loading adapters from {source_lora_path}")

    model.load_adapter(source_lora_path, "default")
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "transform_matrix" in name:
            param.requires_grad_(True)
        # print(name, param.data.requires_grad)
    model.print_trainable_parameters()

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


