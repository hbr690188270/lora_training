"""
Trainer for LoRA fine-tuning on pre-training evaluation tasks, such as GSM8K, ARC, and Sciq.
To view all available configs, run:

```
import src.experiments.pretrain_tasks.pretrain_task_trainer as sp
configs = sp.named_trainer_configs()
print(configs)
```

CUDA_VISIBLE_DEVICES=3,4,5,6 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --main_process_port 29504 --config_file configs/a6000_config.yaml \
    src/experiments/pretrain_tasks/pretrain_task_trainer.py \
    --config_name="ptr-llama3-8b-arc_challenge"

"""

import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import transformers
from absl import app, flags
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, set_seed

import datasets
from src.cmd_parser import (
    DataArguments,
    ModelArguments,
    SFTConfig,
)
from src.data_utils import (
    DataCollatorForInstructLM,
    get_tokenizer,
)
from src.experiments.pretrain_tasks.input_preprocess import (
    pretraining_task_preprocessor,
)

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS

def set_flags():
    flags.DEFINE_string(
        "config_name",
        None,
        help="Name of the trainer config. Must be defined in `named_trainer_configs()` function",
    )

DATASET_TO_TRAIN_EPOCHS = defaultdict(lambda: 3)
DATASET_TO_TRAIN_EPOCHS.update(
    {
        "arc": 3,
        "arc_challenge": 3,
        "arc_easy": 3,
        "gsm8k": 3,
        "hellaswag": 1,
    }
)

def load_pretraining_tasks_configs():
    trainer_configs = {}
    for model_path in ["model_cache/llama3-8b", "model_cache/llama3_1-8b"]:
        model_name_for_shot = os.path.basename(model_path)
        for taskname in [
            "arc", "arc_challenge", "arc_easy", "gsm8k", "hellaswag", "winogrande", "piqa"
        ]:
            model_args = ModelArguments(
                model_name_or_path=model_path,
                model_name_for_short="llama3-8b",
                torch_dtype=torch.bfloat16,
                use_peft=True,
                trust_remote_code=True,
                use_flash_attention_2=True,
                lora_r=64,
                lora_alpha=128,
                lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_modules_to_save=None,
            )
            data_args = DataArguments(
                dataset_name=taskname,
            )

            if taskname == "gsm8k":
                max_seq_length = 2048
                per_device_train_batch_size = 2
                per_device_eval_batch_size = 2
            elif taskname in [
                "arc_challenge", "arc_easy", "arc", "hellaswag", "winogrande", "piqa"
            ]:
                max_seq_length = 768
                per_device_train_batch_size = 4
                per_device_eval_batch_size = 4
            else:
                raise ValueError(f"Unrecognized task {taskname}")
            sft_training_args = SFTConfig(
                max_seq_length=max_seq_length,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                gradient_accumulation_steps=4,
                num_train_epochs=3,
                learning_rate=2.0e-5,
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                do_eval=True,
                bf16=True,
                output_dir=f"ckpt/ptr/{model_name_for_shot}-{taskname}/",
                save_only_model=True,
                save_strategy="steps",
                save_steps=2000,
                save_total_limit=4,
                remove_unused_columns=False,
                report_to="wandb",
                run_name=f"ptr-{model_name_for_shot}-{taskname}",
                warmup_ratio=0.1,
                seed=42,
                push_to_hub=False,
                logging_steps=10,
                log_level="info",
                gradient_checkpointing=False,
            )

            cfg_name = f"ptr-{model_name_for_shot}-{taskname}"
            trainer_configs[cfg_name] = (model_args, data_args, sft_training_args)

    return trainer_configs

def named_trainer_configs():
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
    preprocessor = pretraining_task_preprocessor(
        tokenizer=tokenizer,
        max_len=training_args.max_seq_length,
    )

    data_collator = DataCollatorForInstructLM(
        tokenizer=tokenizer,
    )

    if data_args.dataset_name == "gsm8k":
        dataset = datasets.load_dataset("openai/gsm8k", "main", split="train")
        preprocess_fn = preprocessor.process_gsm8k
        remove_columns=["question", "answer"]
    elif data_args.dataset_name == "arc_challenge":
        dataset = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        preprocess_fn = preprocessor.process_arc
        remove_columns=["question", "id", "choices", "answerKey"]
    elif data_args.dataset_name == "arc_easy":
        dataset = datasets.load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
        preprocess_fn = preprocessor.process_arc
        remove_columns=["question", "id", "choices", "answerKey"]
    elif data_args.dataset_name == "arc":
        arc_challenge = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        arc_easy = datasets.load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
        dataset = datasets.concatenate_datasets([arc_challenge, arc_easy],)
        preprocess_fn = preprocessor.process_arc
        remove_columns=["question", "id", "choices", "answerKey"]
    elif data_args.dataset_name == "hellaswag":
        dataset = datasets.load_dataset("Rowan/hellaswag",   split="train")
        preprocess_fn = preprocessor.process_hellaswag
        remove_columns=[
            "ind", "activity_label", "ctx_a", "ctx_b",
            "ctx", "endings", "split", "split_type", "label"
        ]
    elif data_args.dataset_name == "piqa":
        dataset = datasets.load_dataset("ybisk/piqa",   split="train", trust_remote_code=True)
        preprocess_fn = preprocessor.process_piqa
        remove_columns=["label", "goal", "sol1", "sol2"]
    elif data_args.dataset_name == "winogrande":
        dataset = datasets.load_dataset(
            "allenai/winogrande",
            "winogrande_xl",
            split="train",
            trust_remote_code=True
        )
        preprocess_fn = preprocessor.process_winogrand
        remove_columns=["sentence", "option1", "option2", "answer"]
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



