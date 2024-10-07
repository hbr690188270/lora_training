"""
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29504 --config_file configs/a100_config.yaml instruct_lm_trainer.py configs/instruct_lm_llama3/config.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29504 --config_file configs/a100_ddp_config.yaml instruct_lm_trainer.py configs/instruct_lm_llama3/config.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29504 --config_file configs/a100_zero3_config.yaml instruct_lm_trainer.py configs/instruct_lm_llama3/config.yaml
"""

import logging
import sys

import accelerate
import numpy as np
import torch
import transformers
from peft import LoraConfig, get_peft_model

from transformers import AutoModelForCausalLM, Trainer, set_seed

import datasets
from src.cmd_parser import (
    DataArguments,
    ModelArguments,
    MyArgumentParser,
    SFTConfig,
)
from src.data_utils import (
    DataCollatorForInstructLM,
    get_instruct_lm_tokenizer,
)
from src.experiments.instruct_lm.input_preprocess import (
    instruct_lm_preprocessor,
)

logger = logging.getLogger(__name__)


def main():
    parser = MyArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

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

    apply_chat_template = training_args.apply_chat_template

    tokenizer = get_instruct_lm_tokenizer(
        model_args.model_name_or_path,
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

    dataset = datasets.load_dataset("nvidia/Daring-Anteater", split="train")
    dataset = dataset.map(
        preprocessor.process_daring_anteater,
        num_proc=32,
        remove_columns=['system', 'mask', 'dataset', 'conversations'],
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
        cache_dir='./model_cache'
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

    dataset = dataset.shuffle()
    num_examples = len(dataset)
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

    output_dir = training_args.output_dir
    lora_alpha = model_args.lora_alpha
    lora_r = model_args.lora_r
    if lora_alpha != 4 or lora_r != 16:
        output_dir += f"_alpha{lora_alpha}_r{lora_r}"
    if apply_chat_template:
        output_dir += "_chat"
    training_args.output_dir = output_dir
    runname = f"llama3-instructlm-alpha{lora_alpha}-r{lora_r}"
    training_args.run_name = runname

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    accelerator = accelerate.Accelerator()
    trainer = accelerator.prepare(trainer)


    logger.info("*** Train ***")
    train_result = trainer.train(resume_from_checkpoint=None)
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
    main()


