"""
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29502 --config_file configs/train_config.yaml train.py configs/task3.yaml

CUDA_VISIBLE_DEVICES=5,6 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29504 --config_file configs/ds_config.yaml train.py configs/llama2/task3.yaml
"""

import logging
import sys

import datasets
import torch
import transformers

# from alignment import (
#     get_peft_config,
# )
from peft import get_peft_model
from transformers import AutoModelForCausalLM, Trainer, set_seed

from src.cmd_parser import (
    DataArguments,
    ModelArguments,
    MyArgumentParser,
    SFTConfig,
)
from src.data_utils import (
    DataCollatorCompletionOnly,
    DataCollatorWithPaddingSFT,
    get_tokenizer,
    load_taskdataset,
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

    tokenizer = get_tokenizer(
        model_args.model_name_or_path,
    )
    if apply_chat_template:
        if "llama3-8b-instruct" in model_args.model_name_or_path or "llama3_1-8b-instruct" in model_args.model_name_or_path:
            response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        elif "llama2-7b-chat" in model_args.model_name_or_path:
            response_template = "[/INST]"
            response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        elif "mistral-7b-instruct-v3" in model_args.model_name_or_path:
            response_template = "[/INST]"
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

    assert training_args.task is not None
    dataset_dict = load_taskdataset(
        taskname=training_args.task,
        tokenizer=tokenizer,
        apply_chat_template=apply_chat_template
    )

    logger.info("*** Load pretrained model ***")
    # torch_dtype = (
    #     model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    # )

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
    # model = get_peft_model(model, get_peft_config(model_args))
    # model, tokenizer = setup_chat_format(model, tokenizer)
    model_kwargs = None
    model.print_trainable_parameters()

    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["val"]
    test_dataset = dataset_dict["test"]

    # with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
    #     for index in random.sample(range(len(dataset_dict["train"])), 3):
    #         logger.info(f"Sample {index} of the processed training set:\n\n{dataset_dict['train'][index]['text']}")

    # trainer = SFTTrainer(
    #     model=model,
    #     model_init_kwargs=model_kwargs,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     # max_seq_length=training_args.max_seq_length,
    #     tokenizer=tokenizer,
    #     packing=False,
    #     peft_config=get_peft_config(model_args),
    #     dataset_kwargs=training_args.dataset_kwargs,
    # )

    output_dir = training_args.output_dir
    lora_alpha = model_args.lora_alpha
    lora_r = model_args.lora_r
    if lora_alpha != 4 or lora_r != 16:
        output_dir += f"_alpha{lora_alpha}_r{lora_r}"
    if apply_chat_template:
        output_dir += "_chat"
    training_args.output_dir = output_dir
    runname = f"{model_args.model_name_for_short}-{training_args.task}-alpha{lora_alpha}-r{lora_r}"
    training_args.run_name = runname

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
