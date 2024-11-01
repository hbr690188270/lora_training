from dataclasses import replace

from src.cmd_parser import (
    SFTConfig,
)

DEFAULT_LORA_TRANSFORM_CONFIG = SFTConfig(
    max_seq_length=2048,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=2.0e-5,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    do_eval=True,
    bf16=True,
    output_dir="tmp",
    save_only_model=True,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=4,
    remove_unused_columns=False,
    report_to="wandb",
    run_name="lora_transform-tmp",
    warmup_ratio=0.1,
    seed=42,
    push_to_hub=False,
    logging_steps=10,
    log_level="info",
    gradient_checkpointing=False,
)

sft_train_config = replace(
    DEFAULT_LORA_TRANSFORM_CONFIG,
    num_train_epochs=2,
)

short_seq_train_config = replace(
    DEFAULT_LORA_TRANSFORM_CONFIG,
    max_seq_length=768,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
)

large_lr_short_seq_train_config = replace(
    DEFAULT_LORA_TRANSFORM_CONFIG,
    max_seq_length=768,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
)

lr14_short_seq_train_config = replace(
    DEFAULT_LORA_TRANSFORM_CONFIG,
    max_seq_length=768,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-4,
)

lr54_short_seq_train_config = replace(
    DEFAULT_LORA_TRANSFORM_CONFIG,
    max_seq_length=768,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-4,
)

TRAINING_RECIPE = {
    "sft": sft_train_config,
    "ptr_default": short_seq_train_config,
    "ptr_lr5e-5": large_lr_short_seq_train_config,
    "ptr_lr1e-4": lr14_short_seq_train_config,
    "ptr_lr5e-4": lr54_short_seq_train_config,
}


