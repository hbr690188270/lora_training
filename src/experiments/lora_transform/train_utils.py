from dataclasses import replace

from src.cmd_parser import (
    SFTConfig,
)

FLAN_PATH = "dataset_cache/flan_2021"

TASKSET_ID_TO_TASKS = {
    "v1": ["gsm8k", "winogrande"],
    "v2": ["piqa", "arc", "hellaswag"],
    "v3": [
        "cot_esnli_ii", "cot_sensemaking", "cot_esnli", "stream_aqua", "cot_ecqa", "cot_creak_ii",
        "cot_strategyqa", "cos_e_v1_11_generate_explanation_given_text", "wiki_bio_key_content",
        "race_middle_Select_the_best_answer_no_instructions_", "drop_2_0_0",
        "adversarial_qa_dbidaf_generate_question", "social_i_qa_Show_choices_and_generate_answer",
        "qasc_is_correct_2", "race_middle_Taking_a_test", "quoref_Guess_Title_For_Context",
        "duorc_ParaphraseRC_generate_question_by_answer", "definite_pronoun_resolution_1_1_0",
        "adversarial_qa_droberta_tell_what_it_is", "super_glue_record_1_0_2",
        "adversarial_qa_dbert_tell_what_it_is", "wiki_qa_Topic_Prediction_Question_Only",
        "wiki_hop_original_choose_best_object_affirmative_3", "unified_qa_science_inst",
        "duorc_SelfRC_decide_worth_it", "wiqa_does_the_supposed_perturbation_have_an_effect",
    ]
}

INDEX_TO_DATASET = [
    "arc", "hellaswag", "piqa", "winogrande", "gsm8k",
    "cot_esnli_ii", "cot_sensemaking", "cot_esnli", "stream_aqua", "cot_ecqa", "cot_creak_ii",
    "cot_strategyqa", "cos_e_v1_11_generate_explanation_given_text", "wiki_bio_key_content",
    "race_middle_Select_the_best_answer_no_instructions_", "drop_2_0_0",
    "adversarial_qa_dbidaf_generate_question", "social_i_qa_Show_choices_and_generate_answer",
    "qasc_is_correct_2", "race_middle_Taking_a_test", "quoref_Guess_Title_For_Context",
    "duorc_ParaphraseRC_generate_question_by_answer", "definite_pronoun_resolution_1_1_0",
    "adversarial_qa_droberta_tell_what_it_is", "super_glue_record_1_0_2",
    "adversarial_qa_dbert_tell_what_it_is", "wiki_qa_Topic_Prediction_Question_Only",
    "wiki_hop_original_choose_best_object_affirmative_3", "unified_qa_science_inst",
    "duorc_SelfRC_decide_worth_it", "wiqa_does_the_supposed_perturbation_have_an_effect",
]
DATASET_TO_INDEX = {INDEX_TO_DATASET[x]: x for x in range(len(INDEX_TO_DATASET))}

DEFAULT_LORA_TRANSFORM_CONFIG = SFTConfig(
    max_seq_length=2048,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=2.0e-5,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    do_eval=True,
    bf16=True,
    output_dir="tmp",
    eval_strategy="epoch",
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
    eval_on_start=True,
    label_names=["labels"]
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

lr55_short_seq_train_config = replace(
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

h100_lr54_short_seq_train_config = replace(
    DEFAULT_LORA_TRANSFORM_CONFIG,
    max_seq_length=768,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-4,
    gradient_accumulation_steps=2,
)



test_train_config = replace(
    DEFAULT_LORA_TRANSFORM_CONFIG,
    max_seq_length=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-4,
    gradient_accumulation_steps=1,
)

TRAINING_RECIPE = {
    "a6000": {
        "sft": sft_train_config,
        "ptr_default": short_seq_train_config,
        "ptr_lr5e-5": lr55_short_seq_train_config,
        "ptr_lr1e-4": lr14_short_seq_train_config,
        "ptr_lr5e-4": lr54_short_seq_train_config,
        "test": test_train_config,
    },
    "h100": {
        "sft": None,
        "ptr_default": None,
        "ptr_lr5e-5": None,
        "ptr_lr1e-4": None,
        "ptr_lr5e-4": h100_lr54_short_seq_train_config,
        "test": test_train_config,
    },
}


