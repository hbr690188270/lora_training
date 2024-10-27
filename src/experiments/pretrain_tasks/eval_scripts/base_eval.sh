
launch_lm_lora_eval() {
    local pretrained="$1"
    local TASKS="$2"

    MODEL_NAME="${pretrained#model_cache/}"
    echo "$MODEL_NAME"

    echo "evaluating ${pretrained} with LoRA ${peft}"
    accelerate launch -m lm_eval --model hf \
        --model_args "pretrained=${pretrained},dtype=bfloat16" \
        --tasks $TASKS \
        --batch_size 16 \
        --log_samples \
        --output_path logs/harness_eval/${MODEL_NAME}
}

# Example usage
launch_lm_lora_eval "model_cache/llama3-8b" "arc_challenge,arc_easy"

