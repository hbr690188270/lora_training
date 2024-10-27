
launch_lm_lora_eval() {
    local pretrained="$1"
    local peft="$2"
    local TASKS="$3"

    MODEL_NAME="${peft#ckpt/ptr/}"
    echo "$MODEL_NAME"

    echo "evaluating ${pretrained} with LoRA ${peft}"
    accelerate launch -m lm_eval --model hf \
        --model_args "pretrained=${pretrained},dtype=bfloat16,peft=${peft}" \
        --tasks "$TASKS" \
        --batch_size 16 \
        --log_samples \
        --output_path logs/harness_eval/${MODEL_NAME}
}

# Example usage
launch_lm_lora_eval "model_cache/llama3-8b" "ckpt/ptr/llama3-8b-arc_challenge"  "arc_challenge,arc_easy"
launch_lm_lora_eval "model_cache/llama3_1-8b" "ckpt/ptr/llama3_1-8b-arc_challenge"  "arc_challenge,arc_easy"

