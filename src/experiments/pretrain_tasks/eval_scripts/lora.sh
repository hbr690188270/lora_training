launch_lm_lora_eval() {
    local pretrained="$1"
    local peft="$2"
    local TASKS="$3"

    MODEL_NAME="${pretrained#model_cache//}"
    echo "$MODEL_NAME"

    PEFT_NAME="${peft#ckpt/ptr/}"
    echo "$PEFT_NAME"

    echo "evaluating ${pretrained} with LoRA ${peft}"
    accelerate launch -m lm_eval --model hf \
        --model_args "pretrained=${pretrained},dtype=bfloat16,peft=${peft}" \
        --tasks "$TASKS" \
        --batch_size 16 \
        --log_samples \
        --output_path logs/harness_eval/${MODEL_NAME}/${PEFT_NAME}
}

# Example usage
launch_lm_lora_eval "model_cache/llama3_1-8b" "ckpt/ptr/llama3-8b-arc"   "arc_challenge,arc_easy"

EVAL_TASKS=("gsm8k" "hellaswag" "piqa" "winogrande")
for EVAL_TASK in ${EVAL_TASKS[@]}
do
    launch_lm_lora_eval "model_cache/llama3_1-8b" "ckpt/ptr/llama3-8b-${EVAL_TASK}" "${EVAL_TASK}"
done

