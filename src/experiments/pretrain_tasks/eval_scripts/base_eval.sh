
launch_lm_lora_eval() {
    local pretrained="$1"
    local TASKS="$2"

    MODEL_NAME="${pretrained#model_cache/}"
    echo "$MODEL_NAME"

    echo "evaluating ${pretrained}"
    accelerate launch -m lm_eval --model hf \
        --model_args "pretrained=${pretrained},dtype=bfloat16" \
        --tasks $TASKS \
        --batch_size 16 \
        --output_path logs/harness_eval/${MODEL_NAME}
}

# Example usage
EVAL_TASKS=("arc_challenge" "arc_easy" "gsm8k" "hellaswag" "piqa" "winogrande")
for EVAL_TASK in ${EVAL_TASKS[@]}
do
    launch_lm_lora_eval "model_cache/llama3-8b" "${EVAL_TASK}"
    launch_lm_lora_eval "model_cache/llama3_1-8b" "${EVAL_TASK}"
done
