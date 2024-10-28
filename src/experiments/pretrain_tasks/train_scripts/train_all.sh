TASKS=("gsm8k" "arc"  "hellaswag" "piqa" "winogrande")
for TASK in ${TASKS[@]}
do
    CUDA_VISIBLE_DEVICES=3,4,5,6 ACCELERATE_LOG_LEVEL=info \
        accelerate launch --main_process_port 29504 --config_file configs/a6000_config.yaml \
        src/experiments/pretrain_tasks/pretrain_task_trainer.py \
        --config_name="ptr-llama3-8b-${TASK}"
done


TASKS=("gsm8k" "arc"  "hellaswag" "piqa" "winogrande")
for TASK in ${TASKS[@]}
do
    CUDA_VISIBLE_DEVICES=3,4,5,6 ACCELERATE_LOG_LEVEL=info \
        accelerate launch --main_process_port 29504 --config_file configs/a6000_config.yaml \
        src/experiments/pretrain_tasks/pretrain_task_trainer.py \
        --config_name="ptr-llama3_1-8b-${TASK}"
done
