# CUDA_VISIBLE_DEVICES=5,6 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29504 --config_file configs/ds_config.yaml train.py configs/llama2/task1.yaml

CUDA_VISIBLE_DEVICES=3,4,5,6 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29504 --config_file configs/ds_config.yaml train.py configs/llama2/task2.yaml

CUDA_VISIBLE_DEVICES=3,4,5,6 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29504 --config_file configs/ds_config.yaml train.py configs/llama2/task3.yaml

CUDA_VISIBLE_DEVICES=3,4,5,6 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29504 --config_file configs/ds_config.yaml train.py configs/llama2/task4.yaml
