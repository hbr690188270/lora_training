# CUDA_VISIBLE_DEVICES=1,2,5,6 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29502 --config_file configs/ds_config.yaml train.py configs/llama3/task1.yaml
# CUDA_VISIBLE_DEVICES=1,2,5,6 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29502 --config_file configs/ds_config.yaml train.py configs/llama3_1/task1.yaml

# CUDA_VISIBLE_DEVICES=1,2,5,6 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29502 --config_file configs/ds_config.yaml train.py configs/llama3/task2.yaml
# CUDA_VISIBLE_DEVICES=1,2,5,6 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29502 --config_file configs/ds_config.yaml train.py configs/llama3_1/task2.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29502 --config_file configs/ds_config.yaml train.py configs/llama3/task3.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29502 --config_file configs/ds_config.yaml train.py configs/llama3_1/task3.yaml

# CUDA_VISIBLE_DEVICES=1,2,5,6 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29502 --config_file configs/ds_config.yaml train.py configs/llama3/task4.yaml
# CUDA_VISIBLE_DEVICES=1,2,5,6 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29502 --config_file configs/ds_config.yaml train.py configs/llama3_1/task4.yaml

