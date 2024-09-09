CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29502 --config_file configs/train_config.yaml train.py configs/task4.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29502 --config_file configs/train_config.yaml train.py configs/phi35_task1.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29502 --config_file configs/train_config.yaml train.py configs/phi35_task2.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29502 --config_file configs/train_config.yaml train.py configs/phi35_task3.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29502 --config_file configs/train_config.yaml train.py configs/phi35_task4.yaml
