"""
A simple function to list all tasks in FLAN.

python -m tools.flan_finetuning.get_flan_task_list
"""

import datasets


def load_flan_data():
    datapath = "/mnt/data/bairu/repos/adapter_transfer/modular_artifacts/flan-flat"
    dataset = datasets.load_from_disk(datapath)["train"]
    task_names = dataset["task_name"]
    task_names = list(set(task_names))
    print(task_names)
    print(len(task_names))


if __name__ == "__main__":
    load_flan_data()
