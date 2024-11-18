"""
python tools/download_flan.py
"""
from typing import Dict

import datasets
import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS

def set_flags():
    flags.DEFINE_boolean("verbose", default=True, help="Whether print the dataset infomation.")

def main(argv):
    cutoff = 10000

    dataset = datasets.load_dataset("chiayewken/flan-v2", split="train")

    task_names = dataset.unique("task_name")
    print("Num Tasks: ", len(task_names))

    all_datasets = []
    for task_name in task_names:
        if task_name.startswith("task"):
            continue
        print("Processing task: ", task_name)

        task_dataset = dataset.filter(
            lambda x: (x["task_name"] == task_name and "fs" not in x["template_type"]), num_proc=24
        )

        # if the dataset is too large, we randomly sample "cutoff" examples for training
        task_dataset = task_dataset.shuffle(42)

        if cutoff > 0 and len(task_dataset) > cutoff:
            task_dataset = task_dataset.select(range(cutoff))

        task_dataset = task_dataset.shuffle(42)

        if cutoff and len(task_dataset) > cutoff:
            task_dataset = task_dataset.select(range(cutoff))

        all_datasets.append(task_dataset)

        print("Dumping task", task_name)
        if FLAGS.verbose:
            print("# Train", len(task_dataset))

    concatenated_datasets: datasets.Dataset = datasets.concatenate_datasets(all_datasets)

    def clean_task(x):
        if "task_name" not in x:
            return x

        x["task_name"] = (
            x["task_name"]
            .replace(":", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
        )
        return x

    concatenated_datasets = concatenated_datasets.map(lambda x: clean_task(x))
    concatenated_datasets.save_to_disk("dataset_cache/flan_2021")

if __name__ == "__main__":
    set_flags()
    app.run(main)

