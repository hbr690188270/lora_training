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
        print("Processing task: ", task_name)

        task_dataset = dataset.filter(
            lambda x: x["task_name"] == task_name, num_proc=24
        )

        # if the dataset is too large, we randomly sample "cutoff" examples for training
        task_dataset = task_dataset.shuffle(42)

        if cutoff > 0 and len(task_dataset) > cutoff:
            task_dataset = task_dataset.select(range(cutoff))

        def assign_split(example, idx):
            rng = np.random.RandomState(idx)
            draw = rng.rand()
            if draw < 0.8:
                return {"split": "train"}
            elif draw < 0.9:
                return {"split": "validation"}
            else:
                return {"split": "test"}

        task_dataset = task_dataset.map(assign_split, with_indices=True)
        # randomly cut the dataset again
        task_dataset = task_dataset.shuffle(42)

        if cutoff and len(task_dataset) > cutoff:
            task_dataset = task_dataset.select(range(cutoff))

        all_datasets.append(task_dataset)

        print("Dumping task", task_name)
        if FLAGS.verbose:
            print("# Train", len(task_dataset.filter(lambda x: x["split"] == "train")))
            print("# Test", len(task_dataset.filter(lambda x: x["split"] == "test")))
            print(
                "# Val", len(task_dataset.filter(lambda x: x["split"] == "validation"))
            )

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


