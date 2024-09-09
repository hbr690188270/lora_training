from typing import Any, Optional, Dict, Union, List
import datasets
import numpy as np
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from dataclasses import dataclass
import torch
import os


def load_from_general_dataset(
    taskname: str, 
    n_proc: int = 32,
    datapath: str = "/mnt/data/bairu/repos/adapter_transfer/modular_artifacts/flan-flat",
    savepath: Optional[str] = None,
):
    # columns: 'source', 'target', 'task_name', 'task_source',
    #          'template_type', 'template_idx', 'split'
    dataset = datasets.load_from_disk(datapath)["train"]
    
    task_dataset = dataset.filter(
        lambda x: x["task_name"] == taskname,
        num_proc=n_proc,
        desc="Filtering task names",
    )
    num_examples = len(task_dataset)
    print(f"Found {num_examples} examples in {taskname} data!")

    np.random.seed(1234)
    all_indices = np.random.permutation(num_examples)

    train, val, _ = 0.8, 0.1, 0.1
    train_idxs = all_indices[:int(num_examples * train)]
    val_idxs = all_indices[int(num_examples * train): int(num_examples * (train + val))]
    test_idxs = all_indices[int(num_examples * (train + val)):]
    
    train_set = task_dataset.select(train_idxs)
    val_set = task_dataset.select(val_idxs)
    test_set = task_dataset.select(test_idxs)

    dataset_dict = datasets.DatasetDict(
        train=train_set,
        val=val_set,
        test=test_set
    )
    if savepath is None:
        savepath = f"datasets/processed_data/{taskname}"
    dataset_dict.save_to_disk(savepath)

    return dataset_dict

def preprocess_fn(
    example: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    apply_chat_template: bool = False,
):
    source_text = example["source"]
    target_text = example["target"]
    if apply_chat_template:
        raise NotImplementedError
    else:
        separator = " "
        concate_texts = [
            source_text[i] + separator + target_text[i] for i in range(len(source_text))
        ]
        tokenized_result = tokenizer(
            concate_texts,
            padding=False,
            truncation=True,
            max_length=768,
            add_special_tokens = False,
        )
        input_ids = tokenized_result["input_ids"]

        source_tokenized_result = tokenizer(
            source_text,
            padding=False,
            truncation=True,
            max_length=768,
            add_special_tokens = False,
        )
        source_input_ids = source_tokenized_result["input_ids"]
        source_length = [len(x) for x in source_input_ids]
    processed_examples = {
        "input_ids": input_ids,
        "source_length": source_length,
    }
    return processed_examples

def load_taskdataset(
    taskname: str,
    tokenizer: PreTrainedTokenizer,
    savepath: Optional[str] = None,
    apply_chat_template: bool = False,
    n_proc: int = 32,
):
    if savepath is not None:
        if not os.path.exists(savepath):
            raise ValueError("Please specify a valid path to the dataset!")
    else:
        savepath = f"datasets/processed_data/{taskname}"
        print(f"Automatically create dataset for {taskname}")
        print(f"Dataset will be saved to datasets/processed_data/{taskname}")
        if not os.path.exists(savepath):
            dataset_dict = load_from_general_dataset(taskname=taskname)
        else:
            dataset_dict = datasets.load_from_disk(savepath)
    
    # only "source" and "target" will be left
    dataset_dict = dataset_dict.remove_columns(
        column_names=[
            "task_name", "task_source",
            "template_type", "template_idx", "split"
        ]
    )

    dataset_dict = dataset_dict.map(
        preprocess_fn,
        fn_kwargs={
            "tokenizer": tokenizer,
            "apply_chat_template": apply_chat_template,
        },
        num_proc = n_proc,
        remove_columns=["source", "target"],
        batched=True,
    )
    return dataset_dict

@dataclass
class DataCollatorWithPaddingSFT:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    ignore_index = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        length_list = [len(x['input_ids']) for x in features]
        source_length_list = [x["source_length"] for x in features]
        max_length = max(length_list)
        all_input_ids = []
        for idx in range(len(features)):
            curr_length = len(features[idx]['input_ids'])
            pad_input_ids = features[idx]['input_ids'] + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id] * (max_length - curr_length)
            all_input_ids.append(pad_input_ids)
        all_input_ids = torch.LongTensor(all_input_ids)
        batch['input_ids'] = all_input_ids

        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels


        for i in range(len(features)):
            prefix_length = source_length_list[i] - 1
            batch["labels"][i, :prefix_length] = self.ignore_index

        return batch

