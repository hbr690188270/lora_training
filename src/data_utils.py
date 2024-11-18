import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


def get_tokenizer(
    model_name_or_path,
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    if "llama3-8b-instruct" in model_name_or_path or "llama3_1-8b-instruct" in model_name_or_path:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 128004
    elif "llama3-8b" in model_name_or_path or "llama3_1-8b" in model_name_or_path:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 128004
    elif "llama2-7b-chat" in model_name_or_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif "mistral-7b" in model_name_or_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    else:
        raise NotImplementedError
    tokenizer.truncation_side = "left"

    assert tokenizer.pad_token is not None
    assert tokenizer.pad_token_id is not None

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    return tokenizer

def get_instruct_lm_tokenizer(
    model_name_or_path,
) -> PreTrainedTokenizer:
    """
    Get the tokenizer for the instruct lm.
    Note: we already manually change the tokenizer_config.json and tokenizer.json
        so that token index 128002 is mapped to <turn_end>, a special token for
        the usage of chat template.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    if "llama3-8b" in model_name_or_path or "llama3_1-8b" in model_name_or_path:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 128004
    else:
        raise NotImplementedError
    # tokenizer.truncation_side = "left"

    assert tokenizer.pad_token is not None
    assert tokenizer.pad_token_id is not None

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    return tokenizer


def load_flan_subset(
    flan_dataset: datasets.Dataset,
    taskname: str,
    n_proc: int = 32,
    savepath: Optional[str] = None,
):
    if savepath is None:
        savepath = f"dataset_cache/processed_data/{taskname}"
    if os.path.exists(savepath):
        return datasets.load_from_disk(savepath)

    # columns: 'source', 'target', 'task_name', 'task_source',
    #          'template_type', 'template_idx', 'split'
    task_dataset = flan_dataset.filter(
        lambda x: x["task_name"] == taskname,
        num_proc=n_proc,
        desc="Filtering task names",
    )
    num_examples = len(task_dataset)
    print(f"Found {num_examples} examples in {taskname} data!")

    if savepath is None:
        savepath = f"dataset_cache/processed_data/{taskname}"
    task_dataset.save_to_disk(savepath)

    return task_dataset

def preprocess_fn(
    example: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    apply_chat_template: bool = False,
):
    source_text = example["source"]
    target_text = example["target"]
    if apply_chat_template:
        messages_list = []
        for idx in range(len(source_text)):
            messages = [
                # {"role": "human", "content": source_text[idx]},
                {"role": "user", "content": source_text[idx]},
                {"role": "assistant", "content": target_text[idx]},
            ]
            messages_list.append(messages)
        inputs_with_chat_template = [tokenizer.apply_chat_template(x, tokenize=False) for x in messages_list]
        tokenized_result = tokenizer(
            inputs_with_chat_template,
            padding=False,
            truncation=True,
            max_length=768,
            add_special_tokens=False,
        )
        input_ids = tokenized_result["input_ids"]
        print(tokenized_result.keys())
        processed_examples = {
            "input_ids": input_ids,
        }
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

def preprocess_fn_for_generation(
    example: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    apply_chat_template: bool = False,
):
    source_text = example["source"]
    target_text = example["target"]
    if apply_chat_template:
        messages_list = []
        for idx in range(len(source_text)):
            messages = [
                {"role": "user", "content": source_text[idx]},
            ]
            messages_list.append(messages)
        inputs_with_chat_template = [
            tokenizer.apply_chat_template(x, tokenize=False,add_generation_prompt=True)
            for x in messages_list
        ]
        tokenized_result = tokenizer(
            inputs_with_chat_template,
            padding=False,
            truncation=True,
            max_length=768,
            add_special_tokens=False,
        )
        input_ids = tokenized_result["input_ids"]
        tokneized_completions = tokenizer(
            target_text,
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=False,
        )
        completion_ids = tokneized_completions["input_ids"]
        processed_examples = {
            "input_ids": input_ids,
            "completion_ids": completion_ids,
        }
    else:
        tokenized_result = tokenizer(
            source_text,
            padding=False,
            truncation=True,
            max_length=768,
            add_special_tokens = False,
        )
        input_ids = tokenized_result["input_ids"]

        source_length = [len(x) for x in input_ids]

        target_tokenized_result = tokenizer(
            target_text,
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens = False,
        )
        target_output_ids = target_tokenized_result["input_ids"]

        processed_examples = {
            "input_ids": input_ids,
            "target_output_ids": target_output_ids,
            "source_length": source_length,
        }
    return processed_examples

@dataclass
class DataCollatorWithPaddingSFT:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    ignore_index = -100
    for_generation: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        length_list = [len(x['input_ids']) for x in features]
        source_length_list = [x["source_length"] for x in features]
        max_length = max(length_list)
        all_input_ids = []
        all_attention_masks = []
        for idx in range(len(features)):
            curr_length = len(features[idx]['input_ids'])
            difference = max_length - curr_length
            if self.for_generation:
                # left padding
                attention_mask = [0] * difference + [1] * curr_length
                pad_input_ids = [self.tokenizer.pad_token_id] * (max_length - curr_length) + features[idx]['input_ids']
            else:
                attention_mask = [1] * curr_length + [0] * difference
                pad_input_ids = features[idx]['input_ids'] + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id] * (max_length - curr_length)
            all_input_ids.append(pad_input_ids)
            all_attention_masks.append(attention_mask)

        all_input_ids = torch.LongTensor(all_input_ids)
        all_attention_masks = torch.LongTensor(all_attention_masks)
        batch["attention_mask"] = all_attention_masks

        if self.for_generation:
            batch['prefix'] = all_input_ids
        else:
            batch['input_ids'] = all_input_ids
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
            for i in range(len(features)):
                prefix_length = source_length_list[i] - 1
                batch["labels"][i, :prefix_length] = self.ignore_index

        if "target_output_ids" in features[0]:
            target_output_ids = [x["target_output_ids"] for x in features]
            target_output_ids= torch.LongTensor(target_output_ids)
            batch["target_output_ids"] = target_output_ids


        return batch

@dataclass
class DataCollatorCompletionOnly:
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
    response_token_ids: List[int] = None
    ignore_index: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # batch = self.tokenizer.pad(
        #     features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors=self.return_tensors,
        # )

        batch = {}
        length_list = [len(x['input_ids']) for x in features]
        max_length = max(length_list)
        all_input_ids = []
        all_attention_masks = []
        for idx in range(len(features)):
            curr_length = len(features[idx]['input_ids'])
            difference = max_length - curr_length
            attention_mask = [1] * curr_length + [1] + [0] * difference
            pad_input_ids = features[idx]['input_ids'] + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id] * (max_length - curr_length)
            all_input_ids.append(pad_input_ids)
            all_attention_masks.append(attention_mask)
        all_input_ids = torch.LongTensor(all_input_ids)
        all_attention_masks = torch.LongTensor(all_attention_masks)
        batch['input_ids'] = all_input_ids
        batch["attention_mask"] = all_attention_masks

        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels


        for i in range(len(features)):
            response_token_ids_start_idx = None

            for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                if (
                    self.response_token_ids
                    == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                ):
                    response_token_ids_start_idx = idx

            if response_token_ids_start_idx is None:
                print(
                    f"Could not find response key `{self.response_token_ids}` in the "
                    f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                    f"This instance will be ignored in loss calculation. "
                    f"Note, if this happens often, consider increasing the `max_seq_length`."
                )
                print("Cannot find response template!!!")
                print(batch["input_ids"][i])
                batch["labels"][i, :] = self.ignore_index
            else:
                response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                # Make pytorch loss function ignore all tokens up through the end of the response key
                batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        if "completion_ids" in features[0]:
            completion_ids = [x["completion_ids"] for x in features]
            completion_ids = torch.LongTensor(completion_ids)
            batch["completion_ids"] = completion_ids
        return batch

@dataclass
class DataCollatorForInstructLM:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        length_list = [len(x['input_ids']) for x in features]
        max_length = max(length_list)
        all_input_ids = []
        all_labels = []
        all_attention_masks = []
        for idx in range(len(features)):
            curr_length = len(features[idx]['input_ids'])
            difference = max_length - curr_length
            # left padding
            attention_mask = [0] * difference + [1] * curr_length
            pad_input_ids = [self.tokenizer.pad_token_id] * difference + features[idx]['input_ids']
            labels = [-100] * difference + features[idx]['labels']

            # attention_mask = [1] * curr_length + [0] * difference
            # pad_input_ids = features[idx]['input_ids'] + [self.tokenizer.pad_token_id] * difference
            # labels = features[idx]['labels'] + [-100] * difference

            all_input_ids.append(pad_input_ids)
            all_labels.append(labels)
            all_attention_masks.append(attention_mask)

        all_input_ids = torch.LongTensor(all_input_ids)
        all_labels = torch.LongTensor(all_labels)
        all_attention_masks = torch.LongTensor(all_attention_masks)

        batch = {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_masks,
        }
        if "dataset_index" in features[0]:
            all_dataset_ids = torch.LongTensor([x["dataset_index"] for x in features])
            # batch["dataset_index"] = all_dataset_ids
            batch["adapter_names"] = all_dataset_ids
        return batch

