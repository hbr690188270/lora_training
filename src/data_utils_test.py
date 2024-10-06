import numpy as np
import torch
import transformers
from absl.testing import absltest
from transformers import AutoTokenizer

import datasets

from .data_utils import (
    DataCollatorCompletionOnly,
    DataCollatorWithPaddingSFT,
    preprocess_fn,
)


def make_ds():
    example = {
        "source": ["This is the 1st input:","This is the 1st input:"*1000, "Another input"],
        "target": ["the 1st output.", "Another output."]
    }
    return example

def make_ds_v2():
    source_texts = ["This is the 1st input:", "Another input"]
    targets = ["the 1st output.", "Another"]
    rand_idxs = np.random.choice(2, size=128, replace=True)
    source_texts = [source_texts[x] for x in rand_idxs]
    targets = [targets[x] for x in rand_idxs]
    print(len(source_texts))
    ds = {
        "source": source_texts,
        "target": targets
    }
    return ds

class Tokenizertest(absltest.TestCase):
    @property
    def tokenizer(self,):
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
        )
        return tokenizer

    def test_tokenize(self, ):
        tokenizer=self.tokenizer
        input_text = "good:"
        tokens = tokenizer.tokenize(input_text)
        print(tokens)
        input_text = "good: a"
        tokens = tokenizer.tokenize(input_text)
        print(tokens)

    def test_preprocess(self,):
        example = make_ds()
        tokenizer=self.tokenizer
        processed_examples = preprocess_fn(
            example=example,
            tokenizer=tokenizer,
            apply_chat_template=False
        )
        source_length = processed_examples["source_length"]
        input_ids = processed_examples["input_ids"]
        tokens_1 = self.tokenizer.encode(example["source"][0])
        tokens_2 = self.tokenizer.encode(example["source"][1])

        print(input_ids)
        print(tokens_1)
        print(tokens_2)
        print(source_length)

        ref_source_length = [len(tokens_1), len(tokens_2)]
        for idx in range(len(ref_source_length)):
            assert ref_source_length[idx] == source_length[idx]

        ref_target_tokens = [
            self.tokenizer.encode(example["target"][x]) 
            for x in range(len(example["target"]))
        ]
        target_tokens = [
            input_ids[x][source_length[x]: ] for x in range(len(input_ids))
        ]
        assert ref_target_tokens == target_tokens

class DataCollatorTest(absltest.TestCase):
    @property
    def tokenizer(self,):
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
        )
        return tokenizer

    def test_datacollator(self,):
        data_examples = make_ds_v2()
        dataset = datasets.Dataset.from_dict(data_examples)
        print(dataset)
        dataset = dataset.map(
            preprocess_fn,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "apply_chat_template": False,
            },
            num_proc = 10,
            remove_columns=["source", "target"],
            batched=True,
        )
        print(dataset)
        data_collator = DataCollatorWithPaddingSFT(
            tokenizer=self.tokenizer,
            padding=transformers.utils.PaddingStrategy.LONGEST,
            max_length=768,
        )
        batch_examples = (dataset[0], dataset[1])
        results = data_collator(batch_examples)
        print(results)
        input_ids = results["input_ids"]
        labels = results["labels"]

        input_end_pos1 = torch.where(labels[0] != -100)[0][0]
        input_end_pos2 = torch.where(labels[1] != -100)[0][0]
        print(input_end_pos1)
        seq1 = self.tokenizer.decode(input_ids[0][:input_end_pos1 + 1])
        seq2 = self.tokenizer.decode(input_ids[1][:input_end_pos2 + 1])
        print(seq1)
        print(seq2)

        resp1 = self.tokenizer.decode(input_ids[0][input_end_pos1 + 1: ])
        resp2 = self.tokenizer.decode(input_ids[1][input_end_pos2 + 1: ])
        print(resp1)
        print(resp2)

class DataCollatorCompletionOnlyTest(absltest.TestCase):
    @property
    def tokenizer(self,):
        tokenizer = AutoTokenizer.from_pretrained(
            "model_cache/llama3_1-8b-instruct",
        )
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
            tokenizer.pad_token
        )
        tokenizer.model_max_length = 100
        return tokenizer

    def test_datacollator(self,):
        print(self.tokenizer.eos_token)
        print(self.tokenizer.bos_token)
        print(self.tokenizer.pad_token)

        print(self.tokenizer.all_special_ids)
        data_examples = make_ds_v2()
        dataset = datasets.Dataset.from_dict(data_examples)
        print(dataset)
        dataset = dataset.map(
            preprocess_fn,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "apply_chat_template": True,
            },
            num_proc = 10,
            remove_columns=["source", "target"],
            batched=True,
        )
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        response_token_ids = self.tokenizer.encode(response_template, add_special_tokens=False)
        print(f"response_template: {response_token_ids}")
        data_collator = DataCollatorCompletionOnly(
            response_token_ids=response_token_ids,
            tokenizer=self.tokenizer,
            padding=transformers.utils.PaddingStrategy.LONGEST,
            max_length=100,
        )
        batch_examples = (dataset[0], dataset[1])
        results = data_collator(batch_examples)
        print(results)
        input_ids = results["input_ids"]
        labels = results["labels"]

        input_end_pos1 = torch.where(labels[0] != -100)[0][0]
        input_end_pos2 = torch.where(labels[1] != -100)[0][0]
        print(input_end_pos1)
        seq1 = self.tokenizer.decode(input_ids[0][:input_end_pos1 + 1])
        seq2 = self.tokenizer.decode(input_ids[1][:input_end_pos2 + 1])
        print(seq1)
        print(seq2)

        resp1 = self.tokenizer.decode(input_ids[0][input_end_pos1 + 1: ])
        resp2 = self.tokenizer.decode(input_ids[1][input_end_pos2 + 1: ])
        print(resp1)
        print(resp2)

