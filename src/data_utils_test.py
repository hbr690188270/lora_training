from absl.testing import absltest
from transformers import AutoTokenizer
import torch

from .data_utils import preprocess_fn

def make_ds():
    example = {
        "source": ["This is the 1st input:", "Another input"],
        "target": ["the 1st output.", "Another output."]
    }
    return example

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




