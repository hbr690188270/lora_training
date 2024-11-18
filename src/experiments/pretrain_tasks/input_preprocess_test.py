from absl.testing import absltest
from transformers import AutoTokenizer

from src.experiments.pretrain_tasks.input_preprocess import (
    PretrainingTaskPreprocessor,
)


def make_gsm8k_data():
    example = {
        "question": (
            "Weng earns $12 an hour for babysitting. Yesterday, she "
            "just did 50 minutes of babysitting. How much did she earn?"
        ),
        "answer": (
            "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, "
            "she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10"
        )
    }
    return example

def make_flan_data():
    example = {
        "source": (
            "Weng earns $12 an hour for babysitting. Yesterday, she "
            "just did 50 minutes of babysitting. How much did she earn?"
        ),
        "target": (
            "Weng earns 12/60 = $0.2 per minute. Working 50 minutes, "
            "she earned 0.2 x 50 = $10. #### 10"
        )
    }
    return example

def make_arc_data():
    example = {
        "question": (
            "Which of these would help to prevent infections from occurring "
            "in small cuts and scrapes?"
        ),
        "choices": {
            "text": [
                "apply a cold ice pack",
                "raise the injured area",
                "apply pressure to stop bleeding",
                "wash the area with warm, soapy water"
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "D",
    }
    return example

def make_hellaswag_data():
    example = {
        "ind": 8,
        "activity_label": "Baking cookies",
        "ctx_a": "A female chef in white uniform shows a stack of baking pans in a large kitchen presenting them.",
        "ctx_b": "the pans",
        "ctx": "A female chef in white uniform shows a stack of baking pans in a large kitchen presenting them. the pans",
        "endings": [
            "contain egg yolks and baking soda.",
            "are then sprinkled with brown sugar.",
            "are placed in a strainer on the counter.",
            "are filled with pastries and loaded into the oven."
        ],
        "label": 3,
    }
    return example

def make_piqa_data():
    example = {
        "goal": "How do I ready a guinea pig cage for it's new occupants?",
        "sol1": "Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.",
        "sol2": "Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.",
        "label": 0,
    }
    return example

def make_winogrande_data():
    example = {
        "sentence": (
            "Ian volunteered to eat Dennis's menudo after already having a bowl because"
            " _ despised eating intestine."
        ),
        "option1": "Ian",
        "option2": "Dennis",
        "answer": "2"
    }
    return example


class pretraining_processor_test(absltest.TestCase):
    @property
    def tokenizer(self,):
        tokenizer = AutoTokenizer.from_pretrained(
            "model_cache/llama3-8b",
            # "model_cache/mistral-7b-instruct-v3",
        )
        return tokenizer

    def test_gsm8k(self):
        example = make_gsm8k_data()
        preprocessor = PretrainingTaskPreprocessor(
            tokenizer=self.tokenizer,
            max_len=2048,
        )
        input_ids = preprocessor.process_gsm8k(example)["input_ids"]
        gt_texts = (
            "Question: Weng earns $12 an hour for babysitting. Yesterday, she "
            "just did 50 minutes of babysitting. How much did she earn?\nAnswer: "
            "Weng earns 12/60 = $0.2 per minute. Working 50 minutes, "
            "she earned 0.2 x 50 = $10. #### 10"
        )
        gt_tokenized_input_ids = self.tokenizer(
            gt_texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        gt_tokenized_input_ids.append(self.tokenizer.eos_token_id)
        assert input_ids == gt_tokenized_input_ids

    def test_arc_process(self):
        example = make_arc_data()
        preprocessor = PretrainingTaskPreprocessor(
            tokenizer=self.tokenizer,
            max_len=768
        )
        input_ids = preprocessor.process_arc(example)["input_ids"]

        gt_texts = (
            "Question: Which of these would help to prevent infections from occurring "
            "in small cuts and scrapes?\nAnswer: wash the area with warm, soapy water"
        )
        gt_tokenized_input_ids = self.tokenizer(
            gt_texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        gt_tokenized_input_ids.append(self.tokenizer.eos_token_id)
        assert input_ids == gt_tokenized_input_ids

    def test_hellaswag_process(self):
        example = make_hellaswag_data()
        preprocessor = PretrainingTaskPreprocessor(
            tokenizer=self.tokenizer,
            max_len=768
        )
        input_ids = preprocessor.process_hellaswag(example)["input_ids"]

        gt_texts = (
            "Baking cookies: A female chef in white uniform shows a stack of baking pans in a large "
            "kitchen presenting them. The pans are filled with pastries and loaded into the oven."
        )
        gt_tokenized_input_ids = self.tokenizer(
            gt_texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        gt_tokenized_input_ids.append(self.tokenizer.eos_token_id)
        print(input_ids)
        print(gt_tokenized_input_ids)
        assert input_ids == gt_tokenized_input_ids

    def test_winogrande_process(self):
        example = make_winogrande_data()
        preprocessor = PretrainingTaskPreprocessor(
            tokenizer=self.tokenizer,
            max_len=768
        )
        input_ids = preprocessor.process_winogrand(example)["input_ids"]

        gt_texts = (
            "Ian volunteered to eat Dennis's menudo after already having a bowl because Dennis"
            " despised eating intestine."
        )
        gt_tokenized_input_ids = self.tokenizer(
            gt_texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        gt_tokenized_input_ids.append(self.tokenizer.eos_token_id)
        print(input_ids)
        print(gt_tokenized_input_ids)
        assert input_ids == gt_tokenized_input_ids

    def test_piqa_process(self):
        example = make_piqa_data()
        preprocessor = PretrainingTaskPreprocessor(
            tokenizer=self.tokenizer,
            max_len=768
        )
        input_ids = preprocessor.process_piqa(example)["input_ids"]

        gt_texts = (
            "Question: How do I ready a guinea pig cage for it's new occupants?"
            "\nAnswer: Provide the guinea pig with a cage full of a few inches of "
            "bedding made of ripped paper strips, you will also need to supply it "
            "with a water bottle and a food dish."
        )
        gt_tokenized_input_ids = self.tokenizer(
            gt_texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        gt_tokenized_input_ids.append(self.tokenizer.eos_token_id)
        print(input_ids)
        print(gt_tokenized_input_ids)
        assert input_ids == gt_tokenized_input_ids

    def test_flanv2(self):
        example = make_flan_data()
        preprocessor = PretrainingTaskPreprocessor(
            tokenizer=self.tokenizer,
            max_len=2048,
        )
        input_ids = preprocessor.process_flan(example)["input_ids"]
        gt_texts = (
            "Weng earns $12 an hour for babysitting. Yesterday, she "
            "just did 50 minutes of babysitting. How much did she earn? "
            "Weng earns 12/60 = $0.2 per minute. Working 50 minutes, "
            "she earned 0.2 x 50 = $10. #### 10"
        )
        gt_tokenized_input_ids = self.tokenizer(
            gt_texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        gt_tokenized_input_ids.append(self.tokenizer.eos_token_id)
        assert input_ids == gt_tokenized_input_ids


