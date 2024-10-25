import re
from typing import Any, Dict, List

from transformers import PreTrainedTokenizerBase


class pretraining_task_preprocessor():
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.eos_token_id = self.tokenizer.eos_token_id

    def process_gsm8k(
        self,
        example: Dict[str, str],
    ):
        question = example["question"]
        answer = example["answer"]
        cleaned_answer = re.sub(r"<<.*?>>", "", answer)
        cot, final_number = cleaned_answer.split("####", 1)
        cot = cot.strip()
        final_number = final_number.strip()

        formated_answer = cot + f" The answer is {final_number}."

        input_text = f"Q: {question.strip()}\n\n A: {formated_answer}"

        input_ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        input_ids.append(self.eos_token_id)
        labels = input_ids[:]
        return input_ids, labels

    def process_arc(
        self,
        example: Dict[str, str],
    ):
        question = example["question"]
        choices = example["choices"]
        choice_texts = choices["text"]
        choice_labels: List = choice_texts["label"]
        answerKey = example["answerKey"]

        target_choice_index = choice_labels.index(answerKey)
        target_completion = choice_texts[target_choice_index]
        input_text = f"Question: {question}\nAnswer:{target_completion}"

        input_ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        input_ids.append(self.eos_token_id)
        labels = input_ids[:]
        return input_ids, labels



