import re
from typing import Dict, List

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
        # cot, final_number = cleaned_answer.split("####", 1)
        # cot = cot.strip()
        # final_number = final_number.strip()

        # formated_answer = cot + f" The answer is {final_number}."

        # input_text = f"Q: {question.strip()}\n\n A: {formated_answer}"

        input_text = f"Question: {question}\nAnswer: {cleaned_answer}"
        input_ids: List[int] = self.tokenizer(
            input_text,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None,
        )["input_ids"]

        qa_split_token_ids = self.tokenizer(
            "\nAnswer:", add_special_tokens=False, return_tensors=None
        )["input_ids"][1:]
        qa_split_position = [
            i for i in range(len(input_ids) - len(qa_split_token_ids) + 1)
            if input_ids[i:i + len(qa_split_token_ids)] == qa_split_token_ids
        ]
        assert len(qa_split_position) == 1, f"{qa_split_position}"
        qa_split_position = qa_split_position[0]

        input_ids.append(self.eos_token_id)
        labels = [-100] * qa_split_position + input_ids[qa_split_position: ]
        assert len(input_ids) == len(labels)
        # labels = input_ids[:]
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def process_arc(
        self,
        example: Dict[str, str],
    ):
        question = example["question"]
        choices = example["choices"]
        choice_texts = choices["text"]
        choice_labels: List = choices["label"]
        answerKey = example["answerKey"]

        target_choice_index = choice_labels.index(answerKey)
        target_completion = choice_texts[target_choice_index]
        input_text = f"Question: {question}\nAnswer: {target_completion}"

        input_ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        input_ids.append(self.eos_token_id)
        labels = input_ids[:]
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def process_hellaswag(
        self,
        example: Dict[str, str],
    ):
        def preprocess(text):
            text = text.strip()
            # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
            text = text.replace(" [title]", ". ")
            text = re.sub("\\[.*?\\]", "", text)
            text = text.replace("  ", " ")
            return text

        def _process_doc(doc):
            ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
            out_doc = {
                "query": preprocess(doc["activity_label"] + ": " + ctx),
                "choices": [preprocess(ending) for ending in doc["endings"]],
                "gold": int(doc["label"]),
            }
            return out_doc

        example.update(_process_doc(example))

        choice_id = int(example["label"])
        input_text = example["query"] + " " + example["choices"][choice_id]

        input_ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        input_ids.append(self.eos_token_id)
        labels = input_ids[:]
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def process_piqa(
        self,
        example: Dict[str, str],
    ):
        label: int = example["label"]
        goal: str = example["goal"]
        sol1: str = example["sol1"]
        sol2 = example["sol2"]

        if label == 0:
            input_text = f"Question: {goal}\nAnswer: {sol1}"
        elif label == 1:
            input_text = f"Question: {goal}\nAnswer: {sol2}"
        else:
            raise ValueError()

        input_ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        input_ids.append(self.eos_token_id)
        labels = input_ids[:]
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def process_winogrand(
        self,
        example: Dict[str, str],
    ):
        def doc_to_text(doc):
            answer_to_num = {"1": 0, "2": 1}
            return answer_to_num[doc["answer"]]


        def doc_to_target(doc):
            idx = doc["sentence"].index("_") + 1
            return doc["sentence"][idx:].strip()


        def doc_to_choice(doc):
            idx = doc["sentence"].index("_")
            options = [doc["option1"], doc["option2"]]
            return [doc["sentence"][:idx] + opt for opt in options]

        doc_content = doc_to_text(example)
        doc_target = doc_to_target(example)
        doc_choices = doc_to_choice(example)

        input_text = f"{doc_choices[doc_content]} {doc_target}"
        input_ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        input_ids.append(self.eos_token_id)
        labels = input_ids[:]
        return {
            "input_ids": input_ids,
            "labels": labels,
        }



