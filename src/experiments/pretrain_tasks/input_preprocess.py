import re
from typing import Dict, List, Optional

import datasets
from transformers import PreTrainedTokenizerBase

from src.data_utils import load_flan_subset


class PretrainingTaskPreprocessor():
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

        updated_example = _process_doc(example)

        choice_id = int(example["label"])
        input_text = (
            updated_example["query"] + " " + updated_example["choices"][choice_id]
        )

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

    def process_flan(
        self,
        example: Dict[str, str],
    ):
        source_text = example["source"]
        target_text = example["target"]

        separator = " "
        input_text = source_text + separator + target_text
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

def get_dataset_and_preprocess_fn(
    task: str,
    preprocessor: Optional[PretrainingTaskPreprocessor],
    FLAN_dataset: Optional[datasets.Dataset],
):
    if task == "gsm8k":
        dataset = datasets.load_dataset("openai/gsm8k", "main", split="train")
        preprocess_fn = preprocessor.process_gsm8k
        remove_columns=["question", "answer"]
    elif task == "arc_challenge":
        dataset = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        preprocess_fn = preprocessor.process_arc
        remove_columns=["question", "id", "choices", "answerKey"]
    elif task == "arc_easy":
        dataset = datasets.load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
        preprocess_fn = preprocessor.process_arc
        remove_columns=["question", "id", "choices", "answerKey"]
    elif task == "arc":
        arc_challenge = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        arc_easy = datasets.load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
        dataset = datasets.concatenate_datasets([arc_challenge, arc_easy],)
        preprocess_fn = preprocessor.process_arc
        remove_columns=["question", "id", "choices", "answerKey"]
    elif task == "hellaswag":
        dataset = datasets.load_dataset("Rowan/hellaswag",   split="train")
        preprocess_fn = preprocessor.process_hellaswag
        remove_columns=[
            "ind", "activity_label", "ctx_a", "ctx_b",
            "ctx", "endings", "split", "split_type", "label",
            "source_id",
        ]
    elif task == "piqa":
        dataset = datasets.load_dataset("ybisk/piqa", split="train", trust_remote_code=True)
        preprocess_fn = preprocessor.process_piqa
        remove_columns=["label", "goal", "sol1", "sol2"]
    elif task == "winogrande":
        dataset = datasets.load_dataset(
            "allenai/winogrande",
            "winogrande_xl",
            split="train",
            trust_remote_code=True
        )
        preprocess_fn = preprocessor.process_winogrand
        remove_columns=["sentence", "option1", "option2", "answer"]
    # otherwise the task comes from FLAN_V2 and we use a unified preprocess fn
    else:
        assert FLAN_dataset is not None
        dataset = load_flan_subset(flan_dataset=FLAN_dataset, taskname=task,)
        remove_columns = [
            "source", "target", "task_name", "task_source",
            "template_type", "template_idx", "split",
        ]
        preprocess_fn = preprocessor.process_flan

    return dataset, preprocess_fn, remove_columns




