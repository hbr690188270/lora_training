from typing import Any, Dict, List

from transformers import PreTrainedTokenizerBase

EOT_TOKEN = "<turn_end>"


class instruct_lm_preprocessor():
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
        eot_id: int,
        prepend_eos: bool = False,
    ) -> None:
        """
        Apply chat template to the conversation between a user and the assistant.

        Args:
            texts: the message from user and assistant.
            roles: the sources of the messages in `texts`, where 0 means human and 1 means assistant.
            max_len: the maximum length of the processed text.
            eot_id: the token id of the <turn_end> token.
            prepend_eos: whether prepend an eos token to the processed text.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.eot_id = eot_id
        self.prepend_eos = prepend_eos

        encoded_eot_id = self.tokenizer.encode(EOT_TOKEN, add_special_tokens=False)[0]
        assert encoded_eot_id == self.eot_id, f"{encoded_eot_id} != {self.eot_id}"

    def process_daring_anteater(
        self,
        example: Dict[str, str],
    ):
        """
        The entry point of the preprocessor for nvidia/Daring-Anteater dataset.
        """
        conversation = example["conversations"]
        texts = []
        roles = []
        if len(conversation) % 2 != 0:
            if conversation[0]["from"] == "Assistant":
                conversation = conversation[1:]
            elif conversation[-1]["from"] == "User":
                conversation = conversation[:-1]
            else:
                conversation = conversation[:-1]
        for idx in range(0, len(conversation), 2):
            if (
                conversation[idx]["from"] == "User" and
                conversation[idx + 1]["from"] == "Assistant"
            ):
                texts.append(conversation[idx]["value"])
                texts.append(conversation[idx + 1]["value"])
                roles += [0, 1]

        input_ids, labels = self.process_single_conversation(texts, roles)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def process_single_conversation(
        self,
        texts: List[str],
        roles: List[str],
    ) -> Any:
        """
        Apply chat template to the conversation between a user and the assistant.

        Args:
            texts: the message from user and assistant.
            roles: the sources of the messages in `texts`, where 0 means human and 1 means assistant.

        Returns:
            input_ids: the input_ids with chat template applied
            labels: the labels for training. Simply set the positions of user messages to -100.
        """
        assert roles[0] == 0
        assert roles[1] == 1
        input_ids = []
        labels = []
        system_prompt = "system\nA conversation between a user and a helpful assistant.<turn_end>"
        # system_prompt = system_prompt.replace("\n", "<n>")
        prefix_ids = self.tokenizer(
            system_prompt,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None,
        )["input_ids"]
        input_ids += prefix_ids
        labels += [-100] * len(prefix_ids)

        for idx in range(0, len(texts), 2):
            user_text = texts[idx]
            assistant_text = texts[idx]
            assert roles[idx] == 0
            assert roles[idx + 1] == 1

            user_text = " user\n" + user_text + EOT_TOKEN
            assistant_text = " assistant\n" + assistant_text + EOT_TOKEN

            # replace \n with <n>
            # user_text = user_text.replace("\n", "<n>")
            # assistant_text = assistant_text.replace("\n", "<n>")

            user_text_ids = self.tokenizer(
                user_text,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_tensors=None,
            )["input_ids"]
            assistant_text_ids = self.tokenizer(
                assistant_text,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_tensors=None,
            )["input_ids"]

            remaining_length = self.max_len - len(input_ids)
            if remaining_length <= 0:
                break
            if len(user_text_ids) + len(assistant_text_ids) <= remaining_length:
                input_ids = input_ids + user_text_ids + assistant_text_ids
                labels += [-100] * len(user_text_ids)
                labels += assistant_text_ids
            else:
                if len(input_ids) == 0:
                    if len(user_text_ids) >= remaining_length:
                        input_ids += user_text_ids
                        labels += [-100] * len(user_text_ids)
                        break
                    else:
                        input_ids += user_text_ids
                        labels += [-100] * len(user_text_ids)
                        input_ids += assistant_text_ids[:remaining_length - len(user_text_ids)]
                        labels += assistant_text_ids[:remaining_length - len(user_text_ids)]

        return input_ids, labels


