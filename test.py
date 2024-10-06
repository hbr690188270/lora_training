import datasets
from transformers import AutoTokenizer

text = " user\n\n\n"
tokenizer = AutoTokenizer.from_pretrained("model_cache/llama3-8b-instruct")
res = tokenizer(text,                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_tensors=None,
)

print(res["input_ids"])
