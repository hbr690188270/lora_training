from transformers import AutoTokenizer, PreTrainedTokenizerFast

# tokenizer = AutoTokenizer.from_pretrained("model_cache/llama3-8b-instruct")
tokenizer = AutoTokenizer.from_pretrained("model_cache/llama2-7b-chat")
# tokenizer = AutoTokenizer.from_pretrained("model_cache/mistral-7b-instruct-v3")
# print(tokenizer.chat_template)

messages = [
    {"role": "user", "content": "a test message"},
    # {"role": "human", "content": "a test message"},
    {"role": "assistant", "content": "a response"}
]

outputs = tokenizer.apply_chat_template(messages, tokenize=False)
print(outputs)

response_template = "[/INST]"
# response_template = "[/INST]"

response_tokens = tokenizer.tokenize(response_template)
response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)

tokenized = tokenizer(
    outputs,
    add_special_tokens=False,
)
output_tokens = tokenizer.tokenize(outputs)

print(tokenized["input_ids"])
print(tokenizer.convert_ids_to_tokens(tokenized["input_ids"]))
print(output_tokens)
print(response_token_ids)
print(response_tokens)

