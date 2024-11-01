from huggingface_hub import snapshot_download


def main():
    # snapshot_download(repo_id="microsoft/Phi-3-mini-4k-instruct", local_dir = "model_cache/phi-3-mini-4k-instruct")
    # snapshot_download(repo_id="microsoft/Phi-3.5-mini-instruct", local_dir = "model_cache/phi-3.5-mini-instruct")
    # snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", local_dir = "model_cache/llama3-8b-instruct")
    # snapshot_download(repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct", local_dir = "model_cache/llama3_1-8b-instruct")
    # snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3", local_dir = "model_cache/mistral-7b-instruct-v3")
    # snapshot_download(repo_id="meta-llama/Llama-2-7b-chat-hf", local_dir = "model_cache/llama2-7b-chat")
    # snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B", local_dir = "model_cache/llama3-8b")
    # snapshot_download(repo_id="meta-llama/Llama-3.1-8B", local_dir = "model_cache/llama3_1-8b")
    snapshot_download(repo_id="mistralai/Mistral-7B-v0.3", local_dir = "model_cache/mistral-7b-v3")

if __name__ == "__main__":
    main()

