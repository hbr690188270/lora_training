from huggingface_hub import snapshot_download

snapshot_download(repo_id="microsoft/Phi-3-mini-4k-instruct", local_dir = "model_cache/phi-3-mini-4k-instruct")
snapshot_download(repo_id="microsoft/Phi-3.5-mini-instruct", local_dir = "model_cache/phi-3.5-mini-instruct")
