from transformers import LlamaForCausalLM

import datasets

datapath = "/mnt/data/bairu/repos/adapter_transfer/modular_artifacts/flan-flat"
dataset = datasets.load_from_disk(datapath)

print(dataset)
import ipdb
ipdb.set_trace()

dataset["train"]

