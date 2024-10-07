import json

path1 = "generations/ifeval_logs_llama3_to_llama3.jsonl"
path2 = "generations/ifeval_logs_llama3_to_llama31.jsonl"
path3 = "generations/ifeval_logs_llama3_to_llama312.jsonl"

ref_logs = [json.loads(x) for x in open(path1).readlines()]
to_revise = [json.loads(x) for x in open(path2).readlines()]

ref_prompts = [x["prompt"] for x in ref_logs]
revised = []
for idx, log in enumerate(to_revise):
    log["prompt"] = ref_prompts[idx]
    revised.append(log)

with open(path2, "w") as f:
    for item in revised:
        f.write(json.dumps(item) + "\n")
