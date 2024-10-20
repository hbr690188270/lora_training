"""
python tools/activation_matching/train_permutation_matrix.py \
    --dataset=Daring-Anteater \
    --model=llama31 \
    --adapter_source=llama3_converted \
    --num_examples=60
"""

import torch
from absl import app, flags
from safetensors import safe_open

FLAGS = flags.FLAGS

def set_eval_args():
    flags.DEFINE_enum(
        "dataset",
        None,
        [
            "Daring-Anteater",
        ],
        help="Dataset that used to collect intermediate outputs after equipped with LoRA.",
    )
    flags.DEFINE_enum(
        "model",
        None,
        [
            "llama3",
            "llama31"
        ],
        help="model to be evauated.",
    )
    flags.DEFINE_enum(
        "adapter_source",
        None,
        [
            "llama3",
            "llama3_converted",
            "llama31",
            "none", # none means we do not load adpaters
        ],
        help="which model's adapter to load. None means do not load any adapters",
    )
    flags.DEFINE_integer(
        "num_examples",
        None,
        help="number of examples to be evaluated",
    )

def load_lora_weights():
    gt_adapter_path = f"logs/activations/{FLAGS.model}_{FLAGS.model}_{FLAGS.num_examples}.safetensor"
    transfered_adapter_path = f"logs/activations/{FLAGS.model}_{FLAGS.adapter_source}_{FLAGS.num_examples}.safetensor"

    gt_tensors = {}
    with safe_open(gt_adapter_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            gt_tensors[key] = f.get_tensor(key)

    transfered_tensors = {}
    with safe_open(transfered_adapter_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            transfered_tensors[key] = f.get_tensor(key)

    return gt_tensors, transfered_tensors

def get_naive_solution(
    gt_adapter_attentions: torch.LongTensor,
    transfered_attentions: torch.LongTensor,
):
    hidden_dim = gt_adapter_attentions.size(1)
    return torch.eye(hidden_dim)

def solve_layer_permutation(
    gt_adapter_attentions: torch.LongTensor,
    transfered_attentions: torch.LongTensor,
):
    """
    Args:
        gt_adapter_attentions: [num_examples, hidden_dim],
        transfered_attentions: [num_examples, hidden_dim],
    """
    part1 = torch.matmul(transfered_attentions.T, transfered_attentions)
    rank = torch.linalg.matrix_rank(part1)
    print("rank: ", rank)
    inverse = torch.inverse(part1)
    analytical_solution = torch.matmul(
        torch.matmul(inverse, transfered_attentions.T), gt_adapter_attentions
    )
    return analytical_solution

def main(argv):
    (
        all_layers_gt_adapter_attentions,
        all_layers_transfered_attentions,
    ) = load_lora_weights()

    # device = torch.device("cpu")
    device = torch.device("cuda")

    tgt_layer = 31
    num_examples = all_layers_gt_adapter_attentions[f"layer_{tgt_layer}"].size(0)
    train_gt_adapter_attentions = all_layers_gt_adapter_attentions[f"layer_{tgt_layer}"][:int(num_examples * 0.8)].to(device).float()
    train_transfered_adapter_attentions = all_layers_transfered_attentions[f"layer_{tgt_layer}"][:int(num_examples * 0.8)].to(device).float()

    test_gt_adapter_attentions = all_layers_gt_adapter_attentions[f"layer_{tgt_layer}"][int(num_examples * 0.8):].to(device).float()
    test_transfered_adapter_attentions = all_layers_transfered_attentions[f"layer_{tgt_layer}"][int(num_examples * 0.8):].to(device).float()

    print(train_gt_adapter_attentions.size())

    naive_solution = get_naive_solution(
        train_gt_adapter_attentions,
        train_transfered_adapter_attentions,
    ).to(device)

    analytical_solution = solve_layer_permutation(
        train_gt_adapter_attentions,
        train_transfered_adapter_attentions,
    ).to(device)

    naive_predictions = torch.matmul(test_transfered_adapter_attentions, naive_solution)
    analytical_predictions = torch.matmul(test_transfered_adapter_attentions, analytical_solution)

    naive_mse = ((test_gt_adapter_attentions - naive_predictions)**2).mean()
    analytical_mse = ((test_gt_adapter_attentions - analytical_predictions)**2).mean()

    print("No transformer: ", naive_mse)
    print("Analytical solution: ", analytical_mse)



if __name__ == "__main__":
    set_eval_args()
    app.run(main)



