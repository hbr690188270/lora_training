"""
CUDA_VISIBLE_DEVICES=1 python -m tools.lora_transform.apply_transform \
    --adapter_path=ckpt/instruct_lm/llama3_transform_for_31_alpha128_r64/checkpoint-11500/adapter_model.safetensors \
    --source_path=ckpt/llama3/dream_read_the_following_conversation_and_answer_the_question_alpha128_r64_chat/adapter_model.safetensors \
    --output_path=ckpt/llama3/dream_absorb_transform/adapter_model.safetensors
"""
import os
import shutil

import torch
from absl import app, flags
from safetensors import safe_open
from safetensors.torch import save_file

FLAGS = flags.FLAGS

def set_flags():
    flags.DEFINE_string(
        "adapter_path",
        None,
        help="Path to the `safetensors` file that stores the transformation matrix",
    )
    flags.DEFINE_string(
        "source_path",
        None,
        required=False,
        help="Path to the `safetensors` file that stores the source LoRA. "
        "Will be `adapter_path` if not specified."
    )
    flags.DEFINE_string(
        "output_path",
        None,
        help="Path to save the converted LoRA weights",
    )
    flags.DEFINE_enum(
        "transform_type",
        None,
        ["BTA", "PQBA", "PQBA_m", "PQBAST"],
        help="Path to save the converted LoRA weights",
    )


def main(argv):
    transform_tensors = {}
    with safe_open(FLAGS.adapter_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            print(key)
            transform_tensors[key] = f.get_tensor(key)
    pause = input("???")

    source_tensors = {}
    if FLAGS.source_path is None:
        source_tensors = transform_tensors
    else:
        with safe_open(FLAGS.source_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                source_tensors[key] = f.get_tensor(key)

    device = torch.device("cuda")
    num_layers = 32
    for idx in range(num_layers):
        for module in ["q", "k", "v", "o"]:
            if FLAGS.transform_type == "BTA":
                lora_A_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_A.weight"
                lora_A = source_tensors.pop(lora_A_key).float().to(device)
                lora_transform_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_transform_matrix.weight"
                lora_transform = transform_tensors.pop(lora_transform_key).float().to(device)
                new_lora_A = torch.matmul(lora_transform, lora_A).bfloat16().cpu()
                source_tensors[lora_A_key] = new_lora_A
            elif FLAGS.transform_type == "PQBA":
                lora_B_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_B.weight"
                lora_B = source_tensors.pop(lora_B_key).float().to(device)
                p_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_transform_matrix_p.weight"
                q_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_transform_matrix_q.weight"
                p_transform = transform_tensors.pop(p_key).float().to(device)
                q_transform = transform_tensors.pop(q_key).float().to(device)
                new_lora_B = torch.matmul(
                    p_transform, torch.matmul(
                        q_transform, lora_B
                    )
                )
                source_tensors[lora_B_key] = new_lora_B
            elif FLAGS.transform_type == "PQBA_m":
                lora_B_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_B.weight"
                lora_B = source_tensors.pop(lora_B_key).float().to(device)
                p_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_transform_matrix_p.weight"
                q_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_transform_matrix_q.weight"
                p_transform = transform_tensors.pop(p_key).float().to(device)
                q_transform = transform_tensors.pop(q_key).float().to(device)
                new_lora_B = torch.matmul(
                    p_transform, torch.matmul(
                        q_transform, lora_B
                    )
                )
                source_tensors[lora_B_key] = new_lora_B
            else:
                raise NotImplementedError()

    if not os.path.exists(os.path.dirname(FLAGS.output_path)):
        os.makedirs(os.path.dirname(FLAGS.output_path))
    save_file(source_tensors, FLAGS.output_path)

    adatper_dir = os.path.dirname(FLAGS.adapter_path)
    adater_config_path = os.path.join(adatper_dir, "adapter_config.json")
    output_dir = os.path.dirname(FLAGS.output_path)
    output_config_path = os.path.join(output_dir, "adapter_config.json")
    shutil.copy(adater_config_path, output_config_path)


if __name__ == "__main__":
    set_flags()
    app.run(main)

