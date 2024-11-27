"""
Merge the learned transformation matrices, such as BTA, PQBA, into the original LoRA adapters.
    - BTA: multiple matrix T with A to get the new A
    - PQBA: multiple P Q B to get the new B

CUDA_VISIBLE_DEVICES=1 python -m tools.lora_transform.apply_transform \
    --adapter_path=ckpt/instruct_lm/llama3_transform_for_31_alpha128_r64/checkpoint-11500/adapter_model.safetensors \
    --source_path=ckpt/llama3/dream_read_the_following_conversation_and_answer_the_question_alpha128_r64_chat/adapter_model.safetensors \
    --output_path=ckpt/llama3/dream_absorb_transform/adapter_model.safetensors
"""
import copy
import os
import shutil
from typing import Dict

import torch
from absl import app, flags
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from src.experiments.lora_transform.lora_transform_model import PQBASTEFLoraModel
from src.experiments.lora_transform.multi_task_trainer import LoraConfigV2

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
        ["BTA", "PQBA", "PQBA_m", "PQBAST", "PQBASTEF"],
        help="Path to save the converted LoRA weights",
    )


def absorb_transform(
    transform_tensors: Dict[str, torch.Tensor],
    source_tensors: Dict[str, torch.Tensor],
):
    """
    Apply the transformation matrices stored in `transform_tensors` to the source
    model's LoRAs stored in `source_tensors`.

    Args:
        transform_tensors: A model state dict that maps the parameter names to the weights
            It contains the trained transformation matrices such as T in BTA and PQ in PQBA.
        source_tensors: The state dict of the source model's adapter.
            It only contains the LoRA adapters, B and A in each module and layer.
    Returns:
        updated_tensors: the `source_tensors` with transformation matrices being absorbed.
    """

    updated_tensor = copy.deepcopy(source_tensors)
    device = torch.device("cuda")
    num_layers = 32
    for idx in range(num_layers):
        for module in ["q", "k", "v", "o"]:
            if FLAGS.transform_type == "BTA":
                lora_A_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_A.weight"
                lora_A = updated_tensor.pop(lora_A_key).float().to(device)
                lora_transform_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_transform_matrix.weight"
                lora_transform = transform_tensors.pop(lora_transform_key).float().to(device)
                new_lora_A = torch.matmul(lora_transform, lora_A).bfloat16().cpu()
                updated_tensor[lora_A_key] = new_lora_A
            elif FLAGS.transform_type == "PQBA":
                lora_B_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_B.weight"
                lora_B = updated_tensor.pop(lora_B_key).float().to(device)
                p_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_transform_matrix_default_p.weight"
                q_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_transform_matrix_default_q.weight"
                p_transform = transform_tensors.pop(p_key).float().to(device)
                q_transform = transform_tensors.pop(q_key).float().to(device)
                new_lora_B = torch.matmul(
                    p_transform, torch.matmul(
                        q_transform, lora_B
                    )
                )
                updated_tensor[lora_B_key] = new_lora_B
            elif FLAGS.transform_type == "PQBAST":
                lora_B_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_B.weight"
                lora_B = updated_tensor.pop(lora_B_key).float().to(device)
                lora_A_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_A.weight"
                lora_A = updated_tensor.pop(lora_A_key).float().to(device)
                p_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_transform_matrix_default_p.weight"
                q_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_transform_matrix_default_q.weight"
                s_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_transform_matrix_default_s.weight"
                t_key = f"base_model.model.model.layers.{idx}.self_attn.{module}_proj.lora_transform_matrix_default_t.weight"
                p_transform = transform_tensors.pop(p_key).float().to(device)
                q_transform = transform_tensors.pop(q_key).float().to(device)
                s_transform = transform_tensors.pop(s_key).float().to(device)
                t_transform = transform_tensors.pop(t_key).float().to(device)
                new_lora_B = torch.matmul(
                    p_transform, torch.matmul(
                        q_transform, lora_B
                    )
                )
                new_lora_A = torch.matmul(
                    torch.matmul(
                        lora_A, s_transform
                    ), t_transform
                )

                updated_tensor[lora_B_key] = new_lora_B
                updated_tensor[lora_A_key] = new_lora_A
            else:
                raise NotImplementedError()

    return updated_tensor

def main(argv):
    # PQBAST + EF cannot be absorbed into by the original LoRA BA
    # So we directly merge it into the full model
    if FLAGS.transform_type == "PQBASTEF":
        target_model = "model_cache/mistral-7b-v3"
        model = AutoModelForCausalLM.from_pretrained(target_model, torch_dtype=torch.bfloat16)
        loaded_kwarges = LoraConfigV2.from_json_file(
            os.path.join(FLAGS.adapter_path, "adapter_config.json")
        )
        peft_config = LoraConfigV2(**loaded_kwarges)
        print(peft_config)
        model = PQBASTEFLoraModel(model, peft_config)
        model.load_adapter(FLAGS.source_path, adapter_name="sft")
        model.load_adapter(FLAGS.adapter_path, adapter_name="default")
        merged_model = model.merge_and_unload(adapter_names=["sft"])
        merged_model.save_pretrained(FLAGS.output_path)
    else:
        transform_tensors = {}
        with safe_open(FLAGS.adapter_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                transform_tensors[key] = f.get_tensor(key)
        source_tensors = {}
        if FLAGS.source_path is None:
            source_tensors = transform_tensors
        else:
            with safe_open(FLAGS.source_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    source_tensors[key] = f.get_tensor(key)
        updated_tensors = absorb_transform(transform_tensors, source_tensors)

        if not os.path.exists(os.path.dirname(FLAGS.output_path)):
            os.makedirs(os.path.dirname(FLAGS.output_path))
        save_file(updated_tensors, FLAGS.output_path)

        # adatper_dir = os.path.dirname(FLAGS.adapter_path)
        # adater_config_path = os.path.join(adatper_dir, "adapter_config.json")
        adatper_dir = os.path.dirname(FLAGS.source_path)
        adater_config_path = os.path.join(adatper_dir, "adapter_config.json")
        output_dir = os.path.dirname(FLAGS.output_path)
        output_config_path = os.path.join(output_dir, "adapter_config.json")
        shutil.copy(adater_config_path, output_config_path)


if __name__ == "__main__":
    set_flags()
    app.run(main)

