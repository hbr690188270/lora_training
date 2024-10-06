import torch
import numpy as np
from transformers.tokenization_utils_base import BatchEncoding

def move_to_target_device(object, device):
    if torch.is_tensor(object):
        return object.to(device)
    elif isinstance(object, dict):
        return {k: move_to_target_device(v, device) for k, v in object.items()}
    elif isinstance(object, BatchEncoding):
        return {k: move_to_target_device(v, device) for k, v in object.items()}
    elif isinstance(object, list):
        return [move_to_target_device(x, device) for x in object]
    else:
        return object
