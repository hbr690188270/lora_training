import re
import warnings
from contextlib import contextmanager
from functools import partial
from itertools import chain
from typing import Any, Optional, Union

import peft.tuners.lora.layer as lora_layer
import torch
import torch.nn as nn
from peft import LoraConfig, LoraModel
from peft.tuners.lora.model import _adapter_names_pre_forward_hook
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils import (
    ModulesToSaveWrapper,
)
from peft.utils.other import transpose
from transformers.pytorch_utils import Conv1D

from src.experiments.lora_transform.train_utils import INDEX_TO_DATASET


class Linear(nn.Module, lora_layer.LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        lora_layer.LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.lora_transform_matrix = nn.Linear(r, r, bias=False)
        with torch.no_grad():
            self.lora_transform_matrix.weight.copy_(torch.eye(r))

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        tranform_weight = self.lora_transform_matrix[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
            tranform_weight = tranform_weight.float()

        # Apply transformation matrix
        # output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
        output_tensor = transpose(
            weight_B @ torch.matmul(tranform_weight, weight_A), self.fan_in_fan_out
        ) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)
            self.lora_transform_matrix[adapter].weight.data = tranform_weight.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            # result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
            result = self.base_layer(x, *args, **kwargs)
            if type(adapter_names) is list:
                adapter_names = adapter_names[0]

            torch_result_dtype = result.dtype
            lora_A = self.lora_A[adapter_names]
            lora_B = self.lora_B[adapter_names]
            dropout = self.lora_dropout[adapter_names]
            scaling = self.scaling[adapter_names]
            x = x.to(lora_A.weight.dtype)

            result = result + lora_B(
                self.lora_transform_matrix[adapter_names](lora_A(dropout(x)))
            ) * scaling
            result = result.to(torch_result_dtype)

        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    # Apply the transformation matrix
                    result = result + lora_B(
                        self.lora_transform_matrix(lora_A(dropout(x)))
                    ) * scaling
                else:
                    raise NotImplementedError()
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

class PQBALinear(nn.Module, lora_layer.LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        lora_layer.LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.lora_transform_matrix_default_p = nn.Linear(r, self.out_features, bias=False)
        self.lora_transform_matrix_default_q = nn.Linear(self.out_features, r, bias=False)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        transform_weight_p = self.lora_transform_matrix_default_p.weight
        transform_weight_q = self.lora_transform_matrix_default_q.weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
            transform_weight_p = transform_weight_p.float()
            transform_weight_q = transform_weight_q.float()

        # Apply transformation matrix
        # output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
        lora_delta = weight_B @ weight_A
        q_lora_delta = transform_weight_q @ lora_delta
        pq_lora_delta = transform_weight_p @ q_lora_delta

        output_tensor = transpose(
            pq_lora_delta, self.fan_in_fan_out
        ) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)
            self.lora_transform_matrix_default_p.weight.data = transform_weight_p.to(dtype)
            self.lora_transform_matrix_default_q.weight.data = transform_weight_q.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)

            lora_change = lora_B(lora_A(dropout(sub_batch)))
            transformed_change = self.lora_transform_matrix_default_p(
                self.lora_transform_matrix_default_q(lora_change)
            )

            lora_output = transformed_change * scaling
            # result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)
            result[sub_batch_indices_list[i]] = (
                result[sub_batch_indices_list[i]] + lora_output.to(torch_result_dtype)
            )

        return result

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # import ipdb
        # ipdb.set_trace()
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        # dataset_index = kwargs.pop("dataset_index", None)
        adapter_names = [INDEX_TO_DATASET[x] for x in adapter_names]


        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    # Apply the transformation matrix
                    lora_change = lora_B(lora_A(dropout(x)))
                    transformed_change = self.lora_transform_matrix_default_p(
                        self.lora_transform_matrix_default_q(lora_change)
                    )
                    result = result + transformed_change * scaling
                else:
                    raise NotImplementedError()
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

class PQBASTLinear(nn.Module, lora_layer.LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        transform_r_multiple: int = 1,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        lora_layer.LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        transform_r = transform_r_multiple * r
        self.lora_transform_matrix_default_p = nn.Linear(transform_r, self.out_features, bias=False)
        self.lora_transform_matrix_default_q = nn.Linear(self.out_features, transform_r, bias=False)
        self.lora_transform_matrix_default_s = nn.Linear(transform_r, self.in_features, bias=False)
        self.lora_transform_matrix_default_t = nn.Linear(self.in_features, transform_r, bias=False)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        transform_weight_p = self.lora_transform_matrix_default_p.weight
        transform_weight_q = self.lora_transform_matrix_default_q.weight
        transform_weight_s = self.lora_transform_matrix_default_s.weight
        transform_weight_t = self.lora_transform_matrix_default_t.weight


        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
            transform_weight_p = transform_weight_p.float()
            transform_weight_q = transform_weight_q.float()
            transform_weight_s = transform_weight_s.float()
            transform_weight_t = transform_weight_t.float()

        # Apply transformation matrix
        # output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
        lora_delta = weight_B @ weight_A
        q_lora_delta = transform_weight_q @ lora_delta
        pq_lora_delta = transform_weight_p @ q_lora_delta
        pq_lora_delta_s = pq_lora_delta @ transform_weight_s
        pq_lora_delta_st = pq_lora_delta_s @ transform_weight_t

        output_tensor = transpose(
            pq_lora_delta_st, self.fan_in_fan_out
        ) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)
            self.lora_transform_matrix_default_p.weight.data = transform_weight_p.to(dtype)
            self.lora_transform_matrix_default_q.weight.data = transform_weight_q.to(dtype)
            self.lora_transform_matrix_default_s.weight.data = transform_weight_s.to(dtype)
            self.lora_transform_matrix_default_t.weight.data = transform_weight_t.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)

            ST_transformed_change = self.lora_transform_matrix_default_s(
                self.lora_transform_matrix_default_t(sub_batch)
            )
            lora_change = lora_B(lora_A(dropout(ST_transformed_change)))
            transformed_change = self.lora_transform_matrix_default_p(
                self.lora_transform_matrix_default_q(lora_change)
            )

            lora_output = transformed_change * scaling
            # result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)
            result[sub_batch_indices_list[i]] = (
                result[sub_batch_indices_list[i]] + lora_output.to(torch_result_dtype)
            )

        return result

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # import ipdb
        # ipdb.set_trace()
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        # dataset_index = kwargs.pop("dataset_index", None)
        adapter_names = [INDEX_TO_DATASET[x] for x in adapter_names]


        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            raise NotImplementedError()
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

class PQBASTEFLinear(nn.Module, lora_layer.LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        transform_r_multiple: int = 1,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        lora_layer.LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        transform_r = transform_r_multiple * r
        self.lora_transform_matrix_default_p = nn.Linear(transform_r, self.out_features, bias=False)
        self.lora_transform_matrix_default_q = nn.Linear(self.out_features, transform_r, bias=False)
        self.lora_transform_matrix_default_s = nn.Linear(transform_r, self.in_features, bias=False)
        self.lora_transform_matrix_default_t = nn.Linear(self.in_features, transform_r, bias=False)
        # self.lora_transform_matrix_default_e = nn.Linear(transform_r, self.in_features, bias=False)
        # self.lora_transform_matrix_default_f = nn.Linear(self.in_features, transform_r, bias=False)

        residual_r = r // 4
        self.lora_transform_matrix_default_e = nn.Linear(residual_r, self.out_features, bias=False)
        self.lora_transform_matrix_default_f = nn.Linear(self.in_features, residual_r, bias=False)
        nn.init.zeros_(self.lora_transform_matrix_default_e.weight)
        nn.init.zeros_(self.lora_transform_matrix_default_f.weight)


    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        transform_weight_p = self.lora_transform_matrix_default_p.weight
        transform_weight_q = self.lora_transform_matrix_default_q.weight
        transform_weight_s = self.lora_transform_matrix_default_s.weight
        transform_weight_t = self.lora_transform_matrix_default_t.weight
        transform_weight_e = self.lora_transform_matrix_default_e.weight
        transform_weight_f = self.lora_transform_matrix_default_f.weight


        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
            transform_weight_p = transform_weight_p.float()
            transform_weight_q = transform_weight_q.float()
            transform_weight_s = transform_weight_s.float()
            transform_weight_t = transform_weight_t.float()
            transform_weight_e = transform_weight_e.float()
            transform_weight_f = transform_weight_f.float()

        # Apply transformation matrix
        # output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
        lora_delta = weight_B @ weight_A
        q_lora_delta = transform_weight_q @ lora_delta
        pq_lora_delta = transform_weight_p @ q_lora_delta
        pq_lora_delta_s = pq_lora_delta @ transform_weight_s
        pq_lora_delta_st = pq_lora_delta_s @ transform_weight_t

        residual = transform_weight_e @ transform_weight_f
        pq_lora_delta_st = pq_lora_delta_st + residual

        output_tensor = transpose(
            pq_lora_delta_st, self.fan_in_fan_out
        ) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)
            self.lora_transform_matrix_default_p.weight.data = transform_weight_p.to(dtype)
            self.lora_transform_matrix_default_q.weight.data = transform_weight_q.to(dtype)
            self.lora_transform_matrix_default_s.weight.data = transform_weight_s.to(dtype)
            self.lora_transform_matrix_default_t.weight.data = transform_weight_t.to(dtype)
            self.lora_transform_matrix_default_e.weight.data = transform_weight_e.to(dtype)
            self.lora_transform_matrix_default_f.weight.data = transform_weight_f.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)

            ST_transformed_change = self.lora_transform_matrix_default_s(
                self.lora_transform_matrix_default_t(sub_batch)
            )
            lora_change = lora_B(lora_A(dropout(ST_transformed_change)))
            transformed_change = self.lora_transform_matrix_default_p(
                self.lora_transform_matrix_default_q(lora_change)
            )
            residual_change = self.lora_transform_matrix_default_e(
                self.lora_transform_matrix_default_f(sub_batch)
            )
            transformed_change = transformed_change + residual_change

            lora_output = transformed_change * scaling
            # result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)
            result[sub_batch_indices_list[i]] = (
                result[sub_batch_indices_list[i]] + lora_output.to(torch_result_dtype)
            )

        return result

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # import ipdb
        # ipdb.set_trace()
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        # dataset_index = kwargs.pop("dataset_index", None)
        adapter_names = [INDEX_TO_DATASET[x] for x in adapter_names]


        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            raise NotImplementedError()
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            raise ValueError()
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights += delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight

                self.merged_adapters.append(active_adapter)

class FullRankLinear(nn.Module, lora_layer.LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        transform_r_multiple: int = 1,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        lora_layer.LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.lora_transform_matrix_default_p = nn.Linear(self.out_features, self.out_features, bias=False)
        self.lora_transform_matrix_default_t = nn.Linear(self.in_features, self.in_features, bias=False)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        transform_weight_p = self.lora_transform_matrix_default_p.weight
        transform_weight_t = self.lora_transform_matrix_default_t.weight


        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
            transform_weight_p = transform_weight_p.float()
            transform_weight_t = transform_weight_t.float()

        # Apply transformation matrix
        # output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
        lora_delta = weight_B @ weight_A
        p_lora_delta = transform_weight_p @ lora_delta
        p_lora_delta_t = p_lora_delta @ transform_weight_t

        output_tensor = transpose(
            p_lora_delta_t, self.fan_in_fan_out
        ) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)
            self.lora_transform_matrix_default_p.weight.data = transform_weight_p.to(dtype)
            self.lora_transform_matrix_default_t.weight.data = transform_weight_t.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)

            transformed_change = self.lora_transform_matrix_default_t(sub_batch)
            lora_change = lora_B(lora_A(dropout(transformed_change)))
            transformed_change = self.lora_transform_matrix_default_p(lora_change)

            lora_output = transformed_change * scaling
            # result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)
            result[sub_batch_indices_list[i]] = (
                result[sub_batch_indices_list[i]] + lora_output.to(torch_result_dtype)
            )

        return result

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # import ipdb
        # ipdb.set_trace()
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        # dataset_index = kwargs.pop("dataset_index", None)
        adapter_names = [INDEX_TO_DATASET[x] for x in adapter_names]


        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            raise NotImplementedError()
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep



def dispatch_BTA_transform_lora(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        raise NotImplementedError()
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        raise NotImplementedError()
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        raise NotImplementedError()

    return new_module

def dispatch_PQBA_transform_lora(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        raise NotImplementedError()
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        raise NotImplementedError()
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = PQBALinear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        raise NotImplementedError()

    return new_module

def dispatch_PQBAST_transform_lora(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        raise NotImplementedError()
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        raise NotImplementedError()
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        transform_r_multiple = lora_config.transform_r_multiple
        new_module = PQBASTLinear(
            target, adapter_name, transform_r_multiple=transform_r_multiple, **kwargs
        )
    elif isinstance(target_base_layer, Conv1D):
        raise NotImplementedError()

    return new_module

def dispatch_PQBASTEF_transform_lora(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        raise NotImplementedError()
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        raise NotImplementedError()
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        transform_r_multiple = lora_config.transform_r_multiple
        new_module = PQBASTEFLinear(
            target, adapter_name, transform_r_multiple=transform_r_multiple, **kwargs
        )
    elif isinstance(target_base_layer, Conv1D):
        raise NotImplementedError()

    return new_module

def dispatch_full_rank_transform_lora(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        raise NotImplementedError()
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        raise NotImplementedError()
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        transform_r_multiple = lora_config.transform_r_multiple
        new_module = FullRankLinear(
            target, adapter_name, transform_r_multiple=transform_r_multiple, **kwargs
        )
    elif isinstance(target_base_layer, Conv1D):
        raise NotImplementedError()

    return new_module



class LoraWithBTATransform(LoraModel):
    def __init__(
        self,
        model,
        config,
        adapter_name,
        low_cpu_mem_usage = False,
    ):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = [dispatch_BTA_transform_lora]

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(
                target,
                adapter_name,
                lora_config=lora_config,
                **kwargs
            )
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        if isinstance(target, lora_layer.LoraLayer) and not isinstance(target, AdaLoraLayer):
            raise NotImplementedError()
        else:
            new_module = self._create_new_module(
                lora_config,
                adapter_name,
                target,
                **kwargs
            )
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

class LoraWithPQBATransform(LoraModel):
    def __init__(
        self,
        model,
        config,
        adapter_name,
        low_cpu_mem_usage = False,
    ):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = [dispatch_PQBA_transform_lora]

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(
                target,
                adapter_name,
                lora_config=lora_config,
                **kwargs
            )
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        if isinstance(target, PQBALinear):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
            )
        elif isinstance(target, lora_layer.LoraLayer) and not isinstance(target, AdaLoraLayer):
            raise NotImplementedError()
        else:
            new_module = self._create_new_module(
                lora_config,
                adapter_name,
                target,
                **kwargs
            )
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)


    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If adapter_names is passed as an argument, we inject it into the forward arguments.
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is None:
            # nothing to do
            yield
            return

        # if self.training:
        #     raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

        hook_handles = []
        for module in self.modules():
            if isinstance(module, lora_layer.LoraLayer) or isinstance(module, ModulesToSaveWrapper):
                pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()

class LoraWithPQBASTTransform(LoraModel):
    def __init__(
        self,
        model,
        config,
        adapter_name,
        low_cpu_mem_usage = False,
    ):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = [dispatch_PQBAST_transform_lora]

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(
                target,
                adapter_name,
                lora_config=lora_config,
                **kwargs
            )
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        # Originally here is Linear. We need to replace it with PQBASTLinear.
        if isinstance(target, PQBASTLinear):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
            )
        elif isinstance(target, lora_layer.LoraLayer) and not isinstance(target, AdaLoraLayer):
            raise NotImplementedError()
        else:
            new_module = self._create_new_module(
                lora_config,
                adapter_name,
                target,
                **kwargs
            )
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)


    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If adapter_names is passed as an argument, we inject it into the forward arguments.
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is None:
            # nothing to do
            yield
            return

        # Bairu: comment the following code to allow multi-task lora training.
        # if self.training:
        #     raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

        hook_handles = []
        for module in self.modules():
            if isinstance(module, lora_layer.LoraLayer) or isinstance(module, ModulesToSaveWrapper):
                pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()

class LoraWithPQBASTEFTransform(LoraModel):
    def __init__(
        self,
        model,
        config,
        adapter_name,
        low_cpu_mem_usage = False,
    ):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = [dispatch_PQBASTEF_transform_lora]

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(
                target,
                adapter_name,
                lora_config=lora_config,
                **kwargs
            )
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        # Originally here is Linear. We need to replace it with PQBASTLinear.
        if isinstance(target, PQBASTEFLinear):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
            )
        elif isinstance(target, lora_layer.LoraLayer) and not isinstance(target, AdaLoraLayer):
            raise NotImplementedError()
        else:
            new_module = self._create_new_module(
                lora_config,
                adapter_name,
                target,
                **kwargs
            )
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)


    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If adapter_names is passed as an argument, we inject it into the forward arguments.
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is None:
            # nothing to do
            yield
            return

        # Bairu: comment the following code to allow multi-task lora training.
        # if self.training:
        #     raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

        hook_handles = []
        for module in self.modules():
            if isinstance(module, lora_layer.LoraLayer) or isinstance(module, ModulesToSaveWrapper):
                pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()

class LoraWithFullRankTransform(LoraModel):
    def __init__(
        self,
        model,
        config,
        adapter_name,
        low_cpu_mem_usage = False,
    ):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = [dispatch_full_rank_transform_lora]

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(
                target,
                adapter_name,
                lora_config=lora_config,
                **kwargs
            )
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        # Originally here is Linear. We need to replace it with PQBASTLinear.
        if isinstance(target, FullRankLinear):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
            )
        elif isinstance(target, lora_layer.LoraLayer) and not isinstance(target, AdaLoraLayer):
            raise NotImplementedError()
        else:
            new_module = self._create_new_module(
                lora_config,
                adapter_name,
                target,
                **kwargs
            )
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)


    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If adapter_names is passed as an argument, we inject it into the forward arguments.
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is None:
            # nothing to do
            yield
            return

        # Bairu: comment the following code to allow multi-task lora training.
        # if self.training:
        #     raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

        hook_handles = []
        for module in self.modules():
            if isinstance(module, lora_layer.LoraLayer) or isinstance(module, ModulesToSaveWrapper):
                pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()


