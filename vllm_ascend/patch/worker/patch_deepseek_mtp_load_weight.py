# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing
from typing import Optional
from collections.abc import Callable, Iterable

import torch
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.platforms import current_platform

from vllm.model_executor.models.deepseek_v2 import (
    get_spec_layer_idx_from_weight_name
)
from vllm.attention.layer import MLAAttention
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP


logger = init_logger(__name__)

def remap_C8_kv_scale_name(name: str, params_dict: dict) -> Optional[str]:
    replace_scale_names = [
        "fa_q.scale", "fa_k.scale", "fa_v.scale", "fa_q.offset",
        "fa_k.offset", "fa_v.offset"
    ]
    for scale_name in replace_scale_names:
        if name.endswith(scale_name):
            remap_name = name.replace(scale_name, f"mla_attn.mla_attn.{scale_name}")
            if remap_name in params_dict:
                return remap_name
            else:
                return remap_name.replace("mla_attn", "attn")
    return name



def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    rocm_aiter_moe_shared_expert_enabled = (
        rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
    )
    stacked_params_mapping = [
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
        ("fused_qkv_a_proj", "q_a_proj", 0),
        ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
    ]

    expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.config.n_routed_experts
                    + (
                        self.config.n_shared_experts
                        if rocm_aiter_moe_shared_expert_enabled
                        else 0
                    ),
    )

    params_dict = dict(self.named_parameters())
    print(params_dict.keys())
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:

        if "rotary_emb.inv_freq" in name:
            continue
        spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
        if spec_layer is None:
            continue
        is_fusion_moe_shared_experts_layer = (
                rocm_aiter_moe_shared_expert_enabled and ("mlp.shared_experts" in name)
        )
        name = self._rewrite_spec_layer_name(spec_layer, name)
        for param_name, weight_name, shard_id in stacked_params_mapping:
            # Skip non-stacked layers and experts (experts handled below).
            if weight_name not in name:
                continue
            # We have mlp.experts[0].gate_proj in the checkpoint.
            # Since we handle the experts below in expert_params_mapping,
            # we need to skip here BEFORE we update the name, otherwise
            # name will be updated to mlp.experts[0].gate_up_proj, which
            # will then be updated below in expert_params_mapping
            # for mlp.experts[0].gate_gate_up_proj, which breaks load.
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            if is_fusion_moe_shared_experts_layer:
                continue
            name_mapped = name.replace(weight_name, param_name)

            # QKV fusion is optional, fall back to normal
            # weight loading if it's not enabled
            if (
                    param_name == "fused_qkv_a_proj"
            ) and name_mapped not in params_dict:
                continue
            else:
                name = name_mapped

            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            # Special handling: when AITER fusion_shared_experts is enabled,
            # checkpoints may provide a single widened shared_experts tensor
            # without explicit expert indices
            # (e.g. ...mlp.shared_experts.gate_proj.weight).
            # For models with multiple shared experts, split that tensor
            # evenly into per-shared-expert slices and load them into
            # appended expert slots mlp.experts.{n_routed_experts + j}.*
            # accordingly.
            num_chunks = 1
            if is_fusion_moe_shared_experts_layer:
                num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                # Determine split axis based on op type
                # gate/up: ColumnParallel → split along dim 0
                # down: RowParallel → split along dim 1
                split_dim = 1 if "down_proj.weight" in name else 0
                total = loaded_weight.shape[split_dim]
                assert total % num_chunks == 0, (
                    f"Shared expert weight dim {total} "
                    f"not divisible by num_chunks {num_chunks}"
                )
                chunk_size = total // num_chunks

            for j in range(num_chunks):
                chunk_name = name
                weight_to_load = loaded_weight

                if is_fusion_moe_shared_experts_layer:
                    if split_dim == 0:
                        weight_to_load = loaded_weight[
                                         j * chunk_size : (j + 1) * chunk_size, :
                                         ]
                    else:
                        weight_to_load = loaded_weight[
                                         :, j * chunk_size : (j + 1) * chunk_size
                                         ]
                    # Synthesize an expert-style name so expert mapping
                    # can route it
                    chunk_name = name.replace(
                        "mlp.shared_experts",
                        f"mlp.experts.{self.config.n_routed_experts + j}",
                    )

                # Use expert_params_mapping to locate the destination
                # param and delegate to its expert-aware weight_loader
                # with expert_id.
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in chunk_name:
                        continue

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = chunk_name.replace(weight_name, param_name)

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or
                    # not here since otherwise we may skip experts with
                    # other available replicas.
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        weight_to_load,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        if not is_fusion_moe_shared_experts_layer:
                            name = name_mapped
                        else:
                            loaded_params.add(name_mapped)
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    name = maybe_remap_kv_scale_name(name, params_dict)
                    name = remap_C8_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    # According to DeepSeek-V3 Technical Report, MTP modules
                    # shares embedding layer. We only load the first weights.
                    if (
                            spec_layer != self.model.mtp_start_layer_idx
                            and ".layers" not in name
                    ):
                        continue

                    param = params_dict[name]

                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
        if not is_fusion_moe_shared_experts_layer:
            loaded_params.add(name)

    fa_quant_layers = {
        param.split(".fa_k.scale")[0]
        for param in loaded_params if "fa_k.scale" in param
    }
    print(fa_quant_layers)
    modules_dict = dict(self.named_modules())
    for module_name, module in modules_dict.items():
        if isinstance(module, MLAAttention):
            if module_name in fa_quant_layers:
                print(module_name,"()"*20)
                module.dtype = torch.float8_e4m3fn
                # Due to the existence of the fallback layer, new attributes are added to distinguish
                setattr(module, "fa_quant_layer", True)
            else:
                setattr(module, "fa_quant_layer", False)

    return loaded_params


DeepSeekMTP.load_weights = load_weights
