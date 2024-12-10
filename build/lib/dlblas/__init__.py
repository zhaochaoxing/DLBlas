from typing import Tuple
import torch
from torch import Tensor

# this import all kernels dynamically
import dlblas.kernels
from dlblas.utils import get_op

__version__ = "0.0.1"


# output: l_aux, token_rearranged_ec_idx, token_exp_weights, expert_select_token_idx
def topk_gating(
    logits: Tensor,
    k: int,
    capacity_factor: float = 1.0,
    min_capacity: int = 2,
    higher_precision: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    op = get_op(
        "topk_gating", (logits, k, capacity_factor, min_capacity, higher_precision)
    )
    return op(logits, k, capacity_factor, min_capacity, higher_precision)


def layernorm_gated(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    op = get_op(
        "layernorm_gated",
        (x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm),
    )
    return op(x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm)


def selective_state_update(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    op = get_op(
        "selective_state_update", (state, x, dt, A, B, C, D, z, dt_bias, dt_softplus)
    )
    return op(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus)


def matmul(a: Tensor, b: Tensor, activation=""):
    if activation == "leaky_relu":
        op = get_op("matmul_leaky_relu", (a, b, activation))
        return op(a, b, activation)
    elif activation == "":
        op = get_op("matmul", (a, b))
        return op(a, b)
    else:
        raise f"matmul_{activation} not impl."


def _topk_gating_fwd_part1(logits: Tensor, k: int):
    op = get_op("_topk_gating_fwd_part1", (logits, k))
    return op(logits, k)


def _topk_gating_fwd_part2(gates: Tensor, masks: Tensor, k: int):
    op = get_op("_topk_gating_fwd_part2", (gates, masks, k))
    return op(gates, masks, k)


def _topk_gating_fwd_part3(
    gates: Tensor,
    masks: Tensor,
    locations: Tensor,
    k: int,
    capacity: int,
    enable_token_rearrange_opt,
):
    op = get_op(
        "_topk_gating_fwd_part3",
        (gates, masks, locations, k, capacity, enable_token_rearrange_opt),
    )
    return op(gates, masks, locations, k, capacity, enable_token_rearrange_opt)


def _topk_gating_bwd(grad_l_aux, locations, masks, gates, ce):
    op = get_op("_topk_gating_bwd", (grad_l_aux, locations, masks, gates, ce))
    return op(grad_l_aux, locations, masks, gates, ce)


def paged_attention(
    query: Tensor,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    key_cache: Tensor,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    value_cache: Tensor,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE], required same stride with key_cache
    context_lens: Tensor,  # [num_seqs]
    block_tables: Tensor,  # [num_seqs, max_num_blocks_per_seq]
    attn_scale: float,
    max_context_len: int,
):
    op = get_op(
        "paged_attention",
        (
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            attn_scale,
            max_context_len,
        ),
    )
    return op(
        query,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
    )


def selective_scan(u, delta, A, B, C, D, initial_state):
    op = get_op("selective_scan", (u, delta, A, B, C, D, initial_state))
    return op(u, delta, A, B, C, D, initial_state)


def add_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-6,
):
    op = get_op("add_rms_norm", (hidden_states, weight, eps, residual))
    return op(hidden_states, weight, eps, residual)


def rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
):
    op = get_op("rms_norm", (hidden_states, weight, eps))
    return op(hidden_states, weight, eps)


def fill_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_indices: torch.Tensor,
):
    op = get_op("fill_kv_cache", (key, value, key_cache, value_cache, kv_indices))
    return op(key, value, key_cache, value_cache, kv_indices)


def partial_rotary_emb(q, k_pe, kv, cos, sin):
    op = get_op("partial_rotary_emb", (q, k_pe, kv, cos, sin))
    return op(q, k_pe, kv, cos, sin)


def fused_rotary_and_fa(q, k, v, cos, sin):
    op = get_op("fused_rotary_and_fa", (q, k, v, cos, sin))
    return op(q, k, v, cos, sin)


def flash_attention_v2(q, k, v):
    op = get_op("flash_attention_v2", (q, k, v))
    return op(q, k, v)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids_1d):
    op = get_op("apply_rotary_pos_emb", (q, k, cos, sin, position_ids_1d))
    return op(q, k, cos, sin, position_ids_1d)
