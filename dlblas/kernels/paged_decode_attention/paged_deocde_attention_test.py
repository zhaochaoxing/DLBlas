import math

import pytest

# import torch_mlu
# import torch_mlu.utils.gpu_migration
import torch
import sys

sys.path.insert(0, "/home/aigc/PRJ/dlBLAS/dlblas/kernels/paged_decode_attention")


from paged_deocde_attention import paged_decode_attention_fwd

import triton

from typing import List, Optional, Tuple

import torch

def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]
        seq_len = int(seq_lens_lst[i])

        keys_lst: List[torch.Tensor] = []
        values_lst: List[torch.Tensor] = []
        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset]
            keys_lst.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(seq_len).int()
            alibi_bias = (position_ids - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)


def _conti_input(data, seq_lens):
    data = [x[:l] for x, l in zip(data, seq_lens)]
    data = torch.cat(data, dim=0)
    return data


def _make_bias(seq_lens, history_lens, neg_val):
    full_seq_lens = seq_lens + history_lens
    max_seq_len = seq_lens.max().item()
    max_full_len = full_seq_lens.max().item()
    seq_ranges = [torch.arange(max_seq_len) for _ in seq_lens]
    for r, l in zip(seq_ranges, seq_lens):
        r[l:] = -max_full_len
    seq_ranges = torch.stack(seq_ranges, dim=0).cuda()
    kv_ranges = [torch.arange(max_full_len) for _ in full_seq_lens]
    kv_ranges = torch.stack(kv_ranges, 0).cuda()
    mask = kv_ranges[:, None, :] - seq_ranges[:, :, None] > history_lens[:, None, None]
    return mask.float() * neg_val


def _make_blocked_cache(
    batched_k,
    batched_v,
    seq_lens,
    history_lens,
    block_offsets,
    block_size,
    num_heads_k,
    feat_dim,
    feat_dim_v,
):
    max_blocks_nums = block_offsets.max() + 1
    full_seq_lens = seq_lens + history_lens
    blocked_k = batched_k.new_zeros(max_blocks_nums, block_size, num_heads_k, feat_dim)
    blocked_v = batched_v.new_zeros(
        max_blocks_nums, block_size, num_heads_k, feat_dim_v
    )

    for batch_id, offset in enumerate(block_offsets):
        ori_k = batched_k[batch_id]
        ori_v = batched_v[batch_id]
        seq_len = full_seq_lens[batch_id]
        for block_id, block_start in enumerate(range(0, seq_len, block_size)):
            block_off = offset[block_id]
            tmp_k = ori_k[block_start : block_start + block_size]
            tmp_v = ori_v[block_start : block_start + block_size]
            size = tmp_k.size(0)
            blocked_k[block_off, :size] = tmp_k
            blocked_v[block_off, :size] = tmp_v

    return blocked_k.permute(0, 2, 3, 1), blocked_v.permute(0, 2, 3, 1)


def _naive_attention(batched_q, batched_kv, bias):
    batched_k, batched_v = batched_kv
    # print("bias.shape = \n", bias.shape)
    # print("bias = \n", bias)

    num_heads_q = batched_q.shape[2]
    num_heads_k = batched_k.shape[2]
    head_dim = batched_q.shape[-1]
    group = num_heads_q // num_heads_k

    q = batched_q.transpose(1, 2)
    k = batched_k.permute(0, 2, 3, 1)
    v = batched_v.transpose(1, 2)

    # expand group
    k = k.unsqueeze(2).expand(-1, -1, group, -1, -1).flatten(1, 2)
    v = v.unsqueeze(2).expand(-1, -1, group, -1, -1).flatten(1, 2)

    qk = torch.matmul(q, k) / math.sqrt(head_dim)
    attn_weight = qk + bias[:, None]
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    attn_weight = attn_weight.to(q.dtype)
    attn_output = torch.matmul(attn_weight, v)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output

def _start_loc(seq_lens):
    seq_sum = seq_lens.cumsum(0)
    start_loc = torch.cat([seq_sum.new_zeros(1), seq_sum[:-1]], dim=0)
    return start_loc


def _batched_q(seq_lens, num_heads_q, feat_dim, dtype):
    torch.manual_seed(123)
    batch_size = len(seq_lens)
    max_seq_len = seq_lens.max().item()
    return torch.randn(
        batch_size, max_seq_len, num_heads_q, feat_dim, dtype=dtype, device="cuda"
    )


def _batched_kv(seq_lens, history_lens, num_heads_k, feat_dim, feat_dim_v, dtype):
    torch.manual_seed(123)
    batch_size = len(seq_lens)
    full_seq_lens = seq_lens + history_lens
    max_seq_len = full_seq_lens.max().item()
    k = torch.rand(
        batch_size, max_seq_len, num_heads_k, feat_dim, dtype=dtype, device="cuda"
    )
    v = torch.rand(
        batch_size, max_seq_len, num_heads_k, feat_dim_v, dtype=dtype, device="cuda"
    )
    return k, v


def _conti_q(seq_lens, batched_q):
    return _conti_input(batched_q, seq_lens)


def _block_offsets(seq_lens, history_lens, block_size):
    full_seq_lens = seq_lens + history_lens
    batch_size = full_seq_lens.size(0)
    num_blocks = (full_seq_lens + block_size - 1) // block_size

    offset = [
        torch.arange(size) * batch_size + idx for idx, size in enumerate(num_blocks)
    ]
    max_len = max(len(o) for o in offset)
    new_offset = offset[0].new_zeros(batch_size, max_len)
    for o, no in zip(offset, new_offset):
        len_o = o.size(0)
        no[:len_o] = o

    print(new_offset)

    return new_offset.cuda()


def _conti_kv(batched_kv, seq_lens, history_lens):
    full_seq_lens = seq_lens + history_lens
    conti_k = _conti_input(batched_kv[0], full_seq_lens)
    conti_v = _conti_input(batched_kv[1], full_seq_lens)
    return (conti_k, conti_v)


def _blocked_kv(
    batched_kv,
    seq_lens,
    history_lens,
    block_offsets,
    block_size,
    num_heads_k,
    feat_dim,
    feat_dim_v,
):
    batched_k, batched_v = batched_kv
    return _make_blocked_cache(
        batched_k,
        batched_v,
        seq_lens,
        history_lens,
        block_offsets,
        block_size,
        num_heads_k,
        feat_dim,
        feat_dim_v,
    )


def _mask(seq_lens, history_lens):
    neg_val = -1e30
    return _make_bias(seq_lens, history_lens, neg_val)


def _gt(batched_q, batched_kv, mask):
    return _naive_attention(batched_q, batched_kv, mask)


def _conti_gt(gt, seq_lens):
    return _conti_input(gt, seq_lens)


def test_paged_attention(
    conti_q,
    blocked_kv,
    block_offsets,
    start_loc,
    seq_lens,
    history_lens,
    feat_dim_v,
):
    kv_seq_lens = seq_lens + history_lens
    max_seq_len = seq_lens.max().item()

    blocked_k, blocked_v = blocked_kv
    out = conti_q.new_empty(*conti_q.shape[:-1], feat_dim_v)

    paged_decode_attention_fwd(
        conti_q,
        blocked_k,
        blocked_v,
        out,
        block_offsets=block_offsets,
        q_start_loc=start_loc,
        q_seqlens=seq_lens,
        kv_seqlens=kv_seq_lens,
        max_seqlen=max_seq_len,
    )


def test():
    dtype = torch.float16
    feat_dim = 96
    feat_dim_v = 64
    num_heads_q = 16
    num_heads_k = 4
    seq_lens = torch.tensor([1], device="cuda")
    start_loc = _start_loc(seq_lens)
    block_size = 16
    history_lens = torch.tensor([32], device="cuda")

    batched_q = _batched_q(seq_lens, num_heads_q, feat_dim, dtype)
    batched_kv = _batched_kv(
        seq_lens, history_lens, num_heads_k, feat_dim, feat_dim_v, dtype
    )
    conti_q = _conti_q(seq_lens, batched_q)
    block_offsets = _block_offsets(seq_lens, history_lens, block_size)
    conti_kv = _conti_kv(batched_kv, seq_lens, history_lens)
    blocked_kv = _blocked_kv(
        batched_kv,
        seq_lens,
        history_lens,
        block_offsets,
        block_size,
        num_heads_k,
        feat_dim,
        feat_dim_v,
    )
    mask = _mask(seq_lens, history_lens)
    # _conti_gt(_gt(batched_q, batched_kv, mask), seq_lens)
    kv_seq_lens = seq_lens + history_lens
    max_seq_len = seq_lens.max().item()

    blocked_k, blocked_v = blocked_kv

    out = conti_q.new_empty(*conti_q.shape[:-1], feat_dim_v)

    
    paged_decode_attention_fwd(
        conti_q,
        blocked_k,
        blocked_v,
        out,
        block_tables=block_offsets,
        seq_lens=kv_seq_lens,
        block_size = block_size,
        max_seqlen=max_seq_len,
    )

    ref_output = conti_q.new_empty(*conti_q.shape[:-1], feat_dim_v)
    num_queries_per_kv = num_heads_q // num_heads_k
    scale = float(1.0 / (feat_dim**0.5))
    ref_single_query_cached_kv_attention(
        ref_output,
        conti_q,
        num_queries_per_kv,
        blocked_k, #  (num_blocks, num_heads, head_size , block_size)
        blocked_v, # (num_blocks, num_heads, head_size, block_size)
        block_offsets,
        kv_seq_lens,
        scale,
        None,
    )

    print("max diff", (out - ref_output).abs().max())
    torch.testing.assert_close(ref_output, out, atol=1e-3, rtol=1e-5)




    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["op"],
            x_vals=["fwd"],
            line_arg="provider",
            line_vals=["triton", "pytorch"],
            line_names=["Triton", "PyTorch"],
            ylabel="ms",
            plot_name="",
            args={},
        )
    )

    @triton.testing.perf_report(configs)
    def bench_fn(op, provider, device="cuda"):
        warmup = 100
        rep = 200

        if "triton" in provider:
            # fn = lambda: test_paged_attention(conti_q, blocked_kv, block_offsets, start_loc, seq_lens, history_lens, feat_dim_v)
            fn = lambda: paged_decode_attention_fwd(
                conti_q,
                blocked_k,
                blocked_v,
                out,
                block_tables=block_offsets,
                seq_lens=kv_seq_lens,
                block_size = block_size,
                max_seqlen=max_seq_len,
            )
        if "pytorch" in provider:
            fn = lambda: ref_single_query_cached_kv_attention(
                ref_output,
                conti_q,
                num_queries_per_kv,
                blocked_k, #  (num_blocks, num_heads, head_size , block_size)
                blocked_v, # (num_blocks, num_heads, head_size, block_size)
                block_offsets,
                kv_seq_lens,
                scale,
                None,
            )

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    test()
