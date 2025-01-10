import math
import random
import time
import csv

import torch
import sys
import io

# CSV 文件的路径
CSV_FILE_PATH = "/home/aigc/PRJ/dlBLAS/dlblas/kernels/context_flahsattention/benchmark_results.csv"

sys.path.insert(0, "/home/aigc/PRJ/dlBLAS/dlblas/kernels/context_flahsattention")

from context_flashattention import context_attention_fwd
from xformers import ops as xops
from typing import Any, Dict, List, Optional, Tuple, Type
from xformers.ops.fmha.attn_bias import (AttentionBias,
                                         BlockDiagonalCausalMask,
                                         BlockDiagonalMask,
                                         LowerTriangularMaskWithTensorBias)

NUM_HEADS = [64]
NUM_QUERIES_PER_KV = [1, 8, 64]
HEAD_SIZES = [128, 96, 24]
HEAD_VSIZES = [128, 96, 24]
DTYPES = [torch.float16]
CUDA_DEVICES = ["cuda:1"]
KV_CACHE_DTYPES = ["auto"]

import numpy as np

def seed_everything(seed=0):
    """
    Sets the seed for all common random number generators to ensure reproducibility.
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU

# 使用该函数设置种子
seed_everything(0)

def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    seq_lens: List[int],
) -> List[AttentionBias]:
    attn_biases: List[AttentionBias] = []
    for seq_len in seq_lens:
        bias = torch.arange(seq_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(seq_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        # Calculate a matrix where each element represents ith element- jth
        # element.
        bias = bias[None, :] - bias[:, None]

        padded_len = (seq_len + 7) // 8 * 8
        num_heads = alibi_slopes.shape[0]
        bias = torch.empty(
            1,  # batch size
            num_heads,
            seq_len,
            padded_len,
            device=alibi_slopes.device,
            dtype=dtype,
        )[:, :, :, :seq_len].copy_(bias)
        bias.mul_(alibi_slopes[:, None, None])
        if num_heads != num_kv_heads:
            bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
        attn_biases.append(LowerTriangularMaskWithTensorBias(bias))

    return attn_biases

def capture_output(func, *args, **kwargs):
    """捕获函数的标准输出并返回其输出内容"""
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()  # 替换标准输出为StringIO对象
    try:
        func(*args, **kwargs)
        output = sys.stdout.getvalue()  # 获取打印的内容
    finally:
        sys.stdout = original_stdout  # 恢复原始标准输出
    return output


def extract_times_from_output(output: str):
    """从捕获的打印输出中提取 Triton 时间和 xFormers 时间"""
    triton_time = None
    xformers_time = None
    for line in output.split("\n"):
        if "triton Time:" in line:
            triton_time = float(line.split("triton Time:")[1].strip().split()[0])
        elif "xformers Time:" in line:
            xformers_time = float(line.split("xformers Time:")[1].strip().split()[0])
    return triton_time, xformers_time


# @pytest.mark.parametrize("num_heads", NUM_HEADS)
# @pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
# @pytest.mark.parametrize("head_size", HEAD_SIZES)
# @pytest.mark.parametrize("headv_size", HEAD_VSIZES)
# @pytest.mark.parametrize("dtype", DTYPES)
# @pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
# @pytest.mark.parametrize("device", CUDA_DEVICES)
# @torch.inference_mode()
def test_contexted_kv_attention_alibi(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    headv_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> None:

    seed_everything(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.cuda.set_device(device)

    def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
        # Fork from: vllm/vllm/model_executor/models/bloom.py#L44
        closest_power_of_2 = 2**math.floor(math.log2(total_num_heads))
        base = torch.tensor(
            2**(-(2**-(math.log2(closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != total_num_heads:
            extra_base = torch.tensor(
                2**(-(2**-(math.log2(2 * closest_power_of_2) - 3))),
                dtype=torch.float32,
            )
            num_remaining_heads = min(closest_power_of_2,
                                      total_num_heads - closest_power_of_2)
            extra_powers = torch.arange(start=1,
                                        end=1 + 2 * num_remaining_heads,
                                        step=2,
                                        dtype=torch.int32)
            slopes = torch.cat(
                [slopes, torch.pow(extra_base, extra_powers)], dim=0)
        return slopes

    alibi_slopes = _get_alibi_slopes(num_heads).to(device)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    block_size = 32
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, headv_size, dtype=dtype)

    # kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    # kv.uniform_(-1e-3, 1e-3)
    # key, value = kv.unbind(dim=1)
    key = torch.empty(sum(seq_lens), num_kv_heads, head_size, dtype=dtype).uniform_(-1e-3, 1e-3)
    value = torch.empty(sum(seq_lens), num_kv_heads, headv_size, dtype=dtype).uniform_(-1e-3, 1e-3)
    cache_dtype = dtype
    k_cache = torch.zeros(cache_size,   
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=cache_dtype)
    v_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          headv_size,
                          dtype=cache_dtype)
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, headv_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[:BS * max_block_per_request].view(
        BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens[:-1],
                                            dtype=torch.long),
                               dim=0)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1],
                                                dtype=torch.long),
                                   dim=0)
    for i in range(BS):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] +
                                            j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] +
                                              b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             key[start_loc:end_loc])
            v_cache.view(-1, num_kv_heads,
                         headv_size)[start_slot:end_slot].copy_(
                             value[start_loc:end_loc])
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size, block_size]
    k_cache = k_cache.view(-1, block_size, num_kv_heads, 
                           head_size,).permute(0, 2, 3, 1).contiguous()
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = v_cache.view(-1, block_size, num_kv_heads,
                           headv_size).permute(0, 2, 3, 1).contiguous()

    # Warm up the Triton kernel by calling it once before actually measuring
    # generation time
    context_attention_fwd(query,
                          k,
                          v,
                          output,
                          k_cache,
                          v_cache,
                          block_table,
                          b_start_loc,
                          b_seq_len,
                          b_ctx_len,
                          max_input_len,
                          alibi_slopes=alibi_slopes)
    torch.cuda.synchronize()
    start_time = time.time()
    context_attention_fwd(query,
                          k,
                          v,
                          output,
                          k_cache,
                          v_cache,
                          block_table,
                          b_start_loc,
                          b_seq_len,
                          b_ctx_len,
                          max_input_len,
                          alibi_slopes=alibi_slopes)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time)*1000:.2f} ms")
    scale = float(1.0 / (head_size**0.5))

    # NOTE(DefTruth): In order to reuse _make_alibi_bias function,
    # we have to pad query tensor before MQA/GQA expanding.
    if query.shape[0] != key.shape[0]:
        query_pad = torch.empty(sum(seq_lens),
                                num_heads,
                                head_size,
                                dtype=dtype)
        query_pad.uniform_(-1e-3, 1e-3)
        seq_start = 0
        query_start = 0
        for i, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
            seq_end = seq_start + seq_len
            query_end = query_start + query_len
            query_pad[seq_start:seq_end, ...] = torch.cat([
                torch.zeros(
                    seq_len - query_len, num_heads, head_size, dtype=dtype),
                query[query_start:query_end, ...]
            ],dim=0)

            seq_start += seq_len
            query_start += query_len
        query = query_pad

    if num_kv_heads != num_heads:
        # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
        # project the key and value tensors to the desired number of
        # heads.
        #
        # see also: vllm/model_executor/layers/attention.py
        query = query.view(query.shape[0], num_kv_heads, num_queries_per_kv,
                           query.shape[-1])
        key = key[:, :, None, :].expand(key.shape[0], num_kv_heads,
                                        num_queries_per_kv, key.shape[-1])
        value = value[:, :,
                      None, :].expand(value.shape[0], num_kv_heads,
                                      num_queries_per_kv, value.shape[-1])

    query = query.unsqueeze(0)
    key = key.unsqueeze(0)
    value = value.unsqueeze(0)

    attn_bias = _make_alibi_bias(alibi_slopes, num_kv_heads, dtype, seq_lens)
    output_ref = torch.empty_like(output)
    seq_start = 0
    query_start = 0
    start_time = time.time()
    # Attention with alibi slopes.
    # FIXME(DefTruth): Because xformers does not support dynamic sequence
    # lengths with custom attention bias, we process each prompt one by
    # one. This is inefficient, especially when we have many short prompts.
    # modified from: vllm/attention/backends/xformers.py#L343
    for i, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
        seq_end = seq_start + seq_len
        query_end = query_start + query_len
        out = xops.memory_efficient_attention_forward(query[:,
                                                            seq_start:seq_end],
                                                      key[:,
                                                          seq_start:seq_end],
                                                      value[:,
                                                            seq_start:seq_end],
                                                      attn_bias=attn_bias[i],
                                                      p=0.0,
                                                      scale=scale)
        out = out.view_as(value[:, seq_start:seq_end]).view(
            seq_len, num_heads, headv_size)
        output_ref[query_start:query_end, ...].copy_(out[seq_len - query_len:,
                                                         ...])
        seq_start += seq_len
        query_start += query_len
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"xformers Time: {(end_time - start_time)*1000:.2f} ms")
    atol = 1e-3 if "fp8" in kv_cache_dtype else 1e-6
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)

# Main function to run all parameter combinations
def main():
     # 初始化 CSV 文件并写入表头
    with open(CSV_FILE_PATH, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "num_heads",
            "num_queries_per_kv",
            "head_size",
            "headv_size",
            "dtype",
            "kv_cache_dtype",
            "device",
            "triton_time_ms",
            "xformers_time_ms"
        ])
    # Iterate through all parameter combinations
    for num_heads in NUM_HEADS:
        for num_queries_per_kv in NUM_QUERIES_PER_KV:
            for head_size in HEAD_SIZES:
                for headv_size in HEAD_VSIZES:
                    for dtype in DTYPES:
                        for kv_cache_dtype in KV_CACHE_DTYPES:
                            for device in CUDA_DEVICES:
                                # Call the test function

                                # Print the current parameter configuration
                                config_description = (
                                    f"Current Configuration: "
                                    f"num_heads={num_heads}, "
                                    f"num_queries_per_kv={num_queries_per_kv}, "
                                    f"head_size={head_size}, "
                                    f"headv_size={headv_size}, "
                                    f"dtype={dtype}, "
                                    f"kv_cache_dtype={kv_cache_dtype}, "
                                    f"device={device}"
                                )
                                print(config_description)
                                test_contexted_kv_attention_alibi(
                                    num_heads,
                                    num_queries_per_kv,
                                    head_size,
                                    headv_size,
                                    dtype,
                                    kv_cache_dtype,
                                    device,
                                )
                                # 捕获函数输出并提取时间
                                try:
                                    output = capture_output(
                                        test_contexted_kv_attention_alibi,
                                        num_heads,
                                        num_queries_per_kv,
                                        head_size,
                                        headv_size,
                                        dtype,
                                        kv_cache_dtype,
                                        device,
                                    )
                                    triton_time, xformers_time = extract_times_from_output(
                                        output)
                                except Exception as e:
                                    print(f"Error during execution: {e}")
                                    triton_time, xformers_time = None, None

                                # 将结果写入 CSV 文件
                                with open(CSV_FILE_PATH, mode="a", newline="") as file:
                                    writer = csv.writer(file)
                                    writer.writerow([
                                        num_heads,
                                        num_queries_per_kv,
                                        head_size,
                                        headv_size,
                                        dtype,
                                        kv_cache_dtype,
                                        device,
                                        triton_time,
                                        xformers_time
                                    ])

if __name__ == "__main__":
    main()