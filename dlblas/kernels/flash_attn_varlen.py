import torch
import torch.nn.functional as F
import triton
import triton.language as tl

if triton.__version__ >= "3.0.0":
    from triton.language.extra.cuda.libdevice import fast_expf as tl_exp
    from triton.language.extra.cuda.libdevice import fast_logf as tl_log
else:
    from triton.language.math import fast_expf as tl_exp
    from triton.language.math import fast_logf as tl_log

def flash_attn_varlen_forward_torch(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    softmax_scale, causal=False
):
    """
    Torch-based implementation of multi-head attention with variable length sequences.
    
    Args:
        q: Query tensor of shape [total_q, nheads_q, head_dim].
        k: Key tensor of shape [total_k, nheads_kv, head_dim].
        v: Value tensor of shape [total_k, nheads_kv, head_dim].
        cu_seqlens_q: Cumulative sequence lengths for queries [batch_size + 1].
        cu_seqlens_k: Cumulative sequence lengths for keys [batch_size + 1].
        max_seqlen_q: Maximum sequence length for queries.
        max_seqlen_k: Maximum sequence length for keys.
        nheads_q: Number of attention heads for queries.
        nheads_kv: Number of attention heads for keys/values.
        softmax_scale: Scaling factor for softmax.
        causal: Whether to apply causal masking.
    
    Returns:
        Output tensor of shape [total_q, nheads_q, head_dim].
    """
    batch_size = len(cu_seqlens_q) - 1
    head_dim_qk = q.size(-1)
    head_dim_v = v.size(-1)
    total_q = q.size(0)
    total_k = k.size(0)
    nheads_q = q.size(1) 
    nheads_kv = k.size(1)
    # Initialize output
    output = torch.zeros((total_q, nheads_q, head_dim_v), device = q.device)

    for b in range(batch_size):
        start_q, end_q = cu_seqlens_q[b], cu_seqlens_q[b + 1]
        start_k, end_k = cu_seqlens_k[b], cu_seqlens_k[b + 1]

        seq_len_q = end_q - start_q
        seq_len_k = end_k - start_k

        q_b = q[start_q:end_q]  # [seq_len_q, nheads_q, head_dim]
        k_b = k[start_k:end_k]  # [seq_len_k, nheads_kv, head_dim]
        v_b = v[start_k:end_k]  # [seq_len_k, nheads_kv, head_dim]

        heads_per_kv = nheads_q // nheads_kv
        kv_head_idx = torch.arange(nheads_q) // heads_per_kv  # [nheads_q]

        for h in range(nheads_q):
            q_h = q_b[:, h]  # [seq_len_q, head_dim]
            k_h = k_b[:, kv_head_idx[h]]  # [seq_len_k, head_dim]
            v_h = v_b[:, kv_head_idx[h]]  # [seq_len_k, head_dim]

            attn_scores = torch.matmul(q_h, k_h.T) * softmax_scale  # [seq_len_q, seq_len_k]

            if causal:
                causal_mask = torch.tril(torch.ones(seq_len_q, seq_len_k, device=attn_scores.device)).bool()
                attn_scores = torch.where(causal_mask, attn_scores, float('-inf'))

            attn_probs = F.softmax(attn_scores, dim=-1)  # [seq_len_q, seq_len_k]
            output_h = torch.matmul(attn_probs, v_h)  # [seq_len_q, head_dim]
            output[start_q:end_q, h] = output_h

    return output

@triton.jit
def _flash_attn_varlen_forward(
    q_ptr, 
    k_ptr, 
    v_ptr, 
    cu_seqlens_q_ptr, 
    cu_seqlens_k_ptr,
    max_seqlen_q, 
    max_seqlen_k, 
    nheads_q, 
    nheads_kv, 
    softmax_scale, 
    causal, 
    out_ptr,
    head_dim_qk: tl.constexpr,
    head_dim_v: tl.constexpr,  
    BLOCK_Q: tl.constexpr, 
    BLOCK_K: tl.constexpr
):
    batch_idx = tl.program_id(0) 
    head_idx = tl.program_id(1)  
    block_q_idx = tl.program_id(2)  

    start_q = tl.load(cu_seqlens_q_ptr + batch_idx)
    end_q = tl.load(cu_seqlens_q_ptr + batch_idx + 1)
    start_k = tl.load(cu_seqlens_k_ptr + batch_idx)
    end_k = tl.load(cu_seqlens_k_ptr + batch_idx + 1)

    seq_len_q = end_q - start_q
    seq_len_k = end_k - start_k

    if block_q_idx * BLOCK_Q >= seq_len_q or head_idx >= nheads_q:
        return

    heads_per_kv = nheads_q // nheads_kv
    kv_head_idx = head_idx // heads_per_kv

    q_offsets = (
        (block_q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)[:, None] + start_q) * nheads_q * head_dim_qk +
        head_idx * head_dim_qk +
        tl.arange(0, head_dim_qk)[None, :]
    )
    # tl.device_print("q_offsets", q_offsets)
    q = tl.load(
        q_ptr + q_offsets,
        mask=(
            tl.arange(0, BLOCK_Q)[:, None] + block_q_idx * BLOCK_Q < seq_len_q
        ),
        other=0.0
    )

    m_i = tl.full([BLOCK_Q], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    output = tl.zeros((BLOCK_Q, head_dim_v), dtype=tl.float32)

    for k_block_idx in range((seq_len_k + BLOCK_K -1) // BLOCK_K):
        k_offsets = (
            (k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)[:, None] + start_k) * nheads_kv * head_dim_qk +
            kv_head_idx * head_dim_qk +
            tl.arange(0, head_dim_qk)[None, :]
        )

        k = tl.load(
            k_ptr + k_offsets,
            mask=((tl.arange(0, BLOCK_K)[:, None] + k_block_idx * BLOCK_K) < seq_len_k),
            other=0.0
        )

        v_offsets = (
            (k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)[:, None] + start_k) * nheads_kv * head_dim_v +
            kv_head_idx * head_dim_v +
            tl.arange(0, head_dim_v)[None, :]
        )
        
        v = tl.load(
            v_ptr + v_offsets,
            mask=((tl.arange(0, BLOCK_K)[:, None] + k_block_idx * BLOCK_K) < seq_len_k),
            other=0.0
        )
        
        attn_scores = tl.dot(q, tl.trans(k))
        if causal:
            q_pos = block_q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)[:, None]
            k_pos = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
            causal_mask = q_pos >= k_pos
            attn_scores = tl.where(causal_mask, attn_scores, float("-inf"))

        ### Online Softmax
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(attn_scores, 1))
        alpha = tl_exp((m_i - m_i_new) * softmax_scale)
        p = tl_exp(attn_scores * softmax_scale - m_i_new[:, None] * softmax_scale)

        # -- compute partial sumexpn before applying dropout
        p_sum = tl.sum(p, 1)

        # -- scale and update acc: acc *= alpha[:, None]--
        output *= alpha[:, None]
        output += tl.dot(p, v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + p_sum
        m_i = m_i_new

    output = output * (1.0 / l_i[:, None])
    l = m_i * softmax_scale + tl.log(l_i)

    out_offsets = (
        (block_q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)[:, None] + start_q) * nheads_q * head_dim_v +
        head_idx * head_dim_v +
        tl.arange(0, head_dim_v)[None, :]
    )
    out_block_ptr = out_ptr + out_offsets
    tl.store(out_block_ptr, output, mask=(tl.arange(0, BLOCK_Q)[:, None] + block_q_idx * BLOCK_Q < seq_len_q))

def generate_test_data(batch_size, max_seqlen_q, max_seqlen_k, nheads_q, nheads_kv, head_dim_qk, head_dim_v, device="cuda"):
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)

    for i in range(1, batch_size + 1):
        cu_seqlens_q[i] = cu_seqlens_q[i - 1] + torch.randint(1, max_seqlen_q + 1, (1,), device=device).item()
        cu_seqlens_k[i] = cu_seqlens_k[i - 1] + torch.randint(1, max_seqlen_k + 1, (1,), device=device).item()

    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()

    q = torch.randn(total_q, nheads_q, head_dim_qk, dtype=torch.float32, device='cuda')
    k = torch.randn(total_k, nheads_kv, head_dim_qk, dtype=torch.float32, device='cuda')
    v = torch.randn(total_k, nheads_kv, head_dim_v, dtype=torch.float32, device='cuda')

    return total_q, total_k, q, k, v, cu_seqlens_q, cu_seqlens_k


def flash_attn_varlen_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=False, softmax_scale=None):
    """
    Triton-based implementation of multi-head attention with variable length sequences.
    
    Args:
        q: Query tensor of shape [total_q, nheads_q, head_dim].
        k: Key tensor of shape [total_k, nheads_kv, head_dim].
        v: Value tensor of shape [total_k, nheads_kv, head_dim].
        cu_seqlens_q: Cumulative sequence lengths for queries [batch_size + 1].
        cu_seqlens_k: Cumulative sequence lengths for keys [batch_size + 1].
        max_seqlen_q: Maximum sequence length for queries.
        max_seqlen_k: Maximum sequence length for keys.
        softmax_scale: Scaling factor for softmax.
        causal: Whether to apply causal masking.
    
    Returns:
        Output tensor of shape [total_q, nheads_q, head_dim].
    """
    # shape constraints
    total_q, nheads_q, headdim_qk = q.shape
    total_k, nheads_kv, _, = k.shape
    _, _, headdim_v = v.shape
    assert headdim_qk <= 128 and headdim_v <= 128, "FlashAttention only support head dimensions up to 128"
    assert headdim_qk in {16, 32, 64, 128} and headdim_v in {16, 32, 64, 128}
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    # assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert nheads_q % nheads_kv == 0, "Number of heads must be divisible by nheads_kv"
    assert q.is_cuda and k.is_cuda and v.is_cuda

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim_qk)

    batch_size = len(cu_seqlens_q) - 1

    BLOCK_Q = 16
    BLOCK_K = 16

    result = torch.zeros((total_q, nheads_q, headdim_v), device=q.device)

    grid = lambda META: (batch_size, nheads_q, triton.cdiv(max_seqlen_q, META["BLOCK_Q"]))
    _flash_attn_varlen_forward[grid](
        q, 
        k, 
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q, 
        max_seqlen_k,
        nheads_q, 
        nheads_kv,
        softmax_scale,
        causal,
        result,
        headdim_qk,
        headdim_v,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=1,
    )
    return result

def test():
    batch_size = 2
    max_seqlen_q = 256
    max_seqlen_k = 128
    nheads_q = 6
    nheads_kv = 2
    head_dim_qk = 128
    head_dim_v = 64
    causal = False
    softmax_scale = 1.0 / (head_dim_qk ** 0.5)

    cu_seqlens_q = torch.tensor([0, 64, 256], dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.tensor([0, 32, 96], dtype=torch.int32, device='cuda')
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()

    q = torch.randn(total_q, nheads_q, head_dim_qk, dtype=torch.float32, device='cuda')
    k = torch.randn(total_k, nheads_kv, head_dim_qk, dtype=torch.float32, device='cuda')
    v = torch.randn(total_k, nheads_kv, head_dim_v, dtype=torch.float32, device='cuda')

    result = torch.zeros((total_q, nheads_q, head_dim_v), device=q.device)

    tri_out = flash_attn_varlen_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, softmax_scale)
    ref_out = flash_attn_varlen_forward_torch(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal=False)
    
    # import ipdb; ipdb.set_trace();
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=1e-3)

    # only launch the kernel, no tensor preparation here to remove all overhead
    def triton_perf_fn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, softmax_scale):
        grid = lambda META: (batch_size, nheads_q, triton.cdiv(max_seqlen_q, META["BLOCK_Q"]))
        _flash_attn_varlen_forward[grid](
            q, 
            k, 
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q, 
            max_seqlen_k,
            nheads_q, 
            nheads_kv,
            softmax_scale,
            causal,
            result,
            head_dim_qk,
            head_dim_v,
            BLOCK_Q=32,
            BLOCK_K=32,
            num_warps=4,
            num_stages=1,
        )

    def torch_perf_fn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal=False):
        flash_attn_varlen_forward_torch(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal=False)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            # argument names to use as an x-axis for the plot
            x_names=["batch_size"],
            x_vals=[
                2,4,8
            ],  # different possible values for `x_name`
            line_arg="provider",
            # argument name whose value corresponds to a different line in the plot
            # possible values for `line_arg``
            line_vals=["torch", "triton"],
            # label name for the lines
            line_names=["torch", "Triton"],
            # line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="runtime(ms)",  # label name for the y-axis
            plot_name="flash-attentipn-performance",
            # name for the plot. Used also as a file name for saving the plot.
            args={},
        )
    )
    def benchmark(batch_size, provider):
        # batch_size = batch_size
        max_seqlen_q = 256
        max_seqlen_k = 128
        nheads_q = 6
        nheads_kv = 2
        head_dim_qk = 128
        head_dim_v = 64
        causal = False
        softmax_scale = 1.0 / (head_dim_qk ** 0.5)

        total_q, total_k, q, k, v, cu_seqlens_q, cu_seqlens_k = generate_test_data(
            batch_size, max_seqlen_q, max_seqlen_k, nheads_q, nheads_kv, head_dim_qk, head_dim_v
        )

        
        quantiles = [0.5, 0.2, 0.8]
        if provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch_perf_fn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal=False), quantiles=quantiles
            )
        if provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_perf_fn(
                    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, softmax_scale
                ),
                quantiles=quantiles,
            )
        return ms, max_ms, min_ms

    benchmark.run(show_plots=True, print_data=True)

if __name__ == "__main__":
    test()
