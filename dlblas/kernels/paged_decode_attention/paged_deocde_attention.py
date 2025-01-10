# modify from: https://github.com/InternLM/lmdeploy/blob/v0.6.1/lmdeploy/pytorch/kernels/cuda/pagedattention.py
import torch
from torch import Tensor
import triton
import triton.language as tl
from packaging import version
from dlblas.utils import logger
from dlblas.utils.device_utils import is_mlu_592

TRITON_VERSION = version.parse(triton.__version__)

assert TRITON_VERSION >= version.parse("2.1.0")

if TRITON_VERSION >= version.parse("3.0.0"):

    @triton.jit
    def tanh(x):
        """tanh."""
        return 2 * tl.sigmoid(2 * x) - 1

    fast_expf = tl.math.exp
    fast_dividef = tl.math.fdiv
else:
    tanh = tl.math.tanh
    fast_expf = tl.math.fast_expf
    fast_dividef = tl.math.fast_dividef


def get_autotune_config():
    if is_mlu_592():
        return [triton.Config({}, num_stages=s, num_warps=w) for s in [2] for w in [4]]
    else:
        return [
            triton.Config({}, num_stages=s, num_warps=w)
            for s in [2]
            for w in [4, 8, 16]
        ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["BLOCK_H", "BLOCK_N", "BLOCK_DMODEL", "BLOCK_DV"],
)
@triton.jit
def _fwd_grouped_split_kernel(
    Q,
    K,
    V,
    sm_scale,
    KV_seqlens,
    Block_offsets,
    Acc_out,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kp: tl.constexpr,
    stride_kbs: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vp: tl.constexpr,
    stride_vbs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ok: tl.constexpr,
    stride_obs: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_boffb,
    kv_group_num: tl.constexpr,
    window_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    num_heads_q: tl.constexpr,
    logit_softcapping: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DMODEL1: tl.constexpr,
):
    """first step kernel of split k attention."""
    cur_batch = tl.program_id(2)
    cur_kv_head = tl.program_id(0)
    split_k_id = tl.program_id(1)

    if BLOCK_H < kv_group_num:
        HEAD_PER_CTA: tl.constexpr = BLOCK_H
    else:
        HEAD_PER_CTA: tl.constexpr = kv_group_num
    cur_head = cur_kv_head * HEAD_PER_CTA + tl.arange(0, BLOCK_H)
    mask_h = cur_head < cur_kv_head * HEAD_PER_CTA + HEAD_PER_CTA
    mask_h = mask_h & (cur_head < num_heads_q)

    q_seqlen = 1
    kv_seqlen = tl.load(KV_seqlens + cur_batch)
    if kv_seqlen <= 0:
        return
    history_len = kv_seqlen - q_seqlen

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < head_size
    offs_d = offs_d % head_size
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < head_size_v
    offs_dv = offs_dv % head_size_v
    off_k = (
        cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd
        + offs_n[None, :] * stride_kbs
    )
    off_v = (
        cur_kv_head * stride_vh
        + offs_dv[None, :] * stride_vd
        + offs_n[:, None] * stride_vbs
    )

    off_q = (
        cur_batch * stride_qbs
        + cur_head[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=mask_h[:, None] & mask_d[None, :], other=0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    if BLOCK_DMODEL1 != 0:
        offs_d1 = BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL1)
        mask_d1 = offs_d1 < head_size
        offs_d1 = offs_d1 % head_size
        off_q1 = (
            cur_batch * stride_qbs
            + cur_head[:, None] * stride_qh
            + offs_d1[None, :] * stride_qd
        )
        q1 = tl.load(Q + off_q1, mask=mask_h[:, None] & mask_d1[None, :], other=0)
        off_k1 = (
            cur_kv_head * stride_kh
            + offs_d1[:, None] * stride_kd
            + offs_n[None, :] * stride_kbs
        )
        k1_ptrs = K + off_k1

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    num_total_blocks = tl.cdiv(kv_seqlen, BLOCK_N)
    BLOCK_PER_CTA = tl.cdiv(num_total_blocks, SPLIT_K)
    kv_len_per_prog = BLOCK_PER_CTA * BLOCK_N
    loop_start = kv_len_per_prog * split_k_id
    loop_end = tl.minimum(loop_start + kv_len_per_prog, kv_seqlen)

    # load block offset
    # dirty
    start_block_id = loop_start // BLOCK_N

    loop_start = start_block_id * BLOCK_N
    block_offset_ptrs += start_block_id
    for start_n in range(loop_start, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        b_offset = tl.load(block_offset_ptrs)
        block_offset_ptrs += 1

        # -- compute qk ----
        k = tl.load(k_ptrs + b_offset * stride_kp)
        if BLOCK_DMODEL1 != 0:
            k1 = tl.load(k1_ptrs + b_offset * stride_kp)

        v = tl.load(v_ptrs + b_offset * stride_vp)

        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if BLOCK_DMODEL1 != 0:
            qk += tl.dot(q1, k1)
        qk *= sm_scale
        if logit_softcapping > 0.0:
            qk = qk / logit_softcapping
            qk = tanh(qk)
            qk = qk * logit_softcapping
        
        # print("start_n:", start_n)
        # print("BLOCK_N:", BLOCK_N)
        # print("history_len:", history_len)
        # print("window_size:", window_size)
        # print("offs_n:", offs_n)

        # NOTE: inf - inf = nan, and nan will leads to error
        if start_n + BLOCK_N > history_len:
            qk_mask = history_len >= (start_n + offs_n)
            qk = tl.where(
                qk_mask[None, :],
                qk,
                -float("inf"),
            )

        # -- compute p, m_i and l_i
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = fast_expf(qk - m_i_new[:, None])
        alpha = fast_expf(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, 1)

        # -- update output accumulator --
        # scale acc
        acc = acc * alpha[:, None]

        # update acc
        p, v = _convert_pv(p, v)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # initialize pointers to output
    off_acc = (
        cur_batch * stride_obs
        + split_k_id * stride_ok
        + cur_head[:, None] * stride_oh
        + offs_dv[None, :] * stride_od
    )
    tl.store(Acc_out + off_acc, acc, mask=mask_h[:, None] & mask_dv[None, :])

    off_meta = (
        cur_batch * stride_obs
        + split_k_id * stride_ok
        + cur_head * stride_oh
        + head_size_v
    )
    tl.store(Acc_out + off_meta, m_i, mask=mask_h)
    tl.store(Acc_out + off_meta + 1, l_i, mask=mask_h)


@triton.jit
def _reduce_split_kernel(
    Acc,
    Out,
    stride_ak,
    stride_abs,
    stride_ah,
    stride_ad,
    stride_obs,
    stride_oh,
    stride_od,
    head_size_v: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """second step kernel of split k attention."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # initialize offsets
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_k = tl.arange(0, SPLIT_K)
    mask_dv = offs_dv < head_size_v

    offs_acc = (
        cur_batch * stride_abs
        + cur_head * stride_ah
        + offs_k[:, None] * stride_ak
        + offs_dv[None, :] * stride_ad
    )
    offs_mi = (
        cur_batch * stride_abs + cur_head * stride_ah + stride_ak * offs_k + head_size_v
    )

    acc_k = tl.load(Acc + offs_acc, mask=mask_dv[None, :], other=0.0)
    m_k = tl.load(Acc + offs_mi)
    l_k = tl.load(Acc + offs_mi + 1)

    m_max = tl.max(m_k, 0)
    alpha = fast_expf(m_k - m_max)
    acc_k = acc_k * alpha[:, None]
    l_k = l_k * alpha

    acc = tl.sum(acc_k, 0)
    l_sum = tl.sum(l_k, 0)
    acc = acc / l_sum

    out_offs = cur_batch * stride_obs + cur_head * stride_oh + offs_dv * stride_od
    tl.store(Out + out_offs, acc, mask=mask_dv)


def _get_convert_pv(nv_capability):
    """lazy load convert_pv."""
    if nv_capability[0] >= 8:

        @triton.jit
        def convert_pv(p, v):
            """convert pv."""
            p = p.to(v.dtype)
            return p, v

    else:

        @triton.jit
        def convert_pv(p, v):
            """convert pv."""
            v = v.to(p.dtype)
            return p, v

    return convert_pv


_convert_pv = None


def paged_decode_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    block_tables: Tensor,
    seq_lens: Tensor,
    block_size: int,
    max_seqlen: int,
    window_size: int = None,
    sm_scale: float = None,
    logit_softcapping: float = None,
):
    """Paged Attention forward.

    Args:
        q (Tensor): [batch, num_heads, headdim_qk]
        k (Tensor): [num_blocks, num_kv_heads, headdim_qk, block_size] 
        v (Tensor): [num_blocks, num_kv_heads, headdim_v, block_size]
        o (Tensor): [batch, num_heads, headdim_v]
        block_tables (Tensor): [batch, max_num_blocks_per_seq]
        seq_lens: [batch] (Tensor): Key/Value length for each data in batch.
        max_seqlen (int): The max input length.
        block_size (int): kv cache block size.
    """
    # print("seq_lens:", seq_lens)
    global _convert_pv
    if _convert_pv is None:
        nv_cap = torch.cuda.get_device_capability()
        _convert_pv = _get_convert_pv(nv_cap)

    if window_size is None:
        window_size = -1

    if logit_softcapping is None:
        logit_softcapping = -1.0

    def _get_block_d(Lk):
        """get block d."""
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DMODEL1 = 0
        if BLOCK_DMODEL != Lk:
            BLOCK_DMODEL = BLOCK_DMODEL // 2
            BLOCK_DMODEL1 = max(16, triton.next_power_of_2(Lk - BLOCK_DMODEL))
        BLOCK_DV = triton.next_power_of_2(Lv)
        return BLOCK_DMODEL, BLOCK_DMODEL1, BLOCK_DV

    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-2], v.shape[-2]
    assert Lq == Lk, Lv == o.shape[-1]

    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)
    batch, head = seq_lens.shape[0], q.shape[-2]
    kv_group_num = q.shape[-2] // k.shape[-3]

    BLOCK = block_size
    assert BLOCK >= 16
    if Lk > 512 and BLOCK > 32:
        logger.warning(
            f"`head_dim={Lk}` and `block_size={BLOCK}` "
            "might leads to bad performance. "
            "Please reduce `block_size`."
        )

    is_decoding = q.shape[0] == seq_lens.size(0)
    if is_decoding:
        SPLIT_K = 4
        acc = q.new_empty(batch, head, SPLIT_K, Lv + 2, dtype=torch.float32)
        BLOCK_DMODEL, BLOCK_DMODEL1, BLOCK_DV = _get_block_d(Lk)

        p2_kv_group_num = triton.next_power_of_2(kv_group_num)
        BLOCK_H = max(16, min(BLOCK, p2_kv_group_num))
        grid_1 = triton.cdiv(head, min(BLOCK_H, kv_group_num))
        grid = (
            grid_1,
            SPLIT_K,
            batch,
        )
        _fwd_grouped_split_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            seq_lens,
            block_tables,
            acc,
            stride_qbs=q.stride(-3),
            stride_qh=q.stride(-2),
            stride_qd=q.stride(-1),
            stride_kp=k.stride(-4),
            stride_kbs=k.stride(-1),
            stride_kh=k.stride(-3),
            stride_kd=k.stride(-2),
            stride_vp=v.stride(-4),
            stride_vbs=v.stride(-1),
            stride_vh=v.stride(-3),
            stride_vd=v.stride(-2),
            stride_ok=acc.stride(-2),
            stride_obs=acc.stride(-4),
            stride_oh=acc.stride(-3),
            stride_od=acc.stride(-1),
            stride_boffb=block_tables.stride(0),
            kv_group_num=kv_group_num,
            window_size=window_size,
            head_size=Lk,
            head_size_v=Lv,
            num_heads_q=head,
            logit_softcapping=logit_softcapping,
            SPLIT_K=SPLIT_K,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DV=BLOCK_DV,
            BLOCK_N=BLOCK,
            BLOCK_H=BLOCK_H,
            BLOCK_DMODEL1=BLOCK_DMODEL1,
        )

        num_warps = 4
        grid = (batch, head)
        _reduce_split_kernel[grid](
            acc,
            o,
            stride_ak=acc.stride(-2),
            stride_abs=acc.stride(-4),
            stride_ah=acc.stride(-3),
            stride_ad=acc.stride(-1),
            stride_obs=o.stride(-3),
            stride_oh=o.stride(-2),
            stride_od=o.stride(-1),
            SPLIT_K=SPLIT_K,
            head_size_v=Lv,
            BLOCK_DV=BLOCK_DV,
            num_warps=num_warps,
            num_stages=1,
        )
