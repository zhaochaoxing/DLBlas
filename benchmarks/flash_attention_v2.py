# Copyright (c) 2025, DeepLink.
import torch
import torch.nn.functional as F
import triton

import dlblas
from dlblas.kernels.flash_attention_v2 import _flash_attn_forward as flash_attention_v2
from dlblas.utils.device_utils import get_idle_device, is_cuda, is_muxi

MUXI_CUDA = False #is_muxi() or is_cuda()


def test():
    device_ = 'npu'#torch.device(get_idle_device())
    # torch.cuda.set_device(device_)
    dtype = torch.float16
    if MUXI_CUDA:
        dtype = torch.float32

    seq_len, heads, dim = 25600, 32, 64
    query = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device_)
    key = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device_)
    value = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device_)
    cos = torch.rand([1, seq_len, dim], dtype=dtype, device=device_)
    sin = torch.rand([1, seq_len, dim], dtype=dtype, device=device_)

    tt_out, _, _ = flash_attention_v2(query, key, value)
    ref_out = F.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3).cpu(),
        key.permute(0, 2, 1, 3).cpu(),
        value.permute(0, 2, 1, 3).cpu(),
    ).permute(0, 2, 1, 3)

    print('TEST: ')
    tt_out = tt_out.cpu()
    print('max abs diff: ', torch.max(abs(tt_out - ref_out)))
    assert torch.allclose(tt_out, ref_out, atol=1e-2, rtol=0)

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=['op'],
            x_vals=['fwd'],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=['triton', 'torch'],
            ylabel='ms',
            plot_name=f"flashAttention(batchSize={1}, seqlen:{seq_len}, num_heads:{heads}, dim:{dim})",
            args={'SeqLen': seq_len},
        ))

    @triton.testing.perf_report(configs)
    def bench_fn(SeqLen, op, provider, device=device_):
        warmup = 100
        rep = 200

        if 'triton' in provider:
            fn = lambda: flash_attention_v2(query, key, value)

        if 'torch' in provider:
            fn = lambda: F.scaled_dot_product_attention(
                query.permute(0, 2, 1, 3).cpu(),
                key.permute(0, 2, 1, 3).cpu(),
                value.permute(0, 2, 1, 3).cpu(),
            ).permute(0, 2, 1, 3)

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
    print('sucessfully!')
