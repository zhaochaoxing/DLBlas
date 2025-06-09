# Copyright (c) 2025, DeepLink.
# https://github.com/InternLM/lmdeploy/blob/v0.6.1/tests/pytorch/kernel/test_rms_norm.py
# import torch_mlu
# import torch_mlu.utils.gpu_migration
import torch
import triton

from dlblas.kernels.rms_norm import rms_norm


def _gt(input, weight, eps):
    input_dtype = input.dtype
    input = input.to(torch.float32)
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    return weight * input.to(input_dtype)


def test_rms_norm(input, weight, eps):
    return rms_norm(input, weight, eps)


def test():
    dtype = torch.float16
    input = torch.rand(4, 8, dtype=dtype, device='npu')
    weight = torch.rand(8, dtype=dtype, device='npu')
    eps = 1e-6

    gt = _gt(input, weight, eps)
    tt = test_rms_norm(input, weight, eps)

    print('max diff', (gt - tt).abs().max())

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=['op'],
            x_vals=['fwd'],
            line_arg='provider',
            line_vals=['triton', 'pytorch'],
            line_names=['Triton', 'PyTorch'],
            ylabel='ms',
            plot_name='',
            args={},
        ))

    @triton.testing.perf_report(configs)
    def bench_fn(op, provider, device='npu'):
        warmup = 100
        rep = 200

        if 'triton' in provider:
            fn = lambda: rms_norm(input, weight, eps)
        if 'pytorch' in provider:
            fn = lambda: _gt(input, weight, eps)

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
