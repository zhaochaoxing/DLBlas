# Copyright (c) 2025, DeepLink.
# https://github.com/InternLM/lmdeploy/blob/v0.6.1/tests/pytorch/kernel/test_rms_norm.py
# import torch_mlu
# import torch_mlu.utils.gpu_migration
import torch
import triton

from dlblas.kernels.rms_norm import rms_norm

DEVICE = 'cpu'
TEST_CPU = True
def change_env():
    global DEVICE
    if TEST_CPU:
        from triton.backends.triton_shared.driver import CPUDriver

        def select_cpu_backend():
            triton.runtime.driver.set_active(CPUDriver())

        select_cpu_backend()
        DEVICE = 'cpu'
    else:
        from dlblas.utils.device_utils import get_idle_device
        DEVICE = torch.device(get_idle_device())
        torch.cuda.set_device(DEVICE)

    print(f"zmz debug device={triton.runtime.driver.active.get_current_target()}, DEVICE={DEVICE}")

change_env()


import time

def cpu_do_bench(fn, warmup=25, rep=100):
    # 预热
    for _ in range(warmup):
        fn()
    
    # 实际测量
    start_time = time.perf_counter()
    for _ in range(rep):
        fn()
    end_time = time.perf_counter()
    
    # 计算平均时间 (毫秒)
    return (end_time - start_time) * 1000 / rep



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
    input = torch.rand(4, 8, dtype=dtype, device='cpu')
    weight = torch.rand(8, dtype=dtype, device='cpu')
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
    def bench_fn(op, provider, device='cpu'):
        warmup = 100
        rep = 200

        if 'triton' in provider:
            fn = lambda: rms_norm(input, weight, eps)
        if 'pytorch' in provider:
            fn = lambda: _gt(input, weight, eps)

        # ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        if device == 'cuda':
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        else:
            print('input is cpu, use cpu_do_bench')
            ms = cpu_do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
